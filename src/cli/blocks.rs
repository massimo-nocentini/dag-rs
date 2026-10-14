//! `dag blocks`: builds, verifies and saves the [`BlockIndex`].
//!
//! Writes `<out>.blk.{create,spend,first,len,nin}.u32le`, `<out>.blk.json` (the
//! [`BlockReport`](crate::blocks::BlockReport)) and `<out>.json` (the run record). With
//! `--compare-bicliques DIR` it also compares the index with the reference files of the
//! `bicliques` lens (`DIR/blocks.u32x3`: `(first, |I|, |O|)` per block; `DIR/blk_start.u32`:
//! the first output of the block of each node, or `u32::MAX` for sources) and exits with 1 on
//! any difference. The run record also holds the [`BlockDistributions`] (log2 histograms of the
//! block sizes, the largest blocks, and per node range with `--buckets`) and the time of each
//! phase (open, build, save, summary, compare).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args as ClapArgs;
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use serde_json::json;
use webgraph::utils::MmapHelper;

use super::common::{
    GlobalArgs, GraphOpt, OutOpt, StatsOpt, concurrent_logger, exit_status, logger,
};
use crate::blocks::BlockIndex;
use crate::io::{FileInfo, NONE, RunInfo, suffixed};
use crate::summary::{DistConf, DistReport, distribution_with};
use crate::window::Window;

/// `dag blocks`.
#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Input graph.
    #[command(flatten)]
    pub graph: GraphOpt,
    /// Output prefix.
    #[command(flatten)]
    pub out: OutOpt,
    /// Compare with `DIR/blocks.u32x3` and `DIR/blk_start.u32` (the `bicliques` lens).
    #[arg(long)]
    pub compare_bicliques: Option<PathBuf>,
    /// Aggregate measures (`--buckets`, `--top`).
    #[command(flatten)]
    pub stats: StatsOpt,
}

/// The distributions of the block sizes ([`crate::summary`]): one entry per block, with the
/// absolute block id as id and its first output as node (`argmax_node`, `top[].node`; the
/// buckets are node ranges). They give the log2 histograms, the largest blocks and, with
/// `--buckets`, the number of transactions (`count`), outputs (`len.sum`), spent inputs
/// (`nin.sum`) and arcs (`arcs.sum`) per node range.
#[derive(Debug, serde::Serialize)]
pub struct BlockDistributions {
    /// `len[B] = |O(B)|`: outputs of the transaction.
    pub len: DistReport,
    /// `nin[B] = |I(B)|`: inputs of the transaction.
    pub nin: DistReport,
    /// `len[B]·nin[B]`: arcs of the biclique.
    pub arcs: DistReport,
}

/// Computes the [`BlockDistributions`] of `idx` (three parallel scans of the block arrays).
pub fn distributions(idx: &BlockIndex, conf: &DistConf<'_>) -> BlockDistributions {
    let first = |b: usize| idx.first[b] as usize;
    let id = |b: usize| b;
    let one = |f: &(dyn Fn(usize) -> f64 + Sync)| {
        distribution_with(idx.nb, |b| Some(f(b)), &id, Some(&first), conf)
    };
    BlockDistributions {
        len: one(&|b| f64::from(idx.len[b])),
        nin: one(&|b| f64::from(idx.nin[b])),
        arcs: one(&|b| f64::from(idx.len[b]) * f64::from(idx.nin[b])),
    }
}

/// Result of [`compare_bicliques`].
#[derive(Debug, Default, serde::Serialize)]
pub struct BicliqueComparison {
    /// Blocks in the reference.
    pub reference_blocks: usize,
    /// Blocks whose `(first, |I|, |O|)` differs.
    pub block_mismatches: u64,
    /// First differing block.
    pub first_block_mismatch: Option<usize>,
    /// Nodes whose `blk_start` differs from `first[create[v]]`.
    pub node_mismatches: u64,
    /// First differing node.
    pub first_node_mismatch: Option<usize>,
}

impl BicliqueComparison {
    /// Whether the index and the reference agree.
    pub fn ok(&self) -> bool {
        self.block_mismatches == 0 && self.node_mismatches == 0
    }
}

fn map_u32(path: &Path) -> Result<MmapHelper<u32>> {
    MmapHelper::<u32>::mmap(path, mmap_rs::MmapFlags::SEQUENTIAL)
        .with_context(|| format!("cannot map {}", path.display()))
}

fn count_mismatches(len: usize, bad: impl Fn(usize) -> bool + Sync) -> (u64, Option<usize>) {
    (0..len)
        .into_par_iter()
        .with_min_len(1 << 16)
        .fold(
            || (0u64, None::<usize>),
            |(c, f), i| {
                if bad(i) {
                    (c + 1, Some(f.map_or(i, |f: usize| f.min(i))))
                } else {
                    (c, f)
                }
            },
        )
        .reduce(
            || (0, None),
            |a, b| {
                let f = match (a.1, b.1) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    (x, y) => x.or(y),
                };
                (a.0 + b.0, f)
            },
        )
}

/// Compares `idx` with the `bicliques` lens files in `dir`.
pub fn compare_bicliques(idx: &BlockIndex, dir: &Path) -> Result<BicliqueComparison> {
    let blocks = map_u32(&dir.join("blocks.u32x3"))?;
    let start = map_u32(&dir.join("blk_start.u32"))?;
    let rec: &[u32] = blocks.as_ref();
    let bs: &[u32] = start.as_ref();
    anyhow::ensure!(rec.len() % 3 == 0, "blocks.u32x3 has {} words", rec.len());
    anyhow::ensure!(
        bs.len() == idx.n,
        "blk_start.u32 has {} entries, the graph has {} nodes",
        bs.len(),
        idx.n
    );
    let nb_ref = rec.len() / 3;
    let mut cmp = BicliqueComparison {
        reference_blocks: nb_ref,
        ..Default::default()
    };
    let nb = idx.nb.max(nb_ref);
    // used as timers: the comparisons are single parallel scans bound by reading the files
    let mut pl = logger("block");
    pl.start(format!("comparing {nb} blocks with {}", dir.display()));
    let (c, f) = count_mismatches(nb, |b| {
        b >= idx.nb
            || b >= nb_ref
            || rec[3 * b] != idx.first[b]
            || rec[3 * b + 1] != idx.nin[b]
            || rec[3 * b + 2] != idx.len[b]
    });
    cmp.block_mismatches = c;
    cmp.first_block_mismatch = f;
    pl.done_with_count(nb);
    let mut pl = logger("node");
    pl.start("comparing blk_start with first[create[v]]");
    let (c, f) = count_mismatches(idx.n, |v| {
        let want = match idx.create[v] {
            NONE => NONE,
            b => idx.first[b as usize],
        };
        bs[v] != want
    });
    cmp.node_mismatches = c;
    cmp.first_node_mismatch = f;
    pl.done_with_count(idx.n);
    Ok(cmp)
}

/// Runs `dag blocks`.
pub fn run(args: &Args, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = args.out.prepare()?.to_owned();
    let cg = args.graph.open(g)?;
    let n = cg.num_nodes();
    let buckets = args.stats.buckets.load(n)?;
    let mut info = RunInfo::new(
        Some(args.graph.graph.clone()),
        n,
        cg.num_arcs,
        Window::full(n),
    );
    info.phases_s.add("open", t0.elapsed().as_secs_f64());
    let mut budget = g.budget();
    // create + spend (8 B/node) before pass 1, first + len + nin (12 B/block) once nb is known
    let tb = Instant::now();
    let (idx, report) = {
        let mut pl = concurrent_logger("node");
        let r = BlockIndex::build_budgeted(&cg, &mut pl, &mut budget)?;
        // the logger is dropped here, not at the end of the run, so that any final flush is
        // logged next to the build
        drop(pl);
        r
    };
    let build_s = tb.elapsed().as_secs_f64();
    info.phases_s.add("blocks", build_s);
    log::info!(
        "BlockIndex: nb = {}, sources = {}, sinks = {}, Σ len = {}, Σ nin = {}, Σ len·nin = {} (m = {}), max len = {}, max nin = {}",
        report.nb,
        report.sources,
        report.sinks,
        report.sum_len,
        report.sum_nin,
        report.sum_len_times_nin,
        report.m,
        report.max_len,
        report.max_nin
    );
    let save_s = info.phases_s.time("save", || -> Result<f64> {
        // used as a timer: saving 5 arrays (≈ 26 GB on the PG) is bound by the disk
        let ts = Instant::now();
        let mut pl = logger("node");
        pl.start(format!("saving {}.blk.*", out.display()));
        idx.save(&out)?;
        pl.done_with_count(n);
        Ok(ts.elapsed().as_secs_f64())
    })?;
    let dists = info.phases_s.time("summary", || {
        distributions(
            &idx,
            &DistConf {
                top: args.stats.top,
                buckets: buckets.as_ref(),
            },
        )
    });
    let mut ok = true;
    let comparison = match &args.compare_bicliques {
        Some(dir) => {
            let c = info
                .phases_s
                .time("compare", || compare_bicliques(&idx, dir))?;
            log::info!("comparison with {}: {:?}", dir.display(), c);
            ok &= c.ok();
            Some(c)
        }
        None => None,
    };
    for (name, len, meaning) in [
        (
            "create",
            n,
            "block whose outputs contain v (NONE for sources)",
        ),
        (
            "spend",
            n,
            "block whose outputs are succ(u) (NONE for sinks)",
        ),
        ("first", idx.nb, "first output of the block"),
        ("len", idx.nb, "number of outputs of the block"),
        ("nin", idx.nb, "number of inputs of the block"),
    ] {
        let sentinel = matches!(name, "create" | "spend").then_some("u32::MAX = NONE");
        info.files.push(FileInfo::new::<u32>(
            suffixed(&out, &format!("blk.{name}.u32le")),
            len,
            sentinel,
            meaning,
        ));
    }
    info.params = json!({
        "buckets": args.stats.buckets.buckets,
        "top": args.stats.top,
    });
    info.results = json!({
        "report": report,
        "distributions": dists,
        "build_s": build_s,
        "save_s": save_s,
        "build_and_save_s": build_s + save_s,
        "comparison": comparison,
    });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}
