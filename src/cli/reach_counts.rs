//! `dag reach-counts`: approximate proper ancestor/descendant counts (HyperLogLog).
//!
//! With a [`BlockIndex`] (`--blocks auto|PREFIX`) the counts come
//! from the decode-free block passes of [`counts_blocks`]; otherwise (`--blocks none`, or `auto`
//! on a graph that is not a biclique digraph) from the generic sweeps of [`counts_generic`]. The
//! hash function is [`SeededBuildHasher`] with `--seed` (recorded in the JSON), so runs are
//! reproducible and independent seeds give independent estimates
//! ([\[FFGM07\]](crate::references#ffgm07), [\[QKT16\]](crate::references#qkt16)).
//!
//! Outputs (indexed by `v − lo` for the window `[lo, hi)`):
//! - `<out>.anc.f32le`, `<out>.desc.f32le`: estimated proper counts per node (0 for sources,
//!   resp. sinks, of the window);
//! - block mode: `<out>.blk.anc.f32le`, `<out>.blk.desc.f32le`: estimated `|A(X)|` and `|D(X)|`
//!   for the blocks `X ∈ [blo, bhi)` whose first output lies in the window, indexed by `X − blo`;
//! - `<out>.json`: parameters (`log2m`, `seed`, rsd, hasher, estimator), the pass reports (time,
//!   peak live sketches, merges) and summary statistics (sum, mean, mean / |window|, maximum,
//!   zeros); in block mode also the check that `anc = 0` on every source of the window, i.e.
//!   every node whose creating block is ⊥, lies before the block window, or has no input inside
//!   the window (exit code 1 if it fails).
//!
//! The summary of each array ([`Summary`]) reports absolute ids (node ids, or block ids plus the
//! block's first output), a log2 histogram, the top-k entries and, with `--buckets`, one
//! distribution per node range.

use std::process::ExitCode;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Result;
use clap::Args as ClapArgs;
use rayon::prelude::*;
use serde_json::json;

use super::common::{
    BlocksOpt, GlobalArgs, GraphOpt, OutOpt, StatsOpt, WindowOpt, exit_status, logger,
};
use crate::blocks::BlockIndex;
use crate::io::{FileInfo, NONE, RunInfo, array_path, as_atomic_u64, write_array};
use crate::reach::hash::SeededBuildHasher;
use crate::reach::sketch::{
    self, Counts, ESTIMATE_BATCH, SketchConf, block_window, counts_blocks, counts_generic,
    sketch_bytes,
};
use crate::summary::{BucketDist, DistConf, Log2Hist, NodeOf, Top, distribution_with};
use crate::window::{Which, Window};

/// `dag reach-counts`.
#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Input graph.
    #[command(flatten)]
    pub graph: GraphOpt,
    /// Block index.
    #[command(flatten)]
    pub blocks: BlocksOpt,
    /// Window.
    #[command(flatten)]
    pub window: WindowOpt,
    /// Output prefix.
    #[command(flatten)]
    pub out: OutOpt,
    /// Ancestors, descendants or both.
    #[arg(long, value_enum, default_value = "both")]
    pub which: Which,
    /// log2 of the number of HyperLogLog registers.
    #[arg(long, default_value_t = 10)]
    pub log2m: u32,
    /// Hash seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Aggregate measures (`--buckets`, `--top`).
    #[command(flatten)]
    pub stats: StatsOpt,
}

/// Summary statistics of an estimate array: the [`Dist`](crate::summary::Dist) of the
/// estimates under the field names of the first version (`len`, `sum`, `mean`,
/// `mean_over_window`, `max`, `argmax`, `zeros`), plus the log2 histogram, the top-k entries
/// and, with `--buckets`, one distribution per node range.
///
/// Ids are absolute: node ids for the per-node arrays, block ids for the per-block arrays,
/// whose `argmax_node` and `top[].node` give the first output of the block (the node used for
/// bucketing).
#[derive(Debug, serde::Serialize)]
pub struct Summary {
    /// Number of entries.
    pub len: usize,
    /// Sum of the estimates (added in a fixed order: reproducible).
    pub sum: f64,
    /// Mean estimate.
    pub mean: f64,
    /// Mean estimate divided by the window size.
    pub mean_over_window: f64,
    /// Largest estimate.
    pub max: f64,
    /// Smallest absolute id (node or block) of the largest estimate.
    pub argmax: Option<usize>,
    /// For per-block arrays: the first output of the `argmax` block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argmax_node: Option<usize>,
    /// Entries equal to 0.
    pub zeros: u64,
    /// Log2 histogram of the estimates ([`crate::summary`] bins).
    pub log2_hist: Log2Hist,
    /// The largest estimates (value descending, id ascending).
    pub top: Vec<Top>,
    /// The same statistics per `--buckets` bucket.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub by_bucket: Option<Vec<BucketDist>>,
    /// Entries outside every bucket.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unbucketed: Option<u64>,
}

/// Computes the [`Summary`] of `a` for a window of `n_w` nodes: entry `i` has the absolute id
/// `id(i)` and, for per-block arrays, the node `node(i)`.
pub fn summarize(
    a: &[f32],
    n_w: usize,
    id: &(dyn Fn(usize) -> usize + Sync),
    node: Option<&NodeOf<'_>>,
    conf: &DistConf<'_>,
) -> Summary {
    let r = distribution_with(a.len(), |i| Some(f64::from(a[i])), id, node, conf);
    let d = r.all;
    let len = a.len();
    let mean = d.mean.unwrap_or(0.0);
    Summary {
        len,
        sum: d.sum,
        mean,
        mean_over_window: if n_w == 0 { 0.0 } else { mean / n_w as f64 },
        max: d.max.unwrap_or(0.0),
        argmax: d.argmax,
        argmax_node: d.argmax_node,
        zeros: d.log2_hist.bins.first().copied().unwrap_or(0),
        log2_hist: d.log2_hist,
        top: d.top,
        by_bucket: r.by_bucket,
        unbucketed: r.unbucketed,
    }
}

/// Block mode: the number of sources of the window `[lo, hi)` (the nodes without a
/// predecessor in the window), and how many of them have a non-zero ancestor estimate (must
/// be 0).
///
/// The predecessors of an output `v` of block `B` are the inputs of `B`, so `v` is a source of
/// the window iff `B = ⊥` (no creating transaction), `B` lies before the block window (its first
/// output is `< lo`, so all its inputs are), or no input of `B` lies in `[lo, hi)`; the last
/// condition is evaluated with a bitset over the block window filled from `spend[lo..hi]`.
/// Time `O(n_w)` in parallel, memory `nb_w/8` bytes.
fn check_sources(idx: &BlockIndex, w: Window, anc: &[f32]) -> (usize, usize) {
    let bw = block_window(idx, w);
    let mut bits = vec![0u64; bw.len().div_ceil(64)];
    let set = as_atomic_u64(&mut bits);
    idx.spend[w.lo..w.hi]
        .par_iter()
        .with_min_len(1 << 16)
        .for_each(|&x| {
            if x != NONE && bw.contains(&(x as usize)) {
                let i = x as usize - bw.start;
                set[i / 64].fetch_or(1 << (i % 64), Ordering::Relaxed);
            }
        });
    let has_input = |b: usize| bits[b / 64] >> (b % 64) & 1 == 1;
    idx.create[w.lo..w.hi]
        .par_iter()
        .zip(anc.par_iter())
        .with_min_len(1 << 16)
        .map(|(&c, &a)| {
            let source =
                c == NONE || !bw.contains(&(c as usize)) || !has_input(c as usize - bw.start);
            (usize::from(source), usize::from(source && a != 0.0))
        })
        .reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1))
}

/// Memory reserved for the block mode: the larger of its two phases (see
/// [`sketch`](crate::reach::sketch), "Complexity"), plus the `nb_w/8`-byte bitset of
/// [`check_sources`].
///
/// - Pass phase, per pass: `slot_of` and an `f32` per window block, a slab of
///   `min(nb_w, max(2^20, nb_w/64))` sketches (the measured live peak on the 2022 PG is
///   5.14·10⁶ = nb/151) and the two estimation buffers; `refcnt` for the descendant pass.
/// - Expansion phase, per direction: an `f32` per window block and per window node.
fn blocks_bytes(n_w: usize, nb_w: usize, which: Which, log2m: u32) -> u64 {
    let (n_w, nb_w) = (n_w as u64, nb_w as u64);
    let sk = sketch_bytes(log2m);
    let slab = nb_w.min((1 << 20).max(nb_w / 64)) * sk;
    let buffers = 2 * ESTIMATE_BATCH as u64 * sk;
    let dirs = u64::from(which.forward()) + u64::from(which.backward());
    let refcnt = if which.backward() { 4 * nb_w } else { 0 };
    let pass_phase = dirs * (8 * nb_w + slab + buffers) + refcnt;
    let expand_phase = dirs * (4 * nb_w + 4 * n_w);
    pass_phase.max(expand_phase) + nb_w.div_ceil(8)
}

/// Runs `dag reach-counts`.
pub fn run(args: &Args, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = args.out.prepare()?.to_owned();
    let cg = args.graph.open(g)?;
    let n = cg.num_nodes();
    let w = args.window.window(n)?;
    let buckets = args.stats.buckets.load(n)?;
    let dconf = DistConf {
        top: args.stats.top,
        buckets: buckets.as_ref(),
    };
    // validate the sketch parameters before any expensive work
    sketch::logic(n.max(1), args.log2m, SeededBuildHasher::new(args.seed))?;
    let mut budget = g.budget();
    let t_blocks = Instant::now();
    let idx = args.blocks.resolve(&cg, g, &mut budget)?;
    let blocks_s = t_blocks.elapsed().as_secs_f64();
    let conf = SketchConf {
        log2m: args.log2m,
        hasher: SeededBuildHasher::new(args.seed),
        window: w,
        sweep: g.sweep_conf(),
    };
    let t_counts = Instant::now();
    let counts: Counts = match &idx {
        Some(idx) => {
            let nb_w = block_window(idx, w).len();
            budget.need(
                blocks_bytes(w.len(), nb_w, args.which, args.log2m),
                "reach-counts block passes (slab estimate included)",
            )?;
            counts_blocks(idx, args.which, &conf, &mut logger("node"))?
        }
        None => counts_generic(&cg, args.which, &conf, &mut budget, &mut logger("node"))?,
    };
    let counts_s = t_counts.elapsed().as_secs_f64();
    let mode = if idx.is_some() { "blocks" } else { "generic" };
    log::info!(
        "reach-counts ({mode}): window {w}, peak live sketches {}, {:.1} s",
        counts.peak_live,
        counts_s
    );

    let mut info = RunInfo::new(Some(args.graph.graph.clone()), n, cg.num_arcs, w);
    let mut results = serde_json::Map::new();
    let mut ok = true;
    let t_write = Instant::now();
    let nb_w = counts.blocks.as_ref().map_or(0, |r| r.len());
    let blo = counts.blocks.as_ref().map_or(0, |r| r.start);
    let mut summary_s = 0.0;
    for (field, data, len, meaning) in [
        (
            "anc",
            &counts.anc,
            w.len(),
            "estimated number of proper ancestors in the window (0 for sources), index v - lo",
        ),
        (
            "desc",
            &counts.desc,
            w.len(),
            "estimated number of proper descendants in the window (0 for sinks), index v - lo",
        ),
        (
            "blk.anc",
            &counts.blk_anc,
            nb_w,
            "estimated |A(X)|: proper ancestors of the outputs of block X, index X - blo",
        ),
        (
            "blk.desc",
            &counts.blk_desc,
            nb_w,
            "estimated |D(X)|: outputs of block X and their descendants, index X - blo",
        ),
    ] {
        let Some(a) = data else { continue };
        let path = array_path::<f32>(&out, field);
        write_array(&path, a)?;
        info.files
            .push(FileInfo::new::<f32>(&path, len, None, meaning));
        let ts = Instant::now();
        let summary = if field.starts_with("blk.") {
            let idx = idx.as_ref().expect("block arrays come from a BlockIndex");
            let first = |i: usize| idx.first[blo + i] as usize;
            summarize(a, w.len(), &|i| blo + i, Some(&first), &dconf)
        } else {
            summarize(a, w.len(), &|i| w.lo + i, None, &dconf)
        };
        summary_s += ts.elapsed().as_secs_f64();
        results.insert(field.replace('.', "_"), json!(summary));
    }
    let write_s = t_write.elapsed().as_secs_f64() - summary_s;
    if let (Some(idx), Some(anc)) = (&idx, &counts.anc) {
        let (sources, bad) = check_sources(idx, w, anc);
        log::info!("{sources} sources in the window, {bad} with a non-zero ancestor estimate");
        ok &= bad == 0;
        results.insert(
            "sources_check".into(),
            json!({ "sources": sources, "nonzero_anc": bad, "ok": bad == 0 }),
        );
    }
    results.insert("mode".into(), json!(mode));
    results.insert("peak_live".into(), json!(counts.peak_live));
    results.insert("blocks".into(), json!(counts.blocks));
    results.insert("passes".into(), json!(counts.passes));
    results.insert(
        "timings_s".into(),
        json!({ "blocks": blocks_s, "counts": counts_s, "write": write_s, "summary": summary_s }),
    );
    info.phases_s
        .add("open", t_blocks.duration_since(t0).as_secs_f64());
    info.phases_s.add("blocks", blocks_s);
    info.phases_s.add("kernel", counts_s);
    info.phases_s.add("summary", summary_s);
    info.phases_s.add("write", write_s);
    info.params = json!({
        "which": args.which,
        "log2m": args.log2m,
        "seed": args.seed,
        "rsd": sketch::rsd(args.log2m),
        "sketch_bytes": sketch_bytes(args.log2m),
        "num_elements": n,
        "hasher": "SplitMix64 (dag::reach::hash::SeededBuildHasher)",
        "estimator": "card-est-array 0.6.0 HyperLogLog, 5-bit registers, LogLog-beta",
        "blocks": format!("{:?}", args.blocks.blocks),
        "buckets": args.stats.buckets.buckets,
        "top": args.stats.top,
    });
    info.results = serde_json::Value::Object(results);
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}
