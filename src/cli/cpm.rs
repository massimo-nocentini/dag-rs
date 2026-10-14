//! `dag cpm`: the critical path method ([`crate::cpm`]) from the command line.
//!
//! Writes `<out>.E.u32le` (earliest times, indexed by `v − lo`), `<out>.H.u32le` (heights),
//! optionally `<out>.slack.u32le` (`--slack`) and `<out>.path.u32le` (`--path`: one longest
//! path, source first, absolute ids), and `<out>.json` with the [`CpmSummary`] (project length
//! `T`, critical nodes, critical arcs with `--critical-arcs`, width-1 layers for unit weights,
//! the largest critical layer, its node range and the histogram of the layer widths) and the
//! [`CpmDistributions`] of E, H, slack and the critical nodes (log2 histograms, top-k nodes,
//! and with `--buckets` one distribution per node range). `--compare-e FILE` /
//! `--compare-h FILE` compare E / H with reference `u32` arrays (`n` entries, of which the
//! window is compared, or one per window node); the run exits with 1 on any difference.
//!
//! Weights: `unit` (arc length 1: E is the depth, H the height), `one` (length 0) or
//! `node-u32:FILE` (activity-on-node durations `α(u) = FILE[u]`). Times are `u32`; a
//! saturated `T = u32::MAX − 1` is reported as a warning.
//!
//! Resources: `--blocks` is resolved as in `dp` ([`window_blocks`]): `auto` builds the index
//! only for the full graph (a strict sub-window runs the generic kernels unless
//! `--kernel blocks`), a saved `PREFIX` is memory-mapped for a strict sub-window, and with
//! `--blocks PREFIX` and without `--critical-arcs` the BV graph is not opened at all
//! ([`block_only`]). Node durations are read for the window only (`4·n_W` bytes).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::Args as ClapArgs;
use rayon::prelude::*;
use serde_json::json;

use super::common::{
    BlocksOpt, GlobalArgs, GraphOpt, OutOpt, StatsOpt, WeightsSpec, WindowOpt, concurrent_logger,
    exit_status, logger,
};
use super::dp::{block_only, window_blocks};
use crate::cpm::{CpmResult, CpmSummary, backward_kernel, cpm};
use crate::dp::solve::{estimate_bytes, resolve_kernel, sweep_buffer_bytes};
use crate::dp::{ArcWeights, Const, DpConf, DpInput, Kernel, Node, One};
use crate::graph::{self, BvRa, ChunkedGraph};
use crate::io::{self, FileInfo, LoadMode, RunInfo, U32Array};
use crate::semiring::MaxPlus;
use crate::summary::{Buckets, DistConf, DistReport, distribution};
use crate::window::{Dir, Window};

/// `dag cpm`.
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
    /// Arc weights: unit or node-u32:FILE.
    #[arg(long, default_value = "unit")]
    pub weights: WeightsSpec,
    /// Write the slack array.
    #[arg(long)]
    pub slack: bool,
    /// Write one longest path.
    #[arg(long)]
    pub path: bool,
    /// Count critical arcs (one more parallel scan).
    #[arg(long)]
    pub critical_arcs: bool,
    /// Compare E with this u32 array.
    #[arg(long)]
    pub compare_e: Option<PathBuf>,
    /// Compare H with this u32 array.
    #[arg(long)]
    pub compare_h: Option<PathBuf>,
    /// Kernel of both passes (`auto` uses the block kernels when a BlockIndex is available;
    /// `push` runs E with push and H with pull, as there is no backward push kernel).
    #[arg(long, value_enum, default_value = "auto")]
    pub kernel: Kernel,
    /// Aggregate measures (`--buckets`, `--top`).
    #[command(flatten)]
    pub stats: StatsOpt,
}

/// Result of `--compare-e` / `--compare-h`.
#[derive(Debug, serde::Serialize)]
pub struct ArrayComparison {
    /// The reference file.
    pub file: PathBuf,
    /// Entries that differ.
    pub mismatches: u64,
    /// Smallest differing node (absolute id).
    pub first_mismatch: Option<usize>,
}

/// Compares `val` (window `w` of `n` nodes) with the `u32` array in `file`.
fn compare(val: &[u32], w: Window, n: usize, file: &Path) -> Result<ArrayComparison> {
    let r = U32Array::load(file, LoadMode::Mmap)?;
    let off = if r.len() == n {
        w.lo
    } else if r.len() == w.len() {
        0
    } else {
        bail!(
            "{} has {} entries: expected {n} (graph) or {} (window)",
            file.display(),
            r.len(),
            w.len()
        )
    };
    let reference = &r[off..off + val.len()];
    let mismatches = val
        .par_iter()
        .zip(reference.par_iter())
        .filter(|(a, b)| a != b)
        .count() as u64;
    let first_mismatch = val
        .par_iter()
        .zip(reference.par_iter())
        .position_first(|(a, b)| a != b)
        .map(|i| i + w.lo);
    Ok(ArrayComparison {
        file: file.to_owned(),
        mismatches,
        first_mismatch,
    })
}

/// Runs `dag cpm`.
pub fn run(args: &Args, global: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = args.out.prepare()?.to_owned();
    let mut budget = global.budget();
    let (n, m) = graph::read_properties(&args.graph.graph)?;
    let win = args.window.window(n)?;
    let buckets = args.stats.buckets.load(n)?;
    let cg = if block_only(&args.blocks, args.kernel, args.critical_arcs) {
        log::info!("block-only run: the BV graph is not opened");
        None
    } else {
        Some(args.graph.open(global)?)
    };
    let blocks = window_blocks(
        &args.blocks,
        args.kernel,
        cg.as_ref(),
        n,
        win,
        global,
        &mut budget,
    )?;
    let gt = if args.kernel == Kernel::Pull || (args.kernel == Kernel::Auto && blocks.is_none()) {
        args.graph.open_transpose(global)?
    } else {
        None
    };
    let mut input = match (&cg, &gt, &blocks) {
        (Some(cg), Some(gt), _) => DpInput::with_transpose(cg, gt),
        (Some(cg), None, _) => DpInput::graph(cg),
        (None, _, Some(b)) => DpInput::blocks(b),
        (None, _, None) => bail!("a block-only run needs the BlockIndex"),
    };
    if let Some(b) = &blocks {
        input = input.and_blocks(b);
    }
    let kf = resolve_kernel(&input, Dir::Forward, true, args.kernel)?;
    let kb = resolve_kernel(&input, Dir::Backward, true, backward_kernel(args.kernel))?;
    if args.kernel == Kernel::Push {
        log::info!("--kernel push: E uses push, H uses pull (there is no backward push kernel)");
    }
    let sweep = global.sweep_conf();
    let n_blocks = blocks
        .as_ref()
        .map_or(0, |b| b.block_bound(win.hi) - b.block_bound(win.lo));
    let sb = sweep_buffer_bytes(&sweep, global.chunk_arcs, global.max_chunk_nodes);
    budget.need(
        estimate_bytes::<u32>(win.len(), Dir::Forward, kf, args.path, n_blocks, sb)
            + estimate_bytes::<u32>(win.len(), Dir::Backward, kb, false, n_blocks, sb),
        "cpm E and H",
    )?;
    if args.slack {
        budget.need(4 * win.len() as u64, "slack array")?;
    }
    let conf = DpConf {
        window: win,
        kernel: args.kernel,
        pred: args.path,
        sweep,
    };

    let durations;
    let unit = matches!(args.weights, WeightsSpec::Unit);
    let open_s = t0.elapsed().as_secs_f64();
    let tk = Instant::now();
    let (res, arcs): (CpmResult<u32>, Option<u64>) = match &args.weights {
        WeightsSpec::Unit => {
            run_with(&input, &Const(1u32), &conf, cg.as_ref(), args.critical_arcs)?
        }
        WeightsSpec::One => run_with(&input, &One, &conf, cg.as_ref(), args.critical_arcs)?,
        WeightsSpec::NodeU32(f) => {
            // only α(u) of window nodes is read: map the file, copy the window
            budget.need(4 * win.len() as u64, "durations")?;
            let file = U32Array::load(f, LoadMode::Mmap)
                .with_context(|| format!("cannot load the durations {}", f.display()))?;
            ensure!(
                file.len() == n,
                "{} has {} entries, the graph {n} nodes",
                f.display(),
                file.len()
            );
            durations = {
                let mut v = io::huge_vec(win.len(), 0u32);
                v.par_iter_mut()
                    .zip(file[win.lo..win.hi].par_iter())
                    .with_min_len(1 << 16)
                    .for_each(|(d, &x)| *d = x.min(u32::MAX - 1));
                v
            };
            drop(file);
            let (d, lo): (&[u32], usize) = (&durations, win.lo);
            let w = Node {
                alpha: |u: usize| d[u - lo],
                beta: |_: usize| 0u32,
            };
            run_with(&input, &w, &conf, cg.as_ref(), args.critical_arcs)?
        }
        other => bail!("cpm supports --weights unit, one or node-u32:FILE, not {other:?}"),
    };
    let cpm_s = tk.elapsed().as_secs_f64();
    budget.need(res.summary_bytes(), "CPM layer profile")?;
    let ts = Instant::now();
    let mut summary: CpmSummary = res.summary(unit);
    summary.critical_arcs = arcs;
    let dists = distributions(&res, args.stats.top, buckets.as_ref());
    let summary_s = ts.elapsed().as_secs_f64();
    log::info!(
        "cpm: T = {}, critical nodes = {}, critical arcs = {:?}, width-1 layers = {:?}, max critical width = {}",
        summary.t,
        summary.critical_nodes,
        summary.critical_arcs,
        summary.width1_layers,
        summary.max_critical_width
    );

    let mut info = RunInfo::new(Some(args.graph.graph.clone()), n, m, win);
    info.phases_s.add("open", open_s);
    info.phases_s.add("kernel", cpm_s);
    info.phases_s.add("summary", summary_s);
    let tw = Instant::now();
    info.params = json!({
        "buckets": args.stats.buckets.buckets,
        "top": args.stats.top,
        "weights": format!("{:?}", args.weights),
        "kernel_requested": args.kernel,
        "kernel_forward": kf,
        "kernel_backward": kb,
        "blocks": blocks.is_some(),
        "graph_opened": cg.is_some(),
        "slack": args.slack,
        "path": args.path,
        "critical_arcs": args.critical_arcs,
    });
    for (field, data, meaning) in [
        (
            "E",
            &res.e,
            "earliest time (longest path from a window source)",
        ),
        ("H", &res.h, "height (longest path to a window sink)"),
    ] {
        let p = io::array_path::<u32>(&out, field);
        io::write_array(&p, data)?;
        info.files
            .push(FileInfo::new::<u32>(&p, data.len(), None, meaning));
    }
    if args.slack {
        let sl = res.slacks();
        let p = io::array_path::<u32>(&out, "slack");
        io::write_array(&p, &sl)?;
        info.files
            .push(FileInfo::new::<u32>(&p, sl.len(), None, "T − E − H"));
    }
    let mut path_meta = serde_json::Value::Null;
    if let Some(path) = summary.longest_path.take() {
        let p = io::array_path::<u32>(&out, "path");
        io::write_array(&p, &path)?;
        info.files.push(FileInfo::new::<u32>(
            &p,
            path.len(),
            Some("none"),
            "one longest path, source first (absolute ids)",
        ));
        path_meta = json!({ "nodes": path.len(), "start": path.first(), "end": path.last() });
    }
    info.phases_s.add("write", tw.elapsed().as_secs_f64());
    let tc = Instant::now();
    let mut ok = true;
    let mut comparisons = Vec::new();
    for (what, file, data) in [
        ("E", &args.compare_e, &res.e),
        ("H", &args.compare_h, &res.h),
    ] {
        if let Some(f) = file {
            let c = compare(data, win, n, f)?;
            log::info!("comparison of {what}: {c:?}");
            ok &= c.mismatches == 0;
            comparisons.push(json!({ "array": what, "result": c }));
        }
    }
    info.phases_s.add("compare", tc.elapsed().as_secs_f64());
    info.results = json!({
        "summary": summary,
        "distributions": dists,
        "cpm_s": cpm_s,
        "longest_path": path_meta,
        "comparisons": comparisons,
    });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}

/// The distributions of a CPM run ([`crate::summary`], node ids absolute).
#[derive(Debug, serde::Serialize)]
pub struct CpmDistributions {
    /// Earliest times `E`.
    pub e: DistReport,
    /// Heights `H`.
    pub h: DistReport,
    /// Slacks `T − E − H`.
    pub slack: DistReport,
    /// The `E` values of the critical nodes only (absent elsewhere): `count` per bucket is the
    /// number of critical nodes in it, `min`/`max` the span of the critical layers it holds.
    pub critical_e: DistReport,
}

/// Computes the [`CpmDistributions`] of `res` (four parallel scans, no allocation beyond the
/// accumulators).
pub fn distributions(
    res: &CpmResult<u32>,
    top: usize,
    buckets: Option<&Buckets>,
) -> CpmDistributions {
    let conf = DistConf { top, buckets };
    let (lo, len, t) = (res.window.lo, res.e.len(), res.t);
    let slack = |i: usize| t - res.e[i] - res.h[i];
    CpmDistributions {
        e: distribution(len, |i| Some(f64::from(res.e[i])), lo, &conf),
        h: distribution(len, |i| Some(f64::from(res.h[i])), lo, &conf),
        slack: distribution(len, |i| Some(f64::from(slack(i))), lo, &conf),
        critical_e: distribution(
            len,
            |i| (slack(i) == 0).then(|| f64::from(res.e[i])),
            lo,
            &conf,
        ),
    }
}

fn run_with<W: ArcWeights<MaxPlus<u32>>>(
    input: &DpInput<BvRa>,
    w: &W,
    conf: &DpConf,
    g: Option<&ChunkedGraph<BvRa>>,
    critical_arcs: bool,
) -> Result<(CpmResult<u32>, Option<u64>)> {
    let mut pl = logger("node");
    let res = cpm::<u32, _, _>(input, w, conf, &mut pl)?;
    let arcs = if critical_arcs {
        let g = g.context("--critical-arcs needs the graph")?;
        Some(res.critical_arcs(g, w, &mut concurrent_logger("node"))?)
    } else {
        None
    };
    Ok((res, arcs))
}
