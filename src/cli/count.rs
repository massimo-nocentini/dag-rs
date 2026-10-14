//! `dag count-paths`: path counting ([`crate::paths`]) from the command line.
//!
//! Writes, for each requested direction `D ∈ {fwd, bwd}` and kind, one array indexed by
//! `v − lo`: `<out>.D.log2.f64le` (`extfloat`: `log2` of the count), `<out>.D.mod61.u64le`
//! (`mod61`: the count modulo 2⁶¹ − 1), `<out>.D.sat64.u64le` and `<out>.D.sat128.u128le`
//! (saturating counts, `MAX` = saturated). With `--through` and both directions it also writes
//! the source-to-sink paths through each node, `<out>.through.log2.f64le` (`log2 F + log2 B`)
//! and `<out>.through.mod61.u64le` (`F·B mod (2⁶¹ − 1)`), computed on the fly while writing.
//! `--summary-only` writes no array. `<out>.json` holds the
//! [`PathSummary`]: totals per direction, the duality check, the largest counts, `log2`
//! histograms and the through-count statistics, and the [`CountDistributions`] (top-k nodes by
//! count and, with `--buckets`, one distribution per node range; computed before the arrays
//! are dropped, so also with `--summary-only`).
//!
//! The run exits with 1 when the duality check `Σ_{sinks} F = Σ_{sources} B` fails (exactly
//! in `mod61` and the unsaturated saturating kinds, within relative 10⁻⁶ in `extfloat`).
//!
//! Resources: `--blocks` is resolved as in `dp` ([`window_blocks`]): `auto` builds the index
//! only for the full graph, a saved `PREFIX` is memory-mapped for a strict sub-window, and with
//! `--blocks PREFIX` the BV graph is not opened at all ([`block_only`]). Without a block index
//! the forward direction pulls over the transpose when it exists (else pushes over the graph)
//! and the backward direction pulls over the graph. Memory is checked against `--max-mem`
//! ([`direction_bytes`](crate::paths::direction_bytes)); the two directions run concurrently
//! when both estimates fit.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Result, bail};
use clap::Args as ClapArgs;
use rayon::prelude::*;
use serde_json::json;

use super::common::{
    BlocksOpt, GlobalArgs, GraphOpt, OutOpt, StatsOpt, WindowOpt, exit_status, logger,
};
use super::dp::{block_only, window_blocks};
use crate::dp::{DpConf, DpInput, Kernel};
use crate::graph;
use crate::io::{self, Dump, FileInfo, RunInfo};
use crate::paths::{CountKind, PathCounts, PathSummary, count_paths_budgeted};
use crate::semiring::Semiring;
use crate::semiring::modular::M61;
use crate::summary::{Buckets, DistConf, DistReport, distribution};
use crate::window::Which;

/// Elements converted per batch when writing the through counts.
const THROUGH_BATCH: usize = 1 << 22;

/// `dag count-paths`.
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
    /// Directions.
    #[arg(long, value_enum, default_value = "both")]
    pub dir: Which,
    /// Count representations.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "extfloat,mod61"
    )]
    pub kinds: Vec<CountKind>,
    /// DP kernel (`auto` uses the block kernels when a BlockIndex is available; `push` runs the
    /// backward direction with `pull`).
    #[arg(long, value_enum, default_value = "auto")]
    pub kernel: Kernel,
    /// Also write the through counts F·B (`log2` and `mod61`; needs both directions).
    #[arg(long)]
    pub through: bool,
    /// Write only `<out>.json` (no per-node array).
    #[arg(long)]
    pub summary_only: bool,
    /// Aggregate measures (`--buckets`, `--top`).
    #[command(flatten)]
    pub stats: StatsOpt,
}

/// The distributions of `log2` of the path counts ([`crate::summary`]; ExtFloat kind only):
/// forward `log2 F`, backward `log2 B` and, with both directions, the through counts
/// `log2 F + log2 B`. Their log2 histograms bin the *magnitude of the exponent* (bin `k ≥ 1`:
/// counts in `[2^(2^(k−1)), 2^(2^k))`, bin 0: count 1, as in
/// [`DirSummary::log2_hist`](crate::paths::DirSummary::log2_hist)), and their top-k lists are
/// the nodes with the most paths.
#[derive(Debug, Default, serde::Serialize)]
pub struct CountDistributions {
    /// `log2 F(v)`.
    pub forward_log2: Option<DistReport>,
    /// `log2 B(v)`.
    pub backward_log2: Option<DistReport>,
    /// `log2 (F(v)·B(v))`: source-to-sink paths through `v`.
    pub through_log2: Option<DistReport>,
}

/// Computes the [`CountDistributions`] of `c` (non-finite values count as absent).
pub fn distributions(c: &PathCounts, top: usize, buckets: Option<&Buckets>) -> CountDistributions {
    let conf = DistConf { top, buckets };
    let lo = c.window.lo;
    let fin = |x: f64| x.is_finite().then_some(x);
    let one = |a: &Option<Vec<f64>>| {
        a.as_ref()
            .map(|a| distribution(a.len(), |i| fin(a[i]), lo, &conf))
    };
    CountDistributions {
        forward_log2: one(&c.fwd_log2),
        backward_log2: one(&c.bwd_log2),
        through_log2: match (&c.fwd_log2, &c.bwd_log2) {
            (Some(f), Some(b)) => Some(distribution(f.len(), |i| fin(f[i] + b[i]), lo, &conf)),
            _ => None,
        },
    }
}

/// Writes `data` to `<out>.<field>.<dtype>` and records it in `info`.
fn write<T: Dump>(
    info: &mut RunInfo,
    out: &Path,
    field: &str,
    data: &[T],
    sentinel: Option<&str>,
    meaning: &str,
) -> Result<()> {
    let p = io::array_path::<T>(out, field);
    io::write_array(&p, data)?;
    info.files
        .push(FileInfo::new::<T>(&p, data.len(), sentinel, meaning));
    Ok(())
}

/// Streams `f(i)` for `i < len` into `<out>.<field>.<dtype>`, converting in parallel batches.
fn write_with<T: Dump + Default>(
    info: &mut RunInfo,
    out: &Path,
    field: &str,
    len: usize,
    f: impl Fn(usize) -> T + Sync,
    meaning: &str,
) -> Result<()> {
    let p = io::array_path::<T>(out, field);
    let mut w = io::array_writer::<T>(&p)?;
    let mut buf = Vec::with_capacity(THROUGH_BATCH.min(len));
    for start in (0..len).step_by(THROUGH_BATCH) {
        let end = (start + THROUGH_BATCH).min(len);
        (start..end)
            .into_par_iter()
            .map(&f)
            .collect_into_vec(&mut buf);
        w.extend(&buf)?;
    }
    w.finish()?;
    info.files.push(FileInfo::new::<T>(&p, len, None, meaning));
    Ok(())
}

/// Writes every array of `c` (see the module documentation).
fn write_arrays(info: &mut RunInfo, out: &Path, c: &PathCounts, through: bool) -> Result<()> {
    let sat64 = Some("u64::MAX = saturated (≥ u64::MAX)");
    let sat128 = Some("u128::MAX = saturated (≥ u128::MAX)");
    for (d, log2, mod61, s64, s128, what) in [
        (
            "fwd",
            &c.fwd_log2,
            &c.fwd_mod61,
            &c.fwd_sat64,
            &c.fwd_sat128,
            "paths from a window source to v",
        ),
        (
            "bwd",
            &c.bwd_log2,
            &c.bwd_mod61,
            &c.bwd_sat64,
            &c.bwd_sat128,
            "paths from v to a window sink",
        ),
    ] {
        if let Some(a) = log2 {
            write(
                info,
                out,
                &format!("{d}.log2"),
                a,
                None,
                &format!("log2 of the {what} (ExtFloat)"),
            )?;
        }
        if let Some(a) = mod61 {
            write(
                info,
                out,
                &format!("{d}.mod61"),
                a,
                None,
                &format!("{what}, modulo 2^61 - 1"),
            )?;
        }
        if let Some(a) = s64 {
            write(
                info,
                out,
                &format!("{d}.sat64"),
                a,
                sat64,
                &format!("{what}, saturating u64"),
            )?;
        }
        if let Some(a) = s128 {
            write(
                info,
                out,
                &format!("{d}.sat128"),
                a,
                sat128,
                &format!("{what}, saturating u128"),
            )?;
        }
    }
    if through {
        let n_w = c.window.len();
        let mut any = false;
        if let (Some(f), Some(b)) = (&c.fwd_log2, &c.bwd_log2) {
            write_with(
                info,
                out,
                "through.log2",
                n_w,
                |i| f[i] + b[i],
                "log2 of the source-to-sink paths through v (log2 F + log2 B)",
            )?;
            any = true;
        }
        if let (Some(f), Some(b)) = (&c.fwd_mod61, &c.bwd_mod61) {
            let m = M61::new();
            write_with(
                info,
                out,
                "through.mod61",
                n_w,
                |i| m.times(f[i], b[i]),
                "source-to-sink paths through v, modulo 2^61 - 1 (F·B)",
            )?;
            any = true;
        }
        if !any {
            log::warn!("--through needs both directions and extfloat or mod61: nothing written");
        }
    }
    Ok(())
}

/// Whether a run with this summary succeeds: the [`Duality`](crate::paths::Duality) check
/// passed, or was not run (a single direction). `run` exits with
/// [`exit_status`]`(duality_ok(..))`, i.e. 1 when the check fails.
///
/// ```
/// use dag::cli::count::duality_ok;
/// use dag::paths::{Duality, PathSummary};
/// assert!(duality_ok(&PathSummary::default()));
/// let failed = PathSummary { duality: Some(Duality::default()), ..Default::default() };
/// assert!(!duality_ok(&failed)); // `Duality::default().ok` is false
/// ```
pub fn duality_ok(summary: &PathSummary) -> bool {
    summary.duality.as_ref().is_none_or(|d| d.ok)
}

/// Runs `dag count-paths`.
pub fn run(args: &Args, global: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    if args.kinds.is_empty() {
        bail!("--kinds is empty");
    }
    let out = args.out.prepare()?.to_owned();
    let mut budget = global.budget();
    let (n, m) = graph::read_properties(&args.graph.graph)?;
    let win = args.window.window(n)?;
    let buckets = args.stats.buckets.load(n)?;
    let cg = if block_only(&args.blocks, args.kernel, false) {
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
    let forward = matches!(args.dir, Which::Forward | Which::Both);
    let gt = if forward
        && (args.kernel == Kernel::Pull || (args.kernel == Kernel::Auto && blocks.is_none()))
    {
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
    let conf = DpConf {
        window: win,
        kernel: args.kernel,
        pred: false,
        sweep: global.sweep_conf(),
    };
    let open_s = t0.elapsed().as_secs_f64();
    let tc = Instant::now();
    let mut pl = logger("node");
    let counts = count_paths_budgeted(&input, args.dir, &args.kinds, &conf, &mut budget, &mut pl)?;
    let count_s = tc.elapsed().as_secs_f64();
    let summary: &PathSummary = &counts.summary;
    let ok = duality_ok(summary);

    let mut info = RunInfo::new(Some(args.graph.graph.clone()), n, m, win);
    info.phases_s.add("open", open_s);
    info.phases_s.add("kernel", count_s);
    let dists = info.phases_s.time("summary", || {
        distributions(&counts, args.stats.top, buckets.as_ref())
    });
    info.params = json!({
        "buckets": args.stats.buckets.buckets,
        "top": args.stats.top,
        "dir": args.dir,
        "kinds": args.kinds,
        "kernel_requested": args.kernel,
        "blocks": blocks.is_some(),
        "graph_opened": cg.is_some(),
        "transpose_opened": gt.is_some(),
        "through": args.through,
        "summary_only": args.summary_only,
    });
    let tw = Instant::now();
    if !args.summary_only {
        write_arrays(&mut info, &out, &counts, args.through)?;
    }
    info.phases_s.add("write", tw.elapsed().as_secs_f64());
    info.results = json!({
        "summary": summary,
        "distributions": dists,
        "total_log2": counts.total_log2.is_finite().then_some(counts.total_log2),
        "total_mod61": summary.forward.as_ref().or(summary.backward.as_ref()).and_then(|d| d.total_mod61),
        "saturated_nodes": counts.saturated_nodes,
        "count_s": count_s,
        "write_s": tw.elapsed().as_secs_f64(),
        "ok": ok,
    });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}
