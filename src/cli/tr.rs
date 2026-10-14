//! `dag tr verify|reduce`: transitive-reduction verification and reduction
//! ([`crate::reduce`]).
//!
//! - `tr verify` decides whether the graph (or `--window`) is transitively reduced and writes
//!   `<out>.json` (the [`ReductionReport`]) and `<out>.redundant.u32x2le` (the redundant arcs
//!   `(u, v)`, absolute ids, sorted). Exit code 1 if the graph is not reduced.
//! - `tr reduce` writes the reduced graph (or window, ids shifted by `−lo`) as the BV graph
//!   `--out` (+ `.ef`, `.dcf`, transpose `<out>-t`), plus `<out>.redundant.u32x2le` and
//!   `<out>.json`.
//!
//! `--blocks`: `PREFIX` loads `PREFIX.blk.*` and uses the "blocks" certificate; `auto` runs the
//! line-digraph check restricted to the window
//! ([`line_digraph_window`], 4 B per window node, no full-graph index) and falls back to the
//! generic H3 path with a warning if it fails; `none` always uses the generic path (H1, H3 and
//! the exact re-check of H3 failures).
//!
//! `tr reduce --algo`: `bitset` and `search` load the window as a [`Csr`] and run
//! [`transitive_reduction_bitset`](crate::reduce::transitive_reduction_bitset) or
//! [`transitive_reduction_search`](crate::reduce::transitive_reduction_search) (`--blocks` is
//! then ignored and recorded as `null`; `results.threads` records the thread count, which
//! for `search` is capped by `--max-mem`); `auto` uses `bitset` for
//! windows of at most [`BITSET_MAX_NODES`] nodes and otherwise the verification policy
//! (certificates, then exact searches from the H3-failing nodes only), which is linear on the
//! Payment Graph.
//!
//! Memory: every phase reserves its buffers in the `--max-mem` budget and releases them when
//! they are dropped, so the external sort of the output gets half of what is left after the
//! redundant-arc list (8 B per arc).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use serde_json::json;

use super::common::{
    BlocksOpt, BlocksSpec, GlobalArgs, GraphOpt, OutOpt, WindowOpt, concurrent_logger, exit_status,
    logger,
};
use crate::blocks::BlockIndex;
use crate::csr::Csr;
use crate::graph::{BvRa, ChunkedGraph};
use crate::io::{self, FileInfo, MemBudget, RunInfo, suffixed};
use crate::reduce::{
    BITSET_MAX_NODES, InMemoryOutcome, LineDigraphReport, ReductionReport, line_digraph_window,
    reduce_bv_window, redundant_arcs_bitset, redundant_arcs_search, verify_reduced_window,
};
use crate::topo;
use crate::window::Window;

/// `dag tr`.
#[derive(ClapArgs, Debug)]
pub struct Args {
    /// The tr subcommand.
    #[command(subcommand)]
    pub cmd: TrCmd,
}

/// Transitive-reduction algorithm of `tr reduce`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum TrAlgo {
    /// Bitset for windows of at most 2¹⁶ nodes, otherwise the verification policy
    /// (certificates, then exact searches from the H3-failing nodes).
    Auto,
    /// Triangular bitsets on the in-memory window.
    Bitset,
    /// Per-node searches with a DirtyBitset on the in-memory window.
    Search,
}

/// The tr subcommands.
#[derive(Subcommand, Debug)]
pub enum TrCmd {
    /// Verifies that the graph is transitively reduced.
    Verify {
        /// Input graph.
        #[command(flatten)]
        graph: GraphOpt,
        /// Block index.
        #[command(flatten)]
        blocks: BlocksOpt,
        /// Window.
        #[command(flatten)]
        window: WindowOpt,
        /// Output prefix.
        #[command(flatten)]
        out: OutOpt,
    },
    /// Writes the transitive reduction as a BV graph.
    Reduce {
        /// Input graph.
        #[command(flatten)]
        graph: GraphOpt,
        /// Block index.
        #[command(flatten)]
        blocks: BlocksOpt,
        /// Window.
        #[command(flatten)]
        window: WindowOpt,
        /// Algorithm.
        #[arg(long, value_enum, default_value = "auto")]
        algo: TrAlgo,
        /// Output basename.
        #[command(flatten)]
        out: OutOpt,
    },
}

/// Runs `dag tr`.
pub fn run(args: &Args, g: &GlobalArgs) -> Result<ExitCode> {
    match &args.cmd {
        TrCmd::Verify {
            graph,
            blocks,
            window,
            out,
        } => run_verify(graph, blocks, window, out, g),
        TrCmd::Reduce {
            graph,
            blocks,
            window,
            algo,
            out,
        } => run_reduce(graph, blocks, window, *algo, out, g),
    }
}

/// Report, redundant arcs and (for `--blocks auto`) the line-digraph check of a verification.
type Verified = (ReductionReport, Vec<(u32, u32)>, Option<LineDigraphReport>);

/// The verification policy with the `--blocks` choice; returns the report, the redundant arcs
/// and the line-digraph check (for `auto`).
fn verify(
    cg: &ChunkedGraph<BvRa>,
    w: Window,
    blocks: &BlocksOpt,
    g: &GlobalArgs,
    budget: &mut MemBudget,
) -> Result<Verified> {
    let conf = g.sweep_conf();
    let mut pl = concurrent_logger("node");
    match &blocks.blocks {
        BlocksSpec::Auto => {
            let ld = line_digraph_window(cg, w, budget, &mut pl)?;
            if ld.ok() {
                log::info!(
                    "tr: window {w} is a line digraph ({} blocks, {} arcs): transitively reduced",
                    ld.blocks,
                    ld.arcs
                );
                return Ok((ReductionReport::from_blocks(ld.arcs), Vec::new(), Some(ld)));
            }
            log::warn!(
                "tr: window {w} is not a line digraph ({:?}): falling back to the H3 path",
                ld.violation
            );
            let (r, red) = verify_reduced_window(cg, w, None, &conf, budget, &mut pl)?;
            Ok((r, red, Some(ld)))
        }
        BlocksSpec::Prefix(_) => {
            // resolve reserves BlockIndex::bytes; the index is dropped on return, so release it
            let idx = blocks.resolve(cg, g, budget)?;
            let idx_bytes = idx.as_ref().map_or(0, |i| BlockIndex::bytes(i.n, i.nb));
            let res = verify_reduced_window(cg, w, idx.as_ref(), &conf, budget, &mut pl);
            drop(idx);
            budget.release(idx_bytes);
            let (r, red) = res?;
            Ok((r, red, None))
        }
        BlocksSpec::None => {
            let (r, red) = verify_reduced_window(cg, w, None, &conf, budget, &mut pl)?;
            Ok((r, red, None))
        }
    }
}

fn redundant_file(out: &Path, red: &[(u32, u32)]) -> Result<FileInfo> {
    let path = suffixed(out, "redundant.u32x2le");
    io::write_array(&path, red)?;
    let mut fi = FileInfo::new::<(u32, u32)>(
        path,
        red.len(),
        None,
        "redundant arcs (u, v), absolute ids, sorted",
    );
    fi.dtype = "u32x2le";
    Ok(fi)
}

fn run_verify(
    graph: &GraphOpt,
    blocks: &BlocksOpt,
    window: &WindowOpt,
    out: &OutOpt,
    g: &GlobalArgs,
) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = out.prepare()?.to_owned();
    let cg = graph.open(g)?;
    let n = cg.num_nodes();
    let w = window.window(n)?;
    let mut budget = g.budget();
    let (report, red, ld) = verify(&cg, w, blocks, g, &mut budget)?;
    log::info!("tr verify: {report:?}");
    let mut info = RunInfo::new(Some(graph.graph.clone()), n, cg.num_arcs, w);
    info.files.push(redundant_file(&out, &red)?);
    info.params = json!({ "blocks": blocks_name(&blocks.blocks) });
    info.results = json!({ "report": report, "line_digraph_check": ld });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(report.is_transitively_reduced))
}

fn blocks_name(b: &BlocksSpec) -> String {
    match b {
        BlocksSpec::Auto => "auto".into(),
        BlocksSpec::None => "none".into(),
        BlocksSpec::Prefix(p) => p.display().to_string(),
    }
}

/// Bytes of a [`Csr`] with `n` nodes and `m` arcs.
fn csr_bytes(n: usize, m: usize) -> u64 {
    8 * (n as u64 + 1) + 4 * m as u64
}

/// `tr reduce --algo bitset|search`: loads the window as a [`Csr`] and returns its redundant
/// arcs (absolute ids, sorted) and the thread count, via
/// [`redundant_arcs_bitset`] or [`redundant_arcs_search`] (which cap their own working memory
/// and, for `search`, the thread count by the budget). Every reservation made here is released
/// before returning.
fn reduce_in_memory(
    cg: &ChunkedGraph<BvRa>,
    w: Window,
    algo: TrAlgo,
    budget: &mut MemBudget,
) -> Result<InMemoryOutcome> {
    let n = cg.num_nodes();
    // loading: per-chunk parts plus the final CSR; the window arc count is estimated
    // proportionally to its size, with a factor 2 of slack (capped by m)
    let m_est = ((cg.num_arcs as f64) * (w.len() as f64) / (n.max(1) as f64) * 2.0)
        .min(cg.num_arcs as f64) as u64;
    let load_bytes = 2 * csr_bytes(w.len(), m_est as usize);
    budget.need(load_bytes, "tr reduce: loading the window CSR")?;
    let gw = {
        let mut pl = concurrent_logger("node");
        Csr::from_window(cg, w, &mut pl)
    };
    budget.release(load_bytes);
    let gw = gw?;
    gw.check_topological()?;
    // now that the window is loaded, reserve its exact size
    let held = csr_bytes(gw.n, gw.num_arcs());
    budget.need(held, "tr reduce: window CSR")?;
    let res = if algo == TrAlgo::Bitset {
        redundant_arcs_bitset(&gw, w.lo, budget, &mut logger("node"))
    } else {
        redundant_arcs_search(&gw, w.lo, budget, &mut concurrent_logger("node"))
    };
    drop(gw);
    // the caller reserves the list again for its whole lifetime
    budget.release(held);
    res
}

fn run_reduce(
    graph: &GraphOpt,
    blocks: &BlocksOpt,
    window: &WindowOpt,
    algo: TrAlgo,
    out: &OutOpt,
    g: &GlobalArgs,
) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = out.prepare()?.to_owned();
    let tmp: PathBuf = suffixed(&out, "tmp");
    let cg = graph.open(g)?;
    let n = cg.num_nodes();
    let w = window.window(n)?;
    let mut budget = g.budget();
    let algo = match algo {
        TrAlgo::Auto if w.len() <= BITSET_MAX_NODES => TrAlgo::Bitset,
        a => a,
    };
    let mut threads = None;
    let (red, report, ld, method) = match algo {
        TrAlgo::Auto => {
            let (r, red, ld) = verify(&cg, w, blocks, g, &mut budget)?;
            (red, Some(r), ld, "verify")
        }
        TrAlgo::Bitset | TrAlgo::Search => {
            let res = reduce_in_memory(&cg, w, algo, &mut budget)?;
            let method = if algo == TrAlgo::Bitset {
                "bitset"
            } else {
                "search"
            };
            threads = Some(res.threads);
            (res.redundant, None, None, method)
        }
    };
    // the redundant-arc list lives until the BV graph is written; on an arbitrary DAG it can
    // hold O(m) arcs, so it is accounted for before sizing the sort buffers
    budget.need(
        8 * red.len() as u64,
        "tr reduce: redundant-arc list (8 B/arc)",
    )?;
    log::info!("tr reduce: {} redundant arcs ({method})", red.len());
    let mem = topo::sort_memory(&mut budget, "tr reduce: external-sort buffers")?;
    let mut pl = logger("arc");
    let (nodes, arcs) = reduce_bv_window(&cg, w, &red, &out, &tmp, mem, &mut pl)?;
    let _ = std::fs::remove_dir(&tmp);
    let mut info = RunInfo::new(Some(graph.graph.clone()), n, cg.num_arcs, w);
    info.files.push(redundant_file(&out, &red)?);
    // --blocks only affects the verification policy; the in-memory algorithms ignore it
    let blocks_used = (method == "verify").then(|| blocks_name(&blocks.blocks));
    info.params = json!({ "algo": method, "blocks": blocks_used });
    info.results = json!({
        "output": out,
        "nodes": nodes,
        "arcs": arcs,
        "redundant": red.len(),
        "threads": threads,
        "report": report,
        "line_digraph_check": ld,
    });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(true))
}
