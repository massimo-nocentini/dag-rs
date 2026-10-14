//! `dag path-cover`, `dag frontier` and `dag width` (see [`crate::width`]).
//!
//! All three work on a window (`--window a..b`, default the full graph) and use the
//! [`BlockIndex`] fast path when it is available (`--blocks auto|PREFIX`), else the generic
//! algorithms on the decoded window. Per-node outputs are indexed by `v − lo`; node ids stored
//! as values are absolute.
//!
//! - `path-cover`: MPC_d by the closed form (blocks) or Hopcroft–Karp with the Kőnig
//!   certificate (generic); `--explicit` writes `P.next.u32le` (NONE = end of a path). With
//!   blocks and a window of at most 2²⁰ nodes, Hopcroft–Karp is also run as a cross-check.
//! - `frontier`: the largest prefix antichain; `--materialize` writes `P.antichain.u32le`
//!   after verifying it (size, range and, with blocks, [`check_antichain_blocks`]). With
//!   `--buckets FILE` and/or `--profile-step S` the JSON also holds the frontier curve
//!   `c ↦ |A_c|` ([`CutRange`]: value at the start and end of each node range and its maximum
//!   inside; on the PG, the UTXO set size over time), and with `--cuts FILE` (sorted `u64` node
//!   ids) `P.cuts.u64le`, the value of `|A_c|` at every cut.
//! - `width`: `--method closure` (blocks; self-certified by
//!   [`check_antichain_blocks`], cross-checked with min-flow on windows of at most 10⁶ nodes),
//!   `--method minflow` (generic; writes the chain partition `P.chains.u32le` +
//!   `P.chains_off.u64le`, `n` ids in all, certified by [`check_width_certificate`]) or `auto`
//!   (closure when a BlockIndex is available). On the full graph the closure needs
//!   `--experimental-full`; without it `auto` reports the bracket `[frontier, MPC_d]` and writes
//!   the frontier antichain, certified by [`check_antichain_blocks`]. `--flow push-relabel|dinic` picks the
//!   max-flow algorithm ([`FlowAlgo`]).
//!
//! Exit codes: 0 ok, 1 a certificate or cross-check failed, 2 error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Result, bail};
use clap::{Args, ValueEnum};
use dsi_progress_logger::prelude::*;
use serde_json::json;
use webgraph::prelude::*;

use super::common::{
    BlocksOpt, BucketsOpt, GlobalArgs, GraphOpt, OutOpt, WindowOpt, concurrent_logger, exit_status,
};
use crate::blocks::BlockIndex;
use crate::csr::Csr;
use crate::graph::ChunkedGraph;
use crate::io::{self, FileInfo, MemBudget, NONE, RunInfo};
use crate::summary::{DistConf, DistReport, distribution};
use crate::sweep::par_chunks;
use crate::width::{
    CutRange, Dinic, FlowAlgo, PUSH_RELABEL_MAX_NODES, check_antichain_blocks, check_konig,
    check_path_cover, check_width_certificate, frontier_at_cuts_blocks, frontier_at_cuts_generic,
    frontier_blocks, frontier_curve_blocks, frontier_curve_generic, frontier_generic,
    hopcroft_karp, konig_cover, min_path_cover_blocks, width_closure_blocks_with,
    width_min_flow_with,
};
use crate::window::Window;

/// Largest window on which `path-cover` cross-checks the closed form with Hopcroft–Karp.
pub const PATH_COVER_CROSS_CHECK: usize = 1 << 20;
/// Largest window on which `width --method closure` cross-checks with min-flow.
pub const WIDTH_CROSS_CHECK: usize = 1_000_000;

/// `dag path-cover`.
#[derive(Args, Debug)]
pub struct PathCoverArgs {
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
    /// Also write the cover (`next` array).
    #[arg(long)]
    pub explicit: bool,
}

/// `dag frontier`.
#[derive(Args, Debug)]
pub struct FrontierArgs {
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
    /// Also write the antichain.
    #[arg(long)]
    pub materialize: bool,
    /// Node-range buckets: the JSON reports `|A_c|` at the start and end of each bucket and
    /// its maximum inside (`curve.by_bucket`; on the PG the UTXO set over each half-year).
    #[command(flatten)]
    pub buckets: BucketsOpt,
    /// Also sample the curve `c ↦ |A_c|` on consecutive cut ranges of this many nodes
    /// (`curve.profile`; 0 = off).
    #[arg(long, default_value_t = 0)]
    pub profile_step: usize,
    /// Evaluate `|A_c|` at every cut of this file (little-endian `u64` node ids, sorted; cuts
    /// outside the window are dropped) and write the values to `P.cuts.u64le` (e.g. the PG time
    /// mapping's `block_first_node.u64`: the UTXO-set size at every blockchain block).
    #[arg(long)]
    pub cuts: Option<PathBuf>,
}

/// Width method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WidthMethod {
    /// Closure on blocks when available, else min-flow.
    Auto,
    /// Min-flow on an in-memory window.
    Minflow,
    /// Closure network on the BlockIndex.
    Closure,
}

/// `dag width`.
#[derive(Args, Debug)]
pub struct WidthArgs {
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
    /// Method.
    #[arg(long, value_enum, default_value = "auto")]
    pub method: WidthMethod,
    /// Allow the closure on the full PG (≈ 100+ GB; measured 1225 s on the first 10⁸ nodes
    /// with time growing ≈ n^1.7, so days on the full PG).
    #[arg(long)]
    pub experimental_full: bool,
    /// Max-flow algorithm of the closure and min-flow networks.
    #[arg(long, value_enum, default_value = "push-relabel")]
    pub flow: FlowAlgo,
}

/// Number of arcs of the subgraph induced by `w` (one parallel decode of the window).
fn window_arcs<G: RandomAccessGraph + Sync>(g: &ChunkedGraph<G>, w: Window) -> Result<u64> {
    let mut pl = concurrent_logger("node");
    pl.expected_updates(Some(w.len()));
    pl.start(format!("counting the arcs of {w}"));
    let m = par_chunks(
        g,
        w,
        &mut pl,
        |it| {
            let mut m = 0u64;
            while let Some((_, s)) = it.next_node() {
                m += s.iter().filter(|&&v| v as usize >= w.lo).count() as u64;
            }
            Ok(m)
        },
        |a, b| a + b,
    )?;
    pl.done();
    Ok(m)
}

/// Decodes the window into a [`Csr`] after reserving `8(|W| + 1) + 4m` bytes, and checks its
/// topology. Returns the Csr and the reserved bytes.
fn load_window<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    budget: &mut MemBudget,
) -> Result<(Csr, u64)> {
    let m = window_arcs(g, w)?;
    decode_window(g, w, m, budget)
}

/// [`load_window`] for a window whose arc count `m` is already known (from [`window_arcs`]).
fn decode_window<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    m: u64,
    budget: &mut MemBudget,
) -> Result<(Csr, u64)> {
    let bytes = 8 * (w.len() as u64 + 1) + 4 * m;
    budget.need(bytes, "window CSR")?;
    let mut pl = concurrent_logger("node");
    pl.expected_updates(Some(w.len()));
    pl.start(format!("decoding {w} into a CSR"));
    let csr = Csr::from_window(g, w, &mut pl)?;
    pl.done();
    csr.check_topological()?;
    Ok((csr, bytes))
}

/// Bytes of the Hopcroft–Karp working arrays for `n` nodes (mates, dist, cursor, queue, stack).
fn hk_bytes(n: usize) -> u64 {
    28 * n as u64
}

/// Bytes of [`konig_cover`] and [`check_konig`] for `n` nodes: two flag arrays and a queue for
/// the cover (`6n`), the cover itself (at most `n` ids, `4n`), and two flag arrays for the
/// check (`2n`).
fn konig_bytes(n: usize) -> u64 {
    12 * n as u64
}

/// Window of a generic min-flow run: the network-size guard and the memory estimate need only
/// `|W|` and the arc count, so both are checked before the window is decoded (on a window as
/// large as the PG, decoding alone would take ≈ 50 GB). Fails with the guard's error before any
/// pass over the graph.
fn load_minflow_window<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    flow: FlowAlgo,
    budget: &mut MemBudget,
) -> Result<Csr> {
    check_network_size(2 * w.len() + 2, flow)?;
    let m = window_arcs(g, w)?;
    budget.need(minflow_bytes(w.len(), m as usize), "min-flow network")?;
    Ok(decode_window(g, w, m, budget)?.0)
}

/// Bytes of [`width_min_flow`](crate::width::width_min_flow) on a window with `n` nodes and `m`
/// arcs: the network and its staging, Hopcroft–Karp, the decomposition arrays (`4m + 16n`), and
/// the chain partition (`4n`, `n` bits of marks, and the `4n` absolute-id copy written to disk).
fn minflow_bytes(n: usize, m: usize) -> u64 {
    let edges = 3 * n + m;
    let nn = n as u64;
    Dinic::bytes(2 * n + 2, edges)
        + 24 * edges as u64
        + hk_bytes(n)
        + 16 * nn
        + 4 * m as u64
        + 8 * nn
        + nn.div_ceil(8)
}

/// Fails if a flow network with `vertices` vertices is too large for `algo` (push–relabel
/// stores labels up to `2·vertices` as `u32`; Dinic needs `u32` vertex ids).
fn check_network_size(vertices: usize, algo: FlowAlgo) -> Result<()> {
    let max = match algo {
        FlowAlgo::PushRelabel => PUSH_RELABEL_MAX_NODES,
        FlowAlgo::Dinic => u32::MAX as usize,
    };
    if vertices >= max {
        bail!("the flow network has {vertices} vertices, {algo:?} supports fewer than {max}");
    }
    Ok(())
}

/// Bytes of [`width_closure_blocks`](crate::width::width_closure_blocks) on a window with `n` nodes and `nbw` blocks.
fn closure_bytes(n: usize, nbw: usize) -> u64 {
    let edges = nbw + n;
    Dinic::bytes(nbw + 2, edges) + 24 * edges as u64 + 4 * n as u64 + 16 * nbw as u64
}

/// The positions of an antichain (absolute ids, ascending) as a distribution of the ids
/// ([`crate::summary`]): `count` = size, `min`/`max`/`mean` = where in id order (on the PG: in
/// time) the antichain lies, and the log2 histogram of the ids.
pub fn antichain_ids(a: &[u32]) -> DistReport {
    distribution(
        a.len(),
        |i| Some(f64::from(a[i])),
        0,
        &DistConf {
            top: 0,
            buckets: None,
        },
    )
}

/// The chain lengths (nodes per chain) of a path cover given by `next` (indexed by `v − lo`,
/// absolute ids, [`NONE`] = path end): one entry per window node with a value at the chain
/// heads only, so `count` is the number of chains, `max`/`argmax` the longest chain and its
/// head, `top` the longest chains. Time `O(n_W)`, memory `n_W` bytes.
pub fn cover_chains(next: &[u32], w: Window, top: usize) -> DistReport {
    let mut has_pred = vec![false; next.len()];
    for &x in next {
        if x != NONE {
            has_pred[x as usize - w.lo] = true;
        }
    }
    let len = |i: usize| {
        let (mut v, mut l) = (i, 1u64);
        while next[v] != NONE {
            v = next[v] as usize - w.lo;
            l += 1;
        }
        l as f64
    };
    distribution(
        next.len(),
        |i| (!has_pred[i]).then(|| len(i)),
        w.lo,
        &DistConf { top, buckets: None },
    )
}

/// The chain lengths of a min-flow chain partition (CSR, window-relative ids): one entry per
/// chain, id = chain index, node = its first node (absolute).
fn csr_chains(c: &crate::width::Chains, lo: usize, top: usize) -> DistReport {
    let k = c.offsets.len().saturating_sub(1);
    let head = |i: usize| c.nodes[c.offsets[i] as usize] as usize + lo;
    crate::summary::distribution_with(
        k,
        |i| Some((c.offsets[i + 1] - c.offsets[i]) as f64),
        &|i| i,
        Some(&head),
        &DistConf { top, buckets: None },
    )
}

/// Writes a `u32` array and records it in `info`.
fn write_u32(
    info: &mut RunInfo,
    out: &Path,
    field: &str,
    data: &[u32],
    meaning: &str,
) -> Result<()> {
    let path = io::array_path::<u32>(out, field);
    io::write_array(&path, data)?;
    info.files.push(FileInfo::new::<u32>(
        path,
        data.len(),
        Some("u32::MAX = NONE"),
        meaning,
    ));
    Ok(())
}

/// Opens the graph and resolves the window and the block index (`want_blocks = false` skips the
/// index entirely).
fn setup(
    graph: &GraphOpt,
    blocks: &BlocksOpt,
    window: &WindowOpt,
    want_blocks: bool,
    g: &GlobalArgs,
    budget: &mut MemBudget,
) -> Result<(ChunkedGraph<crate::graph::BvRa>, Window, Option<BlockIndex>)> {
    let cg = graph.open(g)?;
    let w = window.window(cg.num_nodes())?;
    let idx = if want_blocks {
        blocks.resolve(&cg, g, budget)?
    } else {
        None
    };
    Ok((cg, w, idx))
}

fn run_info(graph: &GraphOpt, cg: &ChunkedGraph<crate::graph::BvRa>, w: Window) -> RunInfo {
    RunInfo::new(Some(graph.graph.clone()), cg.num_nodes(), cg.num_arcs, w)
}

/// Runs `dag path-cover`.
pub fn run_path_cover(args: &PathCoverArgs, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out: PathBuf = args.out.prepare()?.to_owned();
    let mut budget = g.budget();
    let (cg, w, idx) = setup(
        &args.graph,
        &args.blocks,
        &args.window,
        true,
        g,
        &mut budget,
    )?;
    let mut info = run_info(&args.graph, &cg, w);
    info.phases_s.add("open", t0.elapsed().as_secs_f64());
    let tk = Instant::now();
    let mut ok = true;
    let lo = w.lo as u32;
    let results = if let Some(b) = &idx {
        let nbw = b.block_bound(w.hi) - b.block_bound(w.lo);
        // explicit: next (4 B) + predecessor flag (1 B) per node, starts (≤ 4 B per node)
        budget.need(
            4 * nbw as u64 + if args.explicit { 9 * w.len() as u64 } else { 0 },
            "path-cover histogram and cover",
        )?;
        let t = Instant::now();
        let (stats, pc) = min_path_cover_blocks(b, w, args.explicit);
        let closed_form_s = t.elapsed().as_secs_f64();
        log::info!(
            "MPC_d({w}) = {} (matching {}, closed form, {closed_form_s:.3} s)",
            stats.mpc_d,
            stats.matching
        );
        let mut cross = serde_json::Value::Null;
        if w.len() <= PATH_COVER_CROSS_CHECK {
            budget.need(
                hk_bytes(w.len())
                    + konig_bytes(w.len())
                    + if args.explicit { 9 * w.len() as u64 } else { 0 },
                "Hopcroft–Karp, Kőnig and explicit-cover checks",
            )?;
            let (csr, _) = load_window(&cg, w, &mut budget)?;
            let m = hopcroft_karp(&csr);
            let cover = konig_cover(&csr, &m);
            let konig = check_konig(&csr, &m, &cover);
            let explicit_ok = pc.as_ref().map(|pc| {
                let rel = crate::width::PathCover {
                    next: pc
                        .next
                        .iter()
                        .map(|&x| if x == NONE { NONE } else { x - lo })
                        .collect(),
                    starts: pc.starts.iter().map(|&x| x - lo).collect(),
                };
                check_path_cover(&csr, &rel).map(|k| k as u64 == stats.mpc_d)
            });
            let agree = m.size as u64 == stats.matching;
            let explicit_good = explicit_ok.as_ref().is_none_or(|r| matches!(r, Ok(true)));
            if !agree || konig.is_err() || !explicit_good {
                log::error!(
                    "path-cover cross-check failed: HK {} vs closed form {}, Kőnig {konig:?}, explicit {explicit_ok:?}",
                    m.size,
                    stats.matching
                );
                ok = false;
            }
            cross = json!({
                "hk_matching": m.size,
                "agree": agree,
                "konig_ok": konig.is_ok(),
                "explicit_ok": explicit_good,
            });
        }
        let mut chains = serde_json::Value::Null;
        if let Some(pc) = &pc {
            write_u32(
                &mut info,
                &out,
                "next",
                &pc.next,
                "successor on the path cover (absolute id; NONE = path end)",
            )?;
            chains = json!(cover_chains(&pc.next, w, 20));
        }
        json!({
            "method": "blocks",
            "n": stats.n,
            "matching": stats.matching,
            "mpc_d": stats.mpc_d,
            "closed_form_s": closed_form_s,
            "cross_check": cross,
            "chains": chains,
        })
    } else {
        let (csr, _) = load_window(&cg, w, &mut budget)?;
        budget.need(
            hk_bytes(w.len()) + konig_bytes(w.len()),
            "Hopcroft–Karp and Kőnig certificate",
        )?;
        let t = Instant::now();
        let m = hopcroft_karp(&csr);
        let hk_s = t.elapsed().as_secs_f64();
        let cover = konig_cover(&csr, &m);
        let konig = check_konig(&csr, &m, &cover);
        if let Err(e) = &konig {
            log::error!("Kőnig certificate failed: {e:#}");
            ok = false;
        }
        let mpc_d = (w.len() - m.size) as u64;
        log::info!(
            "MPC_d({w}) = {mpc_d} (matching {}, Hopcroft–Karp, {hk_s:.3} s)",
            m.size
        );
        let mut chains = serde_json::Value::Null;
        if args.explicit {
            let next: Vec<u32> = m
                .mate_l
                .iter()
                .map(|&x| if x == NONE { NONE } else { x + lo })
                .collect();
            write_u32(
                &mut info,
                &out,
                "next",
                &next,
                "successor on the path cover (absolute id; NONE = path end)",
            )?;
            chains = json!(cover_chains(&next, w, 20));
        }
        json!({
            "method": "hk",
            "chains": chains,
            "n": w.len(),
            "arcs": csr.num_arcs(),
            "matching": m.size,
            "mpc_d": mpc_d,
            "hk_s": hk_s,
            "konig_ok": konig.is_ok(),
            "konig_cover": cover.0.len() + cover.1.len(),
        })
    };
    // closed form or matching, certificates, cross-checks and the cover
    info.phases_s.add("kernel", tk.elapsed().as_secs_f64());
    info.params = json!({ "explicit": args.explicit });
    info.results = results;
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}

/// Runs `dag frontier`.
pub fn run_frontier(args: &FrontierArgs, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out: PathBuf = args.out.prepare()?.to_owned();
    let mut budget = g.budget();
    let (cg, w, idx) = setup(
        &args.graph,
        &args.blocks,
        &args.window,
        true,
        g,
        &mut budget,
    )?;
    let mut info = run_info(&args.graph, &cg, w);
    info.phases_s.add("open", t0.elapsed().as_secs_f64());
    let t = Instant::now();
    let (f, method) = if let Some(b) = &idx {
        let nbw = b.block_bound(w.hi) - b.block_bound(w.lo);
        budget.need(
            4 * nbw as u64
                + if args.materialize {
                    4 * w.len() as u64
                } else {
                    0
                },
            "frontier histogram and members",
        )?;
        (frontier_blocks(b, w, args.materialize), "blocks")
    } else {
        budget.need(
            4 * w.len() as u64
                + if args.materialize {
                    4 * w.len() as u64
                } else {
                    0
                },
            "frontier histogram and members",
        )?;
        let mut pl = concurrent_logger("node");
        (
            frontier_generic(&cg, w, args.materialize, &mut pl)?,
            "generic",
        )
    };
    let frontier_s = t.elapsed().as_secs_f64();
    info.phases_s.add("kernel", frontier_s);
    log::info!(
        "frontier({w}) = {} at cut {} ({method}, {frontier_s:.3} s)",
        f.size,
        f.cut
    );
    let curve = info.phases_s.time("summary", || {
        frontier_curve(args, &cg, idx.as_ref(), w, &mut budget)
    })?;
    let cuts = match &args.cuts {
        Some(f) => {
            let tc = Instant::now();
            let r = frontier_cuts(f, &cg, idx.as_ref(), w, &mut budget, &out, &mut info)?;
            info.phases_s.add("summary", tc.elapsed().as_secs_f64());
            Some(r)
        }
        None => None,
    };
    let cert = verify_frontier(idx.as_ref(), w, &f);
    if let Err(e) = &cert {
        log::error!("frontier certificate failed: {e:#}");
    }
    if let Some(a) = &f.members {
        write_u32(
            &mut info,
            &out,
            "antichain",
            a,
            "members of the frontier antichain A_c (absolute ids, ascending)",
        )?;
    }
    info.params = json!({
        "materialize": args.materialize,
        "buckets": args.buckets.buckets,
        "profile_step": args.profile_step,
        "cuts": args.cuts,
    });
    info.results = json!({
        "method": method,
        "n": w.len(),
        "size": f.size,
        "cut": f.cut,
        "frontier_s": frontier_s,
        "certificate_ok": f.members.as_ref().map(|_| cert.is_ok()),
        "antichain_ids": f.members.as_deref().map(antichain_ids),
        "curve": curve,
        "cuts": cuts,
    });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(cert.is_ok()))
}

/// The frontier curve of `run_frontier`: `|A_c|` statistics per bucket and per
/// `--profile-step` range (one extra scan; `None` when neither is requested).
fn frontier_curve(
    args: &FrontierArgs,
    cg: &ChunkedGraph<crate::graph::BvRa>,
    idx: Option<&BlockIndex>,
    w: Window,
    budget: &mut crate::io::MemBudget,
) -> Result<Option<serde_json::Value>> {
    let buckets = args.buckets.load(cg.num_nodes())?;
    let bucket_ranges: Vec<(String, usize, usize)> = buckets
        .iter()
        .flat_map(|b| b.as_slice())
        .map(|b| (b.label.clone(), b.lo, b.hi))
        .collect();
    let profile_ranges: Vec<(String, usize, usize)> = if args.profile_step > 0 {
        (w.lo..w.hi.max(w.lo + 1))
            .step_by(args.profile_step)
            .map(|a| (a.to_string(), a, (a + args.profile_step).min(w.hi)))
            .collect()
    } else {
        Vec::new()
    };
    if bucket_ranges.is_empty() && profile_ranges.is_empty() {
        return Ok(None);
    }
    let mut curve = |r: &[(String, usize, usize)]| -> Result<Vec<CutRange>> {
        if r.is_empty() {
            return Ok(Vec::new());
        }
        match idx {
            Some(b) => {
                let nbw = b.block_bound(w.hi) - b.block_bound(w.lo);
                budget.need(4 * nbw as u64, "frontier curve histogram")?;
                let c = frontier_curve_blocks(b, w, r);
                budget.release(4 * nbw as u64);
                Ok(c)
            }
            None => {
                budget.need(4 * w.len() as u64, "frontier curve histogram")?;
                let c = frontier_curve_generic(cg, w, r, &mut concurrent_logger("node"))?;
                budget.release(4 * w.len() as u64);
                Ok(c)
            }
        }
    };
    Ok(Some(json!({
        "by_bucket": curve(&bucket_ranges)?,
        "profile": curve(&profile_ranges)?,
    })))
}

/// `--cuts FILE` of `run_frontier`: `|A_c|` at every cut of the file inside the window, written
/// to `P.cuts.u64le` (one `u64` per kept cut, in file order); returns a JSON summary.
fn frontier_cuts(
    file: &Path,
    cg: &ChunkedGraph<crate::graph::BvRa>,
    idx: Option<&BlockIndex>,
    w: Window,
    budget: &mut crate::io::MemBudget,
    out: &Path,
    info: &mut RunInfo,
) -> Result<serde_json::Value> {
    let all: Vec<u64> = io::read_array(file)?;
    let cuts: Vec<usize> = all
        .iter()
        .map(|&c| c as usize)
        .filter(|&c| c >= w.lo && c <= w.hi)
        .collect();
    let bytes = match idx {
        Some(b) => 4 * (b.block_bound(w.hi) - b.block_bound(w.lo)) as u64,
        None => 4 * w.len() as u64,
    } + 16 * cuts.len() as u64;
    budget.need(bytes, "frontier at cuts")?;
    let vals = match idx {
        Some(b) => frontier_at_cuts_blocks(b, w, &cuts)?,
        None => frontier_at_cuts_generic(cg, w, &cuts, &mut concurrent_logger("node"))?,
    };
    budget.release(bytes);
    let path = io::array_path::<u64>(out, "cuts");
    io::write_array(&path, &vals)?;
    info.files.push(FileInfo::new::<u64>(
        &path,
        vals.len(),
        None,
        format!(
            "|A_c| at the cuts of {} inside the window, in file order",
            file.display()
        ),
    ));
    let (imax, vmax) = vals
        .iter()
        .enumerate()
        .fold((0, 0), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    Ok(json!({
        "file": file,
        "cuts_in_file": all.len(),
        "cuts": vals.len(),
        "max": vmax,
        "argmax_cut": cuts.get(imax),
        "last": vals.last(),
    }))
}

/// Verifies a materialized frontier: `|members| == size`, the members lie in `[lo, cut)`, and,
/// with a BlockIndex, they form an antichain ([`check_antichain_blocks`]; the generic members
/// are the nodes of `[lo, cut)` without a successor in `[lo, cut)`, an antichain by
/// construction). Without members there is nothing to verify.
fn verify_frontier(
    blocks: Option<&BlockIndex>,
    w: Window,
    f: &crate::width::Frontier,
) -> Result<()> {
    let Some(a) = &f.members else {
        return Ok(());
    };
    anyhow::ensure!(
        a.len() as u64 == f.size,
        "materialized {} nodes, frontier size {}",
        a.len(),
        f.size
    );
    anyhow::ensure!(
        a.iter().all(|&u| (w.lo..f.cut).contains(&(u as usize))),
        "a member lies outside [{}, {})",
        w.lo,
        f.cut
    );
    if let Some(b) = blocks {
        check_antichain_blocks(b, w, a)?;
    }
    Ok(())
}

/// Writes the chains of a min-flow result (absolute ids) and records them.
fn write_chains(info: &mut RunInfo, out: &Path, c: &crate::width::Chains, lo: u32) -> Result<()> {
    let nodes: Vec<u32> = c.nodes.iter().map(|&x| x + lo).collect();
    write_u32(
        info,
        out,
        "chains",
        &nodes,
        "concatenated chains (paths of the graph, absolute ids)",
    )?;
    let path = io::array_path::<u64>(out, "chains_off");
    io::write_array(&path, &c.offsets)?;
    info.files.push(FileInfo::new::<u64>(
        path,
        c.offsets.len(),
        None,
        "chain i is chains[off[i]..off[i+1]]",
    ));
    Ok(())
}

/// Runs `dag width`.
pub fn run_width(args: &WidthArgs, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out: PathBuf = args.out.prepare()?.to_owned();
    let mut budget = g.budget();
    let (cg, w, idx) = setup(
        &args.graph,
        &args.blocks,
        &args.window,
        args.method != WidthMethod::Minflow,
        g,
        &mut budget,
    )?;
    let n = cg.num_nodes();
    let full = w.is_full(n);
    let mut info = run_info(&args.graph, &cg, w);
    info.params = json!({ "method": format!("{:?}", args.method).to_lowercase(), "experimental_full": args.experimental_full, "flow": args.flow });
    let lo = w.lo as u32;
    let mut ok = true;
    let use_closure = match (args.method, &idx) {
        (WidthMethod::Closure, None) => {
            bail!("--method closure needs a BlockIndex (--blocks auto|PREFIX on a line digraph)")
        }
        (WidthMethod::Closure, Some(_)) => true,
        (WidthMethod::Auto, Some(_)) => true,
        _ => false,
    };
    if use_closure && full && !args.experimental_full {
        if args.method == WidthMethod::Closure {
            bail!(
                "the closure on the full graph is experimental (≈ 100+ GB on the PG): pass --experimental-full, or use a window"
            );
        }
        // the default full-graph answer: the bracket [frontier, MPC_d]
        let b = idx.as_ref().expect("closure implies blocks");
        budget.need(4 * (b.nb + w.len()) as u64, "frontier + path cover")?;
        let f = frontier_blocks(b, w, true);
        let (pc, _) = min_path_cover_blocks(b, w, false);
        log::info!(
            "width bracket: {} (frontier at cut {}) ≤ width ≤ {} (MPC_d)",
            f.size,
            f.cut,
            pc.mpc_d
        );
        let a = f.members.as_deref().unwrap_or(&[]);
        let cert = check_antichain_blocks(b, w, a).and_then(|()| {
            anyhow::ensure!(
                a.len() as u64 == f.size,
                "antichain has {} nodes, frontier {}",
                a.len(),
                f.size
            );
            Ok(())
        });
        if let Err(e) = &cert {
            log::error!("frontier antichain certificate failed: {e:#}");
        }
        write_u32(
            &mut info,
            &out,
            "antichain",
            a,
            "frontier antichain (a lower bound on the width; absolute ids)",
        )?;
        info.results = json!({
            "method": "bracket",
            "exact": false,
            "n": w.len(),
            "lower": f.size,
            "lower_cut": f.cut,
            "upper": pc.mpc_d,
            "certificate_ok": cert.is_ok(),
            "antichain_ids": antichain_ids(a),
        });
        info.elapsed_s = t0.elapsed().as_secs_f64();
        info.write(&out)?;
        return Ok(exit_status(cert.is_ok()));
    }
    let results = if use_closure {
        let b = idx.as_ref().expect("closure implies blocks");
        let nbw = b.block_bound(w.hi) - b.block_bound(w.lo);
        check_network_size(nbw + 2, args.flow)?;
        budget.need(closure_bytes(w.len(), nbw), "closure network")?;
        let t = Instant::now();
        let r = width_closure_blocks_with(b, w, args.flow);
        let closure_s = t.elapsed().as_secs_f64();
        let cert = check_antichain_blocks(b, w, &r.antichain).and_then(|()| {
            anyhow::ensure!(
                r.antichain.len() as u64 == r.width,
                "antichain has {} nodes, width {}",
                r.antichain.len(),
                r.width
            );
            Ok(())
        });
        if let Err(e) = &cert {
            log::error!("closure width certificate failed: {e:#}");
            ok = false;
        }
        log::info!(
            "width({w}) = {} (closure on {nbw} blocks, {closure_s:.3} s)",
            r.width
        );
        let mut cross = serde_json::Value::Null;
        if w.len() <= WIDTH_CROSS_CHECK {
            let csr = load_minflow_window(&cg, w, args.flow, &mut budget)?;
            let t = Instant::now();
            let mf = width_min_flow_with(&csr, args.flow);
            let minflow_s = t.elapsed().as_secs_f64();
            let mf_cert = check_width_certificate(&csr, &mf);
            let agree = mf.width == r.width;
            if !agree || mf_cert.is_err() {
                log::error!(
                    "width cross-check failed: min-flow {} vs closure {}, certificate {mf_cert:?}",
                    mf.width,
                    r.width
                );
                ok = false;
            }
            cross = json!({ "minflow_width": mf.width, "agree": agree, "minflow_certificate_ok": mf_cert.is_ok(), "minflow_s": minflow_s });
        }
        write_u32(
            &mut info,
            &out,
            "antichain",
            &r.antichain,
            "maximum antichain (absolute ids, ascending)",
        )?;
        json!({
            "method": "closure",
            "exact": true,
            "n": w.len(),
            "blocks": nbw,
            "width": r.width,
            "closure_s": closure_s,
            "certificate_ok": cert.is_ok(),
            "cross_check": cross,
            "antichain_ids": antichain_ids(&r.antichain),
        })
    } else {
        let csr = load_minflow_window(&cg, w, args.flow, &mut budget)?;
        let t = Instant::now();
        let r = width_min_flow_with(&csr, args.flow);
        let minflow_s = t.elapsed().as_secs_f64();
        let cert = check_width_certificate(&csr, &r);
        if let Err(e) = &cert {
            log::error!("min-flow width certificate failed: {e:#}");
            ok = false;
        }
        log::info!("width({w}) = {} (min-flow, {minflow_s:.3} s)", r.width);
        let a: Vec<u32> = r.antichain.iter().map(|&x| x + lo).collect();
        write_u32(
            &mut info,
            &out,
            "antichain",
            &a,
            "maximum antichain (absolute ids, ascending)",
        )?;
        let chain_nodes = r.chains.as_ref().map_or(0, |c| c.nodes.len());
        if let Some(c) = &r.chains {
            write_chains(&mut info, &out, c, lo)?;
        }
        let chains = r.chains.as_ref().map(|c| csr_chains(c, w.lo, 20));
        json!({
            "method": "minflow",
            "exact": true,
            "n": w.len(),
            "arcs": csr.num_arcs(),
            "width": r.width,
            "chain_nodes": chain_nodes,
            "chains": chains,
            "minflow_s": minflow_s,
            "certificate_ok": cert.is_ok(),
            "antichain_ids": antichain_ids(&a),
        })
    };
    info.results = results;
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_size_guard() {
        // push-relabel labels reach 2·vertices − 1 in u32
        assert!(check_network_size((1 << 31) - 1, FlowAlgo::PushRelabel).is_ok());
        assert!(check_network_size(1 << 31, FlowAlgo::PushRelabel).is_err());
        assert!(check_network_size(2 * 1_100_000_000 + 2, FlowAlgo::PushRelabel).is_err());
        assert!(check_network_size(1 << 31, FlowAlgo::Dinic).is_ok());
        assert!(check_network_size(u32::MAX as usize, FlowAlgo::Dinic).is_err());
        // labels of the largest accepted network do not overflow
        let n = PUSH_RELABEL_MAX_NODES - 1;
        assert!(2u32.checked_mul(n as u32).is_some());
    }

    /// Regression: a bad materialized frontier is reported as a failed
    /// certificate (exit 1), not a panic.
    #[test]
    fn verify_frontier_rejects_bad_members() {
        use crate::testkit::{self, MiniPgOpts};
        let g = testkit::mini_pg(200, 3, &MiniPgOpts::default()).graph;
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let w = Window::full(g.num_nodes());
        let f = frontier_blocks(&blocks, w, true);
        assert!(f.size > 1);
        verify_frontier(Some(&blocks), w, &f).unwrap();
        verify_frontier(None, w, &f).unwrap();
        // nothing materialized: nothing to verify
        let bare = crate::width::Frontier {
            members: None,
            ..f.clone()
        };
        verify_frontier(Some(&blocks), w, &bare).unwrap();
        // wrong size
        let mut bad = f.clone();
        bad.size += 1;
        assert!(verify_frontier(None, w, &bad).is_err());
        // a member outside [lo, cut)
        let mut bad = f.clone();
        let a = bad.members.as_mut().unwrap();
        a.pop();
        a.push(f.cut as u32);
        assert!(verify_frontier(None, w, &bad).is_err());
        // a comparable pair: replace a member by a node it reaches (its spender's first
        // output), keeping the size
        let a = f.members.as_ref().unwrap();
        let (u, v) = a
            .iter()
            .find_map(|&u| {
                let b = blocks.spend[u as usize];
                (b != NONE).then(|| (u, blocks.first[b as usize]))
            })
            .expect("a member with a spender");
        let mut bad = f.clone();
        let m = bad.members.as_mut().unwrap();
        m.retain(|&x| x != u && x != v);
        m.push(u);
        m.push(v);
        m.sort_unstable();
        bad.cut = bad.cut.max(v as usize + 1);
        bad.size = m.len() as u64;
        assert!(verify_frontier(Some(&blocks), w, &bad).is_err());
        assert_eq!(exit_status(false), ExitCode::from(1));
    }

    #[test]
    fn minflow_estimate_counts_the_chains() {
        // the chain partition (4n ids + n/8 marks + 4n written copy) is part of the estimate
        let (n, m) = (1_000_000, 3_000_000);
        let edges = 3 * n + m;
        let base = Dinic::bytes(2 * n + 2, edges)
            + 24 * edges as u64
            + hk_bytes(n)
            + 16 * n as u64
            + 4 * m as u64;
        assert_eq!(
            minflow_bytes(n, m) - base,
            8 * n as u64 + (n as u64).div_ceil(8)
        );
    }
}
