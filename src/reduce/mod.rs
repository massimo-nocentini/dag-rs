//! Transitive reduction: verification, certificates and exact reduction.
//!
//! The transitive reduction `G⁻` of a DAG is the unique minimal subgraph with the same
//! reachability ([\[AGU72\]](crate::references#agu72)): an arc `(u, v)` is *redundant* iff
//! there is a path `u ⇝ v` of length `≥ 2`. This module decides whether a (possibly huge)
//! topologically numbered graph is already reduced, lists its redundant arcs exactly, and
//! writes the reduced graph.
//!
//! # Verification policy ([`verify_reduced`], [`verify_reduced_window`])
//! ```text
//! 1. blocks:  a BlockIndex (or the window check certificate::line_digraph_window) holds
//!             ⇒ reduced, certificate "blocks"         (line digraph of a DAG [HN60, Heu64])
//! 2. H1:      F = minsucc, L = maxsucc in one parallel scan; any arc v ≤ u is an error
//! 3. H3:      t_u = min{F[w] : w ∈ succ(u), F[w] ≠ NONE, F[w] ≤ L[u]};
//!             no u has a t_u ⇒ reduced, certificate "H3"
//! 4. exact:   for each u with t_u, search from succ(succ(u)) up to the largest candidate
//!             v ∈ succ(u), v ≥ t_u; (u, v) is redundant iff v is reached; certificate "exact"
//! ```
//! Step 4 is complete (every redundant arc is a candidate) and sound (a marked candidate has a
//! path of length `≥ 2`); see [`exact`]. Steps 1 and 3 are proved in [`certificate`].
//!
//! # Windows
//! Ids are topological, so a window `[lo, hi)` is convex: every path between two window nodes
//! stays inside it. The redundant arcs of the induced subgraph are therefore exactly the
//! redundant arcs of `G` with both ends in the window, and every function here takes a window
//! (the plain variants use the whole graph). Node ids in outputs are **absolute**.
//!
//! # Exact reduction
//! [`exact::transitive_reduction_bitset`] (triangular bitsets, windows of at most
//! 2¹⁶ nodes) and [`exact::transitive_reduction_search`] (pruned per-node searches)
//! compute `G⁻` of an in-memory [`Csr`](crate::csr::Csr); they follow the topological-order
//! schemes of [\[GK79\]](crate::references#gk79), [\[Sim88\]](crate::references#sim88) and
//! [\[HMR93\]](crate::references#hmr93). [`reduce_bv_window`] writes a window minus a list of
//! redundant arcs as a BV graph ([\[BV04\]](crate::references#bv04)) with its transpose.
//!
//! # Complexity
//! Certificates: one (blocks, H1) or two (H1 + H3) parallel decodes of the window, O(n + m).
//! Exact step: one bounded search per H3-failing node, O(Σ_u (nodes and arcs in
//! `(u, max candidate]`)) in the worst case, run with a per-thread
//! [`DirtyBitset`](crate::bitset::DirtyBitset) and a thread count capped by the
//! [`MemBudget`]. Memory: `F` and `L` take 8 bytes per window node.
//!
//! # Payment Graph fast path
//! The PG is the line digraph of the transaction graph: every successor set is the output set
//! of the spending transaction, so two successor sets are identical or disjoint and the
//! identity is topological. A successful [`BlockIndex`] build (or the window check)
//! certifies that the PG and every window of it are transitively reduced, without searching
//! (see the crate documentation); the generic H3 path reaches the same verdict with zero
//! failing nodes.
//!
//! # Invariants (tested against brute-force closures)
//! `closure(G⁻) = closure(G)`, `G⁻ ⊆ G`, `(G⁻)⁻ = G⁻`, the bitset and search algorithms agree
//! with each other and with a Floyd–Warshall oracle, H3 ⇒ reduced, and the verifier reports
//! exactly the oracle's redundant arcs.

pub mod certificate;
pub mod exact;

use std::path::Path;

use anyhow::{Context, Result};
use dsi_bitstream::prelude::BE;
use dsi_progress_logger::prelude::*;
use lender::prelude::*;
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::blocks::BlockIndex;
use crate::graph::{BvRa, ChunkedGraph};
use crate::io::{self, MemBudget};
use crate::sweep::SweepConf;
use crate::topo;
use crate::window::Window;

pub use certificate::{
    LineDigraphReport, MinMax, blocks_window_arcs, h3_failures, h3_failures_in,
    line_digraph_window, min_max_succ,
};
pub use exact::{
    ExactOutcome, InMemoryOutcome, exact_redundant, redundant_arcs_bitset, redundant_arcs_in,
    redundant_arcs_of, redundant_arcs_search, search_span, transitive_reduction_bitset,
    transitive_reduction_bitset_logged, transitive_reduction_search,
    transitive_reduction_search_logged,
};

/// Largest window reduced with [`transitive_reduction_bitset`] by the `auto` choice (2¹⁶).
pub const BITSET_MAX_NODES: usize = 1 << 16;

/// Outcome of [`verify_reduced`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ReductionReport {
    /// Arcs of the (window-induced) graph.
    pub arcs: u64,
    /// Arcs violating the topological order. Always 0 in a returned report: a violation is
    /// returned as [`DagError::NotTopological`](crate::DagError::NotTopological) (D1).
    pub h1_violations: u64,
    /// Whether the line-digraph certificate holds (a [`BlockIndex`] was given or built, or
    /// [`line_digraph_window`] succeeded).
    pub line_digraph: bool,
    /// Nodes where the H3 certificate fails (0 when the blocks certificate was used).
    pub h3_failing_nodes: u64,
    /// Candidate arcs re-checked exactly.
    pub exact_checked_arcs: u64,
    /// Redundant arcs found.
    pub redundant: u64,
    /// Whether the graph (window) is transitively reduced.
    pub is_transitively_reduced: bool,
    /// The certificate of the verdict: `"blocks"`, `"H3"` or `"exact"`.
    pub certificate: &'static str,
}

impl ReductionReport {
    /// The report of a graph certified by the line-digraph ("blocks") certificate.
    pub fn from_blocks(arcs: u64) -> Self {
        Self {
            arcs,
            h1_violations: 0,
            line_digraph: true,
            h3_failing_nodes: 0,
            exact_checked_arcs: 0,
            redundant: 0,
            is_transitively_reduced: true,
            certificate: "blocks",
        }
    }
}

/// [`verify_reduced_window`] on the whole graph.
pub fn verify_reduced<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    blocks: Option<&BlockIndex>,
    conf: &SweepConf,
    budget: &mut MemBudget,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<(ReductionReport, Vec<(u32, u32)>)> {
    verify_reduced_window(g, Window::full(g.num_nodes()), blocks, conf, budget, pl)
}

/// Verifies that the subgraph induced by `w` is transitively reduced and returns the report
/// and its redundant arcs (absolute ids, sorted), following the policy of the module
/// documentation: the blocks certificate if `blocks` is given, else H1, H3 and the exact
/// re-check of the H3 failures.
///
/// Memory: with `blocks`, none besides the index; otherwise [`MinMax::bytes`] for `F` and `L`,
/// the list of H3 failures (8 bytes per failing node, reserved as soon as it is built, since its
/// size is only known then) and the exact searches (see [`exact_redundant`]), all reserved in
/// `budget` and released on return. The returned redundant arcs (8 bytes per arc) are not
/// reserved: the caller owns them. Both lists are empty on the Payment Graph and `O(n)`, `O(m)`
/// on an arbitrary DAG. All passes are [`par_chunks`](crate::sweep::par_chunks) scans (no ordered
/// consumer is needed), so `conf` only documents the caller's sweep geometry and is not used.
///
/// Fails with [`DagError::NotTopological`](crate::DagError::NotTopological) if some arc of
/// the window has `v ≤ u`.
pub fn verify_reduced_window<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    blocks: Option<&BlockIndex>,
    _conf: &SweepConf,
    budget: &mut MemBudget,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<(ReductionReport, Vec<(u32, u32)>)> {
    let n = g.num_nodes();
    anyhow::ensure!(w.hi <= n, "window {w} out of range for {n} nodes");
    if let Some(idx) = blocks {
        anyhow::ensure!(
            idx.n == n,
            "the BlockIndex has {} nodes, the graph {n}",
            idx.n
        );
        let arcs = blocks_window_arcs(idx, w);
        log::info!("tr: window {w} certified by the BlockIndex ({arcs} arcs)");
        return Ok((ReductionReport::from_blocks(arcs), Vec::new()));
    }
    let bytes = MinMax::bytes(w.len());
    budget.need(bytes, "tr: F and L (8 B/node)")?;
    let res = (|| {
        let mm = min_max_succ(g, w, pl)?;
        let fails = h3_failures_in(g, w, &mm.f, &mm.l, pl)?;
        log::info!(
            "tr: window {w}: {} arcs, H3 fails at {} nodes",
            mm.arcs,
            fails.len()
        );
        let fail_bytes = 8 * fails.len() as u64;
        budget.need(fail_bytes, "tr: H3 failures (8 B/failing node)")?;
        let ex = exact_redundant(&g.graph, w, &fails, &mm.l, budget, pl);
        budget.release(fail_bytes);
        let ex = ex?;
        let report = ReductionReport {
            arcs: mm.arcs,
            h1_violations: 0,
            line_digraph: false,
            h3_failing_nodes: fails.len() as u64,
            exact_checked_arcs: ex.checked_arcs,
            redundant: ex.redundant.len() as u64,
            is_transitively_reduced: ex.redundant.is_empty(),
            certificate: if fails.is_empty() { "H3" } else { "exact" },
        };
        Ok((report, ex.redundant))
    })();
    budget.release(bytes);
    res
}

/// Writes `g` minus the `redundant` arcs (sorted, absolute ids) as the BV graph `out` (+ `.ef`,
/// `.dcf`) and its transpose `out-t`, using webgraph's default sort memory and `<out>.tmp` for
/// the sort batches (see [`reduce_bv_window`]).
pub fn reduce_bv(g: &ChunkedGraph<BvRa>, redundant: &[(u32, u32)], out: &Path) -> Result<()> {
    let tmp = io::suffixed(out, "tmp");
    reduce_bv_window(
        g,
        Window::full(g.num_nodes()),
        redundant,
        out,
        &tmp,
        MemoryUsage::default(),
        &mut Option::<ProgressLogger>::None,
    )?;
    let _ = std::fs::remove_dir(&tmp);
    Ok(())
}

/// Number of pieces each chunk is split into by [`reduce_bv_window`] (bounds the per-thread
/// arc buffers to `chunk_arcs / REDUCE_SPLIT` arcs on average).
pub const REDUCE_SPLIT: usize = topo::INDUCED_SPLIT;

/// Writes the subgraph induced by `w` minus the `redundant` arcs (sorted, absolute ids) as the
/// BV graph `out` with ids shifted by `−w.lo` (+ `.ef`, `.dcf`) and its transpose `out-t`;
/// `tmp` holds the sort batches. Returns the number of nodes and arcs written.
///
/// The window is decoded in parallel pieces; each arc is dropped iff it is found (binary
/// search) in `redundant`. Heap: the sort buffers `mem` plus, per rayon thread, the arcs of one
/// piece at 16 bytes per arc.
pub fn reduce_bv_window<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    redundant: &[(u32, u32)],
    out: &Path,
    tmp: &Path,
    mem: MemoryUsage,
    pl: &mut impl ProgressLog,
) -> Result<(usize, u64)> {
    anyhow::ensure!(
        redundant.windows(2).all(|p| p[0] < p[1]),
        "the redundant arcs must be sorted and distinct"
    );
    let n = w.len();
    std::fs::create_dir_all(tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
    let graph = &g.graph;
    let (lo, hi) = (w.lo, w.hi);
    let pieces: Vec<(usize, usize)> = g
        .chunks_in(w)
        .into_iter()
        .flat_map(|(s, e)| {
            let step = (e - s).div_ceil(REDUCE_SPLIT).max(1);
            (s..e).step_by(step).map(move |a| (a, (a + step).min(e)))
        })
        .collect();
    let pairs = pieces.into_par_iter().flat_map_iter(move |(s, e)| {
        let mut arcs = Vec::new();
        let mut it = graph.iter_from(s).take(e - s);
        while let Some((u, succ)) = it.next() {
            arcs.extend(
                succ.into_iter()
                    .filter(|&v| {
                        v >= lo && v < hi && redundant.binary_search(&(u as u32, v as u32)).is_err()
                    })
                    .map(|v| (u - lo, v - lo)),
            );
        }
        arcs
    });
    let sorted = ParSortedGraph::config()
        .memory_usage(mem)
        .progress_logger(&mut *pl)
        .par_sort_pairs(n, pairs)?;
    BvCompConf::new(out)
        .tmp_dir(tmp)
        .par_comp::<BE, _>(sorted)?;
    topo::finish_bv(out, n)?;
    let m = crate::graph::load_ra(out)?.num_arcs();
    topo::write_transpose(out, tmp, mem, pl)?;
    Ok((n, m))
}
