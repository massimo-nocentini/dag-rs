//! Transitive-reduction certificates: the line-digraph ("blocks") certificate, the H1 scan
//! with the `minsucc`/`maxsucc` arrays, and the local order certificate H3.
//!
//! All passes here are linear and parallel ([`par_chunks`]); none needs the transpose.
//!
//! # Line-digraph certificate ("blocks")
//! **Theorem** ([\[HN60\]](crate::references#hn60), [\[Heu64\]](crate::references#heu64); F4
//! §0.2): if the identity is a topological order and any two successor sets are identical or
//! disjoint, the graph is transitively reduced. *Proof sketch.* The graph is then the line
//! digraph `L(T)` of an acyclic multidigraph `T`; an arc `e → f` of `L(T)` means
//! `head(e) = tail(f) = X`, and a longer path `e → g₁ → … → g_k → f` would make `g₁ … g_k` a
//! closed walk through `X` in `T`, a cycle. ∎
//!
//! A [`BlockIndex`] that builds successfully certifies the
//! hypotheses for the whole graph and for every window. [`line_digraph_window`] checks the same
//! properties P1–P3 of [`crate::blocks`] on the subgraph *induced by a window only*, with
//! `4` bytes per window node instead of the `8` bytes per node and `12` per block of a full
//! index:
//!
//! ```text
//! pass 1 (parallel over u ∈ W, successors restricted to W, ascending):
//!   a = first, d = |succ_W(u)|
//!   a ≤ u                  → BackwardArc at u
//!   last ≠ a + d − 1       → NonContiguous at u
//!   lenat[a].cas(0, d) finds d' ≠ d → SameStartDifferentLength at a
//! pass 2 (sequential over W): intervals [a, a + lenat[a]) by increasing a must not overlap
//!   (Overlap at the first offending start)
//! ```
//! Violations are reported deterministically as the minimum `(node, kind)` of pass 1, else
//! the first overlap, exactly as [`BlockIndex::build`](crate::blocks::BlockIndex::build).
//! Because a window of a topologically numbered graph is convex (§3.2 of the design), a
//! certified window has no redundant arc *in the whole graph* either.
//!
//! # H1 and H3 (any DAG)
//! [`min_max_succ`] computes `F[u] = minsucc_W(u)` and `L[u] = maxsucc_W(u)` (`NONE` for nodes
//! without successors in the window) and fails with
//! [`DagError::NotTopological`] at the smallest offending `u`
//! if some arc has `v ≤ u` (**H1**).
//!
//! **H3**: for every arc `(u, w)`, `F[w] > L[u]`. If H3 holds at every node the graph
//! is transitively reduced: on a path `u → w₁ → w₂ ⇝ v` ids increase, so
//! `v ≥ w₂ ≥ F[w₁] > L[u] ≥ v` for any arc `(u, v)`, a contradiction. ∎ [`h3_failures`]
//! returns, for every node where H3 fails, the threshold
//!
//! ```text
//! t_u = min { F[w] : w ∈ succ(u), F[w] ≠ NONE, F[w] ≤ L[u] }
//! ```
//!
//! which bounds the candidates for redundancy from below (see
//! [`redundant_arcs_of`](super::exact::redundant_arcs_of) for the completeness argument).
//!
//! Complexity: each pass decodes the window once, O(n_W + m_W) work over all threads;
//! [`h3_failures`] does one random read of `F` per arc. Memory: `F` and `L` take `8` bytes per
//! window node (17.4 GB on the full Payment Graph); the line-digraph check `4` bytes per window
//! node.

use std::sync::atomic::Ordering;

use anyhow::Result;
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::blocks::BlockIndex;
use crate::error::{BlockViolation, DagError};
use crate::graph::ChunkedGraph;
use crate::io::{MemBudget, NONE, as_atomic_u32, huge_vec};
use crate::sweep::par_chunks;
use crate::window::Window;

/// Result of [`line_digraph_window`].
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct LineDigraphReport {
    /// Distinct successor intervals (blocks) found in the window.
    pub blocks: u64,
    /// Arcs of the subgraph induced by the window.
    pub arcs: u64,
    /// The first violation of P1–P3, if any (`None`: the window is certified reduced).
    pub violation: Option<(usize, BlockViolation)>,
}

impl LineDigraphReport {
    /// Whether the window satisfies P1–P3 (and is therefore transitively reduced).
    pub fn ok(&self) -> bool {
        self.violation.is_none()
    }
}

/// The smaller of two optional `(node, kind)` violations.
fn min_violation(
    a: Option<(usize, BlockViolation)>,
    b: Option<(usize, BlockViolation)>,
) -> Option<(usize, BlockViolation)> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, y) => x.or(y),
    }
}

/// Checks the line-digraph properties P1–P3 of [`crate::blocks`] on the subgraph induced by
/// the window `w` (see the module documentation).
///
/// Reserves `4·|W|` bytes in `budget` for the duration of the call. Returns the number of
/// blocks and arcs of the window and the first violation (the minimum `(node, kind)` of pass 1,
/// else the first `Overlap`). A report without violation certifies that every arc of the
/// window is non-redundant, in the window and in the whole graph.
pub fn line_digraph_window<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    budget: &mut MemBudget,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<LineDigraphReport> {
    let (lo, len) = (w.lo, w.len());
    let bytes = 4 * len as u64;
    budget.need(bytes, "tr: line-digraph check (4 B/node)")?;
    let res = (|| {
        let mut lenat = huge_vec(len, 0u32);
        let at = as_atomic_u32(&mut lenat);
        pl.item_name("node").expected_updates(Some(len));
        pl.start(format!("tr: line-digraph check of window {w}"));
        let (arcs, blocks, viol) = par_chunks(
            g,
            w,
            pl,
            |it| {
                let (mut arcs, mut blocks) = (0u64, 0u64);
                let mut viol = None;
                while let Some((u, s)) = it.next_node() {
                    let (Some(&a), Some(&last)) = (s.first(), s.last()) else {
                        continue;
                    };
                    let d = s.len() as u32;
                    arcs += d as u64;
                    let found = if a as usize <= u {
                        Some((u, BlockViolation::BackwardArc))
                    } else if last - a + 1 != d {
                        Some((u, BlockViolation::NonContiguous))
                    } else {
                        match at[a as usize - lo].compare_exchange(
                            0,
                            d,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => {
                                blocks += 1;
                                None
                            }
                            Err(x) if x != d => {
                                Some((a as usize, BlockViolation::SameStartDifferentLength))
                            }
                            Err(_) => None,
                        }
                    };
                    viol = min_violation(viol, found);
                }
                Ok((arcs, blocks, viol))
            },
            |a, b| (a.0 + b.0, a.1 + b.1, min_violation(a.2, b.2)),
        )?;
        pl.done();
        let mut report = LineDigraphReport {
            blocks,
            arcs,
            violation: viol,
        };
        if report.violation.is_none() {
            // pass 2: the intervals, by increasing start, must not overlap
            let mut cur_end = lo;
            for (i, &d) in lenat.iter().enumerate() {
                if d != 0 {
                    let v = lo + i;
                    if v < cur_end {
                        report.violation = Some((v, BlockViolation::Overlap));
                        break;
                    }
                    cur_end = v + d as usize;
                }
            }
        }
        Ok(report)
    })();
    budget.release(bytes);
    res
}

/// `F` (`minsucc`) and `L` (`maxsucc`) of the subgraph induced by a window, see
/// [`min_max_succ`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinMax {
    /// The window.
    pub window: Window,
    /// `f[u − lo]` = smallest successor of `u` inside the window (an absolute id), `NONE` if
    /// `u` has no successor in the window.
    pub f: Vec<u32>,
    /// `l[u − lo]` = largest successor of `u` inside the window (absolute), `NONE` likewise.
    pub l: Vec<u32>,
    /// Arcs of the subgraph induced by the window.
    pub arcs: u64,
}

impl MinMax {
    /// Bytes of heap used for a window of `len` nodes.
    pub fn bytes(len: usize) -> u64 {
        8 * len as u64
    }
}

/// H1 scan: computes `F`, `L` and the arc count of the window `w` in one parallel pass.
///
/// Fails with [`DagError::NotTopological`] at the smallest `u` (and its smallest offending
/// successor) if some arc `(u, v)` of the window has `v ≤ u`; the error context gives the
/// number of offending arcs. The caller reserves [`MinMax::bytes`] in its budget.
pub fn min_max_succ<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<MinMax> {
    let (lo, len) = (w.lo, w.len());
    let mut f = huge_vec(len, NONE);
    let mut l = huge_vec(len, NONE);
    let (fa, la) = (as_atomic_u32(&mut f), as_atomic_u32(&mut l));
    pl.item_name("node").expected_updates(Some(len));
    pl.start(format!("tr: H1 scan and minsucc/maxsucc of window {w}"));
    let (arcs, bad, first_bad) = par_chunks(
        g,
        w,
        pl,
        |it| {
            let (mut arcs, mut bad) = (0u64, 0u64);
            let mut first_bad: Option<(usize, usize)> = None;
            while let Some((u, s)) = it.next_node() {
                let (Some(&a), Some(&z)) = (s.first(), s.last()) else {
                    continue;
                };
                arcs += s.len() as u64;
                if a as usize <= u {
                    bad += s.iter().take_while(|&&v| v as usize <= u).count() as u64;
                    if first_bad.is_none() {
                        first_bad = Some((u, a as usize));
                    }
                    continue;
                }
                fa[u - lo].store(a, Ordering::Relaxed);
                la[u - lo].store(z, Ordering::Relaxed);
            }
            Ok((arcs, bad, first_bad))
        },
        |a, b| {
            let first = match (a.2, b.2) {
                (Some(x), Some(y)) => Some(x.min(y)),
                (x, y) => x.or(y),
            };
            (a.0 + b.0, a.1 + b.1, first)
        },
    )?;
    pl.done();
    if let Some((u, v)) = first_bad {
        return Err(
            anyhow::Error::new(DagError::NotTopological { u, v }).context(format!(
                "H1 fails: {bad} arcs of window {w} violate the topological order"
            )),
        );
    }
    Ok(MinMax {
        window: w,
        f,
        l,
        arcs,
    })
}

/// The nodes where H3 fails, with their thresholds `t_u`, for the whole graph: `f` and `l` are
/// indexed by node id (see [`h3_failures_in`]).
pub fn h3_failures<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    f: &[u32],
    l: &[u32],
    pl: &mut impl ConcurrentProgressLog,
) -> Result<Vec<(u32, u32)>> {
    h3_failures_in(g, Window::full(g.num_nodes()), f, l, pl)
}

/// The nodes `u ∈ w` where H3 fails, as pairs `(u, t_u)` sorted by `u` (absolute ids), with
/// `t_u = min{F[x] : x ∈ succ_W(u), F[x] ≠ NONE, F[x] ≤ L[u]}`.
///
/// `f` and `l` are the arrays of [`min_max_succ`] for the same window (indexed by `u − lo`).
/// One parallel pass with one random read of `f` per arc.
pub fn h3_failures_in<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    f: &[u32],
    l: &[u32],
    pl: &mut impl ConcurrentProgressLog,
) -> Result<Vec<(u32, u32)>> {
    let lo = w.lo;
    anyhow::ensure!(
        f.len() == w.len() && l.len() == w.len(),
        "F and L have {} and {} entries, window {w} has {} nodes",
        f.len(),
        l.len(),
        w.len()
    );
    pl.item_name("node").expected_updates(Some(w.len()));
    pl.start(format!("tr: H3 certificate on window {w}"));
    let mut fails = par_chunks(
        g,
        w,
        pl,
        |it| {
            let mut out = Vec::new();
            while let Some((u, s)) = it.next_node() {
                let lu = l[u - lo];
                if lu == NONE {
                    continue;
                }
                let t = s
                    .iter()
                    .filter(|&&x| x as usize >= lo)
                    .map(|&x| f[x as usize - lo])
                    .filter(|&fx| fx != NONE && fx <= lu)
                    .min();
                if let Some(t) = t {
                    out.push((u as u32, t));
                }
            }
            Ok(out)
        },
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    )?;
    pl.done();
    fails.par_sort_unstable();
    Ok(fails)
}

/// Arcs of the subgraph induced by `w`, from a [`BlockIndex`] alone (no decoding):
/// `Σ_{u ∈ W, spend[u] = B} |O(B) ∩ W|`.
pub fn blocks_window_arcs(idx: &BlockIndex, w: Window) -> u64 {
    (w.lo..w.hi)
        .into_par_iter()
        .with_min_len(1 << 16)
        .map(|u| match idx.spender(u) {
            Some(b) => {
                let o = idx.outputs(b);
                o.end.min(w.hi).saturating_sub(o.start.max(w.lo)) as u64
            }
            None => 0,
        })
        .sum()
}
