//! Dominator and post-dominator tree builders (blocks and generic).
//!
//! All builders compute `idom(v) = NCA_D(pred(v))` in one pass in topological order (F2 §2,
//! see the [module documentation](super)), on the subgraph induced by a window `[lo, hi)`
//! with its own virtual root (sink). Dominators use the index `x = v − lo + 1` and ascending
//! ids; post-dominators the mirror `x = hi − v` and descending ids.
//!
//! | Builder | Input | Pass | Folds |
//! |---|---|---|---|
//! | [`dominators_blocks`] | [`BlockIndex`] | ascending scan of `create`/`spend` | one running NCA per block |
//! | [`post_dominators_blocks`] | [`BlockIndex`] | descending scan of `spend` | one fold per block, memoised |
//! | [`dominators_push`] | forward graph | [`sweep_forward`] | one NCA per arc |
//! | [`post_dominators_pull`] | forward graph | [`sweep_backward`] | one NCA per arc, consecutive identical lists reused |
//!
//! # PG fast path (decode-free)
//! On a biclique digraph every output of a block `B` has the predecessor set `I(B)`, so
//!
//! ```text
//! dominators_blocks (acc[B] = UNSET for every block B with first[B] < hi):
//! for v in lo..hi:
//!     B = create[v]
//!     if B ≠ NONE ∧ v > max(first[B], lo):  add_leaf_like(v−lo+1, v−lo)   // same block as v−1
//!     else: p = (B ≠ NONE ∧ acc[B] ≠ UNSET) ? acc[B] : ROOT;  add_leaf(v−lo+1, p)
//!     X = spend[v]
//!     if X ≠ NONE ∧ first[X] < hi ∧ acc[X] ≠ ROOT:
//!         acc[X] = (acc[X] == UNSET) ? v−lo+1 : nca(acc[X], v−lo+1)
//!
//! post_dominators_blocks (memo[X] = UNSET):
//! for u in (lo..hi).rev():
//!     X = spend[u]
//!     if X == NONE ∨ first[X] ≥ hi:  p = ROOT
//!     else if memo[X] ≠ UNSET:       p = memo[X]
//!     else: p = fold(nca, {hi − w : w ∈ O(X), w < hi}) with early exit at ROOT; memo[X] = p
//!     add_leaf(hi − u, p)
//! ```
//!
//! Inputs of a block that lie before `lo` are simply never pushed, and outputs at or after `hi`
//! are skipped, which is exactly the induced subgraph. The arrays `acc`/`memo` cover only the
//! blocks with `first ∈ [lo, hi)` (4 B per block).
//!
//! # Generic builders
//! `dominators_push` keeps, in the parent slot of every not yet finalised node, the running NCA
//! of the predecessors pushed so far ([`UNSET`] initially): for each `u` in ascending order it
//! first finalises `u` (`add_leaf(u, pending(u))`, or `ROOT` for a source) and then, for every
//! successor `v`, sets `pending(v) = nca(pending(v), u)` (skipped when already `ROOT`). This
//! needs only the forward graph (D5). `post_dominators_pull` folds the mirror indices of
//! `succ(u)` in descending `u`, decoding backward chunks of the forward graph; a list equal to
//! the previous node's list reuses its result (exact on every graph).
//!
//! Complexity: O(n + Σ folds × NCA cost); an NCA costs O(1) at the `top` test, at most
//! `budget` walk steps, then O(log n) (see [`tree`](super::tree)). The generic builders also
//! decode `m` arcs with the parallel sweep engine. Memory: the tree (16 B per node) plus 4 B
//! per block of the window (blocks builders) or the sweep buffers (generic builders).
//!
//! References: dominators [\[LM69\]](crate::references#lm69),
//! [\[Pro59\]](crate::references#pro59), [\[All70\]](crate::references#all70); the iterative
//! algorithm [\[CHK01\]](crate::references#chk01), here in a single topological pass; acyclic
//! dominators by NCA [\[Och83\]](crate::references#och83), [\[RR94\]](crate::references#rr94).

use anyhow::Result;
use dsi_progress_logger::prelude::*;
use webgraph::prelude::*;

use super::tree::{DomTree, ROOT, UNSET};
use crate::blocks::BlockIndex;
use crate::graph::ChunkedGraph;
use crate::io::{self, NONE};
use crate::sweep::{SweepConf, sweep_backward, sweep_forward};
use crate::window::Window;

/// Nodes between two progress updates of the sequential (decode-free) builders.
const PROGRESS_STEP: usize = 1 << 20;

/// Configuration of the builders.
#[derive(Clone, Debug)]
pub struct DomConf {
    /// The window `[lo, hi)`; the trees are those of the induced subgraph.
    pub window: Window,
    /// Walk steps of the NCA before the jump-pointer fallback (default 64).
    pub budget: u32,
    /// Sweep engine configuration (generic builders).
    pub sweep: SweepConf,
}

impl DomConf {
    /// The default configuration (budget 64) for the window `w`.
    pub fn new(window: Window) -> Self {
        Self {
            window,
            budget: 64,
            sweep: SweepConf::default(),
        }
    }
}

/// Checks that the window fits the block index.
fn check_window(blocks: &BlockIndex, w: Window) -> Result<()> {
    anyhow::ensure!(
        w.lo <= w.hi && w.hi <= blocks.n,
        crate::error::DagError::BadWindow {
            lo: w.lo,
            hi: w.hi,
            n: blocks.n
        }
    );
    Ok(())
}

fn start(pl: &mut impl ProgressLog, n: usize, what: &str) {
    pl.item_name("node");
    pl.expected_updates(Some(n));
    pl.start(what);
}

/// Dominator tree of the window on a biclique digraph, from the [`BlockIndex`] alone (no graph
/// decoding; module docs). One running NCA per block.
pub fn dominators_blocks(
    blocks: &BlockIndex,
    conf: &DomConf,
    pl: &mut impl ProgressLog,
) -> Result<DomTree> {
    let w = conf.window;
    check_window(blocks, w)?;
    let (lo, hi) = (w.lo, w.hi);
    let create: &[u32] = &blocks.create;
    let spend: &[u32] = &blocks.spend;
    let first: &[u32] = &blocks.first;
    // blocks with first < lo receive no window input; blocks with first ≥ hi are outside
    let (blo, bhi) = (blocks.block_bound(lo), blocks.block_bound(hi));
    let mut acc = io::huge_vec(bhi - blo, UNSET);
    let mut tree = DomTree::for_window(w, false, conf.budget);
    start(pl, w.len(), "dominators (blocks)");
    let mut since = 0;
    for v in lo..hi {
        let x = (v - lo + 1) as u32;
        let b = create[v];
        if b != NONE && v > (first[b as usize] as usize).max(lo) {
            tree.add_leaf_like(x, x - 1);
        } else {
            let b = b as usize;
            let p = if b != NONE as usize && (blo..bhi).contains(&b) && acc[b - blo] != UNSET {
                acc[b - blo]
            } else {
                ROOT
            };
            tree.add_leaf(x, p);
        }
        let s = spend[v] as usize;
        if s != NONE as usize && s < bhi {
            // first[s] > v ≥ lo, hence s ≥ blo
            let a = acc[s - blo];
            if a == UNSET {
                acc[s - blo] = x;
            } else if a != ROOT {
                acc[s - blo] = tree.nca_tracked(a, x);
            }
        }
        since += 1;
        if since == PROGRESS_STEP {
            pl.update_with_count(since);
            since = 0;
        }
    }
    pl.update_with_count(since);
    pl.done();
    Ok(tree)
}

/// Post-dominator tree of the window on a biclique digraph, from the [`BlockIndex`] alone
/// (module docs). One memoised fold per block, with early exit at the virtual sink.
pub fn post_dominators_blocks(
    blocks: &BlockIndex,
    conf: &DomConf,
    pl: &mut impl ProgressLog,
) -> Result<DomTree> {
    let w = conf.window;
    check_window(blocks, w)?;
    let (lo, hi) = (w.lo, w.hi);
    let spend: &[u32] = &blocks.spend;
    let first: &[u32] = &blocks.first;
    let len: &[u32] = &blocks.len;
    let (blo, bhi) = (blocks.block_bound(lo), blocks.block_bound(hi));
    let mut memo = io::huge_vec(bhi - blo, UNSET);
    let mut tree = DomTree::for_window(w, true, conf.budget);
    start(pl, w.len(), "post-dominators (blocks)");
    let mut since = 0;
    for u in (lo..hi).rev() {
        let s = spend[u] as usize;
        let p = if s == NONE as usize || s >= bhi {
            ROOT
        } else if memo[s - blo] != UNSET {
            memo[s - blo]
        } else {
            let f = first[s] as usize;
            let end = (f + len[s] as usize).min(hi);
            // outputs w ∈ [f, end) have mirror indices hi − w, all smaller than hi − u
            let mut acc = (hi - f) as u32;
            for o in f + 1..end {
                if acc == ROOT {
                    break;
                }
                acc = tree.nca_tracked(acc, (hi - o) as u32);
            }
            memo[s - blo] = acc;
            acc
        };
        tree.add_leaf((hi - u) as u32, p);
        since += 1;
        if since == PROGRESS_STEP {
            pl.update_with_count(since);
            since = 0;
        }
    }
    pl.update_with_count(since);
    pl.done();
    Ok(tree)
}

/// Dominator tree of the window of any DAG in identity topological order, by a push over the
/// forward graph (module docs). Fails with
/// [`DagError::NotTopological`](crate::error::DagError::NotTopological) if an arc of the window
/// goes backward.
pub fn dominators_push<G>(
    g: &ChunkedGraph<G>,
    conf: &DomConf,
    pl: &mut impl ProgressLog,
) -> Result<DomTree>
where
    G: RandomAccessGraph + Sync,
{
    let w = conf.window;
    let lo = w.lo;
    let mut tree = DomTree::for_window(w, false, conf.budget);
    start(pl, w.len(), "dominators (push)");
    sweep_forward(g, w, &conf.sweep, pl, |chunk| {
        for (u, succ) in chunk.iter() {
            let x = (u - lo + 1) as u32;
            let p = match tree.pending(x) {
                UNSET => ROOT,
                p => p,
            };
            tree.add_leaf(x, p);
            for &v in succ {
                // the engine guarantees u < v < hi
                let y = (v as usize - lo + 1) as u32;
                match tree.pending(y) {
                    ROOT => {}
                    UNSET => tree.set_pending(y, x),
                    c => {
                        let c = tree.nca_tracked(c, x);
                        tree.set_pending(y, c);
                    }
                }
            }
        }
        Ok(())
    })?;
    pl.done();
    Ok(tree)
}

/// Post-dominator tree of the window of any DAG in identity topological order, by a pull over
/// the forward graph decoded in backward chunks (module docs).
pub fn post_dominators_pull<G>(
    g: &ChunkedGraph<G>,
    conf: &DomConf,
    pl: &mut impl ProgressLog,
) -> Result<DomTree>
where
    G: RandomAccessGraph + Sync,
{
    let w = conf.window;
    let hi = w.hi;
    let mut tree = DomTree::for_window(w, true, conf.budget);
    let mut prev: Vec<u32> = Vec::new();
    let mut prev_res = ROOT;
    start(pl, w.len(), "post-dominators (pull)");
    sweep_backward(g, w, &conf.sweep, pl, |chunk| {
        for (u, succ) in chunk.iter_rev() {
            let p = if succ.is_empty() {
                ROOT
            } else if succ == prev.as_slice() {
                prev_res
            } else {
                let mirror = |v: u32| (hi - v as usize) as u32;
                let mut acc = mirror(succ[0]);
                for &v in &succ[1..] {
                    if acc == ROOT {
                        break;
                    }
                    acc = tree.nca_tracked(acc, mirror(v));
                }
                prev.clear();
                prev.extend_from_slice(succ);
                prev_res = acc;
                acc
            };
            tree.add_leaf((hi - u) as u32, p);
        }
        Ok(())
    })?;
    pl.done();
    Ok(tree)
}
