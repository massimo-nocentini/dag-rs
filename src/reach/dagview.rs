//! DAG views for the reachability index: the block quotient Q of the Payment Graph, or the
//! graph itself.
//!
//! The index of [`crate::reach::index`] works on any DAG whose node ids `0..n_dag` are a
//! topological order and which can (a) list the successors of one node by random access and
//! (b) stream every successor list in ascending or descending node order. [`SuccDag`] is that
//! interface; it is object safe, so the index code is compiled once.
//!
//! # Implementations
//! - [`BlockQuotient`] (the **PG fast path**, no BV decoding): the DAG is the block quotient Q
//!   of a [`BlockIndex`], with
//!   ```text
//!   succ_Q(X) = sort-dedup{ spend[w] : w ∈ outputs(X), spend[w] ≠ NONE }
//!   ```
//!   Sinks are dropped (`spend[w] = NONE`). Then, for `u ≠ v`,
//!   `u ⇝ v` in G iff `spend[u] ≠ NONE`, `create[v] ≠ NONE` and `spend[u] ⇝* create[v]` in Q,
//!   and by Lemma 2 block ids are a topological order of Q. Streaming passes compute the
//!   successor lists of batches of blocks in parallel (rayon) while the previous batch is
//!   consumed in order (`rayon::join` double buffering, as in [`crate::sweep`]).
//! - [`GraphDag`]: the graph itself (any webgraph DAG in identity topological order).
//!   Streaming passes use [`sweep_forward`]/[`sweep_backward`], which also verify the order;
//!   [`SuccDag::succ_into`] uses random access.
//!
//! # Windows
//! Both views can be restricted to a node window `[lo, hi)`. Ids are chronological and
//! topological, so a window is convex: every path between two window nodes stays inside it
//! (see [`crate::window`]). DAG node `x` is graph node `lo + x` ([`GraphDag`]) or block
//! `blo + x` ([`BlockQuotient`]), where `[blo, bhi)` are the blocks whose first output lies in
//! the window; outputs `≥ hi` and successor blocks `≥ bhi` are dropped. Blocks straddling `lo`
//! cannot be reached from a window node (their first output is `< lo`), so dropping them loses
//! no window path. [`DagInfo`] records the mapping.
//!
//! Complexity: one streaming pass costs O(n + m) on [`GraphDag`] (one parallel BV decode) and
//! O(n_w + Σ_X |outputs(X)| log) on [`BlockQuotient`] (a scan of `spend`), plus the consumer.

use std::ops::Range;

use anyhow::Result;
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::blocks::BlockIndex;
use crate::graph::ChunkedGraph;
use crate::sweep::{DynLog, DynProgressLog, SweepConf, sweep_backward, sweep_forward};
use crate::window::Window;

/// Which DAG an index was built on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Mode {
    /// The block quotient Q of a [`BlockIndex`] (DAG node = block).
    Blocks,
    /// The graph itself (DAG node = graph node).
    Graph,
}

/// How the DAG ids of a [`SuccDag`] map to the underlying graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DagInfo {
    /// What a DAG node is.
    pub mode: Mode,
    /// Number of nodes of the underlying graph.
    pub n: usize,
    /// The node window `[lo, hi)` of the underlying graph.
    pub lo: usize,
    /// See `lo`.
    pub hi: usize,
    /// DAG node `x` is graph node (or block) `base + x`.
    pub base: usize,
}

/// A DAG whose ids `0..num_nodes()` are a topological order, seen through its successor
/// lists (see the module documentation).
///
/// Progress: the streaming methods call `pl.update_with_count` once per batch or chunk;
/// callers call `start`/`done` around them. The closures never see the logger.
pub trait SuccDag: Sync {
    /// Number of DAG nodes.
    fn num_nodes(&self) -> usize;
    /// Replaces the content of `buf` with the successors of `x`: ascending, deduplicated, all
    /// `> x` and `< num_nodes()`, never [`NONE`](crate::io::NONE).
    fn succ_into(&self, x: usize, buf: &mut Vec<u32>);
    /// Calls `f(x, succ(x))` for every `x` in ascending order.
    fn for_each_forward(
        &self,
        f: &mut (dyn FnMut(usize, &[u32]) + Send),
        pl: &mut dyn DynProgressLog,
    ) -> Result<()>;
    /// Calls `f(x, succ(x))` for every `x` in descending order.
    fn for_each_backward(
        &self,
        f: &mut (dyn FnMut(usize, &[u32]) + Send),
        pl: &mut dyn DynProgressLog,
    ) -> Result<()>;
    /// The id mapping (default: a whole graph of `num_nodes()` nodes in [`Mode::Graph`]).
    fn info(&self) -> DagInfo {
        let n = self.num_nodes();
        DagInfo {
            mode: Mode::Graph,
            n,
            lo: 0,
            hi: n,
            base: 0,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Batched producer shared by the streaming passes of BlockQuotient
// ---------------------------------------------------------------------------------------------

/// Nodes per parallel task of a batch.
const SUB: usize = 4096;

/// Successor lists of DAG nodes `[start, start + offs.len() − 1)` in CSR form.
#[derive(Default)]
struct Piece {
    start: usize,
    offs: Vec<u32>,
    succ: Vec<u32>,
}

impl Piece {
    fn len(&self) -> usize {
        self.offs.len() - 1
    }
    fn list(&self, i: usize) -> &[u32] {
        &self.succ[self.offs[i] as usize..self.offs[i + 1] as usize]
    }
}

// ---------------------------------------------------------------------------------------------
// BlockQuotient
// ---------------------------------------------------------------------------------------------

/// The block quotient Q of a [`BlockIndex`], optionally restricted to a node window.
///
/// DAG node `x` is block `blo + x`; see the module documentation for the window semantics.
#[derive(Clone, Copy, Debug)]
pub struct BlockQuotient<'a> {
    /// The block index.
    pub blocks: &'a BlockIndex,
    window: Window,
    blo: usize,
    bhi: usize,
    batch: usize,
}

impl<'a> BlockQuotient<'a> {
    /// The quotient of the whole graph.
    pub fn new(blocks: &'a BlockIndex) -> Self {
        Self::with_window(blocks, Window::full(blocks.n))
    }

    /// The quotient restricted to the node window `w` (blocks `[blo, bhi)` whose first output
    /// lies in `w`).
    pub fn with_window(blocks: &'a BlockIndex, w: Window) -> Self {
        assert!(w.hi <= blocks.n, "window {w:?} exceeds {} nodes", blocks.n);
        let blo = blocks.block_bound(w.lo);
        let bhi = blocks.block_bound(w.hi).max(blo);
        Self {
            blocks,
            window: w,
            blo,
            bhi,
            batch: 1 << 18,
        }
    }

    /// Sets the number of blocks per streaming batch (tests use tiny batches).
    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch.max(1);
        self
    }

    /// The node window.
    pub fn window(&self) -> Window {
        self.window
    }

    /// The blocks `[blo, bhi)` of the DAG.
    pub fn block_range(&self) -> Range<usize> {
        self.blo..self.bhi
    }

    /// DAG node of block `b` (which must lie in [`block_range`](Self::block_range)).
    #[inline]
    pub fn node_of_block(&self, b: usize) -> usize {
        debug_assert!((self.blo..self.bhi).contains(&b));
        b - self.blo
    }

    /// Appends `succ_Q(blo + x)` (DAG ids, sorted, deduplicated) to `out`.
    #[inline]
    fn push_succ(&self, x: usize, out: &mut Vec<u32>) {
        let b = self.blo + x;
        let r = self.blocks.outputs(b);
        let end = r.end.min(self.window.hi);
        let spend: &[u32] = &self.blocks.spend;
        let (blo, bhi) = (self.blo as u32, self.bhi as u32);
        let at = out.len();
        for &s in &spend[r.start..end] {
            // NONE = u32::MAX ≥ bhi, so this also drops sinks
            if s < bhi {
                out.push(s - blo);
            }
        }
        let tail = &mut out[at..];
        if !tail.is_sorted() {
            tail.sort_unstable();
        }
        // dedup the tail in place
        let mut w = at;
        for i in at..out.len() {
            if w == at || out[i] != out[w - 1] {
                out[w] = out[i];
                w += 1;
            }
        }
        out.truncate(w);
    }

    fn produce(&self, r: Range<usize>) -> Vec<Piece> {
        let starts: Vec<usize> = r.clone().step_by(SUB).collect();
        starts
            .into_par_iter()
            .with_max_len(1)
            .map(|s| {
                let e = (s + SUB).min(r.end);
                let mut p = Piece {
                    start: s,
                    offs: Vec::with_capacity(e - s + 1),
                    succ: Vec::new(),
                };
                p.offs.push(0);
                for x in s..e {
                    self.push_succ(x, &mut p.succ);
                    p.offs.push(p.succ.len() as u32);
                }
                p
            })
            .collect()
    }

    fn stream(
        &self,
        backward: bool,
        f: &mut (dyn FnMut(usize, &[u32]) + Send),
        pl: &mut dyn DynProgressLog,
    ) -> Result<()> {
        let n = self.num_nodes();
        let mut ranges: Vec<Range<usize>> = (0..n)
            .step_by(self.batch)
            .map(|s| s..(s + self.batch).min(n))
            .collect();
        if backward {
            ranges.reverse();
        }
        let Some(first) = ranges.first() else {
            return Ok(());
        };
        let mut cur = self.produce(first.clone());
        for i in 0..ranges.len() {
            let nodes = ranges[i].len();
            let next_range = ranges.get(i + 1).cloned();
            let consume = |cur: &Vec<Piece>, f: &mut (dyn FnMut(usize, &[u32]) + Send)| {
                if backward {
                    for p in cur.iter().rev() {
                        for j in (0..p.len()).rev() {
                            f(p.start + j, p.list(j));
                        }
                    }
                } else {
                    for p in cur {
                        for j in 0..p.len() {
                            f(p.start + j, p.list(j));
                        }
                    }
                }
            };
            let (_, next) = rayon::join(
                || consume(&cur, &mut *f),
                || next_range.map(|r| self.produce(r)),
            );
            pl.dyn_update_with_count(nodes);
            match next {
                Some(nx) => cur = nx,
                None => break,
            }
        }
        Ok(())
    }
}

impl SuccDag for BlockQuotient<'_> {
    fn num_nodes(&self) -> usize {
        self.bhi - self.blo
    }

    fn succ_into(&self, x: usize, buf: &mut Vec<u32>) {
        buf.clear();
        self.push_succ(x, buf);
    }

    fn for_each_forward(
        &self,
        f: &mut (dyn FnMut(usize, &[u32]) + Send),
        pl: &mut dyn DynProgressLog,
    ) -> Result<()> {
        self.stream(false, f, pl)
    }

    fn for_each_backward(
        &self,
        f: &mut (dyn FnMut(usize, &[u32]) + Send),
        pl: &mut dyn DynProgressLog,
    ) -> Result<()> {
        self.stream(true, f, pl)
    }

    fn info(&self) -> DagInfo {
        DagInfo {
            mode: Mode::Blocks,
            n: self.blocks.n,
            lo: self.window.lo,
            hi: self.window.hi,
            base: self.blo,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GraphDag
// ---------------------------------------------------------------------------------------------

/// The graph itself as a [`SuccDag`] (identity topological order), optionally restricted to a
/// node window: DAG node `x` is graph node `lo + x`.
pub struct GraphDag<'a, G> {
    /// The chunked graph.
    pub g: &'a ChunkedGraph<G>,
    /// Configuration of the sweep engine used by the streaming passes.
    pub conf: SweepConf,
    window: Window,
}

impl<'a, G: RandomAccessGraph + Sync> GraphDag<'a, G> {
    /// The whole graph.
    pub fn new(g: &'a ChunkedGraph<G>, conf: SweepConf) -> Self {
        let w = Window::full(g.num_nodes());
        Self::with_window(g, conf, w)
    }

    /// The subgraph induced by the window `w`.
    pub fn with_window(g: &'a ChunkedGraph<G>, conf: SweepConf, w: Window) -> Self {
        assert!(
            w.hi <= g.num_nodes(),
            "window {w:?} exceeds {} nodes",
            g.num_nodes()
        );
        Self { g, conf, window: w }
    }

    /// The node window.
    pub fn window(&self) -> Window {
        self.window
    }

    fn stream(
        &self,
        backward: bool,
        f: &mut (dyn FnMut(usize, &[u32]) + Send),
        pl: &mut dyn DynProgressLog,
    ) -> Result<()> {
        let lo = self.window.lo;
        let mut buf = Vec::new();
        let mut call = |u: usize, s: &[u32]| {
            relative(s, lo, &mut buf);
            f(u - lo, &buf);
        };
        let mut log = DynLog(pl);
        if backward {
            sweep_backward(self.g, self.window, &self.conf, &mut log, |c| {
                for (u, s) in c.iter_rev() {
                    call(u, s);
                }
                Ok(())
            })
        } else {
            sweep_forward(self.g, self.window, &self.conf, &mut log, |c| {
                for (u, s) in c.iter() {
                    call(u, s);
                }
                Ok(())
            })
        }
    }
}

/// Writes `s − lo` (deduplicated; `s` is ascending) into `buf`.
#[inline]
fn relative(s: &[u32], lo: usize, buf: &mut Vec<u32>) {
    buf.clear();
    let lo = lo as u32;
    for &v in s {
        let r = v - lo;
        if buf.last() != Some(&r) {
            buf.push(r);
        }
    }
}

impl<G: RandomAccessGraph + Sync> SuccDag for GraphDag<'_, G> {
    fn num_nodes(&self) -> usize {
        self.window.len()
    }

    fn succ_into(&self, x: usize, buf: &mut Vec<u32>) {
        buf.clear();
        let (lo, hi) = (self.window.lo, self.window.hi);
        let u = lo + x;
        for v in self.g.graph.successors(u) {
            if v < hi {
                debug_assert!(v > u, "arc ({u}, {v}) violates the topological order");
                buf.push((v - lo) as u32);
            }
        }
        // BV lists are sorted and duplicate free; other graphs may not be
        if !buf.is_sorted() {
            buf.sort_unstable();
        }
        buf.dedup();
    }

    fn for_each_forward(
        &self,
        f: &mut (dyn FnMut(usize, &[u32]) + Send),
        pl: &mut dyn DynProgressLog,
    ) -> Result<()> {
        self.stream(false, f, pl)
    }

    fn for_each_backward(
        &self,
        f: &mut (dyn FnMut(usize, &[u32]) + Send),
        pl: &mut dyn DynProgressLog,
    ) -> Result<()> {
        self.stream(true, f, pl)
    }

    fn info(&self) -> DagInfo {
        DagInfo {
            mode: Mode::Graph,
            n: self.g.num_nodes(),
            lo: self.window.lo,
            hi: self.window.hi,
            base: self.window.lo,
        }
    }
}

/// Every successor list of `dag` (tests and small graphs).
pub fn succ_lists(dag: &dyn SuccDag) -> Vec<Vec<u32>> {
    let mut buf = Vec::new();
    (0..dag.num_nodes())
        .map(|x| {
            dag.succ_into(x, &mut buf);
            buf.clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{MiniPgOpts, chunked, mini_pg, random_dag};
    use dsi_progress_logger::prelude::*;

    fn quiet() -> Option<ProgressLogger> {
        None
    }

    fn streamed(dag: &dyn SuccDag, backward: bool) -> Vec<(usize, Vec<u32>)> {
        let mut v = Vec::new();
        let mut f = |x: usize, s: &[u32]| v.push((x, s.to_vec()));
        if backward {
            dag.for_each_backward(&mut f, &mut quiet()).unwrap();
        } else {
            dag.for_each_forward(&mut f, &mut quiet()).unwrap();
        }
        v
    }

    #[test]
    fn streams_match_random_access() {
        let pg = mini_pg(300, 7, &MiniPgOpts::default());
        let (bi, _) = BlockIndex::from_vec_graph(&pg.graph).unwrap();
        for batch in [1, 3, 5000] {
            let q = BlockQuotient::new(&bi).with_batch(batch);
            let lists = succ_lists(&q);
            let fwd = streamed(&q, false);
            assert_eq!(fwd.len(), q.num_nodes());
            for (i, (x, s)) in fwd.iter().enumerate() {
                assert_eq!((*x, s), (i, &lists[i]));
                assert!(s.iter().all(|&y| y as usize > *x));
            }
            let mut bwd = streamed(&q, true);
            bwd.reverse();
            assert_eq!(fwd, bwd);
        }
        let g = random_dag(80, 0.1, 3);
        let cg = chunked(&g, 3);
        let gd = GraphDag::with_window(&cg, SweepConf { decoders: 2 }, Window { lo: 10, hi: 70 });
        let lists = succ_lists(&gd);
        let fwd = streamed(&gd, false);
        for (i, (x, s)) in fwd.iter().enumerate() {
            assert_eq!((*x, s), (i, &lists[i]));
        }
        let mut bwd = streamed(&gd, true);
        bwd.reverse();
        assert_eq!(fwd, bwd);
    }
}
