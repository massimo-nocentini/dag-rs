//! `BlockIndex`: the biclique decomposition of the Payment Graph (the shared PG fast path).
//!
//! # Definitions
//! A *block* `B` is a spending transaction with at least one output. Its outputs are the id
//! interval `O(B) = [first[B], first[B] + len[B])` and its inputs are
//! `I(B) = {u : spend[u] = B}`. The arcs of the graph are exactly `I(B) × O(B)` over all blocks,
//! i.e. the graph is a disjoint union of complete bicliques (a line digraph,
//! [\[HN60\]](crate::references#hn60), [\[Heu64\]](crate::references#heu64)). Block ids are
//! `0..nb` in increasing `first` (chronological). The structure holds iff
//!
//! - **P1** every successor list is an interval of consecutive ids;
//! - **P2** two successor lists are identical or disjoint;
//! - **P3** every successor list starts after its node (`first > u`).
//!
//! With these arrays, the block-level DP, dominators, post-dominators, HLL counts,
//! reachability labels, MPC_d, frontier and closure width become array scans **without BV
//! decoding**.
//!
//! # Build (exact verification, deterministic violation reports)
//! ```text
//! 1. par_chunks over g: for u with d = outdeg(u) > 0 and first successor a:
//!      require a > u (BackwardArc) and last = a + d − 1 (NonContiguous);
//!      spend[u] = a (a node id for now);  lenat[a].compare_exchange(0, d), a mismatch
//!      with an existing value ≠ d is SameStartDifferentLength at node a;
//!      count successful exchanges (= nb); each chunk returns its minimal (node, kind).
//! 2. sequential scan v ascending with cur_end: if lenat[v] ≠ 0, require v ≥ cur_end
//!      (Overlap at v), open B: first[B] = v, len[B] = lenat[v], cur_end = v + len;
//!      create[v] = (v < cur_end) ? B : NONE, in place over lenat.
//! 3. parallel: spend[u] = create[spend[u]]; nin[spend[u]] += 1 (atomic view).
//! ```
//! Independent checks recorded in [`BlockReport`] and asserted: `Σ len = n − sources`,
//! `Σ nin = n − sinks`, `Σ len·nin = m` with `m` from the graph (`.properties`), not from the
//! same pass.
//!
//! **Theorem** (see the crate documentation): if ids are topological and any two successor sets are
//! identical or disjoint, the graph is transitively reduced
//! ([\[AGU72\]](crate::references#agu72)). A successful build therefore certifies "transitively
//! reduced", for the full graph and for every window.
//!
//! Memory: 8 B/node + 12 B/block (26.8 GB on the PG). Time: one parallel decode plus two array
//! scans.

use std::ops::Range;
use std::path::Path;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result};
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::error::{BlockViolation, DagError};
use crate::graph::{ChunkConf, ChunkedGraph};
use crate::io::{self, LoadMode, MemBudget, NONE, U32Array, as_atomic_u32, huge_vec};
use crate::sweep::par_chunks;
use crate::window::Window;

/// The biclique decomposition of a line digraph (see the module documentation).
#[derive(Debug)]
pub struct BlockIndex {
    /// Number of nodes.
    pub n: usize,
    /// Number of arcs.
    pub m: u64,
    /// Number of blocks.
    pub nb: usize,
    /// `[n]` block whose outputs contain `v`; [`NONE`] if `indeg(v) = 0` (a source).
    pub create: U32Array,
    /// `[n]` block whose outputs are exactly `succ(u)`; [`NONE`] if `u` is a sink.
    pub spend: U32Array,
    /// `[nb]` first output of each block (strictly increasing).
    pub first: U32Array,
    /// `[nb]` `|O(B)|`, the outdegree of every input of `B`.
    pub len: U32Array,
    /// `[nb]` `|I(B)|`, the indegree of every output of `B`.
    pub nin: U32Array,
}

/// Totals of a [`BlockIndex`] build (also the content of `<p>.blk.json`).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlockReport {
    /// Number of nodes.
    pub n: usize,
    /// Number of arcs.
    pub m: u64,
    /// Number of blocks.
    pub nb: usize,
    /// Nodes with indegree 0 (`create = NONE`).
    pub sources: usize,
    /// Nodes with outdegree 0 (`spend = NONE`).
    pub sinks: usize,
    /// Nodes covered by blocks (`Σ len`).
    pub covered: usize,
    /// `Σ len`.
    pub sum_len: u64,
    /// `Σ nin`.
    pub sum_nin: u64,
    /// `Σ len·nin`, which must equal `m`.
    pub sum_len_times_nin: u64,
    /// Largest block (outputs).
    pub max_len: u32,
    /// Largest number of inputs of a block.
    pub max_nin: u32,
}

/// Nodes between two progress updates of the sequential pass 2 (a power of two).
const PASS2_LOG_STEP: usize = 1 << 22;

#[derive(Default)]
struct Pass1 {
    viol: Option<(usize, BlockViolation)>,
    nb: u64,
    sinks: u64,
}

fn min_viol(
    a: Option<(usize, BlockViolation)>,
    b: Option<(usize, BlockViolation)>,
) -> Option<(usize, BlockViolation)> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

impl BlockIndex {
    /// Builds and verifies the index from the forward graph (see the module documentation).
    ///
    /// Fails with [`DagError::NotBiclique`] at a deterministic violation: the smallest
    /// `(node, kind)` among the pass-1 violations (`BackwardArc`, `NonContiguous`,
    /// `SameStartDifferentLength`); if pass 1 finds none, the first `Overlap` in id order.
    pub fn build<G: RandomAccessGraph + Sync>(
        g: &ChunkedGraph<G>,
        pl: &mut impl ConcurrentProgressLog,
    ) -> Result<(Self, BlockReport)> {
        Self::build_budgeted(g, pl, &mut MemBudget::unlimited())
    }

    /// [`build`](Self::build), reserving the index in `budget`: `8n` bytes (`create`, `spend`)
    /// before pass 1 and `12·nb` bytes (`first`, `len`, `nin`) once pass 1 has counted the
    /// blocks. `nb` is not known in advance and is only bounded by `n − 1` (a chain has
    /// `n − 1` blocks), so reserving it after pass 1 avoids both under- and over-reservation.
    ///
    /// On success `8n + 12·nb` stays reserved; on any error everything reserved here is
    /// released.
    pub fn build_budgeted<G: RandomAccessGraph + Sync>(
        g: &ChunkedGraph<G>,
        pl: &mut impl ConcurrentProgressLog,
        budget: &mut MemBudget,
    ) -> Result<(Self, BlockReport)> {
        let before = budget.used();
        let res = Self::build_inner(g, pl, budget);
        if res.is_err() {
            budget.release(budget.used().saturating_sub(before));
        }
        res
    }

    /// Bytes reserved by [`build_budgeted`](Self::build_budgeted) for an index of `n` nodes
    /// and `nb` blocks.
    pub fn bytes(n: usize, nb: usize) -> u64 {
        8 * n as u64 + 12 * nb as u64
    }

    fn build_inner<G: RandomAccessGraph + Sync>(
        g: &ChunkedGraph<G>,
        pl: &mut impl ConcurrentProgressLog,
        budget: &mut MemBudget,
    ) -> Result<(Self, BlockReport)> {
        let n = g.num_nodes();
        let m = g.num_arcs;
        crate::graph::check_nodes(n)?;
        budget.need(8 * n as u64, "BlockIndex create + spend")?;
        let mut create = huge_vec(n, 0u32);
        let mut spend = huge_vec(n, NONE);
        // ---- pass 1 ----
        pl.item_name("node").expected_updates(Some(n));
        pl.start("BlockIndex pass 1: successor intervals");
        let p1 = {
            let lenat = as_atomic_u32(&mut create);
            let sp = as_atomic_u32(&mut spend);
            par_chunks(
                g,
                Window::full(n),
                pl,
                |it| {
                    let mut acc = Pass1::default();
                    while let Some((u, s)) = it.next_node() {
                        let Some(&a) = s.first() else {
                            acc.sinks += 1;
                            continue;
                        };
                        let d = s.len();
                        let a = a as usize;
                        let last = *s.last().unwrap() as usize;
                        if a <= u {
                            acc.viol = min_viol(acc.viol, Some((u, BlockViolation::BackwardArc)));
                            continue;
                        }
                        if last != a + d - 1 {
                            acc.viol = min_viol(acc.viol, Some((u, BlockViolation::NonContiguous)));
                            continue;
                        }
                        sp[u].store(a as u32, Ordering::Relaxed);
                        match lenat[a].compare_exchange(
                            0,
                            d as u32,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => acc.nb += 1,
                            Err(cur) if cur as usize != d => {
                                acc.viol = min_viol(
                                    acc.viol,
                                    Some((a, BlockViolation::SameStartDifferentLength)),
                                );
                            }
                            Err(_) => {}
                        }
                    }
                    Ok(acc)
                },
                |a, b| Pass1 {
                    viol: min_viol(a.viol, b.viol),
                    nb: a.nb + b.nb,
                    sinks: a.sinks + b.sinks,
                },
            )?
        };
        pl.done();
        if let Some((node, kind)) = p1.viol {
            return Err(DagError::NotBiclique { kind, node }.into());
        }
        let nb = p1.nb as usize;
        let sinks = p1.sinks as usize;
        budget.need(12 * nb as u64, "BlockIndex first + len + nin")?;
        // ---- pass 2 (sequential) ----
        pl.expected_updates(Some(n));
        pl.start(format!("BlockIndex pass 2: {nb} blocks, opening intervals"));
        let mut first = huge_vec(nb, 0u32);
        let mut len = huge_vec(nb, 0u32);
        let mut b = 0usize;
        let mut cur_end = 0usize;
        let mut covered = 0usize;
        for (v, c) in create.iter_mut().enumerate() {
            let l = *c;
            if l != 0 {
                if v < cur_end {
                    return Err(DagError::NotBiclique {
                        kind: BlockViolation::Overlap,
                        node: v,
                    }
                    .into());
                }
                first[b] = v as u32;
                len[b] = l;
                cur_end = v + l as usize;
                b += 1;
            }
            if v < cur_end {
                *c = (b - 1) as u32;
                covered += 1;
            } else {
                *c = NONE;
            }
            if v & (PASS2_LOG_STEP - 1) == PASS2_LOG_STEP - 1 {
                pl.update_with_count(PASS2_LOG_STEP);
            }
        }
        pl.update_with_count(n & (PASS2_LOG_STEP - 1));
        pl.done();
        debug_assert_eq!(b, nb);
        // ---- pass 3 (parallel) ----
        pl.expected_updates(Some(n));
        pl.start("BlockIndex pass 3: spend and nin");
        let mut nin = huge_vec(nb, 0u32);
        {
            let nin_a = as_atomic_u32(&mut nin);
            let create = &create;
            spend
                .par_chunks_mut(1 << 16)
                .for_each_with(pl.clone(), |pl, c| {
                    for s in c.iter_mut() {
                        if *s != NONE {
                            let blk = create[*s as usize];
                            *s = blk;
                            nin_a[blk as usize].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    pl.update_with_count(c.len());
                });
        }
        pl.done();
        let report = Self::report(n, m, nb, sinks, covered, &len, &nin)?;
        let idx = Self {
            n,
            m,
            nb,
            create: U32Array::from_vec(create),
            spend: U32Array::from_vec(spend),
            first: U32Array::from_vec(first),
            len: U32Array::from_vec(len),
            nin: U32Array::from_vec(nin),
        };
        Ok((idx, report))
    }

    /// Computes the totals and asserts the independent checks.
    fn report(
        n: usize,
        m: u64,
        nb: usize,
        sinks: usize,
        covered: usize,
        len: &[u32],
        nin: &[u32],
    ) -> Result<BlockReport> {
        let sum_len: u64 = len.par_iter().map(|&x| x as u64).sum();
        let sum_nin: u64 = nin.par_iter().map(|&x| x as u64).sum();
        let sum_ln: u64 = len
            .par_iter()
            .zip(nin.par_iter())
            .map(|(&l, &i)| l as u64 * i as u64)
            .sum();
        let max_len = len.par_iter().copied().max().unwrap_or(0);
        let max_nin = nin.par_iter().copied().max().unwrap_or(0);
        let r = BlockReport {
            n,
            m,
            nb,
            sources: n - covered,
            sinks,
            covered,
            sum_len,
            sum_nin,
            sum_len_times_nin: sum_ln,
            max_len,
            max_nin,
        };
        anyhow::ensure!(
            sum_len == covered as u64,
            "BlockIndex check failed: Σ len = {sum_len} but {covered} nodes covered"
        );
        anyhow::ensure!(
            sum_nin == (n - sinks) as u64,
            "BlockIndex check failed: Σ nin = {sum_nin} but n − sinks = {}",
            n - sinks
        );
        anyhow::ensure!(
            sum_ln == m,
            "BlockIndex check failed: Σ len·nin = {sum_ln} but the graph has {m} arcs"
        );
        Ok(r)
    }

    /// Builds the index of a small in-memory graph (tests).
    pub fn from_vec_graph(g: &VecGraph) -> Result<(Self, BlockReport)> {
        let cg = ChunkedGraph::new(g, ChunkConf::default())?;
        Self::build(&cg, &mut Option::<ConcurrentWrapper<ProgressLogger>>::None)
    }

    /// The report of a built or loaded index (recomputed from the arrays).
    pub fn block_report(&self) -> Result<BlockReport> {
        let sinks = self.spend.par_iter().filter(|&&s| s == NONE).count();
        let covered = self.create.par_iter().filter(|&&c| c != NONE).count();
        Self::report(
            self.n, self.m, self.nb, sinks, covered, &self.len, &self.nin,
        )
    }

    /// Writes `<p>.blk.{create,spend,first,len,nin}.u32le` and `<p>.blk.json`.
    pub fn save(&self, prefix: &Path) -> Result<()> {
        let report = self.block_report()?;
        for (name, arr) in self.arrays() {
            io::write_array(&io::suffixed(prefix, &format!("blk.{name}.u32le")), arr)?;
        }
        let json = io::suffixed(prefix, "blk.json");
        std::fs::write(&json, serde_json::to_string_pretty(&report)? + "\n")
            .with_context(|| format!("cannot write {}", json.display()))?;
        Ok(())
    }

    /// Loads an index written by [`save`](Self::save).
    pub fn load(prefix: &Path, mode: LoadMode) -> Result<Self> {
        let json = io::suffixed(prefix, "blk.json");
        let report: BlockReport = serde_json::from_str(
            &std::fs::read_to_string(&json)
                .with_context(|| format!("cannot read {}", json.display()))?,
        )
        .with_context(|| format!("cannot parse {}", json.display()))?;
        let ld =
            |name: &str| U32Array::load(&io::suffixed(prefix, &format!("blk.{name}.u32le")), mode);
        let idx = Self {
            n: report.n,
            m: report.m,
            nb: report.nb,
            create: ld("create")?,
            spend: ld("spend")?,
            first: ld("first")?,
            len: ld("len")?,
            nin: ld("nin")?,
        };
        for (name, arr) in idx.arrays() {
            let want = if matches!(name, "create" | "spend") {
                idx.n
            } else {
                idx.nb
            };
            anyhow::ensure!(
                arr.len() == want,
                "{}.blk.{name}.u32le has {} entries, expected {want}",
                prefix.display(),
                arr.len()
            );
        }
        Ok(idx)
    }

    fn arrays(&self) -> [(&'static str, &[u32]); 5] {
        [
            ("create", &self.create),
            ("spend", &self.spend),
            ("first", &self.first),
            ("len", &self.len),
            ("nin", &self.nin),
        ]
    }

    /// Block whose outputs contain `v` (`create[v]`), `None` for sources.
    #[inline]
    pub fn block_of(&self, v: usize) -> Option<usize> {
        let b = self.create[v];
        (b != NONE).then_some(b as usize)
    }

    /// Block that spends `u` (`spend[u]`), `None` for sinks.
    #[inline]
    pub fn spender(&self, u: usize) -> Option<usize> {
        let b = self.spend[u];
        (b != NONE).then_some(b as usize)
    }

    /// The outputs `[first[b], first[b] + len[b])` of block `b`.
    #[inline]
    pub fn outputs(&self, b: usize) -> Range<usize> {
        let f = self.first[b] as usize;
        f..f + self.len[b] as usize
    }

    /// First successor of `u` (`first[spend[u]]`), `None` for sinks.
    #[inline]
    pub fn first_succ(&self, u: usize) -> Option<usize> {
        self.spender(u).map(|b| self.first[b] as usize)
    }

    /// Outdegree of `u` (`len[spend[u]]`, or 0).
    #[inline]
    pub fn outdegree(&self, u: usize) -> usize {
        self.spender(u).map_or(0, |b| self.len[b] as usize)
    }

    /// Indegree of `v` (`nin[create[v]]`, or 0).
    #[inline]
    pub fn indegree(&self, v: usize) -> usize {
        self.block_of(v).map_or(0, |b| self.nin[b] as usize)
    }

    /// `bhi = first.partition_point(|f| f < hi)`: `first[X] < hi ⟺ X < bhi`.
    pub fn block_bound(&self, hi: usize) -> usize {
        self.first.partition_point(|&f| (f as usize) < hi)
    }
}
