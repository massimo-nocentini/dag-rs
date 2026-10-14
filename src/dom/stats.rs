//! Statistics of dominator trees, and the depth bound check.
//!
//! # Tree statistics ([`DomStats`])
//! Root children, *cut nodes* (`size ≥ 2`: removing `v` disconnects `size(v) − 1` nodes from
//! every source, F2 §4.5.4), log2 histograms of depth and size, and the top-k nodes by subtree
//! size (written as TSV `node\tsize\tdepth\tidom`). For a window source `s` (a root child of
//! the dominator tree), `size(s) − 1` counts the nodes whose whole ancestry in the window
//! descends from `s` alone: on the PG, outputs whose entire history comes from one coinbase
//! output (F2 §4.5.5, §8). This is the "retained size" of heap-dominator profilers
//! [\[Mit06\]](crate::references#mit06) and the bottleneck measure of ecological flow
//! networks [\[AB04\]](crate::references#ab04).
//!
//! [`DomDistributions`] add the [`crate::summary`] distributions of depth, size and root-child
//! size (top-k lists, and per node range with `--buckets`).
//!
//! Histogram convention: bucket 0 counts the value 0, bucket `k ≥ 1` the values in
//! `[2^(k−1), 2^k)`.
//!
//! # Depth bound ([`check_depth_bound`])
//! Every strict dominator of `v` lies on every root path of `v`, in particular on a shortest
//! one, so `depth(v) ≤ dmin(v) + 1`, where `dmin(v)` is the number of arcs of a shortest path
//! from a window source to `v` (for post-dominators: from `v` to a window sink). On the full PG
//! this gives `depth ≤ 25,349`. [`dmin_blocks`] and [`dmin_sweep`] compute `dmin` on the
//! window (unit-length min-plus DP, one pass).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::prelude::*;

use super::tree::DomTree;
use crate::blocks::BlockIndex;
use crate::graph::ChunkedGraph;
use crate::io::{self, NONE};
use crate::summary::{DistConf, DistReport, distribution};
use crate::sweep::{SweepConf, sweep_backward, sweep_forward};
use crate::window::Window;

/// One entry of the top-k list of [`DomStats`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TopNode {
    /// Absolute node id.
    pub node: usize,
    /// Number of nodes it (post-)dominates, itself included.
    pub size: u32,
    /// Depth in the tree (root children have depth 1).
    pub depth: u32,
    /// Its immediate (post-)dominator, `None` for the virtual root (sink).
    pub idom: Option<u32>,
}

/// Summary statistics of a (post-)dominator tree of a window.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct DomStats {
    /// Nodes of the window.
    pub n: usize,
    /// Children of the virtual root (sink).
    pub root_children: u64,
    /// Root children with `size ≥ 2`.
    pub root_children_cut: u64,
    /// `Σ (size − 1)` over the root children: nodes dominated by a single root child.
    pub root_children_dominated: u64,
    /// `Σ size` over the root children (equals `n`).
    pub root_children_size_sum: u64,
    /// Nodes with `size ≥ 2` (cut nodes).
    pub cut_nodes: u64,
    /// Maximum depth.
    pub max_depth: u32,
    /// Smallest node of maximum depth.
    pub argmax_depth: Option<usize>,
    /// Mean depth.
    pub mean_depth: f64,
    /// Log2 histogram of depths (see the module documentation).
    pub depth_log2_hist: Vec<u64>,
    /// Log2 histogram of subtree sizes.
    pub size_log2_hist: Vec<u64>,
    /// The nodes with the largest subtrees (size descending, then id ascending).
    pub top: Vec<TopNode>,
}

#[inline]
fn bucket(x: u32) -> usize {
    (u32::BITS - x.leading_zeros()) as usize
}

/// Per-thread accumulator of [`dom_stats`].
#[derive(Clone)]
struct Acc {
    root_children: u64,
    root_children_cut: u64,
    root_children_dominated: u64,
    root_children_size_sum: u64,
    cut_nodes: u64,
    max_depth: (u32, Reverse<usize>),
    depth_sum: u128,
    depth_hist: [u64; 33],
    size_hist: [u64; 33],
    top: BinaryHeap<Reverse<(u32, Reverse<usize>)>>,
}

impl Acc {
    fn new() -> Self {
        Self {
            root_children: 0,
            root_children_cut: 0,
            root_children_dominated: 0,
            root_children_size_sum: 0,
            cut_nodes: 0,
            max_depth: (0, Reverse(usize::MAX)),
            depth_sum: 0,
            depth_hist: [0; 33],
            size_hist: [0; 33],
            top: BinaryHeap::new(),
        }
    }

    fn push_top(&mut self, k: usize, size: u32, v: usize) {
        if k == 0 {
            return;
        }
        let key = (size, Reverse(v));
        if self.top.len() < k {
            self.top.push(Reverse(key));
        } else if self.top.peek().is_some_and(|Reverse(min)| key > *min) {
            self.top.pop();
            self.top.push(Reverse(key));
        }
    }

    fn add(&mut self, k: usize, v: usize, idom: u32, depth: u32, size: u32) {
        if idom == NONE {
            self.root_children += 1;
            self.root_children_size_sum += size as u64;
            self.root_children_dominated += size.saturating_sub(1) as u64;
            if size >= 2 {
                self.root_children_cut += 1;
            }
        }
        if size >= 2 {
            self.cut_nodes += 1;
        }
        self.max_depth = self.max_depth.max((depth, Reverse(v)));
        self.depth_sum += depth as u128;
        self.depth_hist[bucket(depth)] += 1;
        self.size_hist[bucket(size)] += 1;
        self.push_top(k, size, v);
    }

    fn merge(mut self, k: usize, o: Acc) -> Acc {
        self.root_children += o.root_children;
        self.root_children_cut += o.root_children_cut;
        self.root_children_dominated += o.root_children_dominated;
        self.root_children_size_sum += o.root_children_size_sum;
        self.cut_nodes += o.cut_nodes;
        self.max_depth = self.max_depth.max(o.max_depth);
        self.depth_sum += o.depth_sum;
        for i in 0..33 {
            self.depth_hist[i] += o.depth_hist[i];
            self.size_hist[i] += o.size_hist[i];
        }
        for Reverse((s, Reverse(v))) in o.top {
            self.push_top(k, s, v);
        }
        self
    }
}

fn trim(h: &[u64; 33]) -> Vec<u64> {
    let end = h.iter().rposition(|&c| c != 0).map_or(0, |i| i + 1);
    h[..end].to_vec()
}

/// Statistics of a tree given by public arrays indexed by `v − w.lo`: `idom` (absolute ids or
/// [`NONE`]), `depth` and `size`. `top` is the length of the top-k list.
pub fn dom_stats(idom: &[u32], depth: &[u32], size: &[u32], w: Window, top: usize) -> DomStats {
    let n = w.len();
    assert!(idom.len() == n && depth.len() == n && size.len() == n);
    stats_with(n, top, |i| (w.lo + i, idom[i], depth[i], size[i]))
}

/// Statistics of an in-memory tree with its [`sizes`](DomTree::sizes) (internal index space).
pub fn tree_stats(tree: &DomTree, sizes: &[u32], top: usize) -> DomStats {
    let n = tree.len();
    let lo = tree.window().lo;
    stats_with(n, top, |i| {
        let v = lo + i;
        let x = tree.index_of(v);
        (
            v,
            tree.idom_public(v),
            tree.depth_index(x),
            sizes[x as usize],
        )
    })
}

fn stats_with(
    n: usize,
    top: usize,
    get: impl Fn(usize) -> (usize, u32, u32, u32) + Sync,
) -> DomStats {
    let acc = (0..n)
        .into_par_iter()
        .with_min_len(1 << 14)
        .fold(Acc::new, |mut a, i| {
            let (v, idom, depth, size) = get(i);
            a.add(top, v, idom, depth, size);
            a
        })
        .reduce(Acc::new, |a, b| a.merge(top, b));
    let mut best: Vec<(u32, usize)> = acc
        .top
        .into_iter()
        .map(|Reverse((s, Reverse(v)))| (s, v))
        .collect();
    best.sort_unstable_by_key(|&(s, v)| (Reverse(s), v));
    let lo = if n == 0 { 0 } else { get(0).0 };
    DomStats {
        n,
        root_children: acc.root_children,
        root_children_cut: acc.root_children_cut,
        root_children_dominated: acc.root_children_dominated,
        root_children_size_sum: acc.root_children_size_sum,
        cut_nodes: acc.cut_nodes,
        max_depth: acc.max_depth.0,
        argmax_depth: (n > 0).then_some(acc.max_depth.1.0),
        mean_depth: if n == 0 {
            0.0
        } else {
            acc.depth_sum as f64 / n as f64
        },
        depth_log2_hist: trim(&acc.depth_hist),
        size_log2_hist: trim(&acc.size_hist),
        top: best
            .into_iter()
            .map(|(size, v)| {
                let (_, idom, depth, _) = get(v - lo);
                TopNode {
                    node: v,
                    size,
                    depth,
                    idom: (idom != NONE).then_some(idom),
                }
            })
            .collect(),
    }
}

/// The distributions of a (post-)dominator tree ([`crate::summary`], absolute node ids).
///
/// `root_children` has a value only at the children of the virtual root (their subtree size):
/// its `count` is the number of root children (per bucket with `--buckets`), its `sum` the
/// nodes they cover, and its top-k list the largest *distinct* dominating structures, whereas
/// [`DomStats::top`] is usually filled by the nested chains of one large subtree.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct DomDistributions {
    /// Depth of every node (root children have depth 1).
    pub depth: DistReport,
    /// Subtree size of every node (bins ≥ 2 of the histogram are the cut nodes).
    pub size: DistReport,
    /// Subtree size of the root children only.
    pub root_children: DistReport,
}

/// [`DomDistributions`] of a tree given by public arrays indexed by `v − w.lo` (as
/// [`dom_stats`]).
pub fn dom_distributions(
    idom: &[u32],
    depth: &[u32],
    size: &[u32],
    w: Window,
    conf: &DistConf<'_>,
) -> DomDistributions {
    let n = w.len();
    assert!(idom.len() == n && depth.len() == n && size.len() == n);
    distributions_with(n, w.lo, |i| (idom[i], depth[i], size[i]), conf)
}

/// [`DomDistributions`] of an in-memory tree with its [`sizes`](DomTree::sizes) (as
/// [`tree_stats`]).
pub fn tree_distributions(tree: &DomTree, sizes: &[u32], conf: &DistConf<'_>) -> DomDistributions {
    let lo = tree.window().lo;
    distributions_with(
        tree.len(),
        lo,
        |i| {
            let v = lo + i;
            let x = tree.index_of(v);
            (tree.idom_public(v), tree.depth_index(x), sizes[x as usize])
        },
        conf,
    )
}

fn distributions_with(
    n: usize,
    lo: usize,
    get: impl Fn(usize) -> (u32, u32, u32) + Sync,
    conf: &DistConf<'_>,
) -> DomDistributions {
    DomDistributions {
        depth: distribution(n, |i| Some(f64::from(get(i).1)), lo, conf),
        size: distribution(n, |i| Some(f64::from(get(i).2)), lo, conf),
        root_children: distribution(
            n,
            |i| {
                let (idom, _, size) = get(i);
                (idom == NONE).then_some(f64::from(size))
            },
            lo,
            conf,
        ),
    }
}

/// Writes the top-k list as TSV with the header `node\tsize\tdepth\tidom` (`idom` is `-` for
/// the virtual root).
pub fn write_top_tsv(path: &Path, stats: &DomStats) -> Result<()> {
    let f =
        std::fs::File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    writeln!(w, "node\tsize\tdepth\tidom")?;
    for t in &stats.top {
        match t.idom {
            Some(d) => writeln!(w, "{}\t{}\t{}\t{d}", t.node, t.size, t.depth)?,
            None => writeln!(w, "{}\t{}\t{}\t-", t.node, t.size, t.depth)?,
        }
    }
    w.flush()?;
    Ok(())
}

/// Subtree sizes (public order, indexed by `v − w.lo`) from a public parent array alone: one
/// reverse sweep in the tree's index order. `mirrored` selects post-dominators. Entries that do
/// not precede their node are treated as root children.
pub fn sizes_from_idom(idom: &[u32], w: Window, mirrored: bool) -> Vec<u32> {
    let n = w.len();
    assert_eq!(idom.len(), n);
    let mut size = io::huge_vec(n, 1u32);
    // visit children before parents: descending v for dominators, ascending for post
    let mut visit = |v: usize| {
        let p = idom[v - w.lo];
        let ok = if mirrored {
            (p as usize) > v && (p as usize) < w.hi
        } else {
            p != NONE && (p as usize) >= w.lo && (p as usize) < v
        };
        if p != NONE && ok {
            size[p as usize - w.lo] += size[v - w.lo];
        }
    };
    if mirrored {
        (w.lo..w.hi).for_each(&mut visit);
    } else {
        (w.lo..w.hi).rev().for_each(&mut visit);
    }
    size
}

/// Result of [`check_depth_bound`].
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct DepthBound {
    /// Nodes checked.
    pub checked: u64,
    /// Nodes with `depth > dmin + 1`.
    pub violations: u64,
    /// The smallest violating node.
    pub first_violation: Option<usize>,
    /// Maximum of `dmin` over the window.
    pub max_dmin: u32,
    /// Mean of `dmin` over the window.
    pub mean_dmin: f64,
    /// Maximum of `dmin + 1 − depth`.
    pub max_slack: u32,
}

/// Checks `depth(v) ≤ dmin(v) + 1` for every node of the tree (module docs); `dmin` is indexed
/// by `v − lo` and must be in the tree's direction (from sources for dominators, to sinks for
/// post-dominators).
pub fn check_depth_bound(tree: &DomTree, dmin: &[u32]) -> DepthBound {
    let n = tree.len();
    assert_eq!(dmin.len(), n);
    let lo = tree.window().lo;
    let (viol, first, maxd, sum, slack) = (0..n)
        .into_par_iter()
        .with_min_len(1 << 14)
        .map(|i| {
            let d = tree.depth(lo + i) as u64;
            let m = dmin[i] as u64;
            let bad = d > m + 1;
            (
                bad as u64,
                if bad { lo + i } else { usize::MAX },
                dmin[i],
                m,
                (m + 1).saturating_sub(d) as u32,
            )
        })
        .reduce(
            || (0, usize::MAX, 0, 0, 0),
            |a, b| {
                (
                    a.0 + b.0,
                    a.1.min(b.1),
                    a.2.max(b.2),
                    a.3 + b.3,
                    a.4.max(b.4),
                )
            },
        );
    DepthBound {
        checked: n as u64,
        violations: viol,
        first_violation: (first != usize::MAX).then_some(first),
        max_dmin: maxd,
        mean_dmin: if n == 0 { 0.0 } else { sum as f64 / n as f64 },
        max_slack: slack,
    }
}

/// Unit-length shortest distance inside the window, indexed by `v − lo`: from a window source
/// to `v` (`post = false`), or from `v` to a window sink (`post = true`), computed from the
/// [`BlockIndex`] alone (one sequential array scan; 4 B per node and per block). `pl` is
/// updated every 2²⁰ nodes.
pub fn dmin_blocks(
    blocks: &BlockIndex,
    w: Window,
    post: bool,
    pl: &mut impl ProgressLog,
) -> Vec<u32> {
    const STEP: usize = 1 << 20;
    let (lo, hi) = (w.lo, w.hi);
    let create: &[u32] = &blocks.create;
    let spend: &[u32] = &blocks.spend;
    let first: &[u32] = &blocks.first;
    let len: &[u32] = &blocks.len;
    let (blo, bhi) = (blocks.block_bound(lo), blocks.block_bound(hi));
    let mut d = io::huge_vec(w.len(), 0u32);
    // per block: min over the window inputs (forward) or the window outputs (backward)
    let mut blk = io::huge_vec(bhi - blo, u32::MAX);
    pl.item_name("node");
    pl.expected_updates(Some(w.len()));
    if !post {
        pl.start("shortest distances from the window sources (blocks)");
        for v in lo..hi {
            if (v - lo) % STEP == STEP - 1 {
                pl.update_with_count(STEP);
            }
            let b = create[v] as usize;
            d[v - lo] = if b != NONE as usize && (blo..bhi).contains(&b) && blk[b - blo] != u32::MAX
            {
                blk[b - blo] + 1
            } else {
                0
            };
            let s = spend[v] as usize;
            if s != NONE as usize && s < bhi {
                blk[s - blo] = blk[s - blo].min(d[v - lo]);
            }
        }
    } else {
        pl.start("shortest distances to the window sinks (blocks)");
        for u in (lo..hi).rev() {
            if (hi - 1 - u) % STEP == STEP - 1 {
                pl.update_with_count(STEP);
            }
            let s = spend[u] as usize;
            d[u - lo] = if s == NONE as usize || s >= bhi {
                0
            } else {
                if blk[s - blo] == u32::MAX {
                    let f = first[s] as usize;
                    let end = (f + len[s] as usize).min(hi);
                    blk[s - blo] = (f..end).map(|o| d[o - lo]).min().unwrap_or(0);
                }
                blk[s - blo] + 1
            };
        }
    }
    pl.update_with_count(w.len() % STEP);
    pl.done();
    d
}

/// [`dmin_blocks`] for any DAG in identity topological order: a push over the forward graph
/// (`post = false`) or a pull over backward chunks (`post = true`).
pub fn dmin_sweep<G>(
    g: &ChunkedGraph<G>,
    w: Window,
    conf: &SweepConf,
    post: bool,
    pl: &mut impl ProgressLog,
) -> Result<Vec<u32>>
where
    G: RandomAccessGraph + Sync,
{
    let lo = w.lo;
    pl.item_name("node");
    pl.expected_updates(Some(w.len()));
    if !post {
        let mut d = io::huge_vec(w.len(), u32::MAX);
        pl.start("shortest distances from the window sources");
        sweep_forward(g, w, conf, pl, |chunk| {
            for (u, succ) in chunk.iter() {
                let du = &mut d[u - lo];
                if *du == u32::MAX {
                    *du = 0;
                }
                let du = *du;
                for &v in succ {
                    let dv = &mut d[v as usize - lo];
                    *dv = (*dv).min(du + 1);
                }
            }
            Ok(())
        })?;
        pl.done();
        Ok(d)
    } else {
        let mut d = io::huge_vec(w.len(), 0u32);
        pl.start("shortest distances to the window sinks");
        sweep_backward(g, w, conf, pl, |chunk| {
            for (u, succ) in chunk.iter_rev() {
                d[u - lo] = succ
                    .iter()
                    .map(|&v| d[v as usize - lo] + 1)
                    .min()
                    .unwrap_or(0);
            }
            Ok(())
        })?;
        pl.done();
        Ok(d)
    }
}
