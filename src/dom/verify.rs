//! Independent verifier of dominator trees.
//!
//! Given a candidate parent array `t` (absolute ids, [`NONE`] for the virtual root), the
//! verifier decides whether `t` is the dominator tree `D` of the window `[lo, hi)` of a DAG in
//! identity topological order (F2 §7.1, adapted to a parallel push). It shares no code path with
//! the builders: no NCA folds, only preorder intervals and level ancestors on a tree it rebuilds
//! from `t` alone, and a parallel scan of the **forward** graph.
//!
//! ```text
//! 1. rebuild parent, depth, Myers jump, size and preorder from t (F2 §4.5)
//!    V0: every parent precedes its child (t[v] = NONE or lo ≤ t[v] < v), and a window
//!        source has t[v] = NONE
//! 2. par_chunks scan, for every arc (u, v) of the window:
//!    V1: t[v] is an ancestor-or-self of u            (pre[t] ≤ pre[u] < pre[t] + size[t])
//!    minpre[v] = min(pre[u] << 32 | u)  (AtomicU64),  maxpre[v] = max(pre[u])  (AtomicU32)
//!    parent_arc[v] |= (u == t[v])        (atomic bitset)
//! 3. V2, in parallel over v with predecessors: parent_arc[v], or maxpre[v] lies outside the
//!    preorder interval of c = LA(argmin, depth(t[v]) + 1)
//! ```
//!
//! **Theorem.** V0 ∧ V1 ∧ V2 ⟺ `t = D`. V1 is the parent property; V2 says that every node
//! has its parent arc or predecessors in two different child subtrees of its parent ("derived
//! in-degree ≥ 2"), i.e. `t(v) = NCA_T(pred(v))`. The induction lemma of F2 §2 then gives
//! `T = D` (by induction on the topological rank, the prefixes of `T` and `D` coincide, so the
//! NCAs of the predecessors coincide). Derived arcs and their certificates are due to
//! Georgiadis and Tarjan [\[GT15\]](crate::references#gt15); the dominator characterisation by
//! data flow to [\[All70\]](crate::references#all70).
//!
//! Post-dominators use the same steps on the reverse graph with mirrored indices: the arc
//! `(u, v)` is read as `(v, u)`, so the scan still runs over the forward graph and both
//! verifiers need only `g`.
//!
//! Complexity: O(n log n + m) work, the scan in parallel. Memory: about 32 B per node (70 GB on
//! the PG): parent/depth/jump 12 B, size and preorder 8 B, `minpre` 8 B, `maxpre` 4 B and one
//! bit.
//!
//! Counting conventions of [`VerifyReport`]: `v0_fail` and `v2_fail` count nodes, `v1_fail`
//! counts arcs; `first_failure` is the smallest failing node (absolute id) with a message. A
//! node is counted in `v2_fail` only if all its incoming arcs pass V1 (otherwise only those
//! arcs are counted, in `v1_fail`).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::Result;
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::error::DagError;
use crate::graph::ChunkedGraph;
use crate::io::{self, NONE, as_atomic_u32, as_atomic_u64};
use crate::sweep::par_chunks;
use crate::window::Window;

/// Result of [`verify_dominators`] and [`verify_post_dominators`].
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct VerifyReport {
    /// Whether the candidate is the (post-)dominator tree.
    pub ok: bool,
    /// Nodes whose parent does not precede them, or sources with a parent other than the root.
    pub v0_fail: u64,
    /// Arcs `(u, v)` whose head's parent is not an ancestor-or-self of the tail.
    pub v1_fail: u64,
    /// Nodes without their parent arc whose predecessors all lie in one child subtree of their
    /// parent. Nodes with a predecessor outside the subtree of their parent are not counted
    /// here: their arcs already count in `v1_fail`.
    pub v2_fail: u64,
    /// The smallest failing node (absolute id) and a description.
    pub first_failure: Option<(usize, String)>,
}

impl VerifyReport {
    fn note(&mut self, v: usize, msg: impl FnOnce() -> String) {
        if self.first_failure.as_ref().is_none_or(|(f, _)| v < *f) {
            self.first_failure = Some((v, msg()));
        }
    }
}

/// Bytes used by the verifier on a window of `n` nodes.
pub fn verify_bytes(n: usize) -> u64 {
    (n as u64 + 1) * (12 + 4 + 4 + 8 + 4) + (n as u64).div_ceil(64) * 8
}

/// Verifies the immediate dominators `idom` (indexed by `v − lo`, absolute values, [`NONE`]
/// for the virtual root) of the window `w` of `g` (module docs). Returns
/// [`DagError::NotTopological`] if an arc of the window goes backward.
pub fn verify_dominators<G>(
    g: &ChunkedGraph<G>,
    idom: &[u32],
    w: Window,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<VerifyReport>
where
    G: RandomAccessGraph + Sync,
{
    verify(g, idom, w, false, pl)
}

/// Verifies the immediate post-dominators `ipdom` of the window `w` of `g` (same conventions as
/// [`verify_dominators`], [`NONE`] for the virtual sink), scanning the forward graph.
pub fn verify_post_dominators<G>(
    g: &ChunkedGraph<G>,
    ipdom: &[u32],
    w: Window,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<VerifyReport>
where
    G: RandomAccessGraph + Sync,
{
    verify(g, ipdom, w, true, pl)
}

/// Parent, depth and skew-binary jump pointer of a rebuilt node.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VNode {
    parent: u32,
    depth: u32,
    jump: u32,
}

/// The tree rebuilt from the candidate array, in internal index space (0 = root).
struct Rebuilt {
    nodes: Vec<VNode>,
    size: Vec<u32>,
    pre: Vec<u32>,
}

impl Rebuilt {
    fn new(par: Vec<u32>) -> Self {
        let len = par.len();
        let mut nodes = io::huge_vec(len, VNode::default());
        for x in 1..len {
            let p = par[x];
            let pn = nodes[p as usize];
            let j = nodes[pn.jump as usize];
            let jump = if pn.depth - j.depth == j.depth - nodes[j.jump as usize].depth {
                j.jump
            } else {
                p
            };
            nodes[x] = VNode {
                parent: p,
                depth: pn.depth + 1,
                jump,
            };
        }
        drop(par);
        log::info!("rebuild: parents, depths and jump pointers done");
        let mut size = io::huge_vec(len, 1u32);
        for x in (1..len).rev() {
            size[nodes[x].parent as usize] += size[x];
        }
        log::info!("rebuild: subtree sizes done");
        let mut pre = io::huge_vec(len, 0u32);
        let mut next = io::huge_vec(len, 0u32);
        next[0] = 1;
        for x in 1..len {
            let p = nodes[x].parent as usize;
            pre[x] = next[p];
            next[p] += size[x];
            next[x] = pre[x] + 1;
        }
        Self { nodes, size, pre }
    }

    /// Whether `a` is an ancestor-or-self of `x`.
    #[inline]
    fn is_ancestor(&self, a: u32, x: u32) -> bool {
        let (pa, px) = (self.pre[a as usize], self.pre[x as usize]);
        pa <= px && px - pa < self.size[a as usize]
    }

    /// Ancestor of `x` at depth `d ≤ depth(x)`.
    fn level_ancestor(&self, mut x: u32, d: u32) -> u32 {
        loop {
            let n = self.nodes[x as usize];
            if n.depth <= d {
                return x;
            }
            x = if self.nodes[n.jump as usize].depth >= d {
                n.jump
            } else {
                n.parent
            };
        }
    }
}

/// Failures found by one chunk of the scan: V1 count and the smallest `(node, pred)`.
#[derive(Default)]
struct ScanFail {
    v1: u64,
    first: Option<(usize, usize)>,
}

fn verify<G>(
    g: &ChunkedGraph<G>,
    t: &[u32],
    w: Window,
    mirrored: bool,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<VerifyReport>
where
    G: RandomAccessGraph + Sync,
{
    let (lo, hi) = (w.lo, w.hi);
    let n = w.len();
    anyhow::ensure!(
        hi <= g.num_nodes(),
        DagError::BadWindow {
            lo,
            hi,
            n: g.num_nodes()
        }
    );
    anyhow::ensure!(
        t.len() == n,
        "the candidate array has {} entries, the window {n} nodes",
        t.len()
    );
    let index = |v: usize| -> u32 {
        if mirrored {
            (hi - v) as u32
        } else {
            (v - lo + 1) as u32
        }
    };
    let node = |x: u32| -> usize {
        if mirrored {
            hi - x as usize
        } else {
            x as usize - 1 + lo
        }
    };
    // a parent must precede its child in the index space
    let valid = |v: usize, p: u32| -> bool {
        let p = p as usize;
        if mirrored {
            v < p && p < hi
        } else {
            lo <= p && p < v
        }
    };
    let mut report = VerifyReport::default();

    // ---- 1. rebuild the tree; V0 (order) ------------------------------------------------
    let mut par = io::huge_capacity::<u32>(n + 1);
    par.push(0);
    par.par_extend((1..=n as u32).into_par_iter().map(|x| {
        let v = node(x);
        match t[v - lo] {
            NONE => 0,
            p if valid(v, p) => index(p as usize),
            _ => 0, // counted below; the node is attached to the root to keep a tree
        }
    }));
    let (v0, v0_first) = (0..n)
        .into_par_iter()
        .filter(|&i| t[i] != NONE && !valid(lo + i, t[i]))
        .map(|i| (1u64, lo + i))
        .reduce(|| (0, usize::MAX), |a, b| (a.0 + b.0, a.1.min(b.1)));
    report.v0_fail = v0;
    if v0 > 0 {
        let p = t[v0_first - lo];
        report.note(v0_first, || {
            format!("V0: parent {p} of node {v0_first} does not precede it")
        });
    }
    let t_rebuild = std::time::Instant::now();
    log::info!("rebuilding the candidate tree of {n} nodes (three sequential passes)");
    let tree = Rebuilt::new(par);
    log::info!(
        "candidate tree rebuilt in {:.2} s",
        t_rebuild.elapsed().as_secs_f64()
    );

    // ---- 2. scan: V1 and the V2 accumulators ---------------------------------------------
    let mut minpre = io::huge_vec(n + 1, u64::MAX);
    let mut maxpre = io::huge_vec(n + 1, 0u32);
    let mut parent_arc = io::huge_vec((n + 1).div_ceil(64), 0u64);
    let fail = {
        let minpre: &[AtomicU64] = as_atomic_u64(&mut minpre);
        let maxpre: &[AtomicU32] = as_atomic_u32(&mut maxpre);
        let parent_arc: &[AtomicU64] = as_atomic_u64(&mut parent_arc);
        let tree = &tree;
        pl.item_name("node");
        pl.expected_updates(Some(n));
        pl.start(if mirrored {
            "verifying post-dominators"
        } else {
            "verifying dominators"
        });
        let fail = par_chunks(
            g,
            w,
            pl,
            |it| {
                let mut f = ScanFail::default();
                while let Some((u, succ)) = it.next_node() {
                    for &v in succ {
                        let v = v as usize;
                        if v <= u {
                            return Err(DagError::NotTopological { u, v }.into());
                        }
                        let (from, to) = if mirrored { (v, u) } else { (u, v) };
                        let (xf, xt) = (index(from), index(to));
                        let p = tree.nodes[xt as usize].parent;
                        if !tree.is_ancestor(p, xf) {
                            f.v1 += 1;
                            if f.first.is_none_or(|m| (to, from) < m) {
                                f.first = Some((to, from));
                            }
                        }
                        let pf = tree.pre[xf as usize];
                        minpre[xt as usize]
                            .fetch_min((pf as u64) << 32 | xf as u64, Ordering::Relaxed);
                        maxpre[xt as usize].fetch_max(pf, Ordering::Relaxed);
                        if p == xf {
                            parent_arc[xt as usize / 64]
                                .fetch_or(1 << (xt % 64), Ordering::Relaxed);
                        }
                    }
                }
                Ok(f)
            },
            |a, b| ScanFail {
                v1: a.v1 + b.v1,
                first: match (a.first, b.first) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    (x, y) => x.or(y),
                },
            },
        )?;
        pl.done();
        fail
    };
    report.v1_fail = fail.v1;
    if let Some((v, u)) = fail.first {
        let p = t[v - lo];
        report.note(v, || {
            if mirrored {
                format!("V1: arc ({v}, {u}): ipdom {p} of {v} does not post-dominate {u}")
            } else {
                format!("V1: arc ({u}, {v}): idom {p} of {v} does not dominate {u}")
            }
        });
    }

    // ---- 3. V2 (and sources with a parent, counted as V0) --------------------------------
    // 0 = ok, 1 = source with a parent (V0), 2 = derived in-degree 1 (V2)
    let check = |x: u32| -> u8 {
        let p = tree.nodes[x as usize].parent;
        let mp = minpre[x as usize];
        if mp == u64::MAX {
            // no predecessor in the window: only the virtual arc from the root
            return if t[node(x) - lo] == NONE { 0 } else { 1 };
        }
        if parent_arc[x as usize / 64] >> (x % 64) & 1 == 1 {
            return 0;
        }
        let a = mp as u32;
        let d = tree.nodes[p as usize].depth + 1;
        if tree.nodes[a as usize].depth < d || !tree.is_ancestor(p, a) {
            // the parent is not an ancestor of the predecessor a (a ≠ p: no parent arc), so the
            // arc (a, x) already failed V1 in the scan; V2 counts only V1-clean nodes
            return 0;
        }
        let c = tree.level_ancestor(a, d);
        let (pc, q) = (tree.pre[c as usize], maxpre[x as usize]);
        if q >= pc && q - pc < tree.size[c as usize] {
            2
        } else {
            0
        }
    };
    let counts = (1..=n as u32)
        .into_par_iter()
        .map(|x| match check(x) {
            0 => [0u64, 0, usize::MAX as u64, usize::MAX as u64],
            1 => [1, 0, node(x) as u64, usize::MAX as u64],
            _ => [0, 1, usize::MAX as u64, node(x) as u64],
        })
        .reduce(
            || [0, 0, usize::MAX as u64, usize::MAX as u64],
            |a, b| [a[0] + b[0], a[1] + b[1], a[2].min(b[2]), a[3].min(b[3])],
        );
    report.v0_fail += counts[0];
    report.v2_fail = counts[1];
    if counts[0] > 0 {
        let v = counts[2] as usize;
        let p = t[v - lo];
        report.note(v, || {
            format!("V0: node {v} has no predecessor in the window but parent {p}")
        });
    }
    if counts[1] > 0 {
        let v = counts[3] as usize;
        let p = t[v - lo];
        report.note(v, || {
            format!(
                "V2: node {v} lacks the arc from its parent {} and all its predecessors lie below one child of it",
                if p == NONE { "ROOT".to_string() } else { p.to_string() }
            )
        });
    }
    report.ok = report.v0_fail == 0 && report.v1_fail == 0 && report.v2_fail == 0;
    Ok(report)
}
