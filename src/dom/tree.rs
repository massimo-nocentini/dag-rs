//! Dominator tree with hybrid NCA (root-child test, budgeted walk, jump pointers).
//!
//! # Layout
//! A [`DomTree`] stores, for every **internal index** `x ∈ 0..=n` (`x = 0` is the virtual root
//! or virtual sink [`ROOT`]), one 16-byte record `{parent, depth, jump, top}` in a single
//! array-of-structs allocated with [`io::huge_vec`], so that one visit of a
//! node is one cache line access. The index space is chosen so that **every parent has a smaller
//! index than its child** (F2 §1):
//!
//! - dominators of the window `[lo, hi)`: `x = v − lo + 1`;
//! - post-dominators (the *mirror*): `x = hi − v`.
//!
//! The same kernel therefore serves both trees. Memory: 16 B per node (34.9 GB on the PG).
//!
//! # Nearest common ancestor
//! [`DomTree::nca`] is a hybrid of three kernels that all return the same answer:
//!
//! ```text
//! nca(a, b):
//!   1. if top[a] ≠ top[b]: return ROOT          // different root children: two reads
//!   2. budgeted walk: while a ≠ b and steps < budget:
//!        move the deeper node to its parent (both when depths are equal)
//!   3. budget exhausted: lift the deeper node with level_ancestor, then the Myers
//!      jump search for the NCA                   // O(log n) worst case
//! ```
//!
//! `top[x]` is the child of the root above `x` (`x` itself for root children). On the PG the
//! common case is a merge of coins from different coinbase lineages, which ends at the root
//! after the first test. The walk of step 2 is the `intersect` of Cooper, Harvey and Kennedy
//! [\[CHK01\]](crate::references#chk01) guided by depth instead of postorder numbers; it costs
//! O(depth) in the worst case (the *ladder* family). Step 3 uses the skew-binary jump pointers
//! of Myers [\[Mye83\]](crate::references#mye83): [`add_leaf`](DomTree::add_leaf) is O(1) and
//! level-ancestor and NCA queries are O(log n), which suits a tree that grows by leaf
//! insertions in topological order.
//!
//! Every call through [`nca_tracked`](DomTree::nca_tracked) updates [`NcaStats`]: the number of
//! calls, of exits at the `top` test, a log2 histogram of walk steps and the number of jump
//! fallbacks. [`nca`](DomTree::nca) is the same query without statistics (`&self`).
//!
//! # Derived arrays
//! [`sizes`](DomTree::sizes) (number of nodes dominated, including the node itself) is one
//! reverse sweep, [`preorder`](DomTree::preorder) one forward sweep with a temporary `next`
//! array (F2 §4.5); together they answer "`d` dominates `v`" in O(1) with
//! `pre[d] ≤ pre[v] < pre[d] + size[d]`.
//!
//! # Invariants
//! - `parent[x] < x` for every finalised `x ≥ 1`; `parent[ROOT] = ROOT`, `depth[ROOT] = 0`.
//! - Not yet finalised slots hold [`UNSET`] (or, in the push builder, the running NCA of the
//!   predecessors seen so far, see [`pending`](DomTree::pending)); never [`ROOT`].
//! - Queries only climb through finalised nodes.

use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;

use crate::io::{self, NONE, array_path};
use crate::window::Window;

/// Internal index of the virtual root (dominators) or virtual sink (post-dominators).
pub const ROOT: u32 = 0;

/// Parent slot not yet finalised. It is distinct from [`io::NONE`] and from
/// every valid internal index, because [`DomTree::for_window`] admits at most `u32::MAX − 2`
/// nodes (indices `0..=n ≤ u32::MAX − 2 < UNSET`).
pub const UNSET: u32 = u32::MAX - 1;

/// Number of elements converted per parallel batch when writing arrays.
const WRITE_BATCH: usize = 1 << 22;

/// One node of the tree: 16 bytes, one cache-line access per visit.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TNode {
    parent: u32,
    depth: u32,
    jump: u32,
    top: u32,
}

/// Performance counters of the NCA kernel (reported in the JSON of `dag dom build`).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct NcaStats {
    /// Number of NCA queries.
    pub calls: u64,
    /// Queries answered by the root-child (`top`) test.
    pub top_exits: u64,
    /// Histogram of the steps of the budgeted walk for the queries that passed the `top` test:
    /// bucket 0 counts walks of 0 steps, bucket `k ≥ 1` walks of `[2^(k−1), 2^k)` steps.
    /// Serialized as a JSON array of 33 counts.
    #[serde(serialize_with = "serialize_hist")]
    pub walk_steps_log2_hist: [u64; 33],
    /// Queries that exhausted the budget and finished with jump pointers.
    pub jump_fallbacks: u64,
}

impl Default for NcaStats {
    fn default() -> Self {
        Self {
            calls: 0,
            top_exits: 0,
            walk_steps_log2_hist: [0; 33],
            jump_fallbacks: 0,
        }
    }
}

impl NcaStats {
    /// Adds the counters of `other` to `self`.
    pub fn merge(&mut self, other: &NcaStats) {
        self.calls += other.calls;
        self.top_exits += other.top_exits;
        self.jump_fallbacks += other.jump_fallbacks;
        for (a, b) in self
            .walk_steps_log2_hist
            .iter_mut()
            .zip(&other.walk_steps_log2_hist)
        {
            *a += b;
        }
    }

    /// Total number of walk steps, bounded from above by the histogram (each bucket counted at
    /// its upper end); useful as a cost proxy.
    pub fn walk_steps_upper_bound(&self) -> u64 {
        self.walk_steps_log2_hist
            .iter()
            .enumerate()
            .map(|(k, &c)| if k == 0 { 0 } else { c * ((1u64 << k) - 1) })
            .sum()
    }
}

/// serde implements `Serialize` for arrays of at most 32 elements: serialize as a slice.
fn serialize_hist<S: serde::Serializer>(h: &[u64; 33], s: S) -> Result<S::Ok, S::Error> {
    serde::Serialize::serialize(&h[..], s)
}

/// Bucket of the walk-steps histogram (see [`NcaStats::walk_steps_log2_hist`]).
#[inline]
fn log2_bucket(steps: u32) -> usize {
    (u32::BITS - steps.leading_zeros()) as usize
}

/// How a query was answered (internal, for the statistics).
enum NcaPath {
    Top,
    Walk(u32),
    Jump(u32),
}

/// A dominator or post-dominator tree of a window, in internal index space (module docs).
pub struct DomTree {
    nodes: Vec<TNode>,
    mirrored: bool,
    budget: u32,
    lo: usize,
    hi: usize,
    /// Counters of the NCA queries made through [`nca_tracked`](DomTree::nca_tracked).
    pub stats: NcaStats,
}

impl std::fmt::Debug for DomTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomTree")
            .field("window", &(self.lo..self.hi))
            .field("mirrored", &self.mirrored)
            .field("budget", &self.budget)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl DomTree {
    /// A tree for the window `[0, n)`: node 0 is [`ROOT`] (depth 0, top 0) and every other
    /// parent slot is [`UNSET`] (never `ROOT`: the push builder relies on it, F2 §4.1, T13).
    /// `mirrored` selects post-dominator indexing (`x = n − v`); `budget` is the number of walk
    /// steps before the jump-pointer fallback (`u32::MAX` = never fall back).
    pub fn with_capacity(n: usize, mirrored: bool, budget: u32) -> Self {
        Self::for_window(Window { lo: 0, hi: n }, mirrored, budget)
    }

    /// A tree for the window `w` (see [`with_capacity`](Self::with_capacity)).
    ///
    /// # Panics
    /// If the window has more than `u32::MAX − 2` nodes (indices must stay below [`UNSET`]).
    pub fn for_window(w: Window, mirrored: bool, budget: u32) -> Self {
        let n = w.hi - w.lo;
        assert!(
            n < UNSET as usize,
            "window of {n} nodes is too large for u32 tree indices"
        );
        let mut nodes = io::huge_vec(
            n + 1,
            TNode {
                parent: UNSET,
                depth: 0,
                jump: 0,
                top: 0,
            },
        );
        nodes[0] = TNode {
            parent: ROOT,
            depth: 0,
            jump: ROOT,
            top: ROOT,
        };
        Self {
            nodes,
            mirrored,
            budget,
            lo: w.lo,
            hi: w.hi,
            stats: NcaStats::default(),
        }
    }

    /// Bytes used by a tree over `n` nodes (16 B per node plus the root).
    pub fn bytes(n: usize) -> u64 {
        (n as u64 + 1) * std::mem::size_of::<TNode>() as u64
    }

    /// Number of (non-root) nodes, i.e. the window length.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len() - 1
    }

    /// Whether the tree has no node besides the root.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The window of the tree.
    pub fn window(&self) -> Window {
        Window {
            lo: self.lo,
            hi: self.hi,
        }
    }

    /// Whether this is a post-dominator tree (mirrored indices `x = hi − v`).
    pub fn is_mirrored(&self) -> bool {
        self.mirrored
    }

    /// The walk budget of [`nca`](Self::nca).
    pub fn budget(&self) -> u32 {
        self.budget
    }

    /// Sets the walk budget (the answers do not depend on it).
    pub fn set_budget(&mut self, budget: u32) {
        self.budget = budget;
    }

    /// Internal index of the absolute node `v ∈ [lo, hi)`.
    #[inline]
    pub fn index_of(&self, v: usize) -> u32 {
        debug_assert!(v >= self.lo && v < self.hi, "node {v} outside the window");
        if self.mirrored {
            (self.hi - v) as u32
        } else {
            (v - self.lo + 1) as u32
        }
    }

    /// Absolute node of the internal index `x ≥ 1`.
    #[inline]
    pub fn node_of(&self, x: u32) -> usize {
        debug_assert!(x != ROOT);
        if self.mirrored {
            self.hi - x as usize
        } else {
            x as usize - 1 + self.lo
        }
    }

    /// Finalises `x` as a leaf below the finalised node `parent` (O(1)). The jump pointer
    /// follows Myers' skew-binary scheme [\[Mye83\]](crate::references#mye83): if the jumps of
    /// `parent` and of its jump target span equal depth differences, `x` jumps two levels of
    /// the scheme up, otherwise it jumps to `parent`.
    #[inline]
    pub fn add_leaf(&mut self, x: u32, parent: u32) {
        debug_assert!(parent < x, "parent {parent} must precede {x}");
        let p = self.nodes[parent as usize];
        let j = self.nodes[p.jump as usize];
        let jj = j.jump;
        let jump = if p.depth - j.depth == j.depth - self.nodes[jj as usize].depth {
            jj
        } else {
            parent
        };
        let top = if parent == ROOT { x } else { p.top };
        self.nodes[x as usize] = TNode {
            parent,
            depth: p.depth + 1,
            jump,
            top,
        };
    }

    /// Finalises `x` as a sibling of the finalised node `sibling` (copies its parent, depth,
    /// jump and top): the PG blocks builder uses it for all outputs of a transaction but the
    /// first, which share the predecessor set and therefore the immediate dominator.
    #[inline]
    pub fn add_leaf_like(&mut self, x: u32, sibling: u32) {
        debug_assert!(sibling < x && sibling != ROOT);
        let s = self.nodes[sibling as usize];
        self.nodes[x as usize] = if s.parent == ROOT {
            // a root child is its own top
            TNode { top: x, ..s }
        } else {
            s
        };
    }

    /// The parent slot of `x` (finalised or not): [`UNSET`], a running NCA in the push
    /// builder, or the parent of a finalised node.
    #[inline]
    pub fn pending(&self, x: u32) -> u32 {
        self.nodes[x as usize].parent
    }

    /// Writes the parent slot of a not yet finalised `x` (push builder only).
    #[inline]
    pub fn set_pending(&mut self, x: u32, p: u32) {
        self.nodes[x as usize].parent = p;
    }

    /// Parent index of the finalised `x` ([`ROOT`] for root children and for the root).
    #[inline]
    pub fn parent(&self, x: u32) -> u32 {
        self.nodes[x as usize].parent
    }

    /// Depth of the internal index `x` (root 0, root children 1).
    #[inline]
    pub fn depth_index(&self, x: u32) -> u32 {
        self.nodes[x as usize].depth
    }

    /// Nearest common ancestor of the finalised nodes `a` and `b` (internal indices), without
    /// statistics. See the module documentation for the three kernels.
    #[inline]
    pub fn nca(&self, a: u32, b: u32) -> u32 {
        self.nca_impl(a, b).0
    }

    /// [`nca`](Self::nca) that also updates [`stats`](Self::stats).
    #[inline]
    pub fn nca_tracked(&mut self, a: u32, b: u32) -> u32 {
        let (c, path) = self.nca_impl(a, b);
        let s = &mut self.stats;
        s.calls += 1;
        match path {
            NcaPath::Top => s.top_exits += 1,
            NcaPath::Walk(k) => s.walk_steps_log2_hist[log2_bucket(k)] += 1,
            NcaPath::Jump(k) => {
                s.walk_steps_log2_hist[log2_bucket(k)] += 1;
                s.jump_fallbacks += 1;
            }
        }
        c
    }

    #[inline]
    fn nca_impl(&self, mut a: u32, mut b: u32) -> (u32, NcaPath) {
        let nodes = &self.nodes[..];
        let (mut na, mut nb) = (nodes[a as usize], nodes[b as usize]);
        if na.top != nb.top {
            return (ROOT, NcaPath::Top);
        }
        let mut steps = 0u32;
        while a != b {
            if steps == self.budget {
                return (self.nca_jump(a, b), NcaPath::Jump(steps));
            }
            steps += 1;
            let (da, db) = (na.depth, nb.depth);
            // the deeper node cannot be an ancestor of the other one; at equal depths the NCA
            // lies strictly above both
            if da >= db {
                a = na.parent;
                na = nodes[a as usize];
            }
            if db >= da {
                b = nb.parent;
                nb = nodes[b as usize];
            }
        }
        (a, NcaPath::Walk(steps))
    }

    /// NCA with jump pointers only (O(log n)): lifts the deeper node, then climbs both nodes
    /// with equal jumps.
    fn nca_jump(&self, a: u32, b: u32) -> u32 {
        let nodes = &self.nodes[..];
        let (da, db) = (nodes[a as usize].depth, nodes[b as usize].depth);
        let (mut a, mut b) = if da >= db {
            (self.level_ancestor(a, db), b)
        } else {
            (a, self.level_ancestor(b, da))
        };
        if a == b {
            return a;
        }
        // same depth: the jump targets of a and b are at the same depth too
        loop {
            let (na, nb) = (nodes[a as usize], nodes[b as usize]);
            if na.parent == nb.parent {
                return na.parent;
            }
            if na.jump != nb.jump {
                a = na.jump;
                b = nb.jump;
            } else {
                a = na.parent;
                b = nb.parent;
            }
        }
    }

    /// The ancestor of the finalised `x` at depth `d ≤ depth(x)` (O(log n) with jump pointers).
    pub fn level_ancestor(&self, mut x: u32, d: u32) -> u32 {
        let nodes = &self.nodes[..];
        debug_assert!(d <= nodes[x as usize].depth);
        loop {
            let n = nodes[x as usize];
            if n.depth <= d {
                return x;
            }
            x = if nodes[n.jump as usize].depth >= d {
                n.jump
            } else {
                n.parent
            };
        }
    }

    /// Immediate (post-)dominator of the absolute node `v ∈ [lo, hi)` as an absolute public id,
    /// [`NONE`] when it is the virtual root (sink).
    #[inline]
    pub fn idom_public(&self, v: usize) -> u32 {
        self.public_of_index(self.index_of(v))
    }

    /// Public id of the parent of the internal index `x ≥ 1`.
    #[inline]
    fn public_of_index(&self, x: u32) -> u32 {
        match self.nodes[x as usize].parent {
            ROOT => NONE,
            p => self.node_of(p) as u32,
        }
    }

    /// Depth of the absolute node `v` (root children, e.g. window sources, have depth 1).
    #[inline]
    pub fn depth(&self, v: usize) -> u32 {
        self.nodes[self.index_of(v) as usize].depth
    }

    /// Public immediate (post-)dominators of the window, indexed by `v − lo` (absolute values,
    /// [`NONE`] for the virtual root/sink).
    pub fn idoms(&self) -> Vec<u32> {
        let n = self.len();
        let mut out = io::huge_capacity(n);
        out.par_extend(
            (0..n)
                .into_par_iter()
                .map(|i| self.public_of_index(self.index_of(self.lo + i))),
        );
        out
    }

    /// Maximum depth over all nodes (0 for an empty tree).
    pub fn max_depth(&self) -> u32 {
        self.nodes.par_iter().map(|t| t.depth).max().unwrap_or(0)
    }

    /// Subtree sizes in internal index space (length `n + 1`): `size[x]` is the number of nodes
    /// dominated by `x`, itself included; `size[ROOT] = n + 1`. One reverse sweep, valid
    /// because parents precede children (F2 §4.5.2).
    pub fn sizes(&self) -> Vec<u32> {
        let mut size = io::huge_vec(self.nodes.len(), 1u32);
        for x in (1..self.nodes.len()).rev() {
            let p = self.nodes[x].parent as usize;
            size[p] += size[x];
        }
        size
    }

    /// Preorder numbers in internal index space (length `n + 1`, root 0), in which the
    /// children of a node are visited by increasing index. One forward sweep with a temporary
    /// `next` array (F2 §4.5.3): `d` dominates `v` iff `pre[d] ≤ pre[v] < pre[d] + size[d]`.
    pub fn preorder(&self, sizes: &[u32]) -> Vec<u32> {
        let len = self.nodes.len();
        assert_eq!(sizes.len(), len, "sizes of a different tree");
        let mut pre = io::huge_vec(len, 0u32);
        let mut next = io::huge_vec(len, 0u32);
        next[0] = 1;
        for x in 1..len {
            let p = self.nodes[x].parent as usize;
            pre[x] = next[p];
            next[p] += sizes[x];
            next[x] = pre[x] + 1;
        }
        pre
    }

    /// Converts an internal-index array (length `n + 1`) into public order (indexed by
    /// `v − lo`), calling `f` on batches.
    fn write_public(&self, path: &Path, f: impl Fn(u32) -> u32 + Sync) -> Result<()> {
        let n = self.len();
        let mut w = io::array_writer::<u32>(path)?;
        let mut buf: Vec<u32> = Vec::with_capacity(WRITE_BATCH.min(n));
        let mut i = 0;
        while i < n {
            let k = WRITE_BATCH.min(n - i);
            buf.clear();
            buf.par_extend(
                (i..i + k)
                    .into_par_iter()
                    .map(|j| f(self.index_of(self.lo + j))),
            );
            w.extend(&buf)?;
            i += k;
        }
        w.finish()?;
        Ok(())
    }

    /// Names of the output arrays: `("idom", "dom")` or `("ipdom", "pdom")`.
    pub fn field_names(&self) -> (&'static str, &'static str) {
        if self.mirrored {
            ("ipdom", "pdom")
        } else {
            ("idom", "dom")
        }
    }

    /// Writes `<prefix>.idom.u32le` and `<prefix>.dom.depth.u32le` (for a post-dominator tree
    /// `ipdom` and `pdom`), plus `<prefix>.dom.size.u32le` and `<prefix>.dom.pre.u32le` when
    /// given. Every array is indexed by `v − lo`; `idom` values are absolute ids with
    /// [`NONE`] for the virtual root, `pre` values are preorder numbers (root 0).
    pub fn write(&self, prefix: &Path, sizes: Option<&[u32]>, pre: Option<&[u32]>) -> Result<()> {
        let (idom, tree) = self.field_names();
        self.write_public(&array_path::<u32>(prefix, idom), |x| {
            self.public_of_index(x)
        })?;
        self.write_public(&array_path::<u32>(prefix, &format!("{tree}.depth")), |x| {
            self.nodes[x as usize].depth
        })?;
        if let Some(s) = sizes {
            assert_eq!(s.len(), self.nodes.len());
            self.write_public(&array_path::<u32>(prefix, &format!("{tree}.size")), |x| {
                s[x as usize]
            })?;
        }
        if let Some(p) = pre {
            assert_eq!(p.len(), self.nodes.len());
            self.write_public(&array_path::<u32>(prefix, &format!("{tree}.pre")), |x| {
                p[x as usize]
            })?;
        }
        Ok(())
    }
}
