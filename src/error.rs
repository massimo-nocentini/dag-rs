//! Structural errors of the crate.
//!
//! Public functions return [`anyhow::Result`]. Structural failures (a graph that is not
//! topologically numbered, a cycle, a graph that is not a disjoint union of bicliques, a bad
//! window, a memory estimate above the budget) are [`DagError`] values wrapped in
//! [`anyhow::Error`]. Callers and tests recover them with
//! `err.downcast_ref::<DagError>()`.

/// A structural failure detected by a kernel or a verifier.
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    /// An arc `(u, v)` with `v <= u` was found while the identity was assumed to be a
    /// topological order (every kernel assumes it, see the crate documentation).
    #[error("arc ({u}, {v}) violates the topological order (need u < v)")]
    NotTopological {
        /// Tail of the offending arc (the smallest such node of the scanned batch).
        u: usize,
        /// Head of the offending arc (the smallest offending successor of `u`).
        v: usize,
    },
    /// The graph has a directed cycle; `(u, v)` is an arc lying on one.
    #[error("the graph has a cycle (arc ({u}, {v}) lies on one)")]
    Cycle {
        /// Tail of the arc.
        u: usize,
        /// Head of the arc.
        v: usize,
    },
    /// The graph violates one of the properties P1–P3 of a biclique (line) digraph, see
    /// [`crate::blocks`].
    #[error("not a biclique/line digraph: {kind:?} at node {node}")]
    NotBiclique {
        /// Which property is violated.
        kind: BlockViolation,
        /// The node at which the violation is reported (deterministic, see [`crate::blocks`]).
        node: usize,
    },
    /// The graph has more nodes than `u32` ids can represent (`u32::MAX` is reserved for
    /// [`NONE`](crate::io::NONE)).
    #[error("{n} nodes do not fit u32 ids (max {max})")]
    TooManyNodes {
        /// Number of nodes of the graph.
        n: usize,
        /// Largest supported number of nodes.
        max: usize,
    },
    /// A window `lo..hi` that is not contained in `0..n` or has `lo > hi`.
    #[error("invalid window {lo}..{hi} for {n} nodes")]
    BadWindow {
        /// Start of the window.
        lo: usize,
        /// End of the window (exclusive).
        hi: usize,
        /// Number of nodes of the graph.
        n: usize,
    },
    /// The estimated memory of a run exceeds `--max-mem`.
    #[error("estimated memory {need_gb:.1} GB exceeds --max-mem {max_gb:.1} GB ({what})")]
    OverBudget {
        /// What was being allocated.
        what: String,
        /// Running estimate including the refused allocation, in GB (10⁹ bytes).
        need_gb: f64,
        /// The cap, in GB (10⁹ bytes).
        max_gb: f64,
    },
}

/// The kind of a biclique (line digraph) violation found by
/// [`BlockIndex::build`](crate::blocks::BlockIndex::build).
///
/// The variants are listed in priority order: when two violations are reported at the same
/// node, the smaller variant wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum BlockViolation {
    /// A node `u` whose first successor is `<= u` (property P3 fails).
    BackwardArc,
    /// A node whose successor list is not an interval of consecutive ids (P1 fails).
    NonContiguous,
    /// Two successor lists start at the same node but have different lengths (P2 fails);
    /// reported at the shared start.
    SameStartDifferentLength,
    /// Two distinct successor intervals overlap (P2 fails); reported at the start of the later
    /// interval.
    Overlap,
}
