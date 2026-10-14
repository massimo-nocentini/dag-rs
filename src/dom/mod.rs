//! Dominator and post-dominator trees of a DAG with a virtual root (sink).
//!
//! # Definitions
//! Add a virtual root `r` with an arc to every source (in-degree 0 node) of the window
//! `[lo, hi)`. A node `d` *dominates* `v` if every `r → v` path contains `d`
//! [\[Pro59\]](crate::references#pro59), [\[LM69\]](crate::references#lm69); the immediate
//! dominator `idom(v)` is the closest strict dominator, and the arcs `idom(v) → v` form the
//! dominator tree `D`. Post-dominators are the dominators of the reverse graph with a virtual
//! sink `t` reached from every sink. Every node of a DAG is reachable from a source, so both
//! trees span the whole window. Isolated nodes are children of both the root and the sink.
//!
//! # Algorithm
//! **Theorem (F2 §2).** In an acyclic flow graph, `Dom(v) = {v} ∪ ⋂_{p ∈ pred(v)} Dom(p)`, and
//! if all predecessors are already in the tree, `idom(v) = NCA_D(pred(v))` (with `pred(v) = {r}`
//! for a source). *Proof.* The identity is the data-flow characterisation of dominators
//! [\[All70\]](crate::references#all70); on a DAG it is evaluated once in topological order.
//! `Dom(p)` is the root path of `p` in `D`, the intersection of root paths is the root path of
//! their NCA, and `v ∉ Dom(p)` (otherwise `v ⇝ p → v` would be a cycle). ∎
//!
//! This is the iterative algorithm of Cooper, Harvey and Kennedy
//! [\[CHK01\]](crate::references#chk01) collapsed to a single pass, and the acyclic case
//! treated by Ochranová [\[Och83\]](crate::references#och83) and by Ramalingam and Reps
//! [\[RR94\]](crate::references#rr94). Lengauer–Tarjan [\[LT79\]](crate::references#lt79) and
//! the linear-time algorithms [\[AHLT99\]](crate::references#ahlt99),
//! [\[BGKRTW08\]](crate::references#bgkrtw08) need a DFS and several arrays per node and are not
//! needed on DAGs (see [\[GTW06\]](crate::references#gtw06) for an experimental comparison).
//!
//! - [`tree`]: the [`DomTree`] (16 B per node) with the hybrid NCA (root-child test, budgeted
//!   walk, Myers jump pointers [\[Mye83\]](crate::references#mye83)), subtree sizes and preorder.
//! - [`build`]: four builders. On the PG both trees come from the
//!   [`BlockIndex`](crate::blocks::BlockIndex) without decoding the graph, with one NCA fold
//!   per transaction ([`dominators_blocks`], [`post_dominators_blocks`]); on any DAG a push
//!   over the forward graph ([`dominators_push`]) and a pull over backward chunks
//!   ([`post_dominators_pull`]) (design decision D5).
//! - [`verify`]: an independent O(m + n log n) verifier (conditions V0–V2, derived arcs of
//!   Georgiadis and Tarjan [\[GT15\]](crate::references#gt15)) that scans the forward graph.
//! - [`stats`]: root children, cut nodes, histograms, top-k by size, and the depth bound
//!   `depth(v) ≤ dmin(v) + 1`.
//!
//! # Index space
//! Internal indices are `x = v − lo + 1` for dominators and `x = hi − v` for post-dominators
//! (the mirror); 0 is the virtual root (sink). Every parent has a smaller index than its child,
//! so one kernel serves both trees. Public arrays are indexed by `v − lo` and hold absolute ids,
//! with [`NONE`](crate::io::NONE) for the virtual root (sink).
//!
//! # Complexity on the PG
//! Strict dominators of `v` lie on a shortest root path, so `depth(v) ≤ dmin(v) + 1 ≤ 25,349`
//! (mean ≤ 77.3): walks are short, and the `top` test answers most merges of different coinbase
//! lineages in two reads. One NCA fold per block (7.8·10⁸) instead of one per arc (8.6·10⁹).
//! Memory: 16 B per node for each tree (34.9 GB), 4 B per block for the accumulators, 8 B per
//! node for sizes and preorder; the verifier about 32 B per node.
//!
//! # Interpretation (F2 §8)
//! `d` dominates `w`: every chain of transfers from any coinbase output to `w` passes through
//! output `d`, so `d` is a provenance bottleneck of `w` (100 % taint under every
//! value-conserving attribution); `size(d) − 1` is its exclusive influence. `ipdom(u) = w`: every
//! onward flow of `u` towards the current UTXO set passes through `w`.
//!
//! # Example
//! ```
//! use dag::dom::{DomConf, dominators_push, verify_dominators};
//! use dag::{testkit, NONE, Window};
//! use dsi_progress_logger::no_logging;
//!
//! // two diamonds in series: 0 → {1, 2} → 3 → {4, 5} → 6
//! let g = testkit::chunked(&testkit::diamonds(2), 3);
//! let w = Window::full(7);
//! let tree = dominators_push(&g, &DomConf::new(w), no_logging![]).unwrap();
//! assert_eq!(tree.idoms(), [NONE, 0, 0, 0, 3, 3, 3]);
//! let report = verify_dominators(&g, &tree.idoms(), w, no_logging![]).unwrap();
//! assert!(report.ok);
//! ```

pub mod build;
pub mod stats;
pub mod tree;
pub mod verify;

pub use build::{
    DomConf, dominators_blocks, dominators_push, post_dominators_blocks, post_dominators_pull,
};
pub use stats::{
    DepthBound, DomDistributions, DomStats, check_depth_bound, dom_distributions, dom_stats,
    tree_distributions, tree_stats,
};
pub use tree::{DomTree, NcaStats, ROOT, UNSET};
pub use verify::{VerifyReport, verify_dominators, verify_post_dominators};
