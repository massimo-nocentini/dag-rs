//! Matchings, minimum path covers, frontier antichains, flows and width (Dilworth).
//!
//! # Problems
//! For a DAG `G = (V, E)` with topological ids:
//!
//! - **MPC_d**, the minimum number of vertex-disjoint paths covering `V`. With the split
//!   bipartite graph `S(G)` (a left copy `u⁺` and a right copy `v⁻` per node, an edge `u⁺v⁻`
//!   per arc), a matching of `S(G)` is a set of arcs with in- and outdegree at most 1, i.e. a
//!   path cover with `n − |M|` paths, so `MPC_d = n − ν(S(G))`
//!   ([\[BG77\]](crate::references#bg77), [\[Ful56\]](crate::references#ful56)).
//! - **Width**, the size of a maximum antichain (a set of pairwise unreachable nodes). By
//!   Dilworth's theorem ([\[Dil50\]](crate::references#dil50)) it equals the minimum number of
//!   chains covering `V`, and by Fulkerson ([\[Ful56\]](crate::references#ful56)) it is
//!   `n − ν(S(TC(G)))` on the transitive closure. Without the closure it is a minimum flow with
//!   lower bound 1 on every node ([\[NH79\]](crate::references#nh79),
//!   [\[CC04\]](crate::references#cc04)).
//! - **Frontier antichains**: for a cut `c`, the nodes before `c` whose first successor is at
//!   or after `c` are pairwise unreachable, which gives the lower bound `max_c |A_c| ≤ width`.
//!
//! Always `max_c |A_c| ≤ width ≤ MPC_d`.
//!
//! # Modules
//! | Module | Content |
//! |---|---|
//! | [`matching`] | Hopcroft–Karp ([\[HK73\]](crate::references#hk73)) and the Kőnig cover certificate ([\[Kon31\]](crate::references#kon31)) |
//! | [`pathcover`] | MPC_d by Hopcroft–Karp and by the PG closed form; path-cover checks |
//! | [`flow`] | flat flow networks with Dinic ([\[Din70\]](crate::references#din70)) and FIFO push–relabel ([\[AMO93\]](crate::references#amo93)); min-flow width |
//! | [`antichain`] | frontiers, the closure network on the BlockIndex ([\[Pic76\]](crate::references#pic76)), antichain checks, Fulkerson's oracle |
//!
//! # Id conventions
//! Functions that take a [`Csr`](crate::csr::Csr) work in the ids of that `Csr` (window
//! relative when it comes from [`Csr::from_window`](crate::csr::Csr::from_window)). Functions
//! that take a [`BlockIndex`](crate::blocks::BlockIndex) and a [`Window`](crate::window::Window)
//! index per-node arrays by `v − lo` and store **absolute** node ids as values, like every other
//! windowed output of the crate.
//!
//! # PG fast path
//! The PG is a line digraph: its split graph is a disjoint union of complete bicliques
//! `I(B) × O(B)`, one per block ([`crate::blocks`]), so `ν = Σ_B min(i_B, o_B)` and MPC_d, the
//! frontier and the exact width of a window (a maximum-weight closure on the transaction
//! graph, [\[Pic76\]](crate::references#pic76)) are scans of the BlockIndex arrays with no BV
//! decoding. On the full PG the exact closure is behind `--experimental-full`; the default
//! answer is the bracket `[frontier, MPC_d]` (parameterized algorithms such as
//! [\[MTKPGC19\]](crate::references#mtkpgc19) and [\[CCMRT22\]](crate::references#ccmrt22) need
//! time `Ω(k·m)` with `k ≥ 1.36·10⁸` on the PG, so they are only cited).

pub mod antichain;
pub mod flow;
pub mod matching;
pub mod pathcover;

pub use antichain::{
    CutRange, Frontier, check_antichain, check_antichain_blocks, check_width_certificate,
    cut_ranges, frontier_at_cuts_blocks, frontier_at_cuts_generic, frontier_blocks,
    frontier_curve_blocks, frontier_curve_generic, frontier_generic, width_closure_blocks,
    width_closure_blocks_with, width_fulkerson,
};
pub use flow::{
    Chains, Dinic, FlowAlgo, PUSH_RELABEL_MAX_NODES, WidthResult, width_min_flow,
    width_min_flow_with,
};
pub use matching::{Matching, check_konig, hopcroft_karp, konig_cover};
pub use pathcover::{
    LineCoverStats, PathCover, check_path_cover, min_path_cover_blocks, min_path_cover_hk,
};
