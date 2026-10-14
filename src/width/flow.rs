//! Dinic's maximum flow on flat arrays, and the exact width of a DAG as a minimum flow.
//!
//! # Dinic ([\[Din70\]](crate::references#din70), [\[FF56\]](crate::references#ff56))
//! Edges are staged with [`Dinic::add_edge`] (or [`Dinic::add_edge_with_rev`], which also gives
//! the reverse arc a capacity) and laid out by [`Dinic::finalize`] in two passes (degree count,
//! then placement) into flat CSR arrays `to` (u32), `cap` (i64) and `rev` (u64) with `u64`
//! offsets: no per-vertex `Vec`. Each phase builds BFS levels from the source and then finds a
//! blocking flow with a multi-push DFS (a frame keeps pushing the flow that reached its node
//! into further children until it is used up, so a long common prefix is walked once per
//! phase, not once per augmenting path), simulated with an explicit stack and a per-node
//! current-arc cursor; nodes that return no flow are removed from the level graph.
//! Memory: 40 B per edge (two arcs of 20 B) + 16 B per edge of bookkeeping (`earc`, and the
//! original capacity for [`Dinic::flow`]) + 24 B per vertex (offsets, level, cursor); the
//! staging arrays (24 B per edge, reserved with [`huge_capacity`]) are freed by `finalize`.
//! The flat arrays and the per-vertex state of both max-flow algorithms come from [`huge_vec`]
//! (transparent huge pages: their access pattern is random). Progress is logged at `info` level at Dinic phases `2^k` and
//! every 1024th phase, and at each global relabel of push–relabel.
//!
//! # Minimum flow width ([\[NH79\]](crate::references#nh79), [\[CC04\]](crate::references#cc04), [\[AMO93\]](crate::references#amo93))
//! Network: nodes `s`, `t`, `v_in`, `v_out`; arcs `v_in → v_out` (lower bound 1), `u_out → v_in`
//! per arc, `s → v_in`, `v_out → t` (lower bound 0), all with infinite capacity. A flow is a
//! cover of `V` by paths that may share nodes, and its value is the number of paths; the
//! minimum is the width (Dilworth, [\[Dil50\]](crate::references#dil50)).
//! ```text
//! 1. f := the flow of a minimum vertex-disjoint path cover (Hopcroft–Karp), value p
//! 2. residual for decreasing f: arc y → x with capacity f(e) − lower(e) per e = (x → y),
//!    and x → y with capacity ∞ (arcs out of s and into t are not needed)
//! 3. g := max flow from t to s (Dinic)                      width = p − g
//! 4. f' = f − g; decompose f' into width s–t paths and keep, on each path, only the nodes
//!    not already kept on an earlier path: a partition of V into width chains
//! 5. Z := nodes reachable from t in the final residual;  A := {v : v_in ∉ Z, v_out ∈ Z}
//! ```
//! The flow paths may share nodes, and their total length `width + Σ_e f'(e)` can be
//! `Θ(width · n)` (a "broom": `k` sources, a path of `k` nodes, `k` sinks). Step 4 therefore
//! emits a chain **partition**: every node once, so the output is exactly `n` ids (4 B/node)
//! plus `width + 1` offsets. Consecutive kept nodes of a path are comparable through the path,
//! and no chain ends up empty, since otherwise the other `width − 1` chains would cover `V`.
//! The walk itself still takes `O(width + Σ_e f'(e))` time, but only `n` bits of extra memory.
//! `Z` is closed under the original arcs (their increase capacity is infinite), so every arc
//! leaving the complement of `Z` towards `Z` carries exactly its lower bound and the min-flow
//! max-cut theorem gives `|A| = width`; two members of `A` are incomparable because `u ⇝ v`
//! would put `v_in` in `Z`. The output is self-certifying: `width` paths covering `V` and an
//! antichain of the same size (checked by
//! [`check_width_certificate`](super::antichain::check_width_certificate)).
//!
//! Complexity: `O((p − width) · (n + m))` in the worst case, much less with Dinic's phases in
//! practice; used for windows up to ~10⁶–10⁷ nodes. On the PG prefer the closure network of
//! [`width_closure_blocks`](super::antichain::width_closure_blocks), which is smaller.

use anyhow::{Result, ensure};
use sux::prelude::*;
use sux::traits::{BitVecOps, BitVecOpsMut};

use super::pathcover::min_path_cover_hk;
use crate::csr::Csr;
use crate::io::{NONE, huge_capacity, huge_vec};

/// "Infinite" capacity: large enough for any real flow, small enough that adding flows to it
/// never overflows.
pub const INF: i64 = i64::MAX / 4;

/// Largest network accepted by [`Dinic::max_flow_push_relabel`]: its distance labels go up
/// to `2n − 1` and are stored as `u32`.
pub const PUSH_RELABEL_MAX_NODES: usize = 1 << 31;

/// Unreached level in the BFS.
const UNSEEN: u32 = u32::MAX;

/// Working state of [`Dinic::max_flow_push_relabel`].
struct PushRelabel {
    d: Vec<u32>,
    excess: Vec<i64>,
    cursor: Vec<u64>,
    queue: std::collections::VecDeque<u32>,
    active: Vec<bool>,
    bfs: Vec<u32>,
}

/// A frame of the simulated recursive DFS of [`Dinic::blocking_flow`]: flow `limit` arrived at
/// `u`, of which `res` has been pushed further.
#[derive(Clone, Copy, Debug)]
struct Frame {
    u: u32,
    limit: i64,
    res: i64,
}

/// Dinic's max-flow solver on flat CSR arc arrays (see the module documentation).
///
/// ```
/// use dag::width::Dinic;
/// let mut d = Dinic::new(4, 5);
/// let e = d.add_edge(0, 1, 3);
/// d.add_edge(0, 2, 2);
/// d.add_edge(1, 2, 5);
/// d.add_edge(1, 3, 2);
/// d.add_edge(2, 3, 3);
/// d.finalize();
/// assert_eq!(d.max_flow(0, 3), 5);
/// assert!(d.flow(e) <= 3);
/// let side = d.min_cut_side(0);
/// assert!(side[0] && !side[3]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Dinic {
    n: usize,
    // staging (freed by finalize)
    eu: Vec<u32>,
    ev: Vec<u32>,
    erc: Vec<i64>,
    // original capacity of each edge (for flow())
    ecap: Vec<i64>,
    // finalized layout
    off: Vec<u64>,
    to: Vec<u32>,
    cap: Vec<i64>,
    rev: Vec<u64>,
    earc: Vec<u64>,
    finalized: bool,
}

impl Dinic {
    /// A network with `n` vertices, with room for `arcs_hint` edges (the staging arrays are
    /// reserved with [`huge_capacity`], so an exact hint avoids any reallocation).
    pub fn new(n: usize, arcs_hint: usize) -> Self {
        assert!(
            n < u32::MAX as usize,
            "Dinic: {n} vertices do not fit u32 ids"
        );
        Self {
            n,
            eu: huge_capacity(arcs_hint),
            ev: huge_capacity(arcs_hint),
            erc: huge_capacity(arcs_hint),
            ecap: huge_capacity(arcs_hint),
            ..Default::default()
        }
    }

    /// Estimated bytes of a finalized network with `n` vertices and `edges` edges (the peak
    /// during `finalize` is higher by the 24 B/edge staging arrays).
    pub fn bytes(n: usize, edges: usize) -> u64 {
        56 * edges as u64 + 24 * n as u64
    }

    /// Number of vertices.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Number of edges added so far.
    pub fn num_edges(&self) -> usize {
        self.ecap.len()
    }

    /// Adds the edge `u → v` with capacity `cap ≥ 0` (the reverse arc starts at 0) and returns
    /// its id. Must be called before [`finalize`](Self::finalize).
    pub fn add_edge(&mut self, u: usize, v: usize, cap: i64) -> usize {
        self.add_edge_with_rev(u, v, cap, 0)
    }

    /// Adds the arc `u → v` with capacity `cap` and its reverse arc `v → u` with capacity
    /// `rev_cap` (both `≥ 0`), and returns the edge id. [`flow`](Self::flow) then reports the
    /// net flow pushed from `u` to `v` (negative if the net flow went from `v` to `u`).
    pub fn add_edge_with_rev(&mut self, u: usize, v: usize, cap: i64, rev_cap: i64) -> usize {
        assert!(!self.finalized, "Dinic: add_edge after finalize");
        assert!(
            u < self.n && v < self.n,
            "Dinic: edge ({u}, {v}) out of range"
        );
        assert!(cap >= 0 && rev_cap >= 0, "Dinic: negative capacity");
        self.eu.push(u as u32);
        self.ev.push(v as u32);
        self.ecap.push(cap);
        self.erc.push(rev_cap);
        self.ecap.len() - 1
    }

    /// Lays the edges out in flat CSR arrays (two passes: degrees, then placement) and frees
    /// the staging arrays.
    pub fn finalize(&mut self) {
        assert!(!self.finalized, "Dinic: finalize called twice");
        let n = self.n;
        let m = self.ecap.len();
        log::info!("flow network: {n} vertices, {m} edges ({} arcs)", 2 * m);
        let mut off = huge_vec(n + 1, 0u64);
        for e in 0..m {
            off[self.eu[e] as usize + 1] += 1;
            off[self.ev[e] as usize + 1] += 1;
        }
        for i in 0..n {
            off[i + 1] += off[i];
        }
        let mut pos: Vec<u64> = huge_vec(n, 0u64);
        pos.copy_from_slice(&off[..n]);
        let mut to = huge_vec(2 * m, 0u32);
        let mut cap = huge_vec(2 * m, 0i64);
        let mut rev = huge_vec(2 * m, 0u64);
        let mut earc = huge_vec(m, 0u64);
        for (e, ea) in earc.iter_mut().enumerate() {
            let (u, v) = (self.eu[e] as usize, self.ev[e] as usize);
            let a = pos[u];
            pos[u] += 1;
            let b = pos[v];
            pos[v] += 1;
            to[a as usize] = v as u32;
            cap[a as usize] = self.ecap[e];
            to[b as usize] = u as u32;
            cap[b as usize] = self.erc[e];
            rev[a as usize] = b;
            rev[b as usize] = a;
            *ea = a;
        }
        self.eu = Vec::new();
        self.ev = Vec::new();
        self.erc = Vec::new();
        self.off = off;
        self.to = to;
        self.cap = cap;
        self.rev = rev;
        self.earc = earc;
        self.finalized = true;
    }

    /// BFS levels from `s` over arcs with positive residual capacity.
    fn levels(&self, s: usize, level: &mut [u32], queue: &mut Vec<u32>) {
        level.fill(UNSEEN);
        queue.clear();
        level[s] = 0;
        queue.push(s as u32);
        let mut head = 0;
        while head < queue.len() {
            let u = queue[head] as usize;
            head += 1;
            for a in self.off[u]..self.off[u + 1] {
                let a = a as usize;
                let v = self.to[a] as usize;
                if self.cap[a] > 0 && level[v] == UNSEEN {
                    level[v] = level[u] + 1;
                    queue.push(v as u32);
                }
            }
        }
    }

    /// Pushes a maximum flow from `s` to `t` on top of the current residual network and returns
    /// its value. Panics if the flow is unbounded (an `s`–`t` path of [`INF`] arcs).
    pub fn max_flow(&mut self, s: usize, t: usize) -> i64 {
        assert!(self.finalized, "Dinic: max_flow before finalize");
        if s == t {
            return 0;
        }
        let n = self.n;
        let mut level = huge_vec(n, UNSEEN);
        let mut queue: Vec<u32> = Vec::new();
        let mut cursor: Vec<u64> = huge_vec(n, 0u64);
        let mut stack: Vec<Frame> = Vec::new();
        let mut total = 0i64;
        let mut phases = 0u64;
        loop {
            self.levels(s, &mut level, &mut queue);
            if level[t] == UNSEEN {
                log::info!("Dinic: max flow {total} in {phases} phases");
                return total;
            }
            phases += 1;
            cursor.copy_from_slice(&self.off[..n]);
            total += self.blocking_flow(s, t, &mut level, &mut cursor, &mut stack);
            if phases.is_power_of_two() || phases % 1024 == 0 {
                log::info!("Dinic: phase {phases}, flow so far {total}");
            }
        }
    }

    /// One blocking flow on the level graph by a multi-push DFS (each frame keeps pushing into
    /// further children until its incoming amount is used up), simulated with an explicit
    /// stack. Dead nodes get the level [`UNSEEN`].
    fn blocking_flow(
        &mut self,
        s: usize,
        t: usize,
        level: &mut [u32],
        cursor: &mut [u64],
        stack: &mut Vec<Frame>,
    ) -> i64 {
        stack.clear();
        stack.push(Frame {
            u: s as u32,
            limit: INF,
            res: 0,
        });
        loop {
            let top = *stack.last().expect("non-empty stack");
            let u = top.u as usize;
            let mut child = None;
            if u != t {
                let end = self.off[u + 1];
                while top.res < top.limit && cursor[u] < end {
                    let a = cursor[u] as usize;
                    let v = self.to[a] as usize;
                    if self.cap[a] > 0 && level[v] == level[u] + 1 {
                        child = Some(Frame {
                            u: v as u32,
                            limit: (top.limit - top.res).min(self.cap[a]),
                            res: 0,
                        });
                        break;
                    }
                    cursor[u] += 1;
                }
            }
            if let Some(f) = child {
                stack.push(f);
                continue;
            }
            // the frame returns what it pushed (the sink absorbs its whole limit)
            let d = if u == t {
                assert!(top.limit < INF, "Dinic: unbounded flow");
                top.limit
            } else {
                top.res
            };
            stack.pop();
            if d == 0 {
                level[u] = UNSEEN;
            }
            let Some(parent) = stack.last_mut() else {
                return d;
            };
            let p = parent.u as usize;
            let a = cursor[p] as usize;
            if d > 0 {
                self.cap[a] -= d;
                self.cap[self.rev[a] as usize] += d;
                parent.res += d;
            }
            // keep the arc only if the parent is satisfied (the arc may have capacity left);
            // otherwise the arc is saturated or the child is exhausted
            if parent.res < parent.limit {
                cursor[p] += 1;
            }
        }
    }

    /// Pushes a maximum flow from `s` to `t` on top of the current residual network with the
    /// FIFO preflow-push (push–relabel) method ([\[GT88\]](crate::references#gt88);
    /// [\[AMO93\]](crate::references#amo93), ch. 7) and returns its value. Same contract as
    /// [`max_flow`](Self::max_flow): at the end the preflow is a flow (every excess has gone to
    /// `t` or back to `s`), so [`min_cut_side`](Self::min_cut_side) and [`flow`](Self::flow)
    /// apply.
    ///
    /// Distance labels are exact residual distances to `t`, or `n` + the distance to `s` for
    /// nodes that cannot reach `t` (which returns their excess to `s` in the same run); they are
    /// recomputed by a backward BFS (global relabeling) at the start and whenever the relabel
    /// work since the last one exceeds `6n + m`. Unlike Dinic, the cost does not grow with the
    /// number of distinct augmenting-path lengths, which is large on networks with long paths.
    /// There is no gap heuristic: on the PG closure networks the work grows superlinearly
    /// (measured ≈ 29 relabels per vertex on `[0, 10⁸)`; see
    /// [`width_closure_blocks`](super::antichain::width_closure_blocks) for the timings).
    pub fn max_flow_push_relabel(&mut self, s: usize, t: usize) -> i64 {
        assert!(
            self.finalized,
            "Dinic: max_flow_push_relabel before finalize"
        );
        if s == t {
            return 0;
        }
        let n = self.n;
        assert!(
            n < PUSH_RELABEL_MAX_NODES,
            "push-relabel: {n} vertices, the u32 labels allow fewer than 2^31"
        );
        let unset = 2 * (n as u32);
        let mut st = PushRelabel {
            d: huge_vec(n, 0u32),
            excess: huge_vec(n, 0i64),
            cursor: huge_vec(n, 0u64),
            queue: std::collections::VecDeque::new(),
            active: huge_vec(n, false),
            bfs: Vec::new(),
        };
        st.cursor.copy_from_slice(&self.off[..n]);
        // saturate the arcs leaving s
        for a in self.off[s]..self.off[s + 1] {
            let a = a as usize;
            let f = self.cap[a];
            if f > 0 {
                assert!(
                    f < INF,
                    "Dinic: unbounded flow (an INF arc leaves the source)"
                );
                let v = self.to[a] as usize;
                self.cap[a] = 0;
                self.cap[self.rev[a] as usize] += f;
                st.excess[v] += f;
                if v != t && v != s && !st.active[v] {
                    st.active[v] = true;
                    st.queue.push_back(v as u32);
                }
            }
        }
        self.global_relabel(s, t, unset, &mut st);
        let threshold = 6 * n as u64 + self.to.len() as u64 / 2;
        let mut work = 0u64;
        let (mut pushes, mut relabels, mut globals) = (0u64, 0u64, 1u64);
        while let Some(u) = st.queue.pop_front() {
            let u = u as usize;
            st.active[u] = false;
            // discharge u
            let end = self.off[u + 1];
            while st.excess[u] > 0 {
                if st.cursor[u] == end {
                    // relabel
                    let mut dmin = unset;
                    for a in self.off[u]..end {
                        let a = a as usize;
                        if self.cap[a] > 0 {
                            dmin = dmin.min(st.d[self.to[a] as usize]);
                        }
                    }
                    work += end - self.off[u] + 12;
                    relabels += 1;
                    st.cursor[u] = self.off[u];
                    // a node with excess always reaches s in the residual network (the excess
                    // arrived along arcs whose reverses have capacity), so dmin < 2n − 1
                    assert!(
                        dmin < unset - 1,
                        "push-relabel: node {u} with excess has no admissible label"
                    );
                    st.d[u] = dmin + 1;
                    if work > threshold {
                        work = 0;
                        globals += 1;
                        self.global_relabel(s, t, unset, &mut st);
                        log::info!(
                            "push-relabel: global relabel {globals} ({pushes} pushes, {relabels} relabels, excess at the sink {})",
                            st.excess[t]
                        );
                        if !st.active[u] && st.excess[u] > 0 {
                            st.active[u] = true;
                            st.queue.push_back(u as u32);
                        }
                        break;
                    }
                    continue;
                }
                let a = st.cursor[u] as usize;
                let v = self.to[a] as usize;
                if self.cap[a] > 0 && st.d[u] == st.d[v] + 1 {
                    let delta = st.excess[u].min(self.cap[a]);
                    self.cap[a] -= delta;
                    self.cap[self.rev[a] as usize] += delta;
                    st.excess[u] -= delta;
                    st.excess[v] += delta;
                    pushes += 1;
                    if v != s && v != t && !st.active[v] {
                        st.active[v] = true;
                        st.queue.push_back(v as u32);
                    }
                    if self.cap[a] == 0 {
                        st.cursor[u] += 1;
                    }
                } else {
                    st.cursor[u] += 1;
                }
            }
        }
        log::info!(
            "push-relabel: max flow {} ({pushes} pushes, {relabels} relabels, {globals} global relabels)",
            st.excess[t]
        );
        st.excess[t]
    }

    /// Exact labels: backward BFS from `t` over residual arcs (`d` = distance to `t`), then from
    /// `s` with base `n` for the nodes that cannot reach `t`; the rest get `unset`. Resets the
    /// arc cursors.
    fn global_relabel(&self, s: usize, t: usize, unset: u32, st: &mut PushRelabel) {
        let n = self.n;
        st.d.fill(unset);
        for (root, base) in [(t, 0u32), (s, n as u32)] {
            st.d[root] = base;
            st.bfs.clear();
            st.bfs.push(root as u32);
            let mut head = 0;
            while head < st.bfs.len() {
                let x = st.bfs[head] as usize;
                head += 1;
                for a in self.off[x]..self.off[x + 1] {
                    let a = a as usize;
                    let y = self.to[a] as usize;
                    // the residual arc y → x is the reverse of x → y
                    if st.d[y] == unset && self.cap[self.rev[a] as usize] > 0 {
                        st.d[y] = st.d[x] + 1;
                        st.bfs.push(y as u32);
                    }
                }
            }
        }
        st.cursor.copy_from_slice(&self.off[..n]);
    }

    /// The vertices reachable from `s` in the residual network (after
    /// [`max_flow`](Self::max_flow) from `s`, the source side of a minimum cut).
    pub fn min_cut_side(&self, s: usize) -> BitVec {
        assert!(self.finalized, "Dinic: min_cut_side before finalize");
        let mut side = BitVec::new(self.n);
        let mut queue = vec![s as u32];
        side.set(s, true);
        let mut head = 0;
        while head < queue.len() {
            let u = queue[head] as usize;
            head += 1;
            for a in self.off[u]..self.off[u + 1] {
                let a = a as usize;
                let v = self.to[a] as usize;
                if self.cap[a] > 0 && !side.get(v) {
                    side.set(v, true);
                    queue.push(v as u32);
                }
            }
        }
        side
    }

    /// Net flow on edge `e` (original capacity minus residual capacity).
    pub fn flow(&self, e: usize) -> i64 {
        self.ecap[e] - self.residual(e)
    }

    /// Residual capacity of edge `e` in its own direction.
    pub fn residual(&self, e: usize) -> i64 {
        assert!(self.finalized, "Dinic: residual before finalize");
        self.cap[self.earc[e] as usize]
    }
}

/// The max-flow algorithm used by the width computations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, clap::ValueEnum)]
pub enum FlowAlgo {
    /// FIFO preflow-push with global relabeling ([`Dinic::max_flow_push_relabel`]).
    #[default]
    PushRelabel,
    /// Dinic's blocking flows ([`Dinic::max_flow`]).
    Dinic,
}

impl FlowAlgo {
    /// Runs the algorithm on `net` from `s` to `t`.
    pub fn run(self, net: &mut Dinic, s: usize, t: usize) -> i64 {
        match self {
            FlowAlgo::PushRelabel => net.max_flow_push_relabel(s, t),
            FlowAlgo::Dinic => net.max_flow(s, t),
        }
    }
}

/// Chains in CSR form: chain `i` is `nodes[offsets[i]..offsets[i + 1]]` (written as
/// `P.chains.u32le` and `P.chains_off.u64le`). Consecutive members of a chain are comparable
/// (an arc or a longer path).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Chains {
    /// Concatenated chains.
    pub nodes: Vec<u32>,
    /// `k + 1` offsets into `nodes`.
    pub offsets: Vec<u64>,
}

impl Chains {
    /// Number of chains.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Whether there are no chains.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Chain `i`.
    pub fn chain(&self, i: usize) -> &[u32] {
        &self.nodes[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    /// Iterates over the chains.
    pub fn iter(&self) -> impl Iterator<Item = &[u32]> {
        (0..self.len()).map(|i| self.chain(i))
    }
}

/// Width of a DAG with a minimum chain cover and a maximum antichain.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WidthResult {
    /// Size of a maximum antichain = minimum number of chains.
    pub width: u64,
    /// A minimum chain cover, when the method produces one.
    pub chains: Option<Chains>,
    /// A maximum antichain, ascending.
    pub antichain: Vec<u32>,
    /// `"minflow"`, `"closure"` or `"fulkerson"`.
    pub method: &'static str,
}

/// Exact width of the DAG `g` (topological ids) by minimum flow (see the module
/// documentation). The chains partition the nodes (`chains.nodes.len() == n`; each chain is a
/// subsequence of a path of `g`), the antichain is maximum, and
/// `chains.len() == antichain.len() == width`.
///
/// Memory beyond `g`: the flow network ([`Dinic::bytes`] with `2n + 2` vertices and
/// `3n + m` edges, plus its staging), the Hopcroft–Karp arrays, and for the decomposition
/// `4m + 16n` bytes plus the `4n` bytes of the chains.
///
/// ```
/// use dag::csr::Csr;
/// use dag::width::{check_width_certificate, width_min_flow};
/// // two chains 0 → 2 and 1 → 2 → 3: width 2 (antichain {0, 1})
/// let g = Csr::from_arcs(4, [(0, 2), (1, 2), (2, 3)]);
/// let r = width_min_flow(&g);
/// assert_eq!(r.width, 2);
/// check_width_certificate(&g, &r).unwrap();
/// ```
pub fn width_min_flow(g: &Csr) -> WidthResult {
    width_min_flow_with(g, FlowAlgo::default())
}

/// [`width_min_flow`] with an explicit max-flow algorithm (for ablations).
pub fn width_min_flow_with(g: &Csr, algo: FlowAlgo) -> WidthResult {
    let n = g.n;
    let m = g.num_arcs();
    if n == 0 {
        return WidthResult {
            width: 0,
            chains: Some(Chains {
                nodes: Vec::new(),
                offsets: vec![0],
            }),
            antichain: Vec::new(),
            method: "minflow",
        };
    }
    let pc = min_path_cover_hk(g);
    let p = pc.starts.len();
    let vin = |v: usize| 2 * v;
    let vout = |v: usize| 2 * v + 1;
    let (s, t) = (2 * n, 2 * n + 1);
    let mut din = Dinic::new(2 * n + 2, n + m + 2 * p);
    // split edges: ids 0..n
    for v in 0..n {
        din.add_edge_with_rev(vout(v), vin(v), 0, INF);
    }
    // arc edges: ids n..n+m, in CSR order
    for u in 0..n {
        let nu = pc.next[u];
        for &v in g.succ(u) {
            din.add_edge_with_rev(vin(v as usize), vout(u), (nu == v) as i64, INF);
        }
    }
    let s_base = din.num_edges();
    for &v in &pc.starts {
        din.add_edge(vin(v as usize), s, 1);
    }
    let ends: Vec<u32> = (0..n as u32)
        .filter(|&v| pc.next[v as usize] == NONE)
        .collect();
    let t_base = din.num_edges();
    for &v in &ends {
        din.add_edge(t, vout(v as usize), 1);
    }
    din.finalize();
    let reduced = algo.run(&mut din, t, s);
    let width = p as u64 - reduced as u64;
    // ---- antichain from the residual reachable set of t ----
    let z = din.min_cut_side(t);
    let antichain: Vec<u32> = (0..n)
        .filter(|&v| !z.get(vin(v)) && z.get(vout(v)))
        .map(|v| v as u32)
        .collect();
    debug_assert_eq!(antichain.len() as u64, width);
    // ---- final flow f' = residual of each decrease arc ----
    let mut fa: Vec<u32> = huge_vec(m, 0u32);
    {
        use rayon::prelude::*;
        let din = &din;
        fa.par_iter_mut()
            .enumerate()
            .with_min_len(1 << 16)
            .for_each(|(i, f)| *f = din.residual(n + i) as u32);
    }
    let mut from_s = huge_vec(n, 0u32);
    for (i, &v) in pc.starts.iter().enumerate() {
        from_s[v as usize] = din.residual(s_base + i) as u32;
    }
    let mut to_t = huge_vec(n, 0u32);
    for (i, &v) in ends.iter().enumerate() {
        to_t[v as usize] = din.residual(t_base + i) as u32;
    }
    drop(din);
    // ---- decomposition into width paths, kept as a chain partition ----
    let mut cursor: Vec<u64> = huge_vec(n, 0u64);
    cursor.copy_from_slice(&g.offsets[..n]);
    let mut emitted: BitVec = BitVec::new(n);
    let mut nodes = huge_capacity(n);
    let mut offsets = Vec::with_capacity(width as usize + 1);
    offsets.push(0u64);
    let mut walked = 0u64;
    for (v0, &k) in from_s.iter().enumerate() {
        for _ in 0..k {
            let mut x = v0;
            loop {
                walked += 1;
                if !emitted.get(x) {
                    emitted.set(x, true);
                    nodes.push(x as u32);
                }
                let end = g.offsets[x + 1];
                while cursor[x] < end && fa[cursor[x] as usize] == 0 {
                    cursor[x] += 1;
                }
                if cursor[x] < end {
                    let pos = cursor[x] as usize;
                    fa[pos] -= 1;
                    x = g.succ[pos] as usize;
                } else {
                    assert!(
                        to_t[x] > 0,
                        "min-flow decomposition: flow not conserved at {x}"
                    );
                    to_t[x] -= 1;
                    break;
                }
            }
            assert!(
                nodes.len() as u64 > *offsets.last().unwrap(),
                "min-flow decomposition: a flow path adds no new node (the flow is not minimum)"
            );
            offsets.push(nodes.len() as u64);
        }
    }
    log::info!(
        "min-flow: width {width}, {walked} path nodes walked, {} chain nodes kept",
        nodes.len()
    );
    assert_eq!(
        nodes.len(),
        n,
        "min-flow decomposition: the paths miss some node"
    );
    debug_assert_eq!(offsets.len() as u64 - 1, width);
    WidthResult {
        width,
        chains: Some(Chains { nodes, offsets }),
        antichain,
        method: "minflow",
    }
}

/// Checks a [`Chains`] value against `g`: offsets are monotone and in range, every chain is
/// non-empty and consecutive members are comparable (an arc, or else a path found by a pruned
/// search over ids `≤` the later member), and every node of `g` lies on some chain.
///
/// The comparability checks run in parallel over the chains (one [`DirtyBitset`] per worker);
/// the error reported is that of the lowest failing chain.
///
/// [`DirtyBitset`]: crate::bitset::DirtyBitset
pub(crate) fn check_chains(g: &Csr, c: &Chains) -> Result<()> {
    use rayon::prelude::*;
    let n = g.n;
    ensure!(
        c.offsets.first() == Some(&0) && *c.offsets.last().unwrap() == c.nodes.len() as u64,
        "chain offsets do not span the node list"
    );
    let mut covered = vec![false; n];
    for i in 0..c.len() {
        ensure!(
            c.offsets[i] < c.offsets[i + 1],
            "chain {i} is empty or offsets decrease"
        );
        for &x in c.chain(i) {
            ensure!((x as usize) < n, "chain {i}: node {x} out of range");
            covered[x as usize] = true;
        }
    }
    if let Some(v) = covered.iter().position(|&c| !c) {
        anyhow::bail!("node {v} lies on no chain");
    }
    drop(covered);
    let bad = (0..c.len())
        .into_par_iter()
        .map_init(
            || (crate::bitset::DirtyBitset::new(), Vec::<u32>::new()),
            |(bits, stack), i| {
                c.chain(i).windows(2).find_map(|pair| {
                    let (x, y) = (pair[0], pair[1]);
                    let ok = g.succ(x as usize).binary_search(&y).is_ok()
                        || reaches(g, x as usize, y as usize, bits, stack);
                    (!ok).then_some((i, x, y))
                })
            },
        )
        .find_first(Option::is_some)
        .flatten();
    if let Some((i, x, y)) = bad {
        anyhow::bail!("chain {i}: {x} does not reach {y}");
    }
    Ok(())
}

/// Whether `x ⇝ y` (path of length ≥ 1) in a DAG with topological ids, by a DFS pruned to ids
/// `≤ y`.
pub(crate) fn reaches(
    g: &Csr,
    x: usize,
    y: usize,
    bits: &mut crate::bitset::DirtyBitset,
    stack: &mut Vec<u32>,
) -> bool {
    if y <= x {
        return false;
    }
    bits.reset(x + 1, y - x);
    stack.clear();
    stack.push(x as u32);
    while let Some(u) = stack.pop() {
        for &v in g.succ(u as usize) {
            let v = v as usize;
            if v == y {
                return true;
            }
            if v < y && bits.insert(v) {
                stack.push(v as u32);
            }
        }
    }
    false
}
