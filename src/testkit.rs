//! Test kit: seeded graph generators, brute-force oracles and BV round-trip helpers.
//!
//! Hidden from the documentation; used by the unit and integration tests of every unit.
//!
//! - **Generators** (seeded `SmallRng::seed_from_u64`; all deduplicate arcs): [`random_dag`],
//!   [`shuffled_dag`], [`chain`], [`complete_dag`], [`out_star`], [`in_star`], [`isolated`],
//!   [`diamonds`], [`layered`], [`ladder`], [`disjoint_union`], [`with_back_arc`],
//!   [`with_self_loop`] and the PG-like UTXO simulation [`mini_pg`].
//! - **Oracles**: [`closure`], [`all_paths`], [`bfs_reach`], [`bellman_ford`],
//!   [`brute_dominators`], [`brute_width`], [`brute_matching`].
//! - **Helpers**: [`transpose`], [`write_bv`], [`chunked`], [`vec_graph`], [`arcs`].
//!
//! The command-line runner `run_cli` lives in `tests/support/cli.rs` because it needs
//! `CARGO_BIN_EXE_dag`, which only integration tests can see.

use std::collections::VecDeque;
use std::ops::Range;
use std::path::{Path, PathBuf};

use dsi_bitstream::prelude::BE;
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;

use crate::graph::{ChunkConf, ChunkedGraph};

// ---------------------------------------------------------------------------------------------
// Basic helpers
// ---------------------------------------------------------------------------------------------

/// A graph with exactly `n` nodes and the given arcs (sorted and deduplicated).
pub fn vec_graph(n: usize, arcs: impl IntoIterator<Item = (usize, usize)>) -> VecGraph {
    let mut a: Vec<(usize, usize)> = arcs.into_iter().collect();
    for &(u, v) in &a {
        assert!(u < n && v < n, "arc ({u}, {v}) out of range for {n} nodes");
    }
    a.sort_unstable();
    a.dedup();
    let mut g = VecGraph::empty(n);
    g.add_arcs(a);
    g
}

/// All arcs of `g`, sorted.
pub fn arcs<G: RandomAccessGraph>(g: &G) -> Vec<(usize, usize)> {
    (0..g.num_nodes())
        .flat_map(|u| g.successors(u).into_iter().map(move |v| (u, v)))
        .collect()
}

/// Successor lists of `g`.
pub fn succ_lists<G: RandomAccessGraph>(g: &G) -> Vec<Vec<usize>> {
    (0..g.num_nodes())
        .map(|u| g.successors(u).into_iter().collect())
        .collect()
}

/// The transpose of `g`.
pub fn transpose<G: RandomAccessGraph>(g: &G) -> VecGraph {
    vec_graph(g.num_nodes(), arcs(g).into_iter().map(|(u, v)| (v, u)))
}

/// Writes `g` as the BV graph `dir/name` and its transpose as `dir/name-t`, with `.ef` (no
/// `.dcf`, so that cut points come from outdegrees). Returns the basename `dir/name`.
pub fn write_bv(dir: &Path, name: &str, g: &VecGraph) -> PathBuf {
    let base = dir.join(name);
    write_one(&base, g);
    write_one(&dir.join(format!("{name}-t")), &transpose(g));
    base
}

fn write_one(base: &Path, g: &VecGraph) {
    BvComp::with_basename(base)
        .comp_graph::<BE>(g)
        .expect("BV compression");
    store_ef_with_data(
        g.num_nodes(),
        base.with_extension(GRAPH_EXTENSION),
        base.with_extension(OFFSETS_EXTENSION),
        base.with_extension(EF_EXTENSION),
        &mut Option::<ProgressLogger>::None,
    )
    .expect("Elias-Fano offsets");
}

/// A [`ChunkedGraph`] over a copy of `g` with tiny chunks (`chunk_arcs` arcs, at most
/// `chunk_arcs + 1` nodes), which stresses chunk boundaries.
pub fn chunked(g: &VecGraph, chunk_arcs: u64) -> ChunkedGraph<VecGraph> {
    let conf = ChunkConf {
        chunk_arcs: chunk_arcs.max(1),
        max_chunk_nodes: chunk_arcs.max(1) as usize + 1,
    };
    ChunkedGraph::new(g.clone(), conf).expect("chunking")
}

fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

// ---------------------------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------------------------

/// A random DAG in identity order: each arc `u → v` with `u < v` independently with
/// probability `p`.
pub fn random_dag(n: usize, p: f64, seed: u64) -> VecGraph {
    let mut r = rng(seed);
    let mut a = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            if r.random_bool(p) {
                a.push((u, v));
            }
        }
    }
    vec_graph(n, a)
}

/// `random_dag(n, p, seed)` with node ids shuffled. Returns `(g, rank)` where `rank[u]` is the
/// original (topological) id of node `u` of `g`, so `relabel_vec(&g, &rank)` is the
/// identity-ordered DAG.
pub fn shuffled_dag(n: usize, p: f64, seed: u64) -> (VecGraph, Vec<usize>) {
    let base = random_dag(n, p, seed);
    let mut perm: Vec<usize> = (0..n).collect(); // perm[old] = shuffled id
    perm.shuffle(&mut rng(seed ^ 0x9E37_79B9_7F4A_7C15));
    let g = vec_graph(n, arcs(&base).into_iter().map(|(u, v)| (perm[u], perm[v])));
    let mut rank = vec![0; n];
    for (old, &new) in perm.iter().enumerate() {
        rank[new] = old;
    }
    (g, rank)
}

/// The path `0 → 1 → … → n−1`.
pub fn chain(n: usize) -> VecGraph {
    vec_graph(n, (1..n).map(|v| (v - 1, v)))
}

/// The transitive tournament: every arc `u → v` with `u < v`.
pub fn complete_dag(n: usize) -> VecGraph {
    vec_graph(n, (0..n).flat_map(|u| (u + 1..n).map(move |v| (u, v))))
}

/// `0 → v` for every `v ≥ 1`.
pub fn out_star(n: usize) -> VecGraph {
    vec_graph(n, (1..n).map(|v| (0, v)))
}

/// `u → n−1` for every `u < n−1`.
pub fn in_star(n: usize) -> VecGraph {
    vec_graph(n, (0..n.saturating_sub(1)).map(|u| (u, n - 1)))
}

/// `n` nodes, no arcs.
pub fn isolated(n: usize) -> VecGraph {
    VecGraph::empty(n)
}

/// `k` diamonds in series: `s → a, s → b, a → t, b → t`, where `t` is the next `s`
/// (`3k + 1` nodes).
pub fn diamonds(k: usize) -> VecGraph {
    let n = 3 * k + 1;
    vec_graph(
        n,
        (0..k).flat_map(|i| {
            let s = 3 * i;
            [(s, s + 1), (s, s + 2), (s + 1, s + 3), (s + 2, s + 3)]
        }),
    )
}

/// Consecutive layers of the given widths; each arc from layer `i` to layer `i + 1` exists
/// with probability `p`.
pub fn layered(widths: &[usize], p: f64, seed: u64) -> VecGraph {
    let mut r = rng(seed);
    let mut start = vec![0usize];
    for w in widths {
        start.push(start.last().unwrap() + w);
    }
    let n = *start.last().unwrap();
    let mut a = Vec::new();
    for i in 0..widths.len().saturating_sub(1) {
        for u in start[i]..start[i + 1] {
            for v in start[i + 1]..start[i + 2] {
                if r.random_bool(p) {
                    a.push((u, v));
                }
            }
        }
    }
    vec_graph(n, a)
}

/// The CHK worst case (F2 §3.1): a chain `c_0 → … → c_{L−1}` (ids `0..L`) plus nodes `x_i`
/// (id `L + i − 1`, `1 ≤ i < L`) with predecessors `{c_0, c_i}`; `2L − 1` nodes.
pub fn ladder(l: usize) -> VecGraph {
    let l = l.max(1);
    let n = 2 * l - 1;
    let mut a: Vec<(usize, usize)> = (1..l).map(|v| (v - 1, v)).collect();
    for i in 1..l {
        let x = l + i - 1;
        a.push((0, x));
        a.push((i, x));
    }
    vec_graph(n, a)
}

/// The disjoint union of `gs`, with ids shifted by the sizes of the previous graphs.
pub fn disjoint_union(gs: &[VecGraph]) -> VecGraph {
    let mut off = 0;
    let mut a = Vec::new();
    for g in gs {
        a.extend(arcs(g).into_iter().map(|(u, v)| (u + off, v + off)));
        off += g.num_nodes();
    }
    vec_graph(off, a)
}

/// `g` plus the reverse `(v, u)` of its last arc `(u, v)` (a 2-cycle, so the order is no longer
/// topological). If `g` has no arcs (and ≥ 2 nodes), adds the 2-cycle `0 ⇄ 1`.
pub fn with_back_arc(g: &VecGraph) -> VecGraph {
    let n = g.num_nodes();
    let mut a = arcs(g);
    match a.last().copied() {
        Some((u, v)) => a.push((v, u)),
        None => {
            assert!(n >= 2, "with_back_arc needs an arc or two nodes");
            a.extend([(0, 1), (1, 0)]);
        }
    }
    vec_graph(n, a)
}

/// `g` plus the self loop `(n/2, n/2)`.
pub fn with_self_loop(g: &VecGraph) -> VecGraph {
    let n = g.num_nodes();
    assert!(n > 0, "with_self_loop needs a node");
    let mut a = arcs(g);
    a.push((n / 2, n / 2));
    vec_graph(n, a)
}

/// Options of [`mini_pg`].
#[derive(Clone, Debug)]
pub struct MiniPgOpts {
    /// Probability that a transaction is a coinbase (no inputs). Default 0.1.
    pub coinbase_prob: f64,
    /// Maximum number of outputs per transaction (uniform in `1..=max_outputs`). Default 4.
    pub max_outputs: usize,
    /// Maximum number of inputs (geometric with P(1) = 0.6, capped). Default 8.
    pub max_inputs: usize,
    /// Probability that a non-coinbase transaction has no outputs (valid: its inputs become
    /// sinks). Default 0.
    pub zero_output_prob: f64,
    /// Inject a double spend: an output spent by a transaction `T1` with outputs becomes an
    /// input of a later transaction `T2` with inputs and outputs too; its successor list
    /// `O(T1) ∪ O(T2)` then always violates P1 or P2 (`NotBiclique`).
    pub double_spend: bool,
    /// Inject a non-contiguous successor list: one input of a transaction with ≥ 3 outputs loses
    /// its arc to the middle output.
    pub non_contiguous: bool,
}

impl Default for MiniPgOpts {
    fn default() -> Self {
        Self {
            coinbase_prob: 0.1,
            max_outputs: 4,
            max_inputs: 8,
            zero_output_prob: 0.0,
            double_spend: false,
            non_contiguous: false,
        }
    }
}

/// A simulated UTXO graph and its ground truth.
#[derive(Clone, Debug)]
pub struct MiniPg {
    /// The payment graph: one node per output, arcs `I(t) × O(t)` for every transaction `t`.
    pub graph: VecGraph,
    /// Sorted inputs of each transaction (empty for coinbases).
    pub tx_inputs: Vec<Vec<u32>>,
    /// Outputs of each transaction (an id interval, possibly empty).
    pub tx_outputs: Vec<Range<u32>>,
}

impl MiniPg {
    /// Ground-truth blocks `(first, len, nin)` in increasing `first`: the transactions with at
    /// least one input and one output (valid only without injected anomalies).
    pub fn blocks(&self) -> Vec<(u32, u32, u32)> {
        let mut b: Vec<(u32, u32, u32)> = self
            .tx_inputs
            .iter()
            .zip(&self.tx_outputs)
            .filter(|(i, o)| !i.is_empty() && !o.is_empty())
            .map(|(i, o)| (o.start, o.end - o.start, i.len() as u32))
            .collect();
        b.sort_unstable();
        b
    }
}

#[derive(Default)]
struct Sim {
    utxo: Vec<u32>,
    next: u32,
    tx_inputs: Vec<Vec<u32>>,
    tx_outputs: Vec<Range<u32>>,
}

impl Sim {
    fn tx(&mut self, inputs: Vec<u32>, nout: u32) {
        let o = self.next..self.next + nout;
        self.utxo.extend(o.clone());
        self.next += nout;
        self.tx_inputs.push(inputs);
        self.tx_outputs.push(o);
    }
}

/// PG-like UTXO simulation with `txs` transactions (see [`MiniPgOpts`]).
///
/// Outputs get ids in creation order, so ids are chronological and topological; each
/// transaction spends distinct unspent outputs and links every input to every output
/// (complete bipartite arcs, deduplicated as by `pg_el_builder.cpp`). Without anomalies the
/// result satisfies P1–P3 of [`crate::blocks`].
pub fn mini_pg(txs: usize, seed: u64, opts: &MiniPgOpts) -> MiniPg {
    let mut r = rng(seed);
    let mut sim = Sim::default();
    for _ in 0..txs {
        let coinbase = sim.utxo.is_empty() || r.random_bool(opts.coinbase_prob);
        let mut inputs = Vec::new();
        if !coinbase {
            let mut k = 1;
            while k < opts.max_inputs.max(1) && r.random_bool(0.4) {
                k += 1;
            }
            for _ in 0..k.min(sim.utxo.len()) {
                let i = r.random_range(0..sim.utxo.len());
                inputs.push(sim.utxo.swap_remove(i));
            }
            inputs.sort_unstable();
        }
        let nout = if !coinbase && r.random_bool(opts.zero_output_prob) {
            0
        } else {
            r.random_range(1..=opts.max_outputs.max(1)) as u32
        };
        sim.tx(inputs, nout);
    }
    if opts.non_contiguous
        && !sim
            .tx_inputs
            .iter()
            .zip(&sim.tx_outputs)
            .any(|(i, o)| !i.is_empty() && o.len() >= 3)
    {
        // force a qualifying transaction: a coinbase with one output spent by a 3-output tx
        sim.tx(Vec::new(), 1);
        let c = sim.next - 1;
        sim.utxo.retain(|&x| x != c);
        sim.tx(vec![c], 3);
    }
    let Sim {
        next,
        mut tx_inputs,
        tx_outputs,
        ..
    } = sim;
    let n = next as usize;
    let mut a: Vec<(usize, usize)> = Vec::new();
    for (i, o) in tx_inputs.iter().zip(&tx_outputs) {
        for &u in i {
            a.extend(o.clone().map(|v| (u as usize, v as usize)));
        }
    }
    if opts.non_contiguous {
        let t = (0..tx_inputs.len())
            .find(|&t| !tx_inputs[t].is_empty() && tx_outputs[t].len() >= 3)
            .unwrap();
        let (u, mid) = (tx_inputs[t][0] as usize, tx_outputs[t].start as usize + 1);
        a.retain(|&x| x != (u, mid));
    }
    if opts.double_spend {
        // an output u spent by T1, re-spent by a later T2 ≠ T1 with outputs after u
        let spent_by = |u: u32| tx_inputs.iter().position(|i| i.contains(&u));
        let mut done = false;
        'outer: for t2 in (0..tx_inputs.len()).rev() {
            if tx_outputs[t2].is_empty() || tx_inputs[t2].is_empty() {
                continue;
            }
            for t1 in 0..t2 {
                if tx_outputs[t1].is_empty() {
                    continue;
                }
                if let Some(&u) = tx_inputs[t1].first() {
                    if spent_by(u) == Some(t1) && u < tx_outputs[t2].start {
                        a.extend(tx_outputs[t2].clone().map(|v| (u as usize, v as usize)));
                        tx_inputs[t2].push(u);
                        tx_inputs[t2].sort_unstable();
                        tx_inputs[t2].dedup();
                        done = true;
                        break 'outer;
                    }
                }
            }
        }
        assert!(
            done,
            "mini_pg: no place for a double spend; use more transactions"
        );
    }
    MiniPg {
        graph: vec_graph(n, a),
        tx_inputs,
        tx_outputs,
    }
}

// ---------------------------------------------------------------------------------------------
// Oracles
// ---------------------------------------------------------------------------------------------

/// Transitive closure as bitset rows: bit `v` of row `u` is set iff there is a path of length
/// `≥ 1` from `u` to `v` (so `u`'s own bit is set only if `u` lies on a cycle). One BFS per node.
pub fn closure<G: RandomAccessGraph>(g: &G) -> Vec<Vec<u64>> {
    let n = g.num_nodes();
    let words = n.div_ceil(64);
    let succ = succ_lists(g);
    (0..n)
        .map(|u| {
            let mut row = vec![0u64; words];
            let mut q: VecDeque<usize> = succ[u].iter().copied().collect();
            for &v in &succ[u] {
                row[v / 64] |= 1 << (v % 64);
            }
            while let Some(x) = q.pop_front() {
                for &v in &succ[x] {
                    if row[v / 64] & (1 << (v % 64)) == 0 {
                        row[v / 64] |= 1 << (v % 64);
                        q.push_back(v);
                    }
                }
            }
            row
        })
        .collect()
}

/// Whether bit `v` is set in a [`closure`] row.
#[inline]
pub fn bit(row: &[u64], v: usize) -> bool {
    row[v / 64] & (1 << (v % 64)) != 0
}

/// Calls `f` on every directed path of the DAG `g` with at least one node (single nodes
/// included), as a node sequence. Exponential: use only for `n ≤ 12`.
pub fn all_paths<G: RandomAccessGraph>(g: &G, mut f: impl FnMut(&[usize])) {
    let succ = succ_lists(g);
    fn rec(succ: &[Vec<usize>], path: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
        f(path);
        let u = *path.last().unwrap();
        for &v in &succ[u] {
            path.push(v);
            rec(succ, path, f);
            path.pop();
        }
    }
    for u in 0..g.num_nodes() {
        let mut path = vec![u];
        rec(&succ, &mut path, &mut f);
    }
}

/// Whether `v` is reachable from `u` (reflexive: `bfs_reach(g, u, u)` is true).
pub fn bfs_reach<G: RandomAccessGraph>(g: &G, u: usize, v: usize) -> bool {
    let n = g.num_nodes();
    let mut seen = vec![false; n];
    let mut q = VecDeque::from([u]);
    seen[u] = true;
    while let Some(x) = q.pop_front() {
        if x == v {
            return true;
        }
        for y in g.successors(x) {
            if !seen[y] {
                seen[y] = true;
                q.push_back(y);
            }
        }
    }
    false
}

/// Shortest distances from the set `sources` (distance 0) with arc weights `w(u, v)`
/// (Bellman–Ford, [\[Bel58\]](crate::references#bel58)); `None` for unreachable nodes. Panics on
/// a negative cycle reachable from the sources.
pub fn bellman_ford<G: RandomAccessGraph>(
    g: &G,
    w: impl Fn(usize, usize) -> i64,
    sources: &[usize],
) -> Vec<Option<i64>> {
    let n = g.num_nodes();
    let a = arcs(g);
    let mut d: Vec<Option<i64>> = vec![None; n];
    for &s in sources {
        d[s] = Some(0);
    }
    for round in 0..=n {
        let mut changed = false;
        for &(u, v) in &a {
            if let Some(du) = d[u] {
                let c = du + w(u, v);
                if d[v].is_none_or(|dv| c < dv) {
                    d[v] = Some(c);
                    changed = true;
                }
            }
        }
        if !changed {
            return d;
        }
        assert!(round < n, "bellman_ford: negative cycle");
    }
    d
}

/// Immediate dominators with a virtual root connected to every node of indegree 0, by vertex
/// removal and BFS: `None` when the immediate dominator is the virtual root. O(n²(n + m)).
pub fn brute_dominators<G: RandomAccessGraph>(g: &G) -> Vec<Option<usize>> {
    let n = g.num_nodes();
    let succ = succ_lists(g);
    let mut indeg = vec![0usize; n];
    for s in &succ {
        for &v in s {
            indeg[v] += 1;
        }
    }
    let roots: Vec<usize> = (0..n).filter(|&v| indeg[v] == 0).collect();
    // reach[x] = nodes reachable from the virtual root when x is removed (x = n: nothing removed)
    let reach_without = |x: usize| -> Vec<bool> {
        let mut seen = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        for &r in &roots {
            if r != x {
                seen[r] = true;
                q.push_back(r);
            }
        }
        while let Some(u) = q.pop_front() {
            for &v in &succ[u] {
                if v != x && !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }
        seen
    };
    let base = reach_without(n);
    // dom[v] = strict dominators of v
    let mut dom: Vec<Vec<usize>> = vec![Vec::new(); n];
    for d in 0..n {
        let r = reach_without(d);
        for v in 0..n {
            if v != d && base[v] && !r[v] {
                dom[v].push(d);
            }
        }
    }
    (0..n)
        .map(|v| {
            // the strict dominator that every other strict dominator of v dominates
            dom[v]
                .iter()
                .copied()
                .find(|&d| dom[v].iter().all(|&e| e == d || dom[d].contains(&e)))
        })
        .collect()
}

/// Width (size of a maximum antichain) of a DAG by subset enumeration; `n ≤ 16`.
pub fn brute_width<G: RandomAccessGraph>(g: &G) -> usize {
    let n = g.num_nodes();
    assert!(n <= 16, "brute_width: n = {n} > 16");
    let c = closure(g);
    let comparable = |u: usize, v: usize| bit(&c[u], v) || bit(&c[v], u);
    let mut best = 0;
    for mask in 0u32..(1 << n) {
        let k = mask.count_ones() as usize;
        if k <= best {
            continue;
        }
        let nodes: Vec<usize> = (0..n).filter(|&i| mask >> i & 1 == 1).collect();
        if nodes
            .iter()
            .enumerate()
            .all(|(i, &u)| nodes[i + 1..].iter().all(|&v| !comparable(u, v)))
        {
            best = k;
        }
    }
    best
}

/// Size of a maximum matching of the split bipartite graph (left copy `u` — right copy `v` for
/// every arc `u → v`), by exhaustive search with memoization; `n ≤ 16`.
pub fn brute_matching<G: RandomAccessGraph>(g: &G) -> usize {
    let n = g.num_nodes();
    assert!(n <= 16, "brute_matching: n = {n} > 16");
    let succ = succ_lists(g);
    let mut memo = vec![u8::MAX; (n + 1) << n];
    fn rec(i: usize, used: usize, n: usize, succ: &[Vec<usize>], memo: &mut [u8]) -> u8 {
        if i == n {
            return 0;
        }
        let key = (i << n) | used;
        if memo[key] != u8::MAX {
            return memo[key];
        }
        let mut best = rec(i + 1, used, n, succ, memo);
        for &v in &succ[i] {
            if used >> v & 1 == 0 {
                best = best.max(1 + rec(i + 1, used | 1 << v, n, succ, memo));
            }
        }
        memo[key] = best;
        best
    }
    rec(0, 0, n, &succ, &mut memo) as usize
}
