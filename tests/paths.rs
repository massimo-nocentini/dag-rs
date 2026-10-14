//! Acceptance tests of path counting: `dag::paths` and
//! `dag count-paths`.
//!
//! Oracles: brute-force path enumeration (u128, `n ≤ 12`), an independent memoised recursion
//! with checked u128 arithmetic, closed forms (complete DAGs, diamonds, Fibonacci DAGs) and the
//! induced subgraph of a window.

#[path = "support/cli.rs"]
mod cli;

use std::path::Path;

use cli::{p, read_json, run_ok};
use dag::blocks::BlockIndex;
use dag::dp::solve::sweep_buffer_bytes;
use dag::dp::{DpConf, DpInput, Kernel};
use dag::graph::{self, ChunkConf, ChunkedGraph};
use dag::io::{self, Dump, MemBudget};
use dag::paths::*;
use dag::semiring::Semiring;
use dag::semiring::extfloat::{Ext, ExtFloat};
use dag::semiring::modular::{M61, M61_P};
use dag::sweep::SweepConf;
use dag::testkit::*;
use dag::{DagError, Which, Window};
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;

/// Unit roundoff of `f64`.
const U: f64 = 1.0 / 9_007_199_254_740_992.0;

const ALL: [CountKind; 4] = [
    CountKind::ExtFloat,
    CountKind::M61,
    CountKind::SatU64,
    CountKind::SatU128,
];

fn quiet() -> Option<ProgressLogger> {
    None
}

fn cquiet() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

fn conf(w: Window, kernel: Kernel, decoders: usize) -> DpConf {
    DpConf {
        window: w,
        kernel,
        pred: false,
        sweep: SweepConf { decoders },
    }
}

fn random_window(r: &mut SmallRng, n: usize) -> Window {
    if r.random_range(0..3) == 0 {
        return Window::full(n);
    }
    let a = r.random_range(0..=n);
    let b = r.random_range(0..=n);
    Window {
        lo: a.min(b),
        hi: a.max(b),
    }
}

/// Successor lists restricted to the window (absolute ids).
fn window_succ<G: RandomAccessGraph>(g: &G, w: Window) -> Vec<Vec<usize>> {
    (0..g.num_nodes())
        .map(|u| {
            if w.contains(u) {
                g.successors(u)
                    .into_iter()
                    .filter(|&v| w.contains(v))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect()
}

/// Exact counts by path enumeration: `(F, B, total, sources, sinks)` over the window.
fn brute_counts<G: RandomAccessGraph>(g: &G, w: Window) -> (Vec<u128>, Vec<u128>, u128, u64, u64) {
    let succ = window_succ(g, w);
    let mut has_pred = vec![false; g.num_nodes()];
    for s in &succ {
        for &v in s {
            has_pred[v] = true;
        }
    }
    let (mut f, mut b) = (vec![0u128; w.len()], vec![0u128; w.len()]);
    // every path from a window source, enumerated explicitly
    fn rec(
        succ: &[Vec<usize>],
        path: &mut Vec<usize>,
        lo: usize,
        f: &mut [u128],
        ends: &mut Vec<(usize, usize)>,
    ) {
        let (first, last) = (path[0], *path.last().unwrap());
        f[last - lo] += 1;
        if succ[last].is_empty() {
            ends.push((first, last));
        }
        for &v in &succ[last] {
            path.push(v);
            rec(succ, path, lo, f, ends);
            path.pop();
        }
    }
    // B by enumeration from every node to window sinks
    fn rec_b(succ: &[Vec<usize>], u: usize) -> u128 {
        if succ[u].is_empty() {
            1
        } else {
            succ[u].iter().map(|&v| rec_b(succ, v)).sum()
        }
    }
    let mut ends = Vec::new();
    for u in (w.lo..w.hi).filter(|&u| !has_pred[u]) {
        rec(&succ, &mut vec![u], w.lo, &mut f, &mut ends);
    }
    let total = ends.len() as u128;
    for u in w.lo..w.hi {
        b[u - w.lo] = rec_b(&succ, u);
    }
    let sources = (w.lo..w.hi).filter(|&u| !has_pred[u]).count() as u64;
    let sinks = (w.lo..w.hi).filter(|&u| succ[u].is_empty()).count() as u64;
    (f, b, total, sources, sinks)
}

/// Exact counts by a memoised recursion with checked u128 arithmetic (`None` = overflow).
fn memo_counts<G: RandomAccessGraph>(g: &G, w: Window) -> (Vec<Option<u128>>, Vec<Option<u128>>) {
    let succ = window_succ(g, w);
    let mut preds = vec![Vec::new(); g.num_nodes()];
    for (u, s) in succ.iter().enumerate() {
        for &v in s {
            preds[v].push(u);
        }
    }
    let add = |a: Option<u128>, b: Option<u128>| a.zip(b).and_then(|(a, b)| a.checked_add(b));
    let mut f = vec![None; w.len()];
    for v in w.lo..w.hi {
        f[v - w.lo] = if preds[v].is_empty() {
            Some(1)
        } else {
            preds[v]
                .iter()
                .try_fold(0, |acc, &u| add(Some(acc), f[u - w.lo]))
        };
    }
    let mut b = vec![None; w.len()];
    for u in (w.lo..w.hi).rev() {
        b[u - w.lo] = if succ[u].is_empty() {
            Some(1)
        } else {
            succ[u]
                .iter()
                .try_fold(0, |acc, &v| add(Some(acc), b[v - w.lo]))
        };
    }
    (f, b)
}

fn assert_log2_close(got: f64, exact: u128, tol: f64, what: &str) {
    let want = (exact as f64).log2();
    assert!(
        (got - want).abs() <= tol,
        "{what}: log2 {got} vs exact log2 {want} (count {exact})"
    );
}

/// Checks every array of `c` against exact counts (`None` = above u128).
fn check_exact(c: &PathCounts, f: &[Option<u128>], b: &[Option<u128>], tol: f64, what: &str) {
    for (dir, exact, log2, mod61, s64, s128) in [
        (
            "fwd",
            f,
            &c.fwd_log2,
            &c.fwd_mod61,
            &c.fwd_sat64,
            &c.fwd_sat128,
        ),
        (
            "bwd",
            b,
            &c.bwd_log2,
            &c.bwd_mod61,
            &c.bwd_sat64,
            &c.bwd_sat128,
        ),
    ] {
        for (i, x) in exact.iter().enumerate() {
            let w = format!("{what} {dir} node {}", i + c.window.lo);
            let Some(x) = *x else { continue };
            if let Some(a) = log2 {
                assert_log2_close(a[i], x, tol, &w);
            }
            if let Some(a) = mod61 {
                assert_eq!(a[i] as u128, x % M61_P as u128, "{w}: mod61");
            }
            if let Some(a) = s64 {
                assert_eq!(a[i], u64::try_from(x).unwrap_or(u64::MAX), "{w}: sat64");
            }
            if let Some(a) = s128 {
                assert_eq!(a[i], x, "{w}: sat128");
            }
        }
    }
}

/// The inputs of one graph: forward graph, transpose and (if a biclique digraph) blocks.
struct Inputs {
    g: VecGraph,
    cg: ChunkedGraph<VecGraph>,
    cgt: ChunkedGraph<VecGraph>,
    blocks: Option<BlockIndex>,
}

impl Inputs {
    fn new(g: VecGraph, chunk_arcs: u64) -> Self {
        let gt = transpose(&g);
        let blocks = BlockIndex::from_vec_graph(&g).ok().map(|x| x.0);
        Self {
            cg: chunked(&g, chunk_arcs),
            cgt: chunked(&gt, chunk_arcs),
            g,
            blocks,
        }
    }

    /// Every input combination: (graph + transpose), (graph only: push), (transpose +
    /// graph with forced kernels) and (blocks only, if any).
    fn variants(&self) -> Vec<(&'static str, DpInput<'_, VecGraph>, Kernel)> {
        let mut v = vec![
            (
                "pull",
                DpInput::with_transpose(&self.cg, &self.cgt),
                Kernel::Auto,
            ),
            ("push", DpInput::graph(&self.cg), Kernel::Auto),
            (
                "push-forced",
                DpInput::with_transpose(&self.cg, &self.cgt),
                Kernel::Push,
            ),
        ];
        if let Some(b) = &self.blocks {
            let mut only = DpInput::graph(&self.cg);
            only.g = None;
            only.blocks = Some(b);
            v.push(("blocks-only", only, Kernel::Auto));
            v.push((
                "blocks+graph",
                DpInput::with_transpose(&self.cg, &self.cgt).and_blocks(b),
                Kernel::Blocks,
            ));
        }
        v
    }
}

/// The per-node arrays of `c` (for bit-identity checks).
fn arrays(c: &PathCounts) -> Vec<Vec<u8>> {
    fn bytes<T: Dump>(x: &Option<Vec<T>>) -> Vec<u8> {
        let Some(x) = x else { return vec![] };
        let mut out = vec![0u8; x.len() * T::BYTES];
        for (o, v) in out.chunks_exact_mut(T::BYTES).zip(x) {
            v.put(o);
        }
        out
    }
    vec![
        bytes(&c.fwd_log2),
        bytes(&c.fwd_mod61),
        bytes(&c.fwd_sat64),
        bytes(&c.fwd_sat128),
        bytes(&c.bwd_log2),
        bytes(&c.bwd_mod61),
        bytes(&c.bwd_sat64),
        bytes(&c.bwd_sat128),
    ]
}

/// All kernels and inputs give bit-identical arrays and the same summary statistics.
fn assert_same(a: &PathCounts, b: &PathCounts, what: &str) {
    assert_eq!(arrays(a), arrays(b), "{what}: arrays differ");
    assert_eq!(a.total_log2.to_bits(), b.total_log2.to_bits(), "{what}");
    assert_eq!(a.total_mod61, b.total_mod61, "{what}");
    assert_eq!(a.saturated_nodes, b.saturated_nodes, "{what}");
    assert_eq!(a.summary.duality, b.summary.duality, "{what}");
    assert_eq!(a.summary.through, b.summary.through, "{what}");
    assert_eq!(
        (a.summary.sources, a.summary.sinks),
        (b.summary.sources, b.summary.sinks)
    );
}

// ---------------------------------------------------------------------------------------------
// Brute force
// ---------------------------------------------------------------------------------------------

/// `log2` of an exact count, evaluated in `f64`.
fn exact_log2(x: u128) -> f64 {
    (x as f64).log2()
}

/// `⌊log2 x⌋` of a count `x ≥ 1`.
fn floor_log2(x: u128) -> u64 {
    127 - x.leading_zeros() as u64
}

/// Mean of a sequence (`NaN` when empty).
fn oracle_mean(x: impl Iterator<Item = f64>) -> f64 {
    let (s, k) = x.fold((0.0, 0usize), |(s, k), x| (s + x, k + 1));
    s / k as f64
}

/// The oracle `log2_hist` from the exact `⌊log2 count⌋` values, binned with integer
/// arithmetic: count 1 goes to bin 0, and `log2 count ∈ [2^(k−1), 2^k)` ⇔ `⌊log2 count⌋ ∈
/// [2^(k−1), 2^k)` goes to bin `k = ⌊log2 ⌊log2 count⌋⌋ + 1` (capped at `HIST_BINS − 1`).
fn oracle_hist(floor_log2s: impl Iterator<Item = u64>) -> Vec<u64> {
    let mut h = vec![0u64; HIST_BINS];
    for fl in floor_log2s {
        let bin = if fl == 0 {
            0
        } else {
            (64 - fl.leading_zeros() as usize).min(HIST_BINS - 1)
        };
        h[bin] += 1;
    }
    while h.last() == Some(&0) {
        h.pop();
    }
    h
}

/// Checks the aggregate statistics of one direction against the exact counts: the mean of the
/// `log2` counts, the whole histogram and the saturation counts.
fn check_dir_stats(
    ds: &DirSummary,
    floor_log2s: impl Iterator<Item = u64>,
    exact: &[u128],
    what: &str,
) {
    assert_eq!(ds.log2_hist, oracle_hist(floor_log2s), "{what}: log2_hist");
    if exact.is_empty() {
        assert_eq!(ds.mean_log2, None, "{what}: empty window");
    } else {
        let want = oracle_mean(exact.iter().map(|&x| exact_log2(x)));
        let got = ds.mean_log2.unwrap();
        assert!(
            (got - want).abs() <= 1e-12,
            "{what}: mean_log2 {got} vs {want}"
        );
    }
    let sat = |cap: u128| exact.iter().filter(|&&x| x >= cap).count() as u64;
    assert_eq!(
        ds.saturated_sat64,
        Some(sat(u64::MAX as u128)),
        "{what}: sat64"
    );
    assert_eq!(ds.saturated_sat128, Some(sat(u128::MAX)), "{what}: sat128");
}

#[test]
fn counts_match_brute_force_enumeration() {
    for seed in 0..400u64 {
        let mut r = SmallRng::seed_from_u64(seed);
        let g = if seed % 2 == 0 {
            random_dag(r.random_range(0..=12), r.random_range(0.0..0.6), seed)
        } else {
            let opts = MiniPgOpts {
                zero_output_prob: 0.15,
                ..Default::default()
            };
            mini_pg(r.random_range(1..8), seed, &opts).graph
        };
        let n = g.num_nodes();
        let inp = Inputs::new(g, r.random_range(1..6));
        let w = random_window(&mut r, n);
        let (f, b, total, sources, sinks) = brute_counts(&inp.g, w);
        let (fo, bo): (Vec<_>, Vec<_>) = (
            f.iter().map(|&x| Some(x)).collect(),
            b.iter().map(|&x| Some(x)).collect(),
        );
        let mut first: Option<PathCounts> = None;
        for (name, input, kernel) in inp.variants() {
            let what = format!("seed {seed} {name} {w}");
            let c = count_paths(
                &input,
                Which::Both,
                &ALL,
                &conf(w, kernel, r.random_range(1..4)),
                &mut quiet(),
            )
            .unwrap_or_else(|e| panic!("{what}: {e:#}"));
            check_exact(&c, &fo, &bo, 1e-12, &what);
            assert_eq!(c.summary.sources, sources, "{what}: sources");
            assert_eq!(c.summary.sinks, sinks, "{what}: sinks");
            assert_eq!(
                c.total_mod61 as u128,
                total % M61_P as u128,
                "{what}: total"
            );
            if total > 0 {
                assert_log2_close(c.total_log2, total, 1e-12, &what);
            } else {
                assert_eq!(c.total_log2, f64::NEG_INFINITY, "{what}: empty window");
            }
            let d = c.summary.duality.clone().unwrap();
            assert!(d.ok, "{what}: {d:?}");
            assert_eq!(d.mod61_equal, Some(true));
            assert_eq!(d.sat64_equal, Some(true));
            assert_eq!(d.sat128_equal, Some(true));
            assert_eq!(
                d.extfloat_rel_diff,
                Some(0.0),
                "{what}: small counts are exact"
            );
            let fs = c.summary.forward.as_ref().unwrap();
            let bs = c.summary.backward.as_ref().unwrap();
            assert_eq!(fs.total_sat64, Some(total as u64), "{what}");
            assert_eq!(bs.total_sat128, Some(total.to_string()), "{what}");
            assert_eq!((fs.endpoints, fs.starts), (sinks, sources), "{what}");
            assert_eq!((bs.endpoints, bs.starts), (sources, sinks), "{what}");
            assert_eq!(c.saturated_nodes, 0);
            check_dir_stats(fs, f.iter().map(|&x| floor_log2(x)), &f, &what);
            check_dir_stats(bs, b.iter().map(|&x| floor_log2(x)), &b, &what);
            if let Some(t) = &c.summary.through {
                let want = oracle_mean((0..w.len()).map(|i| exact_log2(f[i]) + exact_log2(b[i])));
                assert!((t.mean_log2 - want).abs() <= 1e-12, "{what}: through mean");
                // the through count of v is F(v)·B(v), and at most the total
                let (best, at) = (0..w.len())
                    .map(|i| (f[i] * b[i], i))
                    .fold((0, 0), |a, x| if x.0 > a.0 { x } else { a });
                assert_eq!(t.argmax, at + w.lo, "{what}: through argmax");
                assert_log2_close(t.max_log2, best, 1e-12, &what);
                assert!(t.max_share <= 1.0 + 1e-12 && t.max_share > 0.0, "{what}");
            }
            if let Some(a) = &first {
                assert_same(a, &c, &what);
            } else {
                first = Some(c);
            }
        }
    }
}

#[test]
fn counts_match_memoised_recursion_on_larger_graphs() {
    for seed in 0..60u64 {
        let mut r = SmallRng::seed_from_u64(seed ^ 0xC0FFEE);
        let g = match seed % 3 {
            0 => random_dag(r.random_range(20..200), r.random_range(0.02..0.3), seed),
            1 => mini_pg(r.random_range(10..300), seed, &MiniPgOpts::default()).graph,
            _ => layered(&[3, 5, 8, 5, 8, 9, 4, 7, 6, 9, 8, 7, 5, 8, 9, 6], 0.6, seed),
        };
        let n = g.num_nodes();
        let inp = Inputs::new(g, r.random_range(1..40));
        let w = random_window(&mut r, n);
        let (f, b) = memo_counts(&inp.g, w);
        let mut first: Option<PathCounts> = None;
        for (name, input, kernel) in inp.variants() {
            let what = format!("seed {seed} {name} {w}");
            let c =
                count_paths(&input, Which::Both, &ALL, &conf(w, kernel, 3), &mut quiet()).unwrap();
            check_exact(&c, &f, &b, 1e-12, &what);
            assert!(c.summary.duality.as_ref().unwrap().ok, "{what}");
            if let Some(a) = &first {
                assert_same(a, &c, &what);
            } else {
                first = Some(c);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Closed forms
// ---------------------------------------------------------------------------------------------

/// `2^k mod (2⁶¹ − 1) = 2^(k mod 61)`.
fn pow2_mod61(k: u64) -> u64 {
    1u64 << (k % 61)
}

#[test]
fn complete_dag_endpoints_give_powers_of_two() {
    for n in [1usize, 2, 3, 10, 64, 65, 66, 129, 130, 131, 200] {
        let inp = Inputs::new(complete_dag(n), 64);
        let c = count_paths(
            &DpInput::with_transpose(&inp.cg, &inp.cgt),
            Which::Both,
            &ALL,
            &conf(Window::full(n), Kernel::Auto, 2),
            &mut quiet(),
        )
        .unwrap();
        let (fl, fm) = (c.fwd_log2.as_ref().unwrap(), c.fwd_mod61.as_ref().unwrap());
        let (bl, bm) = (c.bwd_log2.as_ref().unwrap(), c.bwd_mod61.as_ref().unwrap());
        let (f64s, f128) = (
            c.fwd_sat64.as_ref().unwrap(),
            c.fwd_sat128.as_ref().unwrap(),
        );
        for v in 0..n {
            // F(v) = 2^(v−1) for v ≥ 1 (any subset of the nodes strictly between 0 and v)
            let k = v.saturating_sub(1) as u64;
            assert_eq!(fl[v], k as f64, "n {n} F({v})");
            assert_eq!(fm[v], pow2_mod61(k), "n {n} F({v})");
            assert_eq!(f64s[v], if k < 64 { 1 << k } else { u64::MAX });
            assert_eq!(f128[v], if k < 128 { 1u128 << k } else { u128::MAX });
            let kb = (n - 1 - v).saturating_sub(1) as u64;
            assert_eq!(bl[v], kb as f64, "n {n} B({v})");
            assert_eq!(bm[v], pow2_mod61(kb), "n {n} B({v})");
        }
        // the only source is 0 and the only sink n − 1: 2^(n−2) paths
        let t = n.saturating_sub(2) as u64;
        assert_eq!(c.total_log2, t as f64, "n {n}");
        assert_eq!(c.total_mod61, pow2_mod61(t), "n {n}");
        let d = c.summary.duality.unwrap();
        assert!(d.ok);
        assert_eq!(d.extfloat_rel_diff, Some(0.0));
        assert_eq!(d.sat64_equal, (t < 64).then_some(true), "n {n}");
        assert_eq!(d.sat128_equal, (t < 128).then_some(true), "n {n}");
        // saturated nodes: F(v) ≥ 2^64 or B(v) ≥ 2^64
        let want = (0..n)
            .filter(|&v| v.saturating_sub(1) >= 64 || (n - 1 - v).saturating_sub(1) >= 64)
            .count() as u64;
        assert_eq!(c.saturated_nodes, want, "n {n}");
        let fs = c.summary.forward.unwrap();
        assert_eq!(fs.max_log2, Some(t as f64));
        // the smallest node with the largest count (F(0) = F(1) = 1)
        assert_eq!(fs.argmax, Some(if n <= 2 { 0 } else { n - 1 }));
        assert_eq!(
            fs.above_2_64,
            Some((0..n).filter(|&v| v.saturating_sub(1) >= 64).count() as u64)
        );
        assert_eq!(fs.log2_hist.iter().sum::<u64>(), n as u64);
        // F(v) = 2^kf(v), B(v) = 2^kb(v): exact log2, histogram and saturation oracles
        let kf = |v: usize| v.saturating_sub(1) as u64;
        let kb = |v: usize| (n - 1 - v).saturating_sub(1) as u64;
        let bs = c.summary.backward.unwrap();
        for (ds, k, name) in [
            (&fs, &kf as &dyn Fn(usize) -> u64, "fwd"),
            (&bs, &kb, "bwd"),
        ] {
            let what = format!("n {n} {name}");
            assert_eq!(
                ds.log2_hist,
                oracle_hist((0..n).map(k)),
                "{what}: log2_hist"
            );
            let mean = oracle_mean((0..n).map(|v| k(v) as f64));
            assert!(
                (ds.mean_log2.unwrap() - mean).abs() <= 1e-12,
                "{what}: mean_log2"
            );
            let sat = |b: u64| (0..n).filter(|&v| k(v) >= b).count() as u64;
            assert_eq!(ds.saturated_sat64, Some(sat(64)), "{what}: sat64");
            assert_eq!(ds.saturated_sat128, Some(sat(128)), "{what}: sat128");
            assert_eq!(ds.above_2_128, Some(sat(128)), "{what}: above 2^128");
        }
        let th = c.summary.through.unwrap();
        let mean = oracle_mean((0..n).map(|v| (kf(v) + kb(v)) as f64));
        assert!((th.mean_log2 - mean).abs() <= 1e-12, "n {n}: through mean");
        // every path goes through 0 and n − 1
        assert_eq!(th.max_log2, t as f64);
        assert_eq!(th.argmax, 0);
        assert_eq!(th.max_share, 1.0);
    }
}

#[test]
fn diamonds_and_ladder_have_known_counts() {
    // k diamonds in series: 2^k paths, far beyond u128 for k = 3000
    for k in [1usize, 5, 127, 128, 3000] {
        let inp = Inputs::new(diamonds(k), 7);
        for (name, input, kernel) in inp.variants() {
            let c = count_paths(
                &input,
                Which::Both,
                &[CountKind::ExtFloat, CountKind::M61],
                &conf(Window::full(3 * k + 1), kernel, 2),
                &mut quiet(),
            )
            .unwrap();
            assert_eq!(c.total_log2, k as f64, "{name} k {k}");
            assert_eq!(c.total_mod61, pow2_mod61(k as u64), "{name} k {k}");
            let d = c.summary.duality.unwrap();
            assert_eq!(d.extfloat_rel_diff, Some(0.0));
            assert!(d.ok);
            // the cut vertices s_i carry every path
            let th = c.summary.through.unwrap();
            assert_eq!(th.max_share, 1.0, "{name} k {k}");
        }
    }
    // ladder(L): the chain c_0 → … → c_{L−1} plus x_i with predecessors {c_0, c_i}; one
    // source, L − 1 sinks x_1..x_{L−1} (c_{L−1} → x_{L−1}), each reached by 2 paths
    // (c_0 → x_i and c_0 → … → c_i → x_i): 2(L − 1) source-to-sink paths
    let l = 50;
    let inp = Inputs::new(ladder(l), 3);
    let c = count_paths(
        &DpInput::with_transpose(&inp.cg, &inp.cgt),
        Which::Both,
        &[CountKind::SatU64],
        &conf(Window::full(2 * l - 1), Kernel::Auto, 2),
        &mut quiet(),
    )
    .unwrap();
    assert_eq!(
        c.summary.forward.unwrap().total_sat64,
        Some(2 * (l as u64 - 1))
    );
}

/// Fibonacci DAG `i → i+1, i → i+2`: `F(v) = Fib(v + 1)`; deep (depth n − 1) with huge counts.
#[test]
fn fibonacci_dag_extfloat_relative_error_and_exact_residues() {
    let n = 20_000;
    let g = vec_graph(
        n,
        (0..n)
            .flat_map(|i| [(i, i + 1), (i, i + 2)])
            .filter(|&(_, v)| v < n),
    );
    let inp = Inputs::new(g, 1000);
    let c = count_paths(
        &DpInput::with_transpose(&inp.cg, &inp.cgt),
        Which::Both,
        &ALL,
        &conf(Window::full(n), Kernel::Auto, 4),
        &mut quiet(),
    )
    .unwrap();
    let fl = c.fwd_log2.as_ref().unwrap();
    let fm = c.fwd_mod61.as_ref().unwrap();
    let f128 = c.fwd_sat128.as_ref().unwrap();
    // exact u128 while it fits, residues always
    let (mut a, mut b) = (Some(1u128), Some(1u128)); // Fib(1), Fib(2)
    let (mut ra, mut rb) = (1u64, 1u64);
    let m = M61::new();
    let ef = ExtFloat::new();
    let (mut ea, mut eb) = (Ext::ONE, Ext::ONE);
    let mut max_rel = 0.0f64;
    for v in 0..n {
        // F(v) = Fib(v + 1): a holds Fib(v + 1)
        assert_eq!(fm[v], ra, "node {v}: mod61");
        match a {
            Some(x) => {
                assert_eq!(f128[v], x, "node {v}: sat128");
                // relative error of the magnitude vs the exact count, against the
                // first-order bound u·Σ_{w on a path}(indeg(w) − 1) ≤ v·u (indeg ≤ 2; plus the
                // rounding of the two log2 evaluations); 2·v·u leaves a factor 2
                let rel = ((fl[v] - (x as f64).log2()) * std::f64::consts::LN_2).abs();
                let bound = 2.0 * v as f64 * U + 1e-15;
                assert!(rel <= bound, "node {v}: relative error {rel} > {bound}");
                max_rel = max_rel.max(rel);
            }
            None => assert_eq!(f128[v], u128::MAX, "node {v}: saturated"),
        }
        // independent ExtFloat recurrence evaluated in the same order gives the same bits
        assert_eq!(
            fl[v].to_bits(),
            ea.log2().to_bits(),
            "node {v}: ExtFloat recurrence"
        );
        let next = a.zip(b).and_then(|(x, y)| x.checked_add(y));
        (a, b) = (b, next);
        (ra, rb) = (rb, m.plus(ra, rb));
        (ea, eb) = (eb, ef.plus(ea, eb));
    }
    assert!(max_rel < 1e-13, "max relative error {max_rel}");
    // Binet: log2 Fib(k) = k·log2 φ − log2 √5 + O(φ^(−2k)); first-order error bound D·u
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    let k = n as f64; // F(n − 1) = Fib(n)
    let binet = k * phi.log2() - 5f64.sqrt().log2();
    let rel = ((fl[n - 1] - binet) * std::f64::consts::LN_2).abs();
    assert!(
        rel < 2.0 * n as f64 * U + 1e-11,
        "relative error {rel} at depth {n}"
    );
    // one source (0) and one sink (n − 1): total = Fib(n), B(0) = Fib(n)
    let d = c.summary.duality.unwrap();
    assert!(d.ok, "{d:?}");
    assert!(d.extfloat_rel_diff.unwrap() <= 1e-12, "{d:?}");
    assert_eq!(c.bwd_mod61.unwrap()[0], fm[n - 1]);
}

/// A high in-degree node behind a deep layered prefix: D10 folds the predecessors of a node one
/// after the other (recursive summation), whose first-order error is `(k − 1)·u`, not
/// `⌈log2 k⌉·u`. Eighteen complete bipartite layers of width 8 give `F(a) = 8¹⁸ = 2⁵⁴`; the
/// sink `v` has the predecessors `a` and 1000 later sources, so `F(v) = 2⁵⁴ + 1000` but every
/// `+1` after `2⁵⁴` rounds away: relative error `1000/2⁵⁴ ≈ 500·u`. That is within the bound
/// `u·Σ_{w on a path}(indeg(w) − 1) = (18·7 + 1000)·u` of the module documentation, and above
/// the pairwise-style bound `depth·(⌈log2 indeg⌉ + 1)·u = 19·11·u` it replaces.
#[test]
fn high_indegree_node_exercises_the_recursive_summation_bound() {
    let (width, layers, extra) = (8usize, 19usize, 1000usize);
    let a = width * (layers - 1); // the first node of the last layer
    let sources = width * layers..width * layers + extra;
    let v = sources.end;
    let n = v + 1;
    let mut arcs = Vec::new();
    for l in 0..layers - 1 {
        for x in 0..width {
            for y in 0..width {
                arcs.push((l * width + x, (l + 1) * width + y));
            }
        }
    }
    arcs.push((a, v));
    arcs.extend(sources.clone().map(|s| (s, v)));
    let inp = Inputs::new(vec_graph(n, arcs), 64);
    let exact = (1u128 << 54) + extra as u128;
    for (name, input, kernel) in inp.variants() {
        let c = count_paths(
            &input,
            Which::Both,
            &ALL,
            &conf(Window::full(n), kernel, 2),
            &mut quiet(),
        )
        .unwrap();
        assert_eq!(c.fwd_sat128.as_ref().unwrap()[v], exact, "{name}");
        assert_eq!(
            c.fwd_mod61.as_ref().unwrap()[v] as u128,
            exact % M61_P as u128
        );
        let got = c.fwd_log2.as_ref().unwrap()[v];
        let rel = (exact as f64 - got.exp2()).abs() / exact as f64;
        let path_bound = ((layers - 1) * (width - 1) + extra) as f64 * U;
        let old_bound = (layers * (11 + 1 - 1)) as f64 * U; // 19·(⌈log2 1001⌉ + 1)·u
        assert!(
            rel <= path_bound,
            "{name}: relative error {rel} > {path_bound}"
        );
        assert!(
            rel > old_bound,
            "{name}: {rel} no longer exceeds {old_bound}"
        );
        assert!(rel > 400.0 * U, "{name}: expected ≈ 500u, got {}u", rel / U);
        assert!(c.summary.duality.unwrap().ok, "{name}");
    }
}

/// Forged totals: every mismatch makes the check fail, saturated or missing kinds are skipped,
/// and the ExtFloat tolerance is applied on both sides of `EXT_DUALITY_TOL`.
#[test]
fn duality_check_detects_every_mismatch() {
    let one = Ext::from_u64(1);
    let base = DirTotals {
        ext: Some(one.scale_pow2(5000)),
        mod61: Some(7),
        sat64: Some(1000),
        sat128: Some(1 << 100),
    };
    let d = Duality::check(&base, &base);
    assert!(d.ok, "{d:?}");
    assert_eq!(
        (d.mod61_equal, d.sat64_equal, d.sat128_equal),
        (Some(true), Some(true), Some(true))
    );
    assert_eq!(d.extfloat_rel_diff, Some(0.0));
    assert_eq!(d.extfloat_tol, EXT_DUALITY_TOL);
    let fails = |b: DirTotals| {
        let (d1, d2) = (Duality::check(&base, &b), Duality::check(&b, &base));
        assert_eq!(d1.ok, d2.ok, "symmetry");
        !d1.ok
    };
    // exact kinds
    assert!(fails(DirTotals {
        mod61: Some(8),
        ..base
    }));
    assert_eq!(
        Duality::check(
            &base,
            &DirTotals {
                mod61: Some(8),
                ..base
            }
        )
        .mod61_equal,
        Some(false)
    );
    assert!(fails(DirTotals {
        sat64: Some(999),
        ..base
    }));
    assert_eq!(
        Duality::check(
            &base,
            &DirTotals {
                sat64: Some(999),
                ..base
            }
        )
        .sat64_equal,
        Some(false)
    );
    assert!(fails(DirTotals {
        sat128: Some((1 << 100) + 1),
        ..base
    }));
    // a saturated total is not compared
    for b in [
        DirTotals {
            sat64: Some(u64::MAX),
            ..base
        },
        DirTotals {
            sat128: Some(u128::MAX),
            ..base
        },
    ] {
        let d = Duality::check(&base, &b);
        assert!(d.ok, "{d:?}");
        assert!(d.sat64_equal.is_none() || d.sat128_equal.is_none(), "{d:?}");
    }
    // a kind computed in only one direction is not compared
    let d = Duality::check(
        &base,
        &DirTotals {
            mod61: None,
            ..base
        },
    );
    assert_eq!(d.mod61_equal, None);
    assert!(d.ok);
    // ExtFloat: just below and just above the tolerance, at a huge exponent
    let near = |r: f64| Some(Ext::from_f64(1.0 - r).scale_pow2(5000));
    let below = Duality::check(
        &base,
        &DirTotals {
            ext: near(0.99 * EXT_DUALITY_TOL),
            ..base
        },
    );
    assert!(below.ok, "{below:?}");
    let r = below.extfloat_rel_diff.unwrap();
    assert!((r - 0.99 * EXT_DUALITY_TOL).abs() < 1e-15, "{r}");
    assert!(fails(DirTotals {
        ext: near(1.01 * EXT_DUALITY_TOL),
        ..base
    }));
    // a zero total against a non-zero one, and exponents too far apart
    assert!(fails(DirTotals {
        ext: Some(Ext::ZERO),
        ..base
    }));
    assert!(fails(DirTotals {
        ext: Some(one.scale_pow2(4000)),
        ..base
    }));
}

/// `dag count-paths` exits with 1 exactly when the duality check failed.
#[test]
fn cli_exit_status_follows_the_duality_check() {
    use dag::cli::common::{EXIT_FAILED, EXIT_OK, exit_status};
    use dag::cli::count::duality_ok;
    use std::process::ExitCode;
    let t = DirTotals {
        mod61: Some(3),
        ..Default::default()
    };
    let summary = |d: Option<Duality>| PathSummary {
        duality: d,
        ..Default::default()
    };
    let good = summary(Some(Duality::check(&t, &t)));
    let bad = summary(Some(Duality::check(
        &t,
        &DirTotals {
            mod61: Some(4),
            ..t
        },
    )));
    assert!(
        duality_ok(&summary(None)),
        "a single direction has no check"
    );
    assert!(duality_ok(&good));
    assert!(!duality_ok(&bad));
    assert_eq!(exit_status(duality_ok(&bad)), ExitCode::from(EXIT_FAILED));
    assert_eq!(exit_status(duality_ok(&good)), ExitCode::from(EXIT_OK));
}

// ---------------------------------------------------------------------------------------------
// Windows, directions, inputs
// ---------------------------------------------------------------------------------------------

/// The subgraph induced by `w`, ids shifted by `−w.lo`.
fn induced<G: RandomAccessGraph>(g: &G, w: Window) -> VecGraph {
    let arcs: Vec<(usize, usize)> = (w.lo..w.hi)
        .flat_map(|u| {
            g.successors(u)
                .into_iter()
                .filter(|&v| w.contains(v))
                .map(move |v| (u - w.lo, v - w.lo))
                .collect::<Vec<_>>()
        })
        .collect();
    vec_graph(w.len(), arcs)
}

#[test]
fn windowed_runs_equal_runs_on_the_induced_subgraph() {
    for seed in 0..100u64 {
        let mut r = SmallRng::seed_from_u64(seed ^ 0x77);
        let g = mini_pg(r.random_range(5..60), seed, &MiniPgOpts::default()).graph;
        let n = g.num_nodes();
        let w = random_window(&mut r, n);
        let inp = Inputs::new(g, 5);
        let sub = Inputs::new(induced(&inp.g, w), 5);
        let want = count_paths(
            &DpInput::with_transpose(&sub.cg, &sub.cgt),
            Which::Both,
            &ALL,
            &conf(Window::full(w.len()), Kernel::Auto, 2),
            &mut quiet(),
        )
        .unwrap();
        for (name, input, kernel) in inp.variants() {
            let got =
                count_paths(&input, Which::Both, &ALL, &conf(w, kernel, 2), &mut quiet()).unwrap();
            assert_eq!(arrays(&got), arrays(&want), "seed {seed} {name} {w}");
            assert_eq!(got.total_mod61, want.total_mod61);
            assert_eq!(got.total_log2.to_bits(), want.total_log2.to_bits());
            assert_eq!(got.summary.duality, want.summary.duality);
            // argmax ids are absolute
            let (a, b) = (
                got.summary.forward.unwrap(),
                want.summary.forward.clone().unwrap(),
            );
            assert_eq!(a.argmax, b.argmax.map(|x| x + w.lo), "seed {seed} {name}");
        }
    }
}

#[test]
fn single_directions_and_kind_subsets() {
    let g = mini_pg(80, 5, &MiniPgOpts::default()).graph;
    let n = g.num_nodes();
    let inp = Inputs::new(g, 4);
    let input = DpInput::with_transpose(&inp.cg, &inp.cgt);
    let c0 = conf(Window::full(n), Kernel::Auto, 2);
    let both = count_paths(&input, Which::Both, &ALL, &c0, &mut quiet()).unwrap();
    let f = count_paths(&input, Which::Forward, &ALL, &c0, &mut quiet()).unwrap();
    assert!(f.bwd_log2.is_none() && f.bwd_mod61.is_none() && f.bwd_sat64.is_none());
    assert!(f.summary.duality.is_none() && f.summary.through.is_none());
    assert_eq!(f.fwd_log2, both.fwd_log2);
    assert_eq!(f.fwd_sat128, both.fwd_sat128);
    assert_eq!(f.total_mod61, both.total_mod61);
    let b = count_paths(&input, Which::Backward, &ALL, &c0, &mut quiet()).unwrap();
    assert!(b.fwd_log2.is_none() && b.summary.forward.is_none());
    assert_eq!(b.bwd_mod61, both.bwd_mod61);
    assert_eq!(
        b.total_mod61, both.total_mod61,
        "backward total = forward total"
    );
    // kind subsets give the same arrays as the product pass
    for kinds in [
        &[CountKind::ExtFloat][..],
        &[CountKind::M61],
        &[CountKind::SatU64, CountKind::M61],
        &[CountKind::SatU128, CountKind::ExtFloat, CountKind::ExtFloat],
    ] {
        let c = count_paths(&input, Which::Both, kinds, &c0, &mut quiet()).unwrap();
        let has = |k| kinds.contains(&k);
        assert_eq!(c.fwd_log2.is_some(), has(CountKind::ExtFloat));
        assert_eq!(c.bwd_mod61.is_some(), has(CountKind::M61));
        if has(CountKind::ExtFloat) {
            assert_eq!(c.fwd_log2, both.fwd_log2, "{kinds:?}");
            assert_eq!(c.bwd_log2, both.bwd_log2, "{kinds:?}");
        } else {
            assert!(c.total_log2.is_nan());
        }
        if has(CountKind::M61) {
            assert_eq!(c.fwd_mod61, both.fwd_mod61, "{kinds:?}");
        } else {
            assert_eq!(c.total_mod61, 0);
        }
        assert!(c.summary.duality.unwrap().ok);
    }
}

#[test]
fn isolated_nodes_and_empty_windows() {
    let inp = Inputs::new(isolated(5), 2);
    let c = count_paths(
        &DpInput::with_transpose(&inp.cg, &inp.cgt),
        Which::Both,
        &ALL,
        &conf(Window::full(5), Kernel::Auto, 1),
        &mut quiet(),
    )
    .unwrap();
    assert_eq!(c.fwd_sat64.as_deref(), Some(&[1u64; 5][..]));
    assert_eq!(c.bwd_log2.as_deref(), Some(&[0.0; 5][..]));
    assert_eq!(c.total_mod61, 5, "5 single-node paths");
    assert_eq!((c.summary.sources, c.summary.sinks), (5, 5));
    for (name, input, kernel) in inp.variants() {
        let c = count_paths(
            &input,
            Which::Both,
            &ALL,
            &conf(Window { lo: 3, hi: 3 }, kernel, 1),
            &mut quiet(),
        )
        .unwrap();
        assert_eq!(c.fwd_log2.as_deref(), Some(&[][..]), "{name}");
        assert_eq!(c.total_mod61, 0);
        assert_eq!(c.total_log2, f64::NEG_INFINITY);
        assert!(c.summary.duality.unwrap().ok);
        assert!(c.summary.forward.unwrap().max_log2.is_none());
    }
    // n = 0
    let inp = Inputs::new(isolated(0), 2);
    let c = count_paths(
        &DpInput::graph(&inp.cg),
        Which::Both,
        &ALL,
        &conf(Window::full(0), Kernel::Auto, 1),
        &mut quiet(),
    )
    .unwrap();
    assert_eq!(c.saturated_nodes, 0);
}

#[test]
fn endpoints_from_blocks_graph_and_transpose_agree() {
    for seed in 0..200u64 {
        let mut r = SmallRng::seed_from_u64(seed);
        let opts = MiniPgOpts {
            zero_output_prob: 0.2,
            ..Default::default()
        };
        let g = mini_pg(r.random_range(1..60), seed, &opts).graph;
        let n = g.num_nodes();
        let inp = Inputs::new(g, r.random_range(1..9));
        let w = random_window(&mut r, n);
        let b = Endpoints::from_blocks(inp.blocks.as_ref().unwrap(), w);
        let f = Endpoints::from_graph(&inp.cg, w, false, &mut cquiet()).unwrap();
        let t = Endpoints::from_graph(&inp.cgt, w, true, &mut cquiet()).unwrap();
        assert_eq!(b, f, "seed {seed} {w}");
        assert_eq!(b, t, "seed {seed} {w}");
        let succ = window_succ(&inp.g, w);
        for v in w.lo..w.hi {
            let has_pred = (w.lo..v).any(|u| succ[u].contains(&v));
            assert_eq!(b.is_source(v), !has_pred, "seed {seed} node {v}");
            assert_eq!(b.is_sink(v), succ[v].is_empty(), "seed {seed} node {v}");
        }
    }
}

#[test]
fn bv_round_trip_gives_identical_counts() {
    for seed in 0..12u64 {
        let g = if seed % 2 == 0 {
            random_dag(60, 0.08, seed)
        } else {
            mini_pg(40, seed, &MiniPgOpts::default()).graph
        };
        let n = g.num_nodes();
        let inp = Inputs::new(g, 4);
        let dir = tempfile::tempdir().unwrap();
        let base = write_bv(dir.path(), "g", &inp.g);
        let cc = ChunkConf {
            chunk_arcs: 5,
            max_chunk_nodes: 3,
        };
        let bg = graph::open(&base, cc).unwrap();
        let bgt = graph::open(&graph::transpose_basename(&base).unwrap(), cc).unwrap();
        let mut r = SmallRng::seed_from_u64(seed);
        let w = random_window(&mut r, n);
        let want = count_paths(
            &DpInput::with_transpose(&inp.cg, &inp.cgt),
            Which::Both,
            &ALL,
            &conf(w, Kernel::Auto, 2),
            &mut quiet(),
        )
        .unwrap();
        for input in [DpInput::with_transpose(&bg, &bgt), DpInput::graph(&bg)] {
            let got = count_paths(
                &input,
                Which::Both,
                &ALL,
                &conf(w, Kernel::Auto, 3),
                &mut quiet(),
            )
            .unwrap();
            assert_same(&want, &got, &format!("seed {seed} BV"));
        }
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let g = random_dag(3000, 0.004, 42);
    let n = g.num_nodes();
    let inp = Inputs::new(g, 50);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                count_paths(
                    &DpInput::with_transpose(&inp.cg, &inp.cgt),
                    Which::Both,
                    &ALL,
                    &conf(Window::full(n), Kernel::Auto, threads.min(4)),
                    &mut quiet(),
                )
                .unwrap()
            })
    };
    let a = run(1);
    for t in [2, 4, 16] {
        let b = run(t);
        assert_same(&a, &b, &format!("{t} threads"));
        assert_eq!(a.summary.forward, {
            let mut f = b.summary.forward.clone();
            if let (Some(x), Some(y)) = (&mut f, &a.summary.forward) {
                x.elapsed_s = y.elapsed_s;
            }
            f
        });
    }
}

#[test]
fn memory_budget_controls_concurrency() {
    let g = random_dag(500, 0.02, 9);
    let n = g.num_nodes();
    let inp = Inputs::new(g, 20);
    let input = DpInput::with_transpose(&inp.cg, &inp.cgt);
    let c = conf(Window::full(n), Kernel::Auto, 2);
    let sweep = sweep_buffer_bytes(
        &SweepConf { decoders: 2 },
        inp.cg.conf.chunk_arcs,
        inp.cg.conf.max_chunk_nodes,
    );
    let fb = direction_bytes(n, dag::Dir::Forward, Kernel::Pull, &ALL, 0, sweep);
    let bb = direction_bytes(n, dag::Dir::Backward, Kernel::Pull, &ALL, 0, sweep);
    let ends = Endpoints::bytes(n);
    let out = output_bytes(n, &ALL);
    assert_eq!(out, 40 * n as u64);
    assert!(fb > out && bb > out);
    let gb = |b: u64| b as f64 / 1e9;

    let mut unlimited = MemBudget::unlimited();
    let a =
        count_paths_budgeted(&input, Which::Both, &ALL, &c, &mut unlimited, &mut quiet()).unwrap();
    assert!(a.summary.concurrent);

    // room for one direction at a time (the outputs of the first stay reserved):
    // sequential, same results
    let mut one = MemBudget::new(gb(ends + fb.max(out + bb)) + 1e-9);
    let s = count_paths_budgeted(&input, Which::Both, &ALL, &c, &mut one, &mut quiet()).unwrap();
    assert!(!s.summary.concurrent);
    assert_same(&a, &s, "sequential");

    let mut tiny = MemBudget::new(gb(ends) + 1e-9);
    let e =
        count_paths_budgeted(&input, Which::Both, &ALL, &c, &mut tiny, &mut quiet()).unwrap_err();
    assert!(
        matches!(
            e.downcast_ref::<DagError>(),
            Some(DagError::OverBudget { .. })
        ),
        "{e:#}"
    );
}

#[test]
fn errors() {
    let inp = Inputs::new(random_dag(20, 0.2, 1), 3);
    let w = Window::full(20);
    let input = DpInput::with_transpose(&inp.cg, &inp.cgt);
    assert!(
        count_paths(
            &input,
            Which::Both,
            &[],
            &conf(w, Kernel::Auto, 1),
            &mut quiet()
        )
        .is_err()
    );
    let mut c = conf(w, Kernel::Auto, 1);
    c.pred = true;
    assert!(count_paths(&input, Which::Both, &ALL, &c, &mut quiet()).is_err());
    // forward pull without a transpose
    assert!(
        count_paths(
            &DpInput::graph(&inp.cg),
            Which::Forward,
            &ALL,
            &conf(w, Kernel::Pull, 1),
            &mut quiet()
        )
        .is_err()
    );
    // blocks requested without an index
    assert!(
        count_paths(
            &input,
            Which::Both,
            &ALL,
            &conf(w, Kernel::Blocks, 1),
            &mut quiet()
        )
        .is_err()
    );
    // bad window
    assert!(
        count_paths(
            &input,
            Which::Both,
            &ALL,
            &conf(Window { lo: 3, hi: 21 }, Kernel::Auto, 1),
            &mut quiet()
        )
        .is_err()
    );
    // a back arc is reported as NotTopological
    let bad = Inputs::new(with_back_arc(&random_dag(20, 0.2, 1)), 3);
    let e = count_paths(
        &DpInput::graph(&bad.cg),
        Which::Forward,
        &ALL,
        &conf(w, Kernel::Push, 1),
        &mut quiet(),
    )
    .unwrap_err();
    assert!(
        matches!(
            e.downcast_ref::<DagError>(),
            Some(DagError::NotTopological { .. })
        ),
        "{e:#}"
    );
    // Push is accepted for the backward direction (it runs Pull)
    let c = count_paths(
        &input,
        Which::Both,
        &ALL,
        &conf(w, Kernel::Push, 1),
        &mut quiet(),
    )
    .unwrap();
    assert_eq!(c.summary.forward.unwrap().kernel, Some(Kernel::Push));
    assert_eq!(c.summary.backward.unwrap().kernel, Some(Kernel::Pull));
}

#[test]
fn helpers() {
    // pairwise is exact on integers and deterministic
    let x: Vec<f64> = (0..1_000_003).map(|i| (i % 1000) as f64).collect();
    let s = pairwise(0, x.len(), 0.0, &|i| x[i], &|a, b| a + b);
    assert_eq!(s, x.iter().sum::<f64>());
    // pairwise sum of 0.1 is much more accurate than the sequential sum
    let k = 1 << 20;
    let p = pairwise(0, k, 0.0, &|_| 0.1, &|a, b| a + b);
    let seq: f64 = (0..k).map(|_| 0.1).sum();
    let exact = 0.1 * k as f64;
    assert!((p - exact).abs() <= (seq - exact).abs());
    assert!((p - exact).abs() / exact <= 21.0 * 2f64.powi(-53));
    assert_eq!(pairwise(5, 5, 7u32, &|_| 0, &|a, b| a + b), 7);
    // ext_rel_diff
    let a = Ext::from_u64(1000);
    assert_eq!(ext_rel_diff(a, a), 0.0);
    assert!((ext_rel_diff(a, Ext::from_u64(999)) - 1e-3).abs() < 1e-15);
    assert_eq!(ext_rel_diff(Ext::ZERO, Ext::ZERO), 0.0);
    assert_eq!(ext_rel_diff(Ext::ZERO, a), 1.0);
    assert_eq!(ext_rel_diff(a.scale_pow2(5000), a), 1.0);
    let big = Ext::from_u64(3).scale_pow2(1 << 40);
    let big2 = Ext::from_f64(3.0 * (1.0 + 1e-9)).scale_pow2(1 << 40);
    assert!((ext_rel_diff(big, big2) - 1e-9).abs() < 1e-15);
    // histogram bins
    assert_eq!(hist_bin(0.0), 0);
    assert_eq!(hist_bin(1.0), 1);
    assert_eq!(hist_bin(1.99), 1);
    assert_eq!(hist_bin(2.0), 2);
    assert_eq!(hist_bin(3_559_224.0), 22);
    assert_eq!(hist_bin(f64::MAX), HIST_BINS - 1);
}

// ---------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------

fn read<T: Dump>(path: &Path) -> Vec<T> {
    io::read_array(path).unwrap()
}

#[test]
fn cli_count_paths_matches_the_library() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let g = mini_pg(300, 17, &MiniPgOpts::default()).graph;
    let n = g.num_nodes();
    let base = write_bv(root, "g", &g);
    let (b, _) = BlockIndex::from_vec_graph(&g).unwrap();
    let bprefix = root.join("g");
    b.save(&bprefix).unwrap();
    let inp = Inputs::new(g, 7);
    let w = Window {
        lo: n / 5,
        hi: n - n / 7,
    };
    let want = count_paths(
        &DpInput::with_transpose(&inp.cg, &inp.cgt),
        Which::Both,
        &ALL,
        &conf(w, Kernel::Auto, 2),
        &mut quiet(),
    )
    .unwrap();
    let ws = format!("{}..{}", w.lo, w.hi);

    for (name, blocks) in [
        ("generic", "none".to_owned()),
        ("blocks", p(&bprefix)),
        ("auto", "auto".to_owned()),
    ] {
        let out = root.join(format!("run-{name}"));
        run_ok(&[
            "--chunk-arcs",
            "6",
            "count-paths",
            "--graph",
            &p(&base),
            "--window",
            &ws,
            "--blocks",
            &blocks,
            "--kinds",
            "extfloat,mod61,sat64,sat128",
            "--through",
            "--out",
            &p(&out),
        ]);
        let f = |s: &str| io::suffixed(&out, s);
        assert_eq!(
            read::<f64>(&f("fwd.log2.f64le")),
            *want.fwd_log2.as_ref().unwrap(),
            "{name}"
        );
        assert_eq!(
            read::<u64>(&f("bwd.mod61.u64le")),
            *want.bwd_mod61.as_ref().unwrap(),
            "{name}"
        );
        assert_eq!(
            read::<u64>(&f("fwd.sat64.u64le")),
            *want.fwd_sat64.as_ref().unwrap(),
            "{name}"
        );
        assert_eq!(
            read::<u128>(&f("bwd.sat128.u128le")),
            *want.bwd_sat128.as_ref().unwrap(),
            "{name}"
        );
        let tl = read::<f64>(&f("through.log2.f64le"));
        let tm = read::<u64>(&f("through.mod61.u64le"));
        let m = M61::new();
        for i in 0..w.len() {
            assert_eq!(
                tl[i],
                want.fwd_log2.as_ref().unwrap()[i] + want.bwd_log2.as_ref().unwrap()[i]
            );
            assert_eq!(
                tm[i],
                m.times(
                    want.fwd_mod61.as_ref().unwrap()[i],
                    want.bwd_mod61.as_ref().unwrap()[i]
                )
            );
        }
        let j = read_json(&f("json"));
        let res = &j["results"];
        assert_eq!(res["ok"], true, "{name}");
        assert_eq!(res["summary"]["duality"]["ok"], true);
        assert_eq!(res["total_mod61"], want.total_mod61);
        // serde_json parses floats to within an ulp
        let tl = res["total_log2"].as_f64().unwrap();
        assert!(
            (tl - want.total_log2).abs() <= 1e-12 * want.total_log2.abs(),
            "{name}"
        );
        assert_eq!(res["summary"]["sources"], want.summary.sources);
        assert_eq!(j["files"].as_array().unwrap().len(), 10, "{name}");
        assert_eq!(j["params"]["blocks"], name == "blocks", "{name}");
        assert_eq!(j["params"]["graph_opened"], name != "blocks", "{name}");
    }

    // full graph with --blocks auto builds the index; fwd only; summary only
    let out = root.join("fwd");
    run_ok(&[
        "count-paths",
        "--graph",
        &p(&base),
        "--dir",
        "fwd",
        "--summary-only",
        "--out",
        &p(&out),
    ]);
    let j = read_json(&io::suffixed(&out, "json"));
    assert!(j["files"].as_array().unwrap().is_empty());
    assert!(!io::suffixed(&out, "fwd.log2.f64le").exists());
    assert_eq!(j["params"]["blocks"], true);
    assert!(j["results"]["summary"]["backward"].is_null());
    assert!(j["results"]["summary"]["duality"].is_null());
    let full = count_paths(
        &DpInput::graph(&inp.cg),
        Which::Forward,
        &[CountKind::M61],
        &conf(Window::full(n), Kernel::Auto, 2),
        &mut quiet(),
    )
    .unwrap();
    assert_eq!(j["results"]["total_mod61"], full.total_mod61);
}
