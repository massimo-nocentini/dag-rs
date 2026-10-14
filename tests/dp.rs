//! Acceptance tests of the DP kernels.
//!
//! Semirings: MaxPlus u32/u64/i64/f64, MinPlus u32/i64/f64, Boolean, a non-commutative 2×2
//! max-plus matrix semiring and a non-selective real (+, ×) semiring (`tests/support/dp.rs`).
//! Weights: `One`, `Const`, `Node` (random), `Arc` (random).

#[path = "support/dp.rs"]
mod support;

use dag::blocks::BlockIndex;
use dag::dp::certify::{RelErr, Tolerance, UNIT_ROUNDOFF, certify};
use dag::dp::path::path_to;
use dag::dp::pull::{backward_pull, forward_pull};
use dag::dp::push::forward_push;
use dag::dp::solve::{resolve_kernel, solve};
use dag::dp::{Arc, ArcWeights, Const, DpConf, DpInput, DpResult, Init, Kernel, Node, One};
use dag::graph::{self, ChunkConf, ChunkedGraph};
use dag::semiring::{Boolean, MaxPlus, MinPlus, Semiring};
use dag::testkit::*;
use dag::{DagError, Dir, NONE, Window};
use rand::RngExt;
use support::*;
use webgraph::prelude::*;

const SEEDS: u64 = 300;

/// The graphs of one seed: a random DAG and a mini PG (the latter also with its BlockIndex).
fn graphs(seed: u64) -> Vec<(VecGraph, Option<BlockIndex>)> {
    let mut r = rng(seed);
    let n = r.random_range(0..40);
    let p = r.random_range(0.0..0.3);
    let opts = MiniPgOpts {
        zero_output_prob: 0.1,
        ..Default::default()
    };
    let pg = mini_pg(r.random_range(1..30), seed, &opts).graph;
    let b = BlockIndex::from_vec_graph(&pg).unwrap().0;
    vec![(random_dag(n, p, seed), None), (pg, Some(b))]
}

/// Runs every kernel on `(g, gt, b)` and checks bit identity, certification and (`oracle`)
/// brute force.
#[allow(clippy::too_many_arguments)]
fn check_weights<S, W, G>(
    s: &S,
    g: &ChunkedGraph<G>,
    gt: &ChunkedGraph<G>,
    b: Option<&BlockIndex>,
    dir: Dir,
    w: &W,
    init: &Init<S::Elem>,
    c: &DpConf,
    what: &str,
) -> DpResult<S::Elem>
where
    S: TestSemiring,
    S::Elem: RelErr,
    W: ArcWeights<S>,
    G: RandomAccessGraph + Sync,
{
    let res = run_kernels(s, g, gt, b, dir, w, init, c);
    let (k0, r0) = &res[0];
    for (k, r) in &res[1..] {
        assert_bit_identical(r0, r, &format!("{what}: {k0:?} vs {k:?}"));
    }
    let on = if dir == Dir::Forward { gt } else { g };
    let rep = certify(s, on, dir, w, init, r0, Tolerance::Exact, &mut cquiet()).unwrap();
    assert!(rep.ok(), "{what}: certification failed: {rep:?}");
    assert_eq!(rep.checked as usize, c.window.len());
    res.into_iter().next().unwrap().1
}

/// The full kernel-equivalence matrix for one semiring and one seed.
fn kernel_matrix<S>(s: S, seed: u64)
where
    S: TestSemiring,
    S::Elem: RelErr,
{
    let mut r = rng(seed ^ 0xABCD);
    for (gi, (g, b)) in graphs(seed).into_iter().enumerate() {
        let n = g.num_nodes();
        let gt = transpose(&g);
        let k = r.random_range(1..8);
        let (cg, cgt) = (chunked(&g, k), chunked(&gt, k));
        let win = random_window(&mut r, n);
        let decoders = r.random_range(1..4);
        let alpha = node_elems(&s, n, seed * 3 + 1);
        let beta = node_elems(&s, n, seed * 5 + 2);
        let node = Node {
            alpha: |u: usize| alpha[u],
            beta: |v: usize| beta[v],
        };
        let arc = Arc(|u: usize, v: usize| s.elem(mix(seed ^ ((u as u64) << 32 | v as u64))));
        let unit = Const(s.unit().unwrap_or_else(|| s.elem(mix(seed))));
        // a BV round trip for every tenth seed
        let bv = (seed % 10 == 0).then(|| {
            let dir = tempfile::tempdir().unwrap();
            let base = write_bv(dir.path(), "g", &g);
            let conf = ChunkConf {
                chunk_arcs: k,
                max_chunk_nodes: 3,
            };
            let bg = graph::open(&base, conf).unwrap();
            let bgt = graph::open(&graph::transpose_basename(&base).unwrap(), conf).unwrap();
            (dir, bg, bgt)
        });
        for dir in [Dir::Forward, Dir::Backward] {
            let init = random_init(&s, win.len(), seed * 7 + dir as u64);
            let c = conf(win, Kernel::Auto, S::SELECTIVE, decoders);
            let what = |wn: &str| format!("{} seed {seed} graph {gi} {dir:?} {wn} {win}", s.name());
            let b = b.as_ref();
            let r1 = check_weights(&s, &cg, &cgt, b, dir, &One, &init, &c, &what("One"));
            let r2 = check_weights(&s, &cg, &cgt, b, dir, &unit, &init, &c, &what("Const"));
            let r3 = check_weights(&s, &cg, &cgt, b, dir, &node, &init, &c, &what("Node"));
            let r4 = check_weights(&s, &cg, &cgt, b, dir, &arc, &init, &c, &what("Arc"));
            if let Some((_, bg, bgt)) = &bv {
                let rb = check_weights(&s, bg, bgt, b, dir, &One, &init, &c, &what("One/BV"));
                assert_bit_identical(&r1, &rb, &what("One BV vs VecGraph"));
                let rb = check_weights(&s, bg, bgt, b, dir, &unit, &init, &c, &what("Const/BV"));
                assert_bit_identical(&r2, &rb, &what("Const BV vs VecGraph"));
                let rb = check_weights(&s, bg, bgt, b, dir, &node, &init, &c, &what("Node/BV"));
                assert_bit_identical(&r3, &rb, &what("Node BV vs VecGraph"));
                let rb = check_weights(&s, bg, bgt, b, dir, &arc, &init, &c, &what("Arc/BV"));
                assert_bit_identical(&r4, &rb, &what("Arc BV vs VecGraph"));
            }
        }
    }
}

macro_rules! kernel_tests {
    ($($name:ident: $s:expr;)*) => {$(
        #[test]
        fn $name() {
            for seed in 0..SEEDS {
                kernel_matrix($s, seed);
            }
        }
    )*};
}

kernel_tests! {
    kernels_agree_maxplus_u32: MaxPlus::<u32>::new();
    kernels_agree_maxplus_u64: MaxPlus::<u64>::new();
    kernels_agree_maxplus_i64: MaxPlus::<i64>::new();
    kernels_agree_maxplus_f64: MaxPlus::<f64>::new();
    kernels_agree_minplus_u32: MinPlus::<u32>::new();
    kernels_agree_minplus_i64: MinPlus::<i64>::new();
    kernels_agree_minplus_f64: MinPlus::<f64>::new();
    kernels_agree_boolean: Boolean::new();
    kernels_agree_noncommutative_matrix: MatMax;
    kernels_agree_real_sum_product: Real;
}

// ---------------------------------------------------------------------------------------------
// Brute force
// ---------------------------------------------------------------------------------------------

fn brute_force<S: TestSemiring>(s: S) {
    for seed in 0..SEEDS {
        let mut r = rng(seed ^ 0x5151);
        let n = r.random_range(0..=10);
        let g = random_dag(n, r.random_range(0.1..0.6), seed);
        let gt = transpose(&g);
        let (cg, cgt) = (chunked(&g, 2), chunked(&gt, 3));
        let win = random_window(&mut r, n);
        let alpha = node_elems(&s, n, seed + 11);
        let beta = node_elems(&s, n, seed + 13);
        let node = Node {
            alpha: |u: usize| alpha[u],
            beta: |v: usize| beta[v],
        };
        let arc = Arc(|u: usize, v: usize| s.elem(mix(seed ^ ((u as u64) << 32 | v as u64))));
        for dir in [Dir::Forward, Dir::Backward] {
            let init = random_init(&s, win.len(), seed * 3 + dir as u64);
            let c = conf(win, Kernel::Auto, false, 2);
            let check = |got: &[S::Elem], want: &[S::Elem], wn: &str| {
                for (i, (a, b)) in got.iter().zip(want).enumerate() {
                    assert!(
                        s.close(*a, *b),
                        "{} seed {seed} {dir:?} {wn}: node {}: kernel {a:?}, paths {b:?}",
                        s.name(),
                        i + win.lo
                    );
                }
            };
            macro_rules! one_weight {
                ($w:expr, $wn:expr) => {{
                    let res = run_kernels(&s, &cg, &cgt, None, dir, $w, &init, &c);
                    let want = match dir {
                        Dir::Forward => brute_forward(&s, &g, win, $w, &init),
                        Dir::Backward => brute_backward(&s, &g, win, $w, &init),
                    };
                    for (_, r) in &res {
                        check(&r.val, &want, $wn);
                    }
                }};
            }
            one_weight!(&One, "One");
            one_weight!(&node, "Node");
            one_weight!(&arc, "Arc");
        }
    }
}

#[test]
fn brute_force_paths_tropical() {
    brute_force(MaxPlus::<u32>::new());
    brute_force(MaxPlus::<i64>::new());
    brute_force(MaxPlus::<f64>::new());
    brute_force(MinPlus::<u32>::new());
    brute_force(MinPlus::<i64>::new());
    brute_force(MinPlus::<f64>::new());
}

#[test]
fn brute_force_paths_other() {
    brute_force(MaxPlus::<u64>::new());
    brute_force(Boolean::new());
    brute_force(MatMax);
    brute_force(Real);
}

#[test]
fn minplus_negative_weights_match_bellman_ford() {
    let s = MinPlus::<i64>::new();
    for seed in 0..SEEDS {
        let mut r = rng(seed);
        let n = r.random_range(1..60);
        let g = random_dag(n, r.random_range(0.0..0.2), seed);
        let wf = |u: usize, v: usize| (mix(seed ^ ((u as u64) << 32 | v as u64)) % 21) as i64 - 10;
        let sources: Vec<usize> = (0..n)
            .filter(|&v| transpose(&g).outdegree(v) == 0)
            .collect();
        let want = bellman_ford(&g, wf, &sources);
        let cg = chunked(&g, 5);
        let c = conf(Window::full(n), Kernel::Push, true, 3);
        let res = forward_push(&s, &cg, &Arc(wf), &Init::Sources(0), &c, &mut quiet()).unwrap();
        for (v, &d) in want.iter().enumerate() {
            assert_eq!(Some(res.val[v]), d, "seed {seed} node {v}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Duality
// ---------------------------------------------------------------------------------------------

/// `⊕_v F(v) ⊗ c(v) = ⊕_u b(u) ⊗ B(u)` with dense random initial terms.
fn duality<S: TestSemiring>(s: S) {
    for seed in 0..SEEDS {
        let mut r = rng(seed ^ 0xD0A1);
        let opts = MiniPgOpts::default();
        let g = if seed % 2 == 0 {
            random_dag(r.random_range(0..30), r.random_range(0.0..0.3), seed)
        } else {
            mini_pg(r.random_range(1..20), seed, &opts).graph
        };
        let n = g.num_nodes();
        let gt = transpose(&g);
        let (cg, cgt) = (chunked(&g, 3), chunked(&gt, 3));
        let win = random_window(&mut r, n);
        let bt: Vec<S::Elem> = (0..win.len())
            .map(|i| s.elem(mix(seed ^ (i as u64) << 9)))
            .collect();
        let ct: Vec<S::Elem> = (0..win.len())
            .map(|i| s.elem(mix(seed ^ (i as u64) << 17)))
            .collect();
        let alpha = node_elems(&s, n, seed + 1);
        let beta = node_elems(&s, n, seed + 2);
        let node = Node {
            alpha: |u: usize| alpha[u],
            beta: |v: usize| beta[v],
        };
        let arc = Arc(|u: usize, v: usize| s.elem(mix(seed ^ ((u as u64) << 32 | v as u64))));
        let c = conf(win, Kernel::Auto, false, 2);
        macro_rules! dual {
            ($w:expr) => {{
                let f = forward_pull(
                    &s,
                    &cgt,
                    $w,
                    &Init::Dense(bt.clone().into()),
                    &c,
                    &mut quiet(),
                )
                .unwrap();
                let b = backward_pull(
                    &s,
                    &cg,
                    $w,
                    &Init::Dense(ct.clone().into()),
                    &c,
                    &mut quiet(),
                )
                .unwrap();
                let mut lhs = s.zero();
                let mut rhs = s.zero();
                for i in 0..win.len() {
                    lhs = s.plus(lhs, s.times(f.val[i], ct[i]));
                    rhs = s.plus(rhs, s.times(bt[i], b.val[i]));
                }
                assert!(
                    s.close(lhs, rhs),
                    "{} seed {seed}: {lhs:?} vs {rhs:?}",
                    s.name()
                );
            }};
        }
        dual!(&One);
        dual!(&node);
        dual!(&arc);
    }
}

#[test]
fn duality_holds() {
    duality(MaxPlus::<i64>::new());
    duality(MinPlus::<f64>::new());
    duality(Boolean::new());
    duality(MatMax);
    duality(Real);
}

#[test]
fn matrix_semiring_is_not_commutative() {
    // otherwise the duality test above could not catch swapped operands
    let s = MatMax;
    let a = support::M2([[0, 1], [NEG_INF, 0]]);
    let b = support::M2([[0, NEG_INF], [2, 0]]);
    assert_ne!(s.times(a, b), s.times(b, a));
}

const NEG_INF: i64 = i64::MIN;

// ---------------------------------------------------------------------------------------------
// Specific properties
// ---------------------------------------------------------------------------------------------

#[test]
fn zero_pushed_values_do_not_make_sources() {
    // every arc weighs 0̄, so every pushed value is 0̄: nodes with a window predecessor must
    // still get 0̄ (not the source term), in every kernel and direction
    let s = MaxPlus::<u32>::new();
    for seed in 0..50 {
        let pg = mini_pg(20, seed, &MiniPgOpts::default()).graph;
        let n = pg.num_nodes();
        let b = BlockIndex::from_vec_graph(&pg).unwrap().0;
        let gt = transpose(&pg);
        let (cg, cgt) = (chunked(&pg, 4), chunked(&gt, 4));
        let mut r = rng(seed);
        let win = random_window(&mut r, n);
        let succ = window_succ(&pg, win);
        let has_pred: Vec<bool> = (0..n)
            .map(|v| succ.iter().any(|s| s.contains(&v)))
            .collect();
        let c = conf(win, Kernel::Auto, true, 2);
        for dir in [Dir::Forward, Dir::Backward] {
            for (k, res) in run_kernels(
                &s,
                &cg,
                &cgt,
                Some(&b),
                dir,
                &Const(s.zero()),
                &Init::Sources(7),
                &c,
            ) {
                for v in win.lo..win.hi {
                    let has = match dir {
                        Dir::Forward => has_pred[v],
                        Dir::Backward => !succ[v].is_empty(),
                    };
                    let want = if has { s.zero() } else { 7 };
                    assert_eq!(
                        res.val[v - win.lo],
                        want,
                        "seed {seed} {dir:?} {k:?} node {v}"
                    );
                }
            }
        }
    }
}

#[test]
fn chain_unit_weights() {
    let s = MaxPlus::<u32>::new();
    let g = chain(5);
    let gt = transpose(&g);
    let (cg, cgt) = (chunked(&g, 1), chunked(&gt, 1));
    let b = BlockIndex::from_vec_graph(&g).unwrap().0;
    let c = conf(Window::full(5), Kernel::Auto, true, 2);
    for (k, r) in run_kernels(
        &s,
        &cg,
        &cgt,
        Some(&b),
        Dir::Forward,
        &Const(1u32),
        &Init::Sources(0),
        &c,
    ) {
        assert_eq!(r.val, [0, 1, 2, 3, 4], "{k:?}");
        assert_eq!(r.pred.unwrap(), [NONE, 0, 1, 2, 3], "{k:?}");
    }
    for (k, r) in run_kernels(
        &s,
        &cg,
        &cgt,
        Some(&b),
        Dir::Backward,
        &Const(1u32),
        &Init::Sources(0),
        &c,
    ) {
        assert_eq!(r.val, [4, 3, 2, 1, 0], "{k:?}");
        assert_eq!(r.pred.unwrap(), [1, 2, 3, 4, NONE], "{k:?}");
    }
}

#[test]
fn certify_rejects_perturbations() {
    let s = MaxPlus::<u32>::new();
    let mut rejected_values = 0;
    let mut rejected_preds = 0;
    let (mut non_neighbours, mut worse_neighbours) = (0, 0);
    for seed in 0..100 {
        let g = random_dag(30, 0.15, seed);
        let gt = transpose(&g);
        let (cg, cgt) = (chunked(&g, 3), chunked(&gt, 3));
        let c = conf(Window::full(30), Kernel::Auto, true, 2);
        for dir in [Dir::Forward, Dir::Backward] {
            let on = if dir == Dir::Forward { &cgt } else { &cg };
            let w = Const(1u32);
            let init = Init::Sources(0);
            let res = match dir {
                Dir::Forward => forward_push(&s, &cg, &w, &init, &c, &mut quiet()).unwrap(),
                Dir::Backward => backward_pull(&s, &cg, &w, &init, &c, &mut quiet()).unwrap(),
            };
            let ok = certify(
                &s,
                on,
                dir,
                &w,
                &init,
                &res,
                Tolerance::Exact,
                &mut cquiet(),
            )
            .unwrap();
            assert!(ok.ok() && ok.max_rel_err == 0.0, "{ok:?}");
            // one perturbed value
            let x = (seed as usize * 7) % 30;
            let mut bad = res.clone();
            bad.val[x] += 1;
            let rep = certify(
                &s,
                on,
                dir,
                &w,
                &init,
                &bad,
                Tolerance::Exact,
                &mut cquiet(),
            )
            .unwrap();
            assert!(rep.violations >= 1, "seed {seed} {dir:?}: {rep:?}");
            assert!(rep.first_violation.unwrap() <= x || dir == Dir::Forward);
            if dir == Dir::Forward {
                assert_eq!(rep.first_violation, Some(x));
            }
            assert!(rep.max_rel_err > 0.0);
            rejected_values += 1;
            // one perturbed predecessor: point a node to itself
            if let Some(y) = res.pred.as_ref().unwrap().iter().position(|&p| p != NONE) {
                let mut bad = res.clone();
                bad.pred.as_mut().unwrap()[y] = y as u32;
                let rep = certify(
                    &s,
                    on,
                    dir,
                    &w,
                    &init,
                    &bad,
                    Tolerance::Exact,
                    &mut cquiet(),
                )
                .unwrap();
                assert_eq!(rep.violations, 0);
                assert_eq!(rep.pred_violations, 1, "seed {seed} {dir:?}: {rep:?}");
                assert_eq!(rep.first_violation, Some(y));
                rejected_preds += 1;
                // NONE where a predecessor realises the value is also rejected
                let mut bad = res.clone();
                bad.pred.as_mut().unwrap()[y] = NONE;
                let rep = certify(
                    &s,
                    on,
                    dir,
                    &w,
                    &init,
                    &bad,
                    Tolerance::Exact,
                    &mut cquiet(),
                )
                .unwrap();
                assert_eq!(rep.pred_violations, 1, "seed {seed} {dir:?}: {rep:?}");
                let adj = if dir == Dir::Forward { &gt } else { &g };
                let [a, b] = pred_neighbour_cases(&s, on, adj, dir, &res, seed);
                non_neighbours += a;
                worse_neighbours += b;
            }
        }
    }
    assert!(rejected_values > 0 && rejected_preds > 0);
    assert!(
        non_neighbours > 0 && worse_neighbours > 0,
        "{non_neighbours} {worse_neighbours}"
    );
}

/// Corrupts `pred` (MaxPlus, `Const(1)`, `Init::Sources(0)`) in the two ways that only the
/// neighbour and argbest checks of `certify` detect: a non-neighbour whose term `val[x] + 1`
/// equals the stored value, and a neighbour whose term is strictly worse than the best one.
/// Returns how many corruptions of each kind were tried (each must be rejected).
fn pred_neighbour_cases(
    s: &MaxPlus<u32>,
    on: &ChunkedGraph<VecGraph>,
    adj: &VecGraph,
    dir: Dir,
    res: &DpResult<u32>,
    seed: u64,
) -> [usize; 2] {
    let (w, init) = (Const(1u32), Init::Sources(0));
    let n = res.val.len();
    let nbrs = |v: usize| -> Vec<usize> { adj.successors(v).collect() };
    let mut tried = [0; 2];
    for v in 0..n {
        let nv = nbrs(v);
        if nv.is_empty() {
            continue;
        }
        // a non-neighbour realising the value
        if let Some(x) =
            (0..n).find(|&x| x != v && !nv.contains(&x) && res.val[x] + 1 == res.val[v])
        {
            let mut bad = res.clone();
            bad.pred.as_mut().unwrap()[v] = x as u32;
            let rep =
                certify(s, on, dir, &w, &init, &bad, Tolerance::Exact, &mut cquiet()).unwrap();
            assert_eq!(rep.violations, 0);
            assert_eq!(
                rep.pred_violations, 1,
                "seed {seed} {dir:?} v {v} x {x}: {rep:?}"
            );
            assert_eq!(rep.first_violation, Some(v));
            tried[0] += 1;
        }
        // a strictly worse neighbour
        if let Some(&x) = nv.iter().find(|&&x| res.val[x] + 1 < res.val[v]) {
            let mut bad = res.clone();
            bad.pred.as_mut().unwrap()[v] = x as u32;
            let rep =
                certify(s, on, dir, &w, &init, &bad, Tolerance::Exact, &mut cquiet()).unwrap();
            assert_eq!(
                rep.pred_violations, 1,
                "seed {seed} {dir:?} v {v} x {x}: {rep:?}"
            );
            tried[1] += 1;
        }
    }
    tried
}

#[test]
fn certify_per_term_tolerance() {
    // the Real semiring: a value perturbed by one ulp passes PerTerm and fails Exact, a value
    // perturbed by 1e-6 fails both
    let s = Real;
    let g = random_dag(40, 0.2, 9);
    let gt = transpose(&g);
    let cgt = chunked(&gt, 3);
    let c = conf(Window::full(40), Kernel::Auto, false, 2);
    let init = Init::All(1.0);
    let res = forward_pull(&s, &cgt, &One, &init, &c, &mut quiet()).unwrap();
    let v = (0..40).rev().find(|&v| gt.outdegree(v) > 2).unwrap();
    let mut ulp = res.clone();
    ulp.val[v] = f64::from_bits(ulp.val[v].to_bits() + 1);
    let tol = Tolerance::DEFAULT_PER_TERM;
    let rep = certify(
        &s,
        &cgt,
        Dir::Forward,
        &One,
        &init,
        &ulp,
        tol,
        &mut cquiet(),
    )
    .unwrap();
    assert!(rep.ok() && rep.max_rel_err > 0.0, "{rep:?}");
    let rep = certify(
        &s,
        &cgt,
        Dir::Forward,
        &One,
        &init,
        &ulp,
        Tolerance::Exact,
        &mut cquiet(),
    )
    .unwrap();
    assert_eq!(rep.violations, 1);
    let mut far = res.clone();
    far.val[v] *= 1.0 + 1e-6;
    let rep = certify(
        &s,
        &cgt,
        Dir::Forward,
        &One,
        &init,
        &far,
        tol,
        &mut cquiet(),
    )
    .unwrap();
    assert_eq!(rep.violations, 1, "{rep:?}");
}

#[test]
fn certify_per_term_bound_grows_with_the_number_of_terms() {
    // node 39 of the complete DAG has k = 39 terms: its PerTerm bound is c·(k + 3)·u (k terms,
    // the init term and the outer product), well above the k = 0 bound c·2·u
    let s = Real;
    let n = 40;
    let g = complete_dag(n);
    let gt = transpose(&g);
    let cgt = chunked(&gt, 7);
    let c = conf(Window::full(n), Kernel::Auto, false, 2);
    let init = Init::All(1.0);
    let res = forward_pull(&s, &cgt, &One, &init, &c, &mut quiet()).unwrap();
    let v = n - 1;
    let k = gt.outdegree(v);
    assert_eq!(k, n - 1);
    let Tolerance::PerTerm(cc) = Tolerance::DEFAULT_PER_TERM else {
        unreachable!()
    };
    let bound = cc * (k as f64 + 3.0) * UNIT_ROUNDOFF;
    assert!(bound > 10.0 * cc * 2.0 * UNIT_ROUNDOFF);
    let at = |rel: f64| {
        let mut p = res.clone();
        p.val[v] *= 1.0 + rel;
        certify(
            &s,
            &cgt,
            Dir::Forward,
            &One,
            &init,
            &p,
            Tolerance::DEFAULT_PER_TERM,
            &mut cquiet(),
        )
        .unwrap()
    };
    let inside = at(0.5 * bound);
    assert!(
        inside.ok() && inside.max_rel_err > 4.0 * cc * UNIT_ROUNDOFF,
        "{inside:?}"
    );
    let outside = at(1.5 * bound);
    assert_eq!(outside.violations, 1, "{outside:?}");
    assert_eq!(outside.first_violation, Some(v));
}

/// `(Real, counting mod 2⁶¹ − 1)` componentwise: a product with a rounded and an exact
/// component, the shape of the `Product<ExtFloat, M61>` count-paths instance.
#[derive(Clone, Copy, Debug, Default)]
struct RealMod;

const P61: u64 = (1 << 61) - 1;

impl Semiring for RealMod {
    type Elem = (f64, u64);
    fn name(&self) -> &'static str {
        "real×mod"
    }
    fn zero(&self) -> (f64, u64) {
        (0.0, 0)
    }
    fn one(&self) -> (f64, u64) {
        (1.0, 1)
    }
    fn plus(&self, a: (f64, u64), b: (f64, u64)) -> (f64, u64) {
        (Real.plus(a.0, b.0), (a.1 + b.1) % P61)
    }
    fn times(&self, a: (f64, u64), b: (f64, u64)) -> (f64, u64) {
        (
            Real.times(a.0, b.0),
            (a.1 as u128 * b.1 as u128 % P61 as u128) as u64,
        )
    }
}

#[test]
fn exact_components_are_never_within_a_per_term_tolerance() {
    // integers above 2⁵³ that round to the same f64
    let (a, b) = (P61 - 1, P61 - 2);
    assert_eq!(a as f64, b as f64);
    assert!(a.rel_err(&b) > 0.0 && !a.within(&b, 1.0));
    assert!(a.within(&a, 0.0));
    assert!((u128::MAX).rel_err(&(u128::MAX - 1)) > 0.0);
    assert!(i64::MIN.rel_err(&(i64::MIN + 1)) > 0.0 && !i64::MIN.within(&i64::MAX, 2.0));
    assert!(!true.within(&false, 1.0));
    assert!(
        !(1.0f64, 5u64).within(&(1.0, 6), 1.0) && (1.0f64, 5u64).within(&(1.0 + 1e-12, 5), 1e-9)
    );
    // certify: a perturbed exact component is a violation under PerTerm, a 1-ulp float one is not
    let s = RealMod;
    let g = chain(3);
    let gt = transpose(&g);
    let cgt = chunked(&gt, 2);
    let c = conf(Window::full(3), Kernel::Auto, false, 2);
    let init = Init::All((1.0, 1u64 << 59));
    let res = forward_pull(&s, &cgt, &One, &init, &c, &mut quiet()).unwrap();
    let tol = Tolerance::DEFAULT_PER_TERM;
    let ok = certify(
        &s,
        &cgt,
        Dir::Forward,
        &One,
        &init,
        &res,
        tol,
        &mut cquiet(),
    )
    .unwrap();
    assert!(ok.ok() && ok.max_rel_err == 0.0, "{ok:?}");
    let mut bad = res.clone();
    bad.val[2].1 = (bad.val[2].1 + 7) % P61;
    assert!(bad.val[2].1 > 1 << 53);
    let rep = certify(
        &s,
        &cgt,
        Dir::Forward,
        &One,
        &init,
        &bad,
        tol,
        &mut cquiet(),
    )
    .unwrap();
    assert_eq!(rep.violations, 1, "{rep:?}");
    assert!(rep.max_rel_err > 0.0, "{rep:?}");
    let mut ulp = res.clone();
    ulp.val[2].0 = f64::from_bits(ulp.val[2].0.to_bits() + 1);
    let rep = certify(
        &s,
        &cgt,
        Dir::Forward,
        &One,
        &init,
        &ulp,
        tol,
        &mut cquiet(),
    )
    .unwrap();
    assert!(rep.ok() && rep.max_rel_err > 0.0, "{rep:?}");
}

#[test]
fn pred_paths_are_optimal_with_the_analysis_path_tie_break() {
    let s = MaxPlus::<u32>::new();
    for seed in 0..SEEDS {
        let mut r = rng(seed);
        let n = r.random_range(1..60);
        let g = if seed % 2 == 0 {
            random_dag(n, r.random_range(0.0..0.3), seed)
        } else {
            mini_pg(n / 2 + 1, seed, &MiniPgOpts::default()).graph
        };
        let n = g.num_nodes();
        let gt = transpose(&g);
        let (cg, cgt) = (chunked(&g, 4), chunked(&gt, 4));
        let b = BlockIndex::from_vec_graph(&g).ok().map(|x| x.0);
        let win = random_window(&mut r, n);
        let c = conf(win, Kernel::Auto, true, 2);
        let res = run_kernels(
            &s,
            &cg,
            &cgt,
            b.as_ref(),
            Dir::Forward,
            &Const(1u32),
            &Init::Sources(0),
            &c,
        );
        let (_, r0) = &res[0];
        let pred = r0.pred.as_ref().unwrap();
        for v in win.lo..win.hi {
            let preds: Vec<usize> = gt.successors(v).filter(|&u| win.contains(u)).collect();
            // analysis_path: strict >, ascending u: the smallest u with the largest d(u)
            let want = preds
                .iter()
                .copied()
                .max_by(|&a, &b| r0.val[a - win.lo].cmp(&r0.val[b - win.lo]).then(b.cmp(&a)))
                .map_or(NONE, |u| u as u32);
            assert_eq!(pred[v - win.lo], want, "seed {seed} node {v}");
            // the path is a real path of the window, from a window source, of length val[v]
            let mut path = path_to(pred, win, v);
            path.reverse();
            assert_eq!(path.len() as u32 - 1, r0.val[v - win.lo]);
            assert!(
                gt.successors(path[0] as usize)
                    .into_iter()
                    .all(|u| !win.contains(u))
            );
            for a in path.windows(2) {
                assert!(
                    g.successors(a[0] as usize)
                        .into_iter()
                        .any(|x| x == a[1] as usize)
                );
            }
        }
    }
}

#[test]
fn backward_pred_is_the_smallest_argbest_successor() {
    let s = MinPlus::<i64>::new();
    for seed in 0..100 {
        let pg = mini_pg(25, seed, &MiniPgOpts::default()).graph;
        let n = pg.num_nodes();
        let b = BlockIndex::from_vec_graph(&pg).unwrap().0;
        let gt = transpose(&pg);
        let (cg, cgt) = (chunked(&pg, 3), chunked(&gt, 3));
        let beta = node_elems(&s, n, seed);
        let w = Node {
            alpha: |_: usize| 0i64,
            beta: |v: usize| beta[v],
        };
        let c = conf(Window::full(n), Kernel::Auto, true, 2);
        let res = run_kernels(
            &s,
            &cg,
            &cgt,
            Some(&b),
            Dir::Backward,
            &w,
            &Init::Sources(0),
            &c,
        );
        let r0 = &res[0].1;
        for u in 0..n {
            let terms: Vec<(usize, i64)> = pg
                .successors(u)
                .map(|v| (v, s.times(beta[v], r0.val[v])))
                .collect();
            let want = terms
                .iter()
                .min_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)))
                .map_or(NONE, |&(v, _)| v as u32);
            for (k, r) in &res {
                assert_eq!(
                    r.pred.as_ref().unwrap()[u],
                    want,
                    "seed {seed} {k:?} node {u}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Windows and edge cases
// ---------------------------------------------------------------------------------------------

#[test]
fn windowed_runs_equal_runs_on_the_induced_subgraph() {
    let s = MaxPlus::<i64>::new();
    for seed in 0..SEEDS {
        for (g, b) in graphs(seed) {
            let n = g.num_nodes();
            let mut r = rng(seed ^ 0x77);
            let win = random_window(&mut r, n);
            let h = induced(&g, win);
            let (gt, ht) = (transpose(&g), transpose(&h));
            let (cg, cgt, ch, cht) = (
                chunked(&g, 3),
                chunked(&gt, 2),
                chunked(&h, 5),
                chunked(&ht, 1),
            );
            let hb = b
                .as_ref()
                .map(|_| BlockIndex::from_vec_graph(&h).unwrap().0);
            let alpha = node_elems(&s, n, seed);
            let beta = node_elems(&s, n, seed + 1);
            let wg = Node {
                alpha: |u: usize| alpha[u],
                beta: |v: usize| beta[v],
            };
            let wh = Node {
                alpha: |u: usize| alpha[u + win.lo],
                beta: |v: usize| beta[v + win.lo],
            };
            for dir in [Dir::Forward, Dir::Backward] {
                for init in [Init::Sources(3i64), random_init(&s, win.len(), seed + 2)] {
                    let cw = conf(win, Kernel::Auto, true, 2);
                    let cf = conf(Window::full(win.len()), Kernel::Auto, true, 3);
                    let rw = run_kernels(&s, &cg, &cgt, b.as_ref(), dir, &wg, &init, &cw);
                    let rh = run_kernels(&s, &ch, &cht, hb.as_ref(), dir, &wh, &init, &cf);
                    for ((k, a), (_, z)) in rw.iter().zip(&rh) {
                        let shifted = DpResult {
                            window: win,
                            val: z.val.clone(),
                            pred: z.pred.as_ref().map(|p| shift_pred(p, win.lo)),
                        };
                        assert_bit_identical(
                            a,
                            &shifted,
                            &format!("seed {seed} {dir:?} {k:?} {win}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn edge_cases() {
    let s = MaxPlus::<u32>::new();
    for g in [
        vec_graph(0, []),
        vec_graph(1, []),
        isolated(5),
        out_star(6),
        in_star(6),
        complete_dag(7),
    ] {
        let n = g.num_nodes();
        let gt = transpose(&g);
        let (cg, cgt) = (chunked(&g, 1), chunked(&gt, 1));
        let b = BlockIndex::from_vec_graph(&g).ok().map(|x| x.0);
        let windows = [
            Window::full(n),
            Window {
                lo: n / 2,
                hi: n / 2,
            },
            Window { lo: n / 3, hi: n },
        ];
        for win in windows {
            for dir in [Dir::Forward, Dir::Backward] {
                let c = conf(win, Kernel::Auto, true, 2);
                let res = run_kernels(
                    &s,
                    &cg,
                    &cgt,
                    b.as_ref(),
                    dir,
                    &Const(1u32),
                    &Init::Sources(0),
                    &c,
                );
                let want = match dir {
                    Dir::Forward => brute_forward(&s, &g, win, &Const(1u32), &Init::Sources(0)),
                    Dir::Backward => brute_backward(&s, &g, win, &Const(1u32), &Init::Sources(0)),
                };
                for (k, r) in &res {
                    assert_eq!(r.val, want, "n {n} {win} {dir:?} {k:?}");
                    assert_eq!(r.val.len(), win.len());
                }
            }
        }
    }
    // isolated nodes are sources and sinks: Init::Sources(e) gives e everywhere
    let g = isolated(4);
    let cg = chunked(&g, 1);
    let r = solve(
        &s,
        &DpInput::graph(&cg),
        Dir::Forward,
        &Const(1u32),
        &Init::Sources(9),
        &conf(Window::full(4), Kernel::Auto, false, 1),
        &mut quiet(),
    )
    .unwrap();
    assert_eq!(r.val, [9; 4]);
}

#[test]
fn kernels_are_identical_across_decoders_and_chunk_sizes() {
    let s = Real;
    let pg = mini_pg(200, 3, &MiniPgOpts::default()).graph;
    let n = pg.num_nodes();
    let gt = transpose(&pg);
    let alpha = node_elems(&s, n, 1);
    let w = Node {
        alpha: |u: usize| alpha[u],
        beta: |_: usize| 1.0,
    };
    let base = forward_push(
        &s,
        &chunked(&pg, 1),
        &w,
        &Init::Sources(1.0),
        &conf(Window::full(n), Kernel::Push, false, 1),
        &mut quiet(),
    )
    .unwrap();
    for (k, d) in [(1, 7), (5, 2), (64, 16), (1 << 20, 4)] {
        let c = conf(Window::full(n), Kernel::Auto, false, d);
        for (kk, r) in run_kernels(
            &s,
            &chunked(&pg, k),
            &chunked(&gt, k),
            None,
            Dir::Forward,
            &w,
            &Init::Sources(1.0),
            &c,
        ) {
            assert_bit_identical(&base, &r, &format!("chunk {k} decoders {d} {kk:?}"));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Errors and kernel resolution
// ---------------------------------------------------------------------------------------------

fn dag_err(e: anyhow::Error) -> DagError {
    match e.downcast::<DagError>() {
        Ok(d) => d,
        Err(e) => panic!("not a DagError: {e:#}"),
    }
}

#[test]
fn errors() {
    let s = MaxPlus::<u32>::new();
    let g = random_dag(20, 0.2, 1);
    let bad = with_back_arc(&g);
    let (cbad, cbadt) = (chunked(&bad, 3), chunked(&transpose(&bad), 3));
    let c = conf(Window::full(20), Kernel::Auto, false, 2);
    let w = Const(1u32);
    let init = Init::Sources(0);
    assert!(matches!(
        dag_err(forward_push(&s, &cbad, &w, &init, &c, &mut quiet()).unwrap_err()),
        DagError::NotTopological { .. }
    ));
    assert!(matches!(
        dag_err(forward_pull(&s, &cbadt, &w, &init, &c, &mut quiet()).unwrap_err()),
        DagError::NotTopological { .. }
    ));
    assert!(matches!(
        dag_err(backward_pull(&s, &cbad, &w, &init, &c, &mut quiet()).unwrap_err()),
        DagError::NotTopological { .. }
    ));
    let cg = chunked(&g, 3);
    let gt = transpose(&g);
    let cgt = chunked(&gt, 3);
    // window out of range
    let cw = conf(Window { lo: 0, hi: 21 }, Kernel::Auto, false, 2);
    assert!(matches!(
        dag_err(forward_push(&s, &cg, &w, &init, &cw, &mut quiet()).unwrap_err()),
        DagError::BadWindow { .. }
    ));
    // dense init of the wrong length
    assert!(
        forward_push(
            &s,
            &cg,
            &w,
            &Init::Dense(vec![0; 3].into()),
            &c,
            &mut quiet()
        )
        .is_err()
    );
    // predecessors of a non-selective semiring
    let cp = conf(Window::full(20), Kernel::Auto, true, 2);
    assert!(forward_push(&Real, &cg, &One, &Init::Sources(1.0), &cp, &mut quiet()).is_err());
    // blocks kernels need separable weights, a BlockIndex, and a biclique graph
    let pg = mini_pg(10, 1, &MiniPgOpts::default()).graph;
    let b = BlockIndex::from_vec_graph(&pg).unwrap().0;
    let cpg = chunked(&pg, 2);
    let cpw = conf(Window::full(pg.num_nodes()), Kernel::Blocks, false, 2);
    let arc = Arc(|_: usize, _: usize| 1u32);
    assert!(dag::dp::blocks::forward_blocks(&s, &b, &arc, &init, &cpw, &mut quiet()).is_err());
    let input = DpInput::graph(&cpg).and_blocks(&b);
    assert!(solve(&s, &input, Dir::Forward, &arc, &init, &cpw, &mut quiet()).is_err());
    assert!(
        solve(
            &s,
            &DpInput::graph(&cpg),
            Dir::Forward,
            &w,
            &init,
            &cpw,
            &mut quiet()
        )
        .is_err()
    );
    // resolution
    let both = DpInput::with_transpose(&cg, &cgt);
    assert_eq!(
        resolve_kernel(&DpInput::graph(&cg), Dir::Forward, true, Kernel::Auto).unwrap(),
        Kernel::Push
    );
    assert_eq!(
        resolve_kernel(&both, Dir::Forward, true, Kernel::Auto).unwrap(),
        Kernel::Pull
    );
    assert_eq!(
        resolve_kernel(&both, Dir::Backward, true, Kernel::Auto).unwrap(),
        Kernel::Pull
    );
    assert_eq!(
        resolve_kernel(&input, Dir::Forward, true, Kernel::Auto).unwrap(),
        Kernel::Blocks
    );
    assert_eq!(
        resolve_kernel(&input, Dir::Backward, true, Kernel::Auto).unwrap(),
        Kernel::Blocks
    );
    assert_eq!(
        resolve_kernel(&input, Dir::Forward, false, Kernel::Auto).unwrap(),
        Kernel::Push
    );
    assert!(resolve_kernel(&DpInput::graph(&cg), Dir::Forward, true, Kernel::Pull).is_err());
    assert!(resolve_kernel(&both, Dir::Backward, true, Kernel::Push).is_err());
    assert!(resolve_kernel(&both, Dir::Forward, true, Kernel::Blocks).is_err());
    // solve dispatches and agrees with the direct kernels
    for (k, inp) in [
        (Kernel::Push, &both),
        (Kernel::Pull, &both),
        (Kernel::Blocks, &input),
    ] {
        let n = inp.n;
        let cc = conf(Window::full(n), k, true, 2);
        let r = solve(&s, inp, Dir::Forward, &w, &init, &cc, &mut quiet()).unwrap();
        let direct = match k {
            Kernel::Blocks => {
                dag::dp::blocks::forward_blocks(&s, &b, &w, &init, &cc, &mut quiet()).unwrap()
            }
            _ => forward_push(&s, &cg, &w, &init, &cc, &mut quiet()).unwrap(),
        };
        assert_bit_identical(&direct, &r, &format!("solve {k:?}"));
    }
}
