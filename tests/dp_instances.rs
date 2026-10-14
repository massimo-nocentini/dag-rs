//! The DP kernel-equivalence, certification, brute-force and duality matrix of `tests/dp.rs`,
//! re-run for **every** semiring instance of `dag::semiring`.
//!
//! Instances: `SatU64`, `SatU128`, `M61`, `Goldilocks`, `ExtFloat`, `Log2Sum`, `Probability`,
//! `Viterbi`, `MaxMin<u32|u64|i64|f64>`, `Product<ExtFloat, M61>`, `Product<M61, Goldilocks>`,
//! `Product<MaxPlus<u32>, SatU64>`, `Matrix2<MinPlus<i64>>` and `Matrix2<M61>` (both
//! non-commutative). Weights: `One`, `Const`, `Node` (random), `Arc` (random) and, for the
//! probability-like instances, `inv-outdeg` (`Node { alpha: 1/outdeg(u), beta: 1̄ }`).
//!
//! Contract checked (D10): forward push == pull == blocks and backward pull == blocks are
//! bit-identical (values and, for selective instances, predecessors) for `One`, `Const`,
//! `Node` and `inv-outdeg`; push == pull for `Arc`; `certify` accepts every result (exactly, or
//! within the per-term bound for non-selective float instances); the values agree with
//! brute-force path enumeration; and `⊕_v F(v)⊗c(v) = ⊕_u b(u)⊗B(u)`.

#[path = "support/dp.rs"]
mod support;

use dag::blocks::BlockIndex;
use dag::dp::certify::{RelErr, Tolerance, certify};
use dag::dp::pull::{backward_pull, forward_pull};
use dag::dp::{Arc, ArcWeights, Const, DpResult, Init, Kernel, Node, One};
use dag::graph::{self, ChunkConf, ChunkedGraph};
use dag::semiring::bottleneck::MaxMin;
use dag::semiring::counting::{SatU64, SatU128};
use dag::semiring::extfloat::{Ext, ExtFloat};
use dag::semiring::logsum::Log2Sum;
use dag::semiring::matrix::{Mat2, Matrix2};
use dag::semiring::modular::{GOLDILOCKS_P, Goldilocks, M61, M61_P};
use dag::semiring::prob::{Probability, Viterbi};
use dag::semiring::product::Product;
use dag::semiring::{MaxPlus, MinPlus, Semiring};
use dag::testkit::*;
use dag::{Dir, Window};
use rand::RngExt;
use support::*;
use webgraph::prelude::*;

const SEEDS: u64 = 200;

// ---------------------------------------------------------------------------------------------
// Instances under test
// ---------------------------------------------------------------------------------------------

/// What the matrix needs to know about an instance besides [`TestSemiring`].
trait Inst: TestSemiring<Elem: RelErr> {
    /// The certification tolerance: exact for exact and selective instances, per-term for
    /// non-selective float instances.
    const TOL: Tolerance = Tolerance::Exact;
    /// `1/d` as an element, for the `inv-outdeg` weights (`None` if not meaningful).
    fn inv(&self, _d: usize) -> Option<Self::Elem> {
        None
    }
}

/// A dyadic rational in `[1/8, 2]` (products and sums of few of them are exact in `f64`).
fn dyadic(key: u64) -> f64 {
    ((key % 16) + 1) as f64 / 8.0
}

fn close_rel(a: f64, b: f64, tol: f64) -> bool {
    a == b || (a - b).abs() <= tol * a.abs().max(b.abs())
}

impl TestSemiring for SatU64 {
    fn elem(&self, key: u64) -> u64 {
        match key % 16 {
            0 => 0,
            1 => 1,
            2 => u64::MAX,
            3 => 1 << 40,
            _ => (key >> 4) % 10,
        }
    }
}
impl Inst for SatU64 {}

impl TestSemiring for SatU128 {
    fn elem(&self, key: u64) -> u128 {
        match key % 16 {
            0 => 0,
            1 => 1,
            2 => u128::MAX,
            3 => 1 << 90,
            _ => ((key >> 4) % 10) as u128,
        }
    }
}
impl Inst for SatU128 {}

impl TestSemiring for M61 {
    fn elem(&self, key: u64) -> u64 {
        match key % 16 {
            0 => 0,
            1 => 1,
            2 => M61_P - 1,
            _ => mix(key) % M61_P,
        }
    }
}
impl Inst for M61 {}

impl TestSemiring for Goldilocks {
    fn elem(&self, key: u64) -> u64 {
        match key % 16 {
            0 => 0,
            1 => 1,
            2 => GOLDILOCKS_P - 1,
            3 => GOLDILOCKS_P - (1 << 32),
            _ => mix(key) % GOLDILOCKS_P,
        }
    }
}
impl Inst for Goldilocks {}

impl TestSemiring for ExtFloat {
    fn elem(&self, key: u64) -> Ext {
        match key % 16 {
            0 => Ext::ZERO,
            1 => Ext::ONE,
            // far outside the f64 range
            2 => Ext::from_f64(dyadic(key >> 4)).scale_pow2(5000),
            3 => Ext::from_f64(dyadic(key >> 4)).scale_pow2(-5000),
            _ => Ext::from_f64(dyadic(key >> 4)).scale_pow2(((key >> 8) % 41) as i64 - 20),
        }
    }
    fn close(&self, a: Ext, b: Ext) -> bool {
        a.rel_err(&b) <= 1e-9
    }
}
impl Inst for ExtFloat {
    const TOL: Tolerance = Tolerance::DEFAULT_PER_TERM;
    fn inv(&self, d: usize) -> Option<Ext> {
        Some(Ext::from_f64(1.0 / d as f64))
    }
}

impl TestSemiring for Log2Sum {
    fn elem(&self, key: u64) -> f64 {
        match key % 16 {
            0 => f64::NEG_INFINITY,
            1 => 0.0,
            _ => ((key >> 4) % 41) as f64 / 4.0 - 5.0,
        }
    }
    fn close(&self, a: f64, b: f64) -> bool {
        a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
    }
}
impl Inst for Log2Sum {
    const TOL: Tolerance = Tolerance::DEFAULT_PER_TERM;
    fn inv(&self, d: usize) -> Option<f64> {
        Some(-(d as f64).log2())
    }
}

impl TestSemiring for Probability {
    fn elem(&self, key: u64) -> f64 {
        match key % 16 {
            0 => 0.0,
            1 => 1.0,
            _ => dyadic(key >> 4),
        }
    }
    fn close(&self, a: f64, b: f64) -> bool {
        close_rel(a, b, 1e-9)
    }
}
impl Inst for Probability {
    const TOL: Tolerance = Tolerance::DEFAULT_PER_TERM;
    fn inv(&self, d: usize) -> Option<f64> {
        Some(1.0 / d as f64)
    }
}

impl TestSemiring for Viterbi {
    fn elem(&self, key: u64) -> f64 {
        Probability.elem(key)
    }
    fn close(&self, a: f64, b: f64) -> bool {
        close_rel(a, b, 1e-9)
    }
}
impl Inst for Viterbi {
    fn inv(&self, d: usize) -> Option<f64> {
        Some(1.0 / d as f64)
    }
}

macro_rules! maxmin_int {
    ($t:ty) => {
        impl TestSemiring for MaxMin<$t> {
            fn elem(&self, key: u64) -> $t {
                match key % 16 {
                    0 => self.zero(),
                    1 => self.one(),
                    _ => ((key >> 4) % 10) as $t,
                }
            }
        }
        impl Inst for MaxMin<$t> {}
    };
}
maxmin_int!(u32);
maxmin_int!(u64);
maxmin_int!(i64);

impl TestSemiring for MaxMin<f64> {
    fn elem(&self, key: u64) -> f64 {
        match key % 16 {
            0 => self.zero(),
            1 => self.one(),
            _ => ((key >> 4) % 21) as f64 / 2.0 - 5.0,
        }
    }
}
impl Inst for MaxMin<f64> {}

impl<A: TestSemiring, B: TestSemiring> TestSemiring for Product<A, B> {
    fn elem(&self, key: u64) -> Self::Elem {
        // keep 0̄ and 1̄ reachable as whole pairs
        match key % 16 {
            0 => self.zero(),
            1 => self.one(),
            _ => (self.0.elem(key), self.1.elem(mix(key))),
        }
    }
    fn close(&self, a: Self::Elem, b: Self::Elem) -> bool {
        self.0.close(a.0, b.0) && self.1.close(a.1, b.1)
    }
}
impl Inst for Product<ExtFloat, M61> {
    const TOL: Tolerance = Tolerance::DEFAULT_PER_TERM;
}
impl Inst for Product<M61, Goldilocks> {}
impl Inst for Product<MaxPlus<u32>, SatU64> {}

impl<S: TestSemiring> TestSemiring for Matrix2<S>
where
    Mat2<S::Elem>: dag::io::Dump,
{
    fn elem(&self, key: u64) -> Mat2<S::Elem> {
        match key % 16 {
            0 => self.zero(),
            1 => self.one(),
            _ => Mat2(core::array::from_fn(|i| {
                core::array::from_fn(|j| self.0.elem(mix(key ^ (2 * i + j) as u64)))
            })),
        }
    }
}

// `MinPlus<i64>` elements come from `tests/support/dp.rs` (small integers, +∞ now and then)
impl Inst for Matrix2<MinPlus<i64>> {}
impl Inst for Matrix2<M61> {}

// ---------------------------------------------------------------------------------------------
// The matrix
// ---------------------------------------------------------------------------------------------

/// A random DAG and a mini PG with its BlockIndex.
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

/// Runs every kernel, checks bit identity across kernels and certification; returns the
/// first result.
#[allow(clippy::too_many_arguments)]
fn check_weights<S, W, G>(
    s: &S,
    g: &ChunkedGraph<G>,
    gt: &ChunkedGraph<G>,
    b: Option<&BlockIndex>,
    dir: Dir,
    w: &W,
    init: &Init<S::Elem>,
    c: &dag::dp::DpConf,
    what: &str,
) -> DpResult<S::Elem>
where
    S: Inst,
    W: ArcWeights<S>,
    G: RandomAccessGraph + Sync,
{
    let res = run_kernels(s, g, gt, b, dir, w, init, c);
    let expected = match (dir, b.is_some() && W::SEPARABLE) {
        (Dir::Forward, true) => 3,
        (Dir::Forward, false) | (Dir::Backward, true) => 2,
        (Dir::Backward, false) => 1,
    };
    assert_eq!(res.len(), expected, "{what}: kernels run");
    let (k0, r0) = &res[0];
    for (k, r) in &res[1..] {
        assert_bit_identical(r0, r, &format!("{what}: {k0:?} vs {k:?}"));
    }
    let on = if dir == Dir::Forward { gt } else { g };
    let rep = certify(s, on, dir, w, init, r0, S::TOL, &mut cquiet()).unwrap();
    assert!(rep.ok(), "{what}: certification failed: {rep:?}");
    assert_eq!(rep.checked as usize, c.window.len());
    res.into_iter().next().unwrap().1
}

/// The kernel-equivalence matrix of one instance and seed.
fn kernel_matrix<S: Inst>(s: S, seed: u64) {
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
        // inv-outdeg over the whole graph (as `--weights inv-outdeg`), 1̄ on sinks
        let inv: Option<Vec<S::Elem>> = s.inv(1).map(|_| {
            (0..n)
                .map(|u| match g.outdegree(u) {
                    0 => s.one(),
                    d => s.inv(d).unwrap(),
                })
                .collect()
        });
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
            if let Some(inv) = &inv {
                let w = Node {
                    alpha: |u: usize| inv[u],
                    beta: |_: usize| s.one(),
                };
                check_weights(&s, &cg, &cgt, b, dir, &w, &init, &c, &what("inv-outdeg"));
            }
            if let Some((_, bg, bgt)) = &bv {
                for (wn, rv) in [
                    (
                        "One",
                        check_weights(&s, bg, bgt, b, dir, &One, &init, &c, &what("One/BV")),
                    ),
                    (
                        "Const",
                        check_weights(&s, bg, bgt, b, dir, &unit, &init, &c, &what("Const/BV")),
                    ),
                    (
                        "Node",
                        check_weights(&s, bg, bgt, b, dir, &node, &init, &c, &what("Node/BV")),
                    ),
                    (
                        "Arc",
                        check_weights(&s, bg, bgt, b, dir, &arc, &init, &c, &what("Arc/BV")),
                    ),
                ] {
                    let base = match wn {
                        "One" => &r1,
                        "Const" => &r2,
                        "Node" => &r3,
                        _ => &r4,
                    };
                    assert_bit_identical(base, &rv, &what(&format!("{wn} BV vs VecGraph")));
                }
            }
        }
    }
}

/// Brute-force path enumeration (`n ≤ 10`) against every kernel.
fn brute_force<S: Inst>(s: S) {
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
        // every Init variant (`random_init` picks it by `seed % 3`) in both directions
        for (dir, variant) in [Dir::Forward, Dir::Backward]
            .into_iter()
            .flat_map(|d| (0..3u64).map(move |v| (d, v)))
        {
            let init = random_init(&s, win.len(), 3 * (2 * seed + dir as u64) + variant);
            let c = conf(win, Kernel::Auto, false, 2);
            macro_rules! one_weight {
                ($w:expr, $wn:expr) => {{
                    let want = match dir {
                        Dir::Forward => brute_forward(&s, &g, win, $w, &init),
                        Dir::Backward => brute_backward(&s, &g, win, $w, &init),
                    };
                    for (k, res) in run_kernels(&s, &cg, &cgt, None, dir, $w, &init, &c) {
                        for (i, (a, b)) in res.val.iter().zip(&want).enumerate() {
                            assert!(
                                s.close(*a, *b),
                                "{} seed {seed} {dir:?} init {variant} {} {k:?}: node {}: kernel {a:?}, paths {b:?}",
                                s.name(),
                                $wn,
                                i + win.lo
                            );
                        }
                    }
                }};
            }
            one_weight!(&One, "One");
            one_weight!(&node, "Node");
            one_weight!(&arc, "Arc");
        }
    }
}

/// `⊕_v F(v) ⊗ c(v) = ⊕_u b(u) ⊗ B(u)` with dense random initial terms.
fn duality<S: Inst>(s: S) {
    for seed in 0..SEEDS {
        let mut r = rng(seed ^ 0xD0A1);
        let g = if seed % 2 == 0 {
            random_dag(r.random_range(0..30), r.random_range(0.0..0.3), seed)
        } else {
            mini_pg(r.random_range(1..20), seed, &MiniPgOpts::default()).graph
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
            ($w:expr, $wn:expr) => {{
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
                    "{} seed {seed} {}: {lhs:?} vs {rhs:?}",
                    s.name(),
                    $wn
                );
            }};
        }
        dual!(&One, "One");
        dual!(&node, "Node");
        dual!(&arc, "Arc");
    }
}

/// The whole matrix for one instance.
fn sweep<S: Inst>(s: S) {
    for seed in 0..SEEDS {
        kernel_matrix(s, seed);
    }
    brute_force(s);
    duality(s);
}

macro_rules! instance_tests {
    ($($name:ident: $s:expr;)*) => {$(
        #[test]
        fn $name() {
            sweep($s);
        }
    )*};
}

instance_tests! {
    sat_u64: SatU64::new();
    sat_u128: SatU128::new();
    mod61: M61::new();
    goldilocks: Goldilocks::new();
    extfloat: ExtFloat::new();
    log2sum: Log2Sum::new();
    probability: Probability::new();
    viterbi: Viterbi::new();
    maxmin_u32: MaxMin::<u32>::new();
    maxmin_u64: MaxMin::<u64>::new();
    maxmin_i64: MaxMin::<i64>::new();
    maxmin_f64: MaxMin::<f64>::new();
    product_extfloat_mod61: Product::new(ExtFloat::new(), M61::new());
    product_mod61_goldilocks: Product::new(M61::new(), Goldilocks::new());
    product_maxplus_sat64: Product::new(MaxPlus::<u32>::new(), SatU64::new());
    matrix2_minplus_i64: Matrix2::new(MinPlus::<i64>::new());
    matrix2_mod61: Matrix2::new(M61::new());
}

// ---------------------------------------------------------------------------------------------
// Specific properties of the instances
// ---------------------------------------------------------------------------------------------

#[test]
fn matrix_instances_are_not_commutative() {
    // otherwise the duality matrix could not catch swapped operands
    let s = Matrix2::new(M61::new());
    let a = Mat2([[1, 2], [0, 1]]);
    let b = Mat2([[1, 0], [3, 1]]);
    assert_ne!(s.times(a, b), s.times(b, a));
    let t = Matrix2::new(MinPlus::<i64>::new());
    let a = Mat2([[0, 5], [i64::MAX, i64::MAX]]);
    let b = Mat2([[i64::MAX, i64::MAX], [0, 0]]);
    assert_ne!(t.times(a, b), t.times(b, a));
}

#[test]
fn inv_outdeg_probability_conserves_mass() {
    // uniform-split mass from every source: what reaches the sinks equals the number of
    // sources (each non-sink forwards exactly its mass), on every kernel
    for seed in 0..100u64 {
        let pg = mini_pg(rng(seed).random_range(1..80), seed, &MiniPgOpts::default()).graph;
        let n = pg.num_nodes();
        let b = BlockIndex::from_vec_graph(&pg).unwrap().0;
        let gt = transpose(&pg);
        let (cg, cgt) = (chunked(&pg, 4), chunked(&gt, 4));
        let s = Probability::new();
        let inv: Vec<f64> = (0..n)
            .map(|u| match pg.outdegree(u) {
                0 => 1.0,
                d => 1.0 / d as f64,
            })
            .collect();
        let w = Node {
            alpha: |u: usize| inv[u],
            beta: |_: usize| 1.0,
        };
        let c = conf(Window::full(n), Kernel::Auto, false, 2);
        let sources = (0..n).filter(|&v| gt.outdegree(v) == 0).count() as f64;
        for (k, r) in run_kernels(
            &s,
            &cg,
            &cgt,
            Some(&b),
            Dir::Forward,
            &w,
            &Init::Sources(1.0),
            &c,
        ) {
            let at_sinks: f64 = (0..n)
                .filter(|&v| pg.outdegree(v) == 0)
                .map(|v| r.val[v])
                .sum();
            assert!(
                (at_sinks - sources).abs() <= 1e-12 * sources.max(1.0),
                "seed {seed} {k:?}: {at_sinks} at sinks vs {sources} sources"
            );
        }
        // Viterbi with the same weights: the most probable random-walk path is at most 1
        for (k, r) in run_kernels(
            &Viterbi::new(),
            &cg,
            &cgt,
            Some(&b),
            Dir::Forward,
            &w,
            &Init::Sources(1.0),
            &c,
        ) {
            assert!(
                r.val.iter().all(|&x| x > 0.0 && x <= 1.0),
                "seed {seed} {k:?}"
            );
        }
    }
}
