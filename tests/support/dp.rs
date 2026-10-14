//! Helpers of the DP tests (`tests/dp.rs`, `tests/cpm.rs`, `tests/cli_dp.rs`): test-local
//! semirings (a non-commutative 2×2 max-plus matrix semiring and a non-selective real
//! (+, ×) semiring), random weights, brute-force oracles and kernel runners.
//!
//! Include with `#[path = "support/dp.rs"] mod support;`.
#![allow(dead_code)]

use dag::blocks::BlockIndex;
use dag::dp::blocks::{backward_blocks, forward_blocks};
use dag::dp::certify::bit_eq;
use dag::dp::pull::{backward_pull, forward_pull};
use dag::dp::push::forward_push;
use dag::dp::{ArcWeights, DpConf, DpResult, Init, Kernel};
use dag::graph::ChunkedGraph;
use dag::io::Dump;
use dag::semiring::{Boolean, MaxPlus, MinPlus, Semiring};
use dag::sweep::SweepConf;
use dag::{Dir, NONE, Window};
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;

/// A silent sequential logger.
pub fn quiet() -> Option<ProgressLogger> {
    None
}

/// A silent concurrent logger.
pub fn cquiet() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

/// SplitMix64 finaliser: a deterministic pseudo-random `u64` from a key.
pub fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// ---------------------------------------------------------------------------------------------
// Test-local semirings
// ---------------------------------------------------------------------------------------------

/// A 2×2 matrix over (max, +) on `i64` (MIN = −∞): a non-commutative idempotent semiring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct M2(pub [[i64; 2]; 2]);

impl Dump for M2 {
    const DTYPE: &'static str = "m2i64";
    const BYTES: usize = 32;
    fn put(&self, out: &mut [u8]) {
        for (k, x) in self.0.iter().flatten().enumerate() {
            out[8 * k..8 * k + 8].copy_from_slice(&x.to_le_bytes());
        }
    }
    fn get(inp: &[u8]) -> Self {
        let g = |k: usize| i64::from_le_bytes(inp[8 * k..8 * k + 8].try_into().unwrap());
        M2([[g(0), g(1)], [g(2), g(3)]])
    }
}

impl dag::dp::certify::RelErr for M2 {
    fn rel_err(&self, other: &Self) -> f64 {
        if self == other { 0.0 } else { 1.0 }
    }
    fn within(&self, other: &Self, _tol: f64) -> bool {
        self == other
    }
}

/// The semiring of [`M2`] matrices.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatMax;

const NEG: i64 = i64::MIN;

fn tadd(a: i64, b: i64) -> i64 {
    if a == NEG || b == NEG { NEG } else { a + b }
}

impl Semiring for MatMax {
    type Elem = M2;
    const IDEMPOTENT: bool = true;
    fn name(&self) -> &'static str {
        "mat2-maxplus"
    }
    fn zero(&self) -> M2 {
        M2([[NEG; 2]; 2])
    }
    fn one(&self) -> M2 {
        M2([[0, NEG], [NEG, 0]])
    }
    fn plus(&self, a: M2, b: M2) -> M2 {
        let mut c = a;
        for i in 0..2 {
            for j in 0..2 {
                c.0[i][j] = a.0[i][j].max(b.0[i][j]);
            }
        }
        c
    }
    fn times(&self, a: M2, b: M2) -> M2 {
        let mut c = [[NEG; 2]; 2];
        for (i, row) in c.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = tadd(a.0[i][0], b.0[0][j]).max(tadd(a.0[i][1], b.0[1][j]));
            }
        }
        M2(c)
    }
}

/// Real numbers (+, ×) on `f64` with exact identities: a non-selective float semiring whose
/// sums are not associative, which makes the kernels' bit identity a strong test.
#[derive(Clone, Copy, Debug, Default)]
pub struct Real;

impl Semiring for Real {
    type Elem = f64;
    fn name(&self) -> &'static str {
        "real"
    }
    fn zero(&self) -> f64 {
        0.0
    }
    fn one(&self) -> f64 {
        1.0
    }
    fn plus(&self, a: f64, b: f64) -> f64 {
        if a.to_bits() == 0 {
            b
        } else if b.to_bits() == 0 {
            a
        } else {
            a + b
        }
    }
    fn times(&self, a: f64, b: f64) -> f64 {
        if a.to_bits() == 0 || b.to_bits() == 0 {
            0.0
        } else {
            a * b
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Random elements
// ---------------------------------------------------------------------------------------------

/// Semirings exercised by the tests, with a generator of random weights.
pub trait TestSemiring: Semiring + Copy + Default {
    /// A pseudo-random element (a weight or an initial term) from a 64-bit key; may be 0̄ and
    /// 1̄ now and then.
    fn elem(&self, key: u64) -> Self::Elem;
    /// "Arc length 1" for the tropical semirings.
    fn unit(&self) -> Option<Self::Elem> {
        None
    }
    /// Whether two results agree (bitwise, except for the non-associative [`Real`], where a
    /// relative 1e-9 is allowed against oracles that associate differently).
    fn close(&self, a: Self::Elem, b: Self::Elem) -> bool {
        bit_eq(&a, &b)
    }
}

fn pick(
    key: u64,
    zero: impl Fn() -> f64,
    one: impl Fn() -> f64,
    other: impl Fn(u64) -> f64,
) -> f64 {
    match key % 16 {
        0 => zero(),
        1 => one(),
        _ => other(key >> 4),
    }
}

macro_rules! test_unsigned {
    ($s:ty, $t:ty) => {
        impl TestSemiring for $s {
            fn elem(&self, key: u64) -> $t {
                match key % 16 {
                    0 => self.zero(),
                    1 => self.one(),
                    _ => ((key >> 4) % 10) as $t,
                }
            }
            fn unit(&self) -> Option<$t> {
                Some(1)
            }
        }
    };
}
test_unsigned!(MaxPlus<u32>, u32);
test_unsigned!(MaxPlus<u64>, u64);
test_unsigned!(MinPlus<u32>, u32);

macro_rules! test_signed {
    ($s:ty) => {
        impl TestSemiring for $s {
            fn elem(&self, key: u64) -> i64 {
                match key % 16 {
                    0 => self.zero(),
                    1 => self.one(),
                    _ => ((key >> 4) % 11) as i64 - 5,
                }
            }
            fn unit(&self) -> Option<i64> {
                Some(1)
            }
        }
    };
}
test_signed!(MaxPlus<i64>);
test_signed!(MinPlus<i64>);

macro_rules! test_float {
    ($s:ty) => {
        impl TestSemiring for $s {
            fn elem(&self, key: u64) -> f64 {
                // integer-valued: sums are exact, so path oracles agree bitwise
                pick(
                    key,
                    || self.zero(),
                    || self.one(),
                    |k| (k % 11) as f64 - 5.0,
                )
            }
            fn unit(&self) -> Option<f64> {
                Some(1.0)
            }
        }
    };
}
test_float!(MaxPlus<f64>);
test_float!(MinPlus<f64>);

impl TestSemiring for Boolean {
    fn elem(&self, key: u64) -> bool {
        key % 4 != 0
    }
}

impl TestSemiring for MatMax {
    fn elem(&self, key: u64) -> M2 {
        match key % 16 {
            0 => self.zero(),
            1 => self.one(),
            _ => {
                let mut m = [[0i64; 2]; 2];
                for (k, x) in m.iter_mut().flatten().enumerate() {
                    let r = mix(key ^ k as u64);
                    *x = if r % 5 == 0 { NEG } else { (r % 9) as i64 - 4 };
                }
                M2(m)
            }
        }
    }
}

impl TestSemiring for Real {
    fn elem(&self, key: u64) -> f64 {
        pick(
            key,
            || 0.0,
            || 1.0,
            |k| 0.1 + (k % 1_000_003) as f64 / 500_000.0,
        )
    }
    fn close(&self, a: f64, b: f64) -> bool {
        a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
    }
}

/// Random per-node factors (absolute ids).
pub fn node_elems<S: TestSemiring>(s: &S, n: usize, seed: u64) -> Vec<S::Elem> {
    (0..n)
        .map(|u| s.elem(mix(seed ^ (u as u64) << 20)))
        .collect()
}

/// A random `Init` (`Sources`, `All` or `Dense`) for a window of `len` nodes.
pub fn random_init<S: TestSemiring>(s: &S, len: usize, seed: u64) -> Init<S::Elem> {
    match seed % 3 {
        0 => Init::Sources(s.elem(mix(seed + 1) | 2)),
        1 => Init::All(s.elem(mix(seed + 2))),
        _ => Init::Dense(
            (0..len)
                .map(|i| s.elem(mix(seed ^ (i as u64) << 8)))
                .collect(),
        ),
    }
}

/// `b(v)` of `init` for window node `v`.
pub fn init_term<S: Semiring>(
    s: &S,
    init: &Init<S::Elem>,
    lo: usize,
    v: usize,
    has: bool,
) -> S::Elem {
    match init {
        Init::Sources(e) => {
            if has {
                s.zero()
            } else {
                *e
            }
        }
        Init::All(e) => *e,
        Init::Dense(d) => d[v - lo],
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel runners
// ---------------------------------------------------------------------------------------------

/// The configuration of a run.
pub fn conf(window: Window, kernel: Kernel, pred: bool, decoders: usize) -> DpConf {
    DpConf {
        window,
        kernel,
        pred,
        sweep: SweepConf { decoders },
    }
}

/// Runs every applicable kernel for `dir` and returns `(kernel, result)` pairs: push/pull
/// (forward), pull (backward), and blocks when `b` is given and `W` is separable.
#[allow(clippy::too_many_arguments)]
pub fn run_kernels<S, W, G>(
    s: &S,
    g: &ChunkedGraph<G>,
    gt: &ChunkedGraph<G>,
    b: Option<&BlockIndex>,
    dir: Dir,
    w: &W,
    init: &Init<S::Elem>,
    c: &DpConf,
) -> Vec<(Kernel, DpResult<S::Elem>)>
where
    S: Semiring,
    W: ArcWeights<S>,
    G: RandomAccessGraph + Sync,
{
    let mut out = Vec::new();
    match dir {
        Dir::Forward => {
            out.push((
                Kernel::Push,
                forward_push(s, g, w, init, c, &mut quiet()).unwrap(),
            ));
            out.push((
                Kernel::Pull,
                forward_pull(s, gt, w, init, c, &mut quiet()).unwrap(),
            ));
            if let (Some(b), true) = (b, W::SEPARABLE) {
                out.push((
                    Kernel::Blocks,
                    forward_blocks(s, b, w, init, c, &mut quiet()).unwrap(),
                ));
            }
        }
        Dir::Backward => {
            out.push((
                Kernel::Pull,
                backward_pull(s, g, w, init, c, &mut quiet()).unwrap(),
            ));
            if let (Some(b), true) = (b, W::SEPARABLE) {
                out.push((
                    Kernel::Blocks,
                    backward_blocks(s, b, w, init, c, &mut quiet()).unwrap(),
                ));
            }
        }
    }
    out
}

/// Asserts that two results are bit-identical (values and predecessors).
pub fn assert_bit_identical<E: Dump + std::fmt::Debug>(
    a: &DpResult<E>,
    b: &DpResult<E>,
    what: &str,
) {
    assert_eq!(a.window, b.window, "{what}: windows");
    assert_eq!(a.val.len(), b.val.len(), "{what}: lengths");
    for (i, (x, y)) in a.val.iter().zip(&b.val).enumerate() {
        assert!(
            bit_eq(x, y),
            "{what}: node {}: {x:?} vs {y:?}",
            i + a.window.lo
        );
    }
    assert_eq!(a.pred, b.pred, "{what}: predecessors");
}

// ---------------------------------------------------------------------------------------------
// Brute-force oracles
// ---------------------------------------------------------------------------------------------

/// Successor lists of `g` restricted to the window.
pub fn window_succ<G: RandomAccessGraph>(g: &G, w: Window) -> Vec<Vec<usize>> {
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

/// Every path inside the window (single nodes included), as node sequences.
pub fn window_paths<G: RandomAccessGraph>(g: &G, w: Window) -> Vec<Vec<usize>> {
    let succ = window_succ(g, w);
    let mut out = Vec::new();
    fn rec(succ: &[Vec<usize>], path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        out.push(path.clone());
        let u = *path.last().unwrap();
        for &v in &succ[u] {
            path.push(v);
            rec(succ, path, out);
            path.pop();
        }
    }
    for u in w.lo..w.hi {
        rec(&succ, &mut vec![u], &mut out);
    }
    out
}

/// Forward values by path enumeration: `F(v) = ⊕_{x₀ → … → v} b(x₀) ⊗ w(x₀, x₁) ⊗ …`,
/// evaluated left to right; `b` follows `init` (`Sources` = no window predecessor).
pub fn brute_forward<S: Semiring, W: ArcWeights<S>, G: RandomAccessGraph>(
    s: &S,
    g: &G,
    win: Window,
    w: &W,
    init: &Init<S::Elem>,
) -> Vec<S::Elem> {
    let succ = window_succ(g, win);
    let mut has_pred = vec![false; g.num_nodes()];
    for s_ in &succ {
        for &v in s_ {
            has_pred[v] = true;
        }
    }
    let mut f = vec![s.zero(); win.len()];
    for p in window_paths(g, win) {
        let mut acc = init_term(s, init, win.lo, p[0], has_pred[p[0]]);
        for a in p.windows(2) {
            acc = s.times(acc, w.arc(s, a[0], a[1]));
        }
        let v = *p.last().unwrap() - win.lo;
        f[v] = s.plus(f[v], acc);
    }
    f
}

/// Backward values by path enumeration: `B(u) = ⊕_{u → … → x_k} w(u, x₁) ⊗ … ⊗ c(x_k)`,
/// evaluated right to left; `c` follows `init` (`Sources` = no window successor).
pub fn brute_backward<S: Semiring, W: ArcWeights<S>, G: RandomAccessGraph>(
    s: &S,
    g: &G,
    win: Window,
    w: &W,
    init: &Init<S::Elem>,
) -> Vec<S::Elem> {
    let succ = window_succ(g, win);
    let mut b = vec![s.zero(); win.len()];
    for p in window_paths(g, win) {
        let last = *p.last().unwrap();
        let mut acc = init_term(s, init, win.lo, last, !succ[last].is_empty());
        for a in p.windows(2).rev() {
            acc = s.times(w.arc(s, a[0], a[1]), acc);
        }
        let u = p[0] - win.lo;
        b[u] = s.plus(b[u], acc);
    }
    b
}

/// A deterministic random seed generator for test loops.
pub fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

/// A random window of `[0, n)` (sometimes the full range, sometimes empty).
pub fn random_window(r: &mut SmallRng, n: usize) -> Window {
    match r.random_range(0..4) {
        0 => Window { lo: 0, hi: n },
        _ => {
            let a = r.random_range(0..=n);
            let b = r.random_range(0..=n);
            Window {
                lo: a.min(b),
                hi: a.max(b),
            }
        }
    }
}

/// The subgraph induced by `w`, with ids shifted by `−w.lo`.
pub fn induced<G: RandomAccessGraph>(g: &G, w: Window) -> VecGraph {
    let arcs: Vec<(usize, usize)> = (w.lo..w.hi)
        .flat_map(|u| {
            g.successors(u)
                .into_iter()
                .filter(|&v| w.contains(v))
                .map(move |v| (u - w.lo, v - w.lo))
                .collect::<Vec<_>>()
        })
        .collect();
    dag::testkit::vec_graph(w.len(), arcs)
}

/// Shifts absolute predecessor ids by `+lo` (NONE stays).
pub fn shift_pred(p: &[u32], lo: usize) -> Vec<u32> {
    p.iter()
        .map(|&x| if x == NONE { NONE } else { x + lo as u32 })
        .collect()
}
