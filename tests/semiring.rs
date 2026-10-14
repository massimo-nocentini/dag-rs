//! Acceptance tests of the semiring instances.
//!
//! Every instance is checked on 10⁴ random triples for the semiring laws: the exact identities
//! of [`dag::semiring::Semiring`] **bitwise** (compared through the `Dump` encoding), `0̄ ⊕ 0̄ = 0̄`, exact
//! commutativity of ⊕, and associativity and distributivity on both sides (exact for exact
//! semirings, relative 1e-12 for the floating-point ones). Oracles: u128 / checked arithmetic
//! for the saturating and modular instances, exact u128 counts for `ExtFloat` (also at 2²⁰⁰
//! scale), `log2(Σ 2^x)` for `Log2Sum`. This file uses only the public API, so it also proves
//! that every instance is constructible from another crate.

use core::marker::PhantomData;

use dag::io::Dump;
use dag::semiring::bottleneck::MaxMin;
use dag::semiring::counting::{SatU64, SatU128};
use dag::semiring::extfloat::{Ext, ExtFloat};
use dag::semiring::logsum::Log2Sum;
use dag::semiring::matrix::{Mat2, Matrix2};
use dag::semiring::modular::{GOLDILOCKS_P, Goldilocks, M61, M61_P, ModP};
use dag::semiring::prob::{Probability, Viterbi};
use dag::semiring::product::Product;
use dag::semiring::{Boolean, MaxPlus, MinPlus, Semiring};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const TRIPLES: usize = 10_000;
const REL: f64 = 1e-12;

// ---------------------------------------------------------------------------------------------
// Generic law checker
// ---------------------------------------------------------------------------------------------

/// The `Dump` encoding of `x`: equality of encodings is bitwise equality.
fn bytes<T: Dump>(x: T) -> Vec<u8> {
    let mut out = vec![0u8; T::BYTES];
    x.put(&mut out);
    out
}

/// Checks the semiring laws of `s` on `TRIPLES` random triples drawn by `sample`.
///
/// Identities, annihilation and commutativity of ⊕ are checked bitwise; associativity and
/// distributivity with `eq` (exact `==` for exact semirings, a tolerance for floats).
fn laws<S: Semiring>(
    s: &S,
    sample: &mut dyn FnMut() -> S::Elem,
    eq: &dyn Fn(S::Elem, S::Elem) -> bool,
) {
    let (z, o) = (s.zero(), s.one());
    let name = s.name();
    assert_eq!(bytes(s.plus(z, z)), bytes(z), "{name}: 0̄ ⊕ 0̄");
    assert_eq!(bytes(s.times(o, o)), bytes(o), "{name}: 1̄ ⊗ 1̄");
    for _ in 0..TRIPLES {
        let (a, b, c) = (sample(), sample(), sample());
        for x in [a, b, z, o] {
            assert_eq!(bytes(s.plus(z, x)), bytes(x), "{name}: 0̄ ⊕ {x:?}");
            assert_eq!(bytes(s.plus(x, z)), bytes(x), "{name}: {x:?} ⊕ 0̄");
            assert_eq!(bytes(s.times(o, x)), bytes(x), "{name}: 1̄ ⊗ {x:?}");
            assert_eq!(bytes(s.times(x, o)), bytes(x), "{name}: {x:?} ⊗ 1̄");
            assert_eq!(bytes(s.times(z, x)), bytes(z), "{name}: 0̄ ⊗ {x:?}");
            assert_eq!(bytes(s.times(x, z)), bytes(z), "{name}: {x:?} ⊗ 0̄");
        }
        assert_eq!(
            bytes(s.plus(a, b)),
            bytes(s.plus(b, a)),
            "{name}: ⊕ commutativity {a:?} {b:?}"
        );
        let (l, r) = (s.plus(s.plus(a, b), c), s.plus(a, s.plus(b, c)));
        assert!(
            eq(l, r),
            "{name}: ⊕ associativity {a:?} {b:?} {c:?}: {l:?} vs {r:?}"
        );
        let (l, r) = (s.times(s.times(a, b), c), s.times(a, s.times(b, c)));
        assert!(
            eq(l, r),
            "{name}: ⊗ associativity {a:?} {b:?} {c:?}: {l:?} vs {r:?}"
        );
        let (l, r) = (
            s.times(a, s.plus(b, c)),
            s.plus(s.times(a, b), s.times(a, c)),
        );
        assert!(
            eq(l, r),
            "{name}: left distributivity {a:?} {b:?} {c:?}: {l:?} vs {r:?}"
        );
        let (l, r) = (
            s.times(s.plus(a, b), c),
            s.plus(s.times(a, c), s.times(b, c)),
        );
        assert!(
            eq(l, r),
            "{name}: right distributivity {a:?} {b:?} {c:?}: {l:?} vs {r:?}"
        );
        if S::IDEMPOTENT {
            assert_eq!(bytes(s.plus(a, a)), bytes(a), "{name}: idempotence {a:?}");
        }
        if S::SELECTIVE {
            let p = s.plus(a, b);
            assert!(p == a || p == b, "{name}: selectivity {a:?} {b:?}");
            assert!(
                !(s.better(a, b) && s.better(b, a)),
                "{name}: better is asymmetric"
            );
            assert!(!s.better(a, a), "{name}: better is strict");
        }
    }
}

fn exact<E: PartialEq>(a: E, b: E) -> bool {
    a == b
}

/// Relative closeness of two floats (equal infinities included).
fn close(a: f64, b: f64, tol: f64) -> bool {
    a == b || (a - b).abs() <= tol * a.abs().max(b.abs())
}

/// Relative closeness of two `Ext` values.
fn ext_close(a: Ext, b: Ext, tol: f64) -> bool {
    if a.is_zero() || b.is_zero() {
        return a == b;
    }
    let d = a.e - b.e;
    d.abs() <= 1 && (a.m / b.m * 2f64.powi(d as i32) - 1.0).abs() <= tol
}

/// Absolute-or-relative closeness in the log domain: `tol · max(1, |a|, |b|)`.
fn log_close(a: f64, b: f64, tol: f64) -> bool {
    a == b || (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

// ---------------------------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------------------------

fn sample_u64(r: &mut SmallRng) -> u64 {
    match r.random_range(0..10) {
        0 => [0, 1, 2, u64::MAX, u64::MAX - 1][r.random_range(0..5)],
        1 => u64::MAX - r.random_range(0..1u64 << 20),
        2 => r.random::<u64>(),
        3 | 4 => r.random_range(0..1u64 << 32),
        _ => r.random_range(0..1u64 << 16),
    }
}

fn sample_u128(r: &mut SmallRng) -> u128 {
    match r.random_range(0..10) {
        0 => [0, 1, 2, u128::MAX, u128::MAX - 1][r.random_range(0..5)],
        1 => u128::MAX - r.random_range(0..1u128 << 20),
        2 => r.random::<u128>(),
        3 | 4 => r.random_range(0..1u128 << 64),
        _ => r.random_range(0..1u128 << 32),
    }
}

fn sample_mod(r: &mut SmallRng, p: u64) -> u64 {
    match r.random_range(0..8) {
        0 => [0, 1, 2, p - 1, p - 2, p / 2, p / 2 + 1][r.random_range(0..7)] % p,
        1 => r.random_range(0..(1u64 << 32).min(p)),
        _ => r.random_range(0..p),
    }
}

fn sample_ext(r: &mut SmallRng) -> Ext {
    match r.random_range(0..12) {
        0 => Ext::ZERO,
        1 => Ext::ONE,
        2 => Ext::from_u64(r.random_range(1..1u64 << 53))
            .scale_pow2(r.random_range(-1_000_000..1_000_000)),
        _ => Ext::from_u64(r.random::<u64>() >> r.random_range(0..64))
            .scale_pow2(r.random_range(-300..300)),
    }
}

// ---------------------------------------------------------------------------------------------
// Laws, instance by instance
// ---------------------------------------------------------------------------------------------

#[test]
fn saturating_counting_laws() {
    let mut r = SmallRng::seed_from_u64(1);
    laws(&SatU64::new(), &mut || sample_u64(&mut r), &exact);
    let mut r = SmallRng::seed_from_u64(2);
    laws(&SatU128::new(), &mut || sample_u128(&mut r), &exact);
}

#[test]
fn modular_laws() {
    let mut r = SmallRng::seed_from_u64(3);
    laws(&M61::new(), &mut || sample_mod(&mut r, M61_P), &exact);
    laws(
        &Goldilocks::new(),
        &mut || sample_mod(&mut r, GOLDILOCKS_P),
        &exact,
    );
    laws(
        &ModP::<GOLDILOCKS_P>::new(),
        &mut || sample_mod(&mut r, GOLDILOCKS_P),
        &exact,
    );
    laws(
        &ModP::<1_000_000_007>::new(),
        &mut || sample_mod(&mut r, 1_000_000_007),
        &exact,
    );
    const P64: u64 = u64::MAX - 58; // the largest 64-bit prime 2⁶⁴ − 59
    laws(&ModP::<P64>::new(), &mut || sample_mod(&mut r, P64), &exact);
    laws(&ModP::<2>::new(), &mut || sample_mod(&mut r, 2), &exact);
}

#[test]
fn extfloat_laws() {
    let mut r = SmallRng::seed_from_u64(4);
    let s = ExtFloat::new();
    laws(&s, &mut || sample_ext(&mut r), &|a, b| ext_close(a, b, REL));
    // Every result is normalised.
    for _ in 0..TRIPLES {
        let (a, b) = (sample_ext(&mut r), sample_ext(&mut r));
        assert!(
            s.plus(a, b).is_normalized() && s.times(a, b).is_normalized(),
            "{a:?} {b:?}"
        );
    }
}

#[test]
fn log2sum_laws() {
    let mut r = SmallRng::seed_from_u64(5);
    let mut sample = || match r.random_range(0..12) {
        0 => f64::NEG_INFINITY,
        1 => [0.0, -0.0, 1.0, -1.0][r.random_range(0..4)],
        2 => r.random_range(-1e6..1e6),
        _ => r.random_range(-60.0..60.0),
    };
    laws(&Log2Sum::new(), &mut sample, &|a, b| log_close(a, b, REL));
}

#[test]
fn probability_and_viterbi_laws() {
    let mut r = SmallRng::seed_from_u64(6);
    let mut sample = || match r.random_range(0..20) {
        0 => 0.0,
        1 => 1.0,
        2 => f64::INFINITY,
        3 => r.random_range(0.0..1e3),
        4 => -0.0,
        _ => r.random_range(0.0..1.0),
    };
    // −0.0 is a domain value: 1̄ ⊗ (−0.0) keeps its sign bitwise, 0̄ = +0.0 still annihilates.
    fn neg_zero<S: Semiring<Elem = f64>>(s: &S) {
        assert_eq!(s.times(1.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(s.times(-0.0, 1.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(s.times(0.0, -0.0).to_bits(), 0);
        assert_eq!(s.times(-0.0, f64::INFINITY), 0.0);
    }
    neg_zero(&Probability::new());
    neg_zero(&Viterbi::new());
    laws(&Probability::new(), &mut sample, &|a, b| close(a, b, REL));
    laws(&Viterbi::new(), &mut sample, &|a, b| close(a, b, REL));
    // Viterbi's ⊕ never rounds and rounding is monotone: distributivity is exact.
    let v = Viterbi::new();
    for _ in 0..TRIPLES {
        let (a, b, c) = (sample(), sample(), sample());
        assert_eq!(
            v.times(a, v.plus(b, c)),
            v.plus(v.times(a, b), v.times(a, c))
        );
        assert_eq!(
            v.times(v.plus(a, b), c),
            v.plus(v.times(a, c), v.times(b, c))
        );
    }
}

#[test]
fn bottleneck_laws() {
    let mut r = SmallRng::seed_from_u64(7);
    laws(
        &MaxMin::<u32>::new(),
        &mut || match r.random_range(0..6) {
            0 => [0, u32::MAX][r.random_range(0..2)],
            _ => r.random_range(0..1000),
        },
        &exact,
    );
    laws(&MaxMin::<u64>::new(), &mut || sample_u64(&mut r), &exact);
    laws(
        &MaxMin::<i64>::new(),
        &mut || match r.random_range(0..6) {
            0 => [i64::MIN, i64::MAX, 0][r.random_range(0..3)],
            _ => r.random::<i64>(),
        },
        &exact,
    );
    laws(
        &MaxMin::<f64>::new(),
        &mut || match r.random_range(0..6) {
            0 => [f64::NEG_INFINITY, f64::INFINITY, 0.0, -0.0][r.random_range(0..4)],
            _ => r.random_range(-1e3..1e3),
        },
        &exact,
    );
}

#[test]
fn product_laws() {
    let mut r = SmallRng::seed_from_u64(8);
    laws(
        &Product::new(ExtFloat::new(), M61::new()),
        &mut || (sample_ext(&mut r), sample_mod(&mut r, M61_P)),
        &|a, b| ext_close(a.0, b.0, REL) && a.1 == b.1,
    );
    laws(
        &Product::new(M61::new(), Goldilocks::new()),
        &mut || (sample_mod(&mut r, M61_P), sample_mod(&mut r, GOLDILOCKS_P)),
        &exact,
    );
    let tropical = |r: &mut SmallRng| match r.random_range(0..8) {
        0 => u32::MAX,
        _ => r.random_range(0..1_000_000u32),
    };
    laws(
        &Product::new(MaxPlus::<u32>::new(), MinPlus::<u32>::new()),
        &mut || (tropical(&mut r), tropical(&mut r)),
        &exact,
    );
}

#[test]
fn matrix_laws_and_non_commutativity() {
    let mut r = SmallRng::seed_from_u64(9);
    let lim = 1i64 << 60;
    let entry = |r: &mut SmallRng| {
        if r.random_bool(0.15) {
            i64::MAX
        } else {
            r.random_range(-lim..=lim)
        }
    };
    let s = Matrix2::new(MinPlus::<i64>::new());
    laws(
        &s,
        &mut || {
            Mat2([
                [entry(&mut r), entry(&mut r)],
                [entry(&mut r), entry(&mut r)],
            ])
        },
        &exact,
    );
    let noncommuting = (0..1000)
        .filter(|_| {
            let a = Mat2([
                [entry(&mut r), entry(&mut r)],
                [entry(&mut r), entry(&mut r)],
            ]);
            let b = Mat2([
                [entry(&mut r), entry(&mut r)],
                [entry(&mut r), entry(&mut r)],
            ]);
            s.times(a, b) != s.times(b, a)
        })
        .count();
    assert!(
        noncommuting > 900,
        "Matrix2 ⊗ should almost never commute ({noncommuting}/1000)"
    );

    let b = Matrix2::new(Boolean::new());
    laws(
        &b,
        &mut || {
            Mat2([
                [r.random_bool(0.5), r.random_bool(0.5)],
                [r.random_bool(0.5), r.random_bool(0.5)],
            ])
        },
        &exact,
    );
    let c = Matrix2::new(SatU64::new());
    laws(
        &c,
        &mut || {
            Mat2([
                [sample_u64(&mut r), sample_u64(&mut r)],
                [sample_u64(&mut r), sample_u64(&mut r)],
            ])
        },
        &exact,
    );
}

// ---------------------------------------------------------------------------------------------
// Oracles
// ---------------------------------------------------------------------------------------------

/// A natural number that may exceed `u128` (`None` = "≥ 2¹²⁸").
type Nat = Option<u128>;

fn nat_add(a: Nat, b: Nat) -> Nat {
    a?.checked_add(b?)
}

fn nat_mul(a: Nat, b: Nat) -> Nat {
    if a == Some(0) || b == Some(0) {
        Some(0)
    } else {
        a?.checked_mul(b?)
    }
}

/// A random expression over four leaves, evaluated in ℕ (as `Nat`) and in `s` (through `h`).
fn random_expression<S: Semiring>(
    r: &mut SmallRng,
    s: &S,
    leaves: [(Nat, S::Elem); 4],
) -> (Nat, S::Elem) {
    let [(a, x), (b, y), (c, z), (d, w)] = leaves;
    match r.random_range(0..4) {
        0 => (
            nat_add(nat_mul(a, b), nat_mul(c, d)),
            s.plus(s.times(x, y), s.times(z, w)),
        ),
        1 => (
            nat_mul(nat_add(a, b), nat_add(c, d)),
            s.times(s.plus(x, y), s.plus(z, w)),
        ),
        2 => (
            nat_mul(nat_mul(a, b), nat_mul(c, d)),
            s.times(s.times(x, y), s.times(z, w)),
        ),
        _ => (
            nat_add(nat_mul(nat_add(a, b), c), d),
            s.plus(s.times(s.plus(x, y), z), w),
        ),
    }
}

#[test]
fn saturating_is_a_homomorphic_image_of_the_naturals() {
    let mut r = SmallRng::seed_from_u64(10);
    let (s64, s128) = (SatU64::new(), SatU128::new());
    for _ in 0..TRIPLES {
        // Single operations against u128 arithmetic.
        let (a, b) = (sample_u64(&mut r), sample_u64(&mut r));
        let cap = |x: u128| x.min(u64::MAX as u128) as u64;
        assert_eq!(s64.plus(a, b), cap(a as u128 + b as u128));
        assert_eq!(s64.times(a, b), cap(a as u128 * b as u128));
        // Expressions: h(expr over ℕ) == expr over the saturating semiring.
        let leaves = core::array::from_fn(|_| {
            let x = sample_u64(&mut r);
            (Some(x as u128), x)
        });
        let (nat, sat) = random_expression(&mut r, &s64, leaves);
        assert_eq!(sat, nat.map_or(u64::MAX, cap), "SatU64 vs ℕ");

        let (a, b) = (sample_u128(&mut r), sample_u128(&mut r));
        assert_eq!(
            s128.plus(a, b),
            nat_add(Some(a), Some(b)).unwrap_or(u128::MAX)
        );
        assert_eq!(
            s128.times(a, b),
            nat_mul(Some(a), Some(b)).unwrap_or(u128::MAX)
        );
        // Leaves below u128::MAX, so that a saturated leaf is not mistaken for a true value.
        let leaves = core::array::from_fn(|_| {
            let x = sample_u128(&mut r).min(u128::MAX - 1);
            (Some(x), x)
        });
        let (nat, sat) = random_expression(&mut r, &s128, leaves);
        assert_eq!(sat, nat.unwrap_or(u128::MAX), "SatU128 vs ℕ");
    }
}

fn modp_against_u128<S: Semiring<Elem = u64>>(s: &S, p: u64, seed: u64) {
    let mut r = SmallRng::seed_from_u64(seed);
    let edges = [
        0,
        1,
        2,
        p - 1,
        p - 2,
        p / 2,
        (1u64 << 32) % p,
        ((1u64 << 32) - 1) % p,
        ((1u64 << 61) - 2) % p,
    ];
    let check = |a: u64, b: u64| {
        assert_eq!(
            s.plus(a, b) as u128,
            (a as u128 + b as u128) % p as u128,
            "{} {a} + {b}",
            s.name()
        );
        assert_eq!(
            s.times(a, b) as u128,
            (a as u128 * b as u128) % p as u128,
            "{} {a} * {b}",
            s.name()
        );
    };
    for &a in &edges {
        for &b in &edges {
            check(a, b);
        }
    }
    for _ in 0..100 * TRIPLES {
        check(sample_mod(&mut r, p), sample_mod(&mut r, p));
    }
}

#[test]
fn modular_matches_u128_reference() {
    modp_against_u128(&M61::new(), M61_P, 11);
    modp_against_u128(&Goldilocks::new(), GOLDILOCKS_P, 12);
    modp_against_u128(&ModP::<GOLDILOCKS_P>::new(), GOLDILOCKS_P, 13);
    modp_against_u128(&ModP::<1_000_000_007>::new(), 1_000_000_007, 14);
    modp_against_u128(&ModP::<{ u64::MAX - 58 }>::new(), u64::MAX - 58, 15);
    // Mersenne edges: 2⁶¹ ≡ 1, (P − 1)² ≡ 1, 2⁶⁰ · 2 ≡ 2⁶¹ ≡ 1.
    let m = M61::new();
    assert_eq!(m.times(M61_P - 1, M61_P - 1), 1);
    assert_eq!(m.times(1 << 60, 2), 1);
    assert_eq!(m.plus(M61_P - 1, 1), 0);
    assert_eq!(m.plus(M61_P - 1, M61_P - 1), M61_P - 2);
    // Goldilocks edges: 2⁶⁴ ≡ 2³² − 1, 2⁹⁶ ≡ −1.
    let g = Goldilocks::new();
    assert_eq!(g.times(1 << 32, 1 << 32), (1 << 32) - 1);
    assert_eq!(g.times(1 << 48, 1 << 48), GOLDILOCKS_P - 1);
    assert_eq!(g.times(GOLDILOCKS_P - 1, GOLDILOCKS_P - 1), 1);
}

#[test]
fn modular_residues_of_path_counts_and_crt() {
    // Expressions over ℕ evaluated in Product<M61, Goldilocks> give the true residues.
    let mut r = SmallRng::seed_from_u64(16);
    let s = Product::new(M61::new(), Goldilocks::new());
    for _ in 0..TRIPLES {
        let leaves = core::array::from_fn(|_| {
            let x = r.random_range(0..1u128 << 31);
            (
                Some(x),
                (
                    (x % M61_P as u128) as u64,
                    (x % GOLDILOCKS_P as u128) as u64,
                ),
            )
        });
        let (nat, (a, b)) = random_expression(&mut r, &s, leaves);
        let n = nat.expect("leaves < 2³¹: every expression fits u128");
        assert_eq!(
            (a as u128, b as u128),
            (n % M61_P as u128, n % GOLDILOCKS_P as u128)
        );
    }
}

/// Exact `Ext` of a u128 that may not be representable in f64: the value is rounded once.
#[test]
fn extfloat_against_exact_u128_counts() {
    let mut r = SmallRng::seed_from_u64(17);
    let s = ExtFloat::new();
    // Each operand conversion rounds once (≤ u) and the operation once (≤ u): ≤ 3u < 1e-15.
    let tol = 1e-15;
    for _ in 0..TRIPLES {
        let (a, b) = (
            r.random::<u64>() >> r.random_range(0..64),
            r.random::<u64>() >> r.random_range(0..64),
        );
        let (x, y) = (Ext::from_u64(a), Ext::from_u64(b));
        let sum = a as u128 + b as u128;
        let prod = a as u128 * b as u128;
        assert!(
            ext_close(s.plus(x, y), Ext::from_u128(sum), tol),
            "{a} + {b}"
        );
        assert!(
            ext_close(s.times(x, y), Ext::from_u128(prod), tol),
            "{a} * {b}"
        );
        // The same at 2²⁰⁰ scale (and 2⁴⁰⁰ for the product), far beyond u128 and f64 · 2⁵³.
        let k = 200 + r.random_range(-8..8);
        let (xs, ys) = (x.scale_pow2(k), y.scale_pow2(k));
        assert!(
            ext_close(s.plus(xs, ys), Ext::from_u128(sum).scale_pow2(k), tol),
            "2^{k}·({a} + {b})"
        );
        assert!(
            ext_close(s.times(xs, ys), Ext::from_u128(prod).scale_pow2(2 * k), tol),
            "2^{k}·{a} * 2^{k}·{b}"
        );
        // Mixed scales: the smaller operand shifted by d bits.
        let d = r.random_range(0..64);
        let exact = ((a as u128) << d) + b as u128;
        assert!(
            ext_close(s.plus(x.scale_pow2(d), y), Ext::from_u128(exact), tol),
            "{a}·2^{d} + {b}"
        );
    }
    // Correct rounding, bitwise: operands below 2⁵³ convert exactly, so the one rounding of each
    // operation must give exactly the correctly rounded exact result (an off-by-one-ulp rounding
    // or a premature absorption of the smaller operand fails here).
    for _ in 0..TRIPLES * 4 {
        let (a, b) = (
            r.random::<u64>() >> (11 + r.random_range(0..53)),
            r.random::<u64>() >> (11 + r.random_range(0..53)),
        );
        let (x, y) = (Ext::from_u64(a), Ext::from_u64(b));
        let d = r.random_range(0..75);
        let exact = ((a as u128) << d) + b as u128;
        assert_eq!(
            s.plus(x.scale_pow2(d), y),
            Ext::from_u128(exact),
            "{a}·2^{d} + {b}"
        );
        assert_eq!(
            s.plus(y, x.scale_pow2(d)),
            Ext::from_u128(exact),
            "{b} + {a}·2^{d}"
        );
        assert_eq!(
            s.times(x, y),
            Ext::from_u128(a as u128 * b as u128),
            "{a} * {b}"
        );
    }
    // Small integers are exact.
    for a in 0..200u64 {
        for b in 0..200u64 {
            assert_eq!(
                s.plus(Ext::from_u64(a), Ext::from_u64(b)),
                Ext::from_u64(a + b)
            );
            assert_eq!(
                s.times(Ext::from_u64(a), Ext::from_u64(b)),
                Ext::from_u64(a * b)
            );
        }
    }
}

#[test]
fn extfloat_counts_paths_of_a_complete_dag() {
    // Paths from 0 to v in the complete DAG on n nodes: 2^(v−1) for v ≥ 1. Fold the recurrence
    // F(v) = Σ_{u<v} F(u) in both u128 (exact while it fits) and ExtFloat.
    let s = ExtFloat::new();
    let n = 127;
    let mut exact = vec![0u128; n];
    let mut ext = vec![Ext::ZERO; n];
    exact[0] = 1;
    ext[0] = Ext::ONE;
    for v in 1..n {
        let mut acc = s.zero();
        for u in 0..v {
            exact[v] += exact[u];
            acc = s.plus(acc, ext[u]);
        }
        ext[v] = acc;
        assert_eq!(exact[v], 1u128 << (v - 1));
        assert_eq!(
            ext[v],
            Ext::ONE.scale_pow2(v as i64 - 1),
            "powers of two are exact"
        );
    }
    // A non-trivial count: Fibonacci-like DAG (v ← v−1, v−2) at 2²⁰⁰ scale is exact in the
    // magnitude to ≈ (depth) · u.
    let mut fib = (1u128, 1u128);
    let mut fe = (Ext::ONE.scale_pow2(200), Ext::ONE.scale_pow2(200));
    for _ in 0..180 {
        fib = (fib.1, fib.0 + fib.1);
        fe = (fe.1, s.plus(fe.0, fe.1));
    }
    assert!(ext_close(
        fe.1,
        Ext::from_u128(fib.1).scale_pow2(200),
        200.0 * f64::EPSILON
    ));
    assert!((fe.1.log2() - (200.0 + (fib.1 as f64).log2())).abs() < 1e-12);
}

#[test]
fn ext_conversions_and_log2() {
    let mut r = SmallRng::seed_from_u64(18);
    for _ in 0..TRIPLES {
        let x: f64 = r.random_range(0.0..1.0) * 2f64.powi(r.random_range(-1070..1020));
        let e = Ext::from_f64(x);
        assert!(e.is_normalized());
        assert_eq!(e.to_f64(), x);
        if x > 0.0 {
            assert!(
                (e.log2() - x.log2()).abs() <= 1e-15 * x.log2().abs().max(1.0),
                "{x}"
            );
        }
        let k = r.random_range(-1_000_000..1_000_000);
        assert!((e.scale_pow2(k).log2() - (e.log2() + k as f64)).abs() < 1e-9 || x == 0.0);
    }
    assert_eq!(Ext::ZERO.log2(), f64::NEG_INFINITY);
    assert_eq!(Ext::from_u128(u128::MAX), Ext::ONE.scale_pow2(128));
}

#[test]
fn log2sum_against_log2_of_sum() {
    let mut r = SmallRng::seed_from_u64(19);
    let s = Log2Sum::new();
    assert_eq!(
        s.plus(f64::NEG_INFINITY, f64::NEG_INFINITY),
        f64::NEG_INFINITY
    );
    assert!(!s.plus(s.zero(), s.zero()).is_nan());
    for _ in 0..TRIPLES {
        let k = r.random_range(1..20);
        let xs: Vec<f64> = (0..k).map(|_| r.random_range(-40.0..40.0)).collect();
        let folded = xs.iter().fold(s.zero(), |acc, &x| s.plus(acc, x));
        let direct = xs.iter().map(|x| x.exp2()).sum::<f64>().log2();
        assert!(
            (folded - direct).abs() <= 1e-12,
            "{xs:?}: {folded} vs {direct}"
        );
        // ⊗ is +, so a common offset does not change the relative shape.
        let off = r.random_range(-1e5..1e5);
        let shifted = xs
            .iter()
            .fold(s.zero(), |acc, &x| s.plus(acc, s.times(x, off)));
        assert!((shifted - (folded + off)).abs() <= 1e-9, "offset {off}");
    }
    // Far-apart operands: the small one vanishes, equal ones add one bit.
    assert_eq!(s.plus(1000.0, 0.0), 1000.0);
    assert_eq!(s.plus(7.0, 7.0), 8.0);
    // The correction term keeps full relative accuracy when it is tiny: log2(1 + 2^−t) for
    // large t, where a plain `ln(1 + 2^−t)` would round 1 + 2^−t to 1 and return 0.
    for t in [20.0f64, 40.0, 60.0, 100.0, 500.0] {
        let exact = (-t).exp2().ln_1p() / core::f64::consts::LN_2;
        for (a, b) in [(0.0, -t), (-t, 0.0)] {
            let got = s.plus(a, b);
            assert!(
                (got - exact).abs() <= 4.0 * f64::EPSILON * exact,
                "log2(1 + 2^-{t}): {got} vs {exact}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Flags, names, storage, constructibility
// ---------------------------------------------------------------------------------------------

fn flags<S: Semiring>(_: &S) -> (bool, bool) {
    (S::IDEMPOTENT, S::SELECTIVE)
}

#[test]
fn flags_and_names() {
    assert_eq!(flags(&SatU64::new()), (false, false));
    assert_eq!(flags(&SatU128::new()), (false, false));
    assert_eq!(flags(&M61::new()), (false, false));
    assert_eq!(flags(&Goldilocks::new()), (false, false));
    assert_eq!(flags(&ExtFloat::new()), (false, false));
    assert_eq!(flags(&Log2Sum::new()), (false, false));
    assert_eq!(flags(&Probability::new()), (false, false));
    assert_eq!(flags(&Viterbi::new()), (true, true));
    assert_eq!(flags(&MaxMin::<u32>::new()), (true, true));
    assert_eq!(flags(&MaxMin::<f64>::new()), (true, true));
    // Product: IDEMPOTENT = A ∧ B, SELECTIVE = false always.
    assert_eq!(
        flags(&Product::new(MaxPlus::<u32>::new(), MinPlus::<u32>::new())),
        (true, false)
    );
    assert_eq!(
        flags(&Product::new(MaxPlus::<u32>::new(), M61::new())),
        (false, false)
    );
    assert_eq!(
        flags(&Product::new(Boolean::new(), Viterbi::new())),
        (true, false)
    );
    assert_eq!(flags(&Matrix2::new(MinPlus::<i64>::new())), (true, false));
    assert_eq!(flags(&Matrix2::new(SatU64::new())), (false, false));
    // Product is not selective even for selective components: (5, 3) is neither operand.
    let p = Product::new(MaxPlus::<u32>::new(), MinPlus::<u32>::new());
    assert_eq!(p.plus((3, 3), (5, 5)), (5, 3));

    let names = [
        SatU64::new().name(),
        SatU128::new().name(),
        M61::new().name(),
        Goldilocks::new().name(),
        ExtFloat::new().name(),
        Log2Sum::new().name(),
        Probability::new().name(),
        Viterbi::new().name(),
        MaxMin::<u32>::new().name(),
        MaxMin::<u64>::new().name(),
        MaxMin::<i64>::new().name(),
        MaxMin::<f64>::new().name(),
    ];
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        names.len(),
        "instance names are distinct: {names:?}"
    );
    // The MaxMin quirk: 1̄ is u32::MAX (= NONE as a bit pattern), a value.
    assert_eq!(MaxMin::<u32>::new().one(), dag::NONE);
}

fn round_trip<T: Dump + PartialEq + core::fmt::Debug>(x: T) {
    let b = bytes(x);
    assert_eq!(b.len(), T::BYTES);
    assert_eq!(T::get(&b), x, "{}", T::DTYPE);
}

#[test]
fn dump_round_trips() {
    let mut r = SmallRng::seed_from_u64(20);
    assert_eq!((Ext::DTYPE, Ext::BYTES), ("extf", 16));
    assert_eq!((<Mat2<i64>>::DTYPE, <Mat2<i64>>::BYTES), ("m2i64le", 32));
    for _ in 0..1000 {
        round_trip(sample_ext(&mut r));
        round_trip(Mat2([
            [r.random::<i64>(), r.random::<i64>()],
            [r.random::<i64>(), r.random::<i64>()],
        ]));
        round_trip(Mat2([[r.random::<u32>(), 1], [2, 3]]));
        round_trip(Mat2([[r.random::<u64>(), 1], [2, 3]]));
        round_trip(Mat2([[r.random::<f64>(), 1.0], [2.0, 3.0]]));
        round_trip(Mat2([[r.random_bool(0.5), true], [false, true]]));
        round_trip((sample_ext(&mut r), sample_mod(&mut r, M61_P)));
    }
    // The documented layout: f64 mantissa then i64 exponent, little endian.
    let b = bytes(Ext { m: 1.5, e: -3 });
    assert_eq!(&b[..8], &1.5f64.to_le_bytes());
    assert_eq!(&b[8..], &(-3i64).to_le_bytes());
}

#[test]
fn constructible_from_another_crate() {
    // `new()`, `Default` and literal construction (public fields or unit structs) all work
    // outside the crate, also in const context.
    const _M61: M61 = M61::new();
    const _EXT: ExtFloat = ExtFloat::new();
    const _P: Product<SatU64, Goldilocks> = Product::new(SatU64::new(), Goldilocks::new());
    const _MM: Matrix2<MinPlus<i64>> = Matrix2::new(MinPlus::new());
    const _B: MaxMin<u32> = MaxMin::new();
    let _ = (
        SatU64,
        SatU128,
        Goldilocks,
        ExtFloat,
        Log2Sum,
        Probability,
        Viterbi,
    );
    let _ = ModP::<7>;
    let _ = MaxMin::<u64>(PhantomData);
    let _ = Product(Log2Sum, Probability);
    let _ = Matrix2(Boolean);
    let _ = Ext { m: 1.0, e: 0 };
    let _ = Mat2([[0i64; 2]; 2]);
    let _: (
        SatU64,
        SatU128,
        M61,
        Goldilocks,
        ExtFloat,
        Log2Sum,
        Probability,
        Viterbi,
    ) = Default::default();
    let _: (
        MaxMin<f64>,
        Product<M61, Goldilocks>,
        Matrix2<MinPlus<i64>>,
        Ext,
    ) = Default::default();
    assert_eq!(Ext::default(), ExtFloat::new().zero());
}
