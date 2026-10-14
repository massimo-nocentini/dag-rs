//! Semirings for path problems on DAGs.
//!
//! A semiring `(E, ⊕, ⊗, 0̄, 1̄)` turns a path problem into one topological pass: the value of a
//! node is the ⊕ over its incoming paths of the ⊗-product of arc weights
//! ([\[Leh77\]](crate::references#leh77), [\[Moh02\]](crate::references#moh02),
//! [\[GM08\]](crate::references#gm08), [\[Goo99\]](crate::references#goo99)). On a DAG no
//! closure (star) is needed, so the only axioms used are those of a semiring.
//!
//! This module freezes the [`Semiring`] trait and the core tropical and Boolean instances
//! ([`MaxPlus`], [`MinPlus`], [`Boolean`]); the other instances live in the submodules.
//!
//! # Contract for every instance
//! - Constructible outside its module: public fields or a unit struct, `Default`, and
//!   `pub const fn new()`.
//! - **Exact identities**: `plus(zero, x) == x` and `plus(x, zero) == x` bitwise,
//!   `times(one, x) == x` and `times(x, one) == x` bitwise, and
//!   `times(zero, x) == times(x, zero) == zero`. Float instances special-case these. This is what
//!   makes the canonical evaluation order of [`crate::dp`] bit-reproducible across kernels.
//! - `⊗` is **not** assumed commutative: forward kernels compute `F(u) ⊗ α(u) ⊗ β(v)`, backward
//!   ones `α(u) ⊗ β(v) ⊗ B(v)`.
//!
//! # Tropical sentinels
//! - **Signed and float types** (`i64`, `f64`): `MaxPlus` uses 0̄ = `MIN`/−∞, `MinPlus` uses
//!   0̄ = `MAX`/+∞; 1̄ = 0; ⊗ is a saturating add in which 0̄ absorbs. Saturating `i64` ⊗ is
//!   associative only away from the bounds (`|x| ≤ 2⁶⁰` is safe for any path of ≤ 2³ terms of that
//!   size; the axiom tests sample there).
//! - **Unsigned types** (`u32`, `u64`): both instances use 0̄ = `T::MAX`, which keeps outputs
//!   byte-compatible with `longest.u32le` and `analysis_path`'s `UNSET`. `MaxPlus<uN>::plus`
//!   treats `MAX` as −∞ (if either operand is `MAX` it returns the other, otherwise the max);
//!   `times` returns `MAX` if either operand is `MAX`, otherwise `min(a + b, MAX − 1)`.
//!   `MinPlus<uN>` is the natural +∞ = `MAX` with saturating add.
//!
//! Longest paths in DAGs are the critical-path method [\[KW59\]](crate::references#kw59),
//! [\[MRCF59\]](crate::references#mrcf59); shortest paths with arbitrary weights follow
//! [\[Bel58\]](crate::references#bel58).

use core::marker::PhantomData;

pub mod bottleneck;
pub mod counting;
pub mod extfloat;
pub mod logsum;
pub mod matrix;
pub mod modular;
pub mod prob;
pub mod product;

/// A semiring over `Copy` elements (see the module documentation for the contract).
pub trait Semiring: Send + Sync {
    /// The element type.
    type Elem: Copy + Send + Sync + PartialEq + core::fmt::Debug + crate::io::Dump;
    /// Whether `a ⊕ a = a`.
    const IDEMPOTENT: bool = false;
    /// Whether `a ⊕ b ∈ {a, b}`; enables argbest (`pred`) tracking.
    const SELECTIVE: bool = false;
    /// A short name (used in file metadata).
    fn name(&self) -> &'static str;
    /// The additive identity 0̄, which annihilates ⊗.
    fn zero(&self) -> Self::Elem;
    /// The multiplicative identity 1̄.
    fn one(&self) -> Self::Elem;
    /// `a ⊕ b`.
    fn plus(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    /// `a ⊗ b` (not assumed commutative).
    fn times(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    /// Whether `cand` is strictly better than `cur`: `cand ≠ cur ∧ cand ⊕ cur = cand`.
    #[inline]
    fn better(&self, cand: Self::Elem, cur: Self::Elem) -> bool {
        cand != cur && self.plus(cand, cur) == cand
    }
}

/// The (max, +) semiring: longest paths. See the module documentation for the sentinels.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxPlus<T>(pub PhantomData<T>);

impl<T> MaxPlus<T> {
    /// The instance.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// The (min, +) semiring: shortest paths. See the module documentation for the sentinels.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinPlus<T>(pub PhantomData<T>);

impl<T> MinPlus<T> {
    /// The instance.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// The Boolean semiring (∨, ∧): reachability.
#[derive(Clone, Copy, Debug, Default)]
pub struct Boolean;

impl Boolean {
    /// The instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Semiring for Boolean {
    type Elem = bool;
    const IDEMPOTENT: bool = true;
    const SELECTIVE: bool = true;
    fn name(&self) -> &'static str {
        "bool"
    }
    #[inline(always)]
    fn zero(&self) -> bool {
        false
    }
    #[inline(always)]
    fn one(&self) -> bool {
        true
    }
    #[inline(always)]
    fn plus(&self, a: bool, b: bool) -> bool {
        a | b
    }
    #[inline(always)]
    fn times(&self, a: bool, b: bool) -> bool {
        a & b
    }
}

macro_rules! tropical_unsigned {
    ($t:ty, $max:literal, $min:literal) => {
        impl Semiring for MaxPlus<$t> {
            type Elem = $t;
            const IDEMPOTENT: bool = true;
            const SELECTIVE: bool = true;
            fn name(&self) -> &'static str {
                $max
            }
            /// `MAX`, read as −∞.
            #[inline(always)]
            fn zero(&self) -> $t {
                <$t>::MAX
            }
            #[inline(always)]
            fn one(&self) -> $t {
                0
            }
            #[inline(always)]
            fn plus(&self, a: $t, b: $t) -> $t {
                if a == <$t>::MAX {
                    b
                } else if b == <$t>::MAX {
                    a
                } else {
                    a.max(b)
                }
            }
            #[inline(always)]
            fn times(&self, a: $t, b: $t) -> $t {
                if a == <$t>::MAX || b == <$t>::MAX {
                    <$t>::MAX
                } else {
                    a.saturating_add(b).min(<$t>::MAX - 1)
                }
            }
        }

        impl Semiring for MinPlus<$t> {
            type Elem = $t;
            const IDEMPOTENT: bool = true;
            const SELECTIVE: bool = true;
            fn name(&self) -> &'static str {
                $min
            }
            /// `MAX` = +∞.
            #[inline(always)]
            fn zero(&self) -> $t {
                <$t>::MAX
            }
            #[inline(always)]
            fn one(&self) -> $t {
                0
            }
            #[inline(always)]
            fn plus(&self, a: $t, b: $t) -> $t {
                a.min(b)
            }
            #[inline(always)]
            fn times(&self, a: $t, b: $t) -> $t {
                a.saturating_add(b)
            }
        }
    };
}

tropical_unsigned!(u32, "maxplus-u32", "minplus-u32");
tropical_unsigned!(u64, "maxplus-u64", "minplus-u64");

impl Semiring for MaxPlus<i64> {
    type Elem = i64;
    const IDEMPOTENT: bool = true;
    const SELECTIVE: bool = true;
    fn name(&self) -> &'static str {
        "maxplus-i64"
    }
    /// `i64::MIN` = −∞.
    #[inline(always)]
    fn zero(&self) -> i64 {
        i64::MIN
    }
    #[inline(always)]
    fn one(&self) -> i64 {
        0
    }
    #[inline(always)]
    fn plus(&self, a: i64, b: i64) -> i64 {
        a.max(b)
    }
    /// Saturating add; `MIN` absorbs.
    #[inline(always)]
    fn times(&self, a: i64, b: i64) -> i64 {
        if a == i64::MIN || b == i64::MIN {
            i64::MIN
        } else {
            a.saturating_add(b).max(i64::MIN + 1)
        }
    }
}

impl Semiring for MinPlus<i64> {
    type Elem = i64;
    const IDEMPOTENT: bool = true;
    const SELECTIVE: bool = true;
    fn name(&self) -> &'static str {
        "minplus-i64"
    }
    /// `i64::MAX` = +∞.
    #[inline(always)]
    fn zero(&self) -> i64 {
        i64::MAX
    }
    #[inline(always)]
    fn one(&self) -> i64 {
        0
    }
    #[inline(always)]
    fn plus(&self, a: i64, b: i64) -> i64 {
        a.min(b)
    }
    /// Saturating add; `MAX` absorbs.
    #[inline(always)]
    fn times(&self, a: i64, b: i64) -> i64 {
        if a == i64::MAX || b == i64::MAX {
            i64::MAX
        } else {
            a.saturating_add(b).min(i64::MAX - 1)
        }
    }
}

impl Semiring for MaxPlus<f64> {
    type Elem = f64;
    const IDEMPOTENT: bool = true;
    const SELECTIVE: bool = true;
    fn name(&self) -> &'static str {
        "maxplus-f64"
    }
    #[inline(always)]
    fn zero(&self) -> f64 {
        f64::NEG_INFINITY
    }
    #[inline(always)]
    fn one(&self) -> f64 {
        0.0
    }
    /// `max`, returning the other operand bitwise when one is −∞.
    #[inline(always)]
    fn plus(&self, a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY {
            b
        } else if b == f64::NEG_INFINITY || a >= b {
            a
        } else {
            b
        }
    }
    /// `a + b`; −∞ absorbs (also against +∞); `x ⊗ 1̄` and `1̄ ⊗ x` (1̄ = +0.0) return `x` bitwise.
    #[inline(always)]
    fn times(&self, a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY || b == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else if b.to_bits() == 0 {
            a
        } else if a.to_bits() == 0 {
            b
        } else {
            a + b
        }
    }
}

impl Semiring for MinPlus<f64> {
    type Elem = f64;
    const IDEMPOTENT: bool = true;
    const SELECTIVE: bool = true;
    fn name(&self) -> &'static str {
        "minplus-f64"
    }
    #[inline(always)]
    fn zero(&self) -> f64 {
        f64::INFINITY
    }
    #[inline(always)]
    fn one(&self) -> f64 {
        0.0
    }
    /// `min`, returning the other operand bitwise when one is +∞.
    #[inline(always)]
    fn plus(&self, a: f64, b: f64) -> f64 {
        if a == f64::INFINITY {
            b
        } else if b == f64::INFINITY || a <= b {
            a
        } else {
            b
        }
    }
    /// `a + b`; +∞ absorbs (also against −∞); `x ⊗ 1̄` and `1̄ ⊗ x` (1̄ = +0.0) return `x` bitwise.
    #[inline(always)]
    fn times(&self, a: f64, b: f64) -> f64 {
        if a == f64::INFINITY || b == f64::INFINITY {
            f64::INFINITY
        } else if b.to_bits() == 0 {
            a
        } else if a.to_bits() == 0 {
            b
        } else {
            a + b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_maxplus_sentinels() {
        let s = MaxPlus::<u32>::new();
        assert_eq!(s.plus(s.zero(), 5), 5);
        assert_eq!(s.plus(5, s.zero()), 5);
        assert_eq!(s.times(s.zero(), 5), s.zero());
        assert_eq!(s.times(u32::MAX - 2, 7), u32::MAX - 1);
        assert!(s.better(7, 5));
        assert!(!s.better(5, 5));
    }

    #[test]
    fn float_identities_bitwise() {
        let s = MaxPlus::<f64>::new();
        let x = -0.0f64;
        assert_eq!(s.times(x, s.one()).to_bits(), x.to_bits());
        assert_eq!(s.times(s.one(), x).to_bits(), x.to_bits());
        assert_eq!(s.plus(s.zero(), x).to_bits(), x.to_bits());
        let t = MinPlus::<f64>::new();
        assert_eq!(t.times(f64::NEG_INFINITY, t.zero()), t.zero());
        assert_eq!(t.plus(t.zero(), x).to_bits(), x.to_bits());
    }
}
