//! The log2-sum-exp semiring: counts and probabilities in the log domain.
//!
//! # Algorithm
//! [`Log2Sum`] is the base-2 log semiring `(ℝ ∪ {−∞}, ⊕, +, −∞, 0)` with
//! `a ⊕ b = log2(2^a + 2^b) = M + log2(1 + 2^(m − M))`, `M = max(a, b)`, `m = min(a, b)`,
//! evaluated as `M + ln_1p(exp2(m − M)) / ln 2` so that no intermediate overflows and the
//! correction term keeps full relative accuracy when `2^(m − M)` is tiny. It is the image of the
//! counting semiring `(ℝ≥0, +, ×)` under `log2` (Mohri's log semiring in base 2).
//!
//! # Invariants
//! - **Exact identities**: `−∞` is special-cased, so `−∞ ⊕ x = x ⊕ −∞ = x`
//!   bitwise and `−∞ ⊕ −∞ = −∞` (never `NaN`); `−∞ ⊗ x = x ⊗ −∞ = −∞` (also for `x = +∞`);
//!   `x ⊗ 0 = 0 ⊗ x = x` bitwise (including `x = −0.0`, which a plain `x + 0.0` would turn into
//!   `+0.0`).
//! - `⊕` is exactly commutative (operands are ordered before the one evaluation); associativity
//!   and distributivity hold up to rounding.
//! - Accuracy: `a ⊕ b` has absolute error ≈ `(|M| + 1) · u` with `u = 2⁻⁵³`: the absolute
//!   resolution of the log itself degrades with its magnitude (≈ 10⁻⁹ at `|M| ≈ 10⁷`), which is
//!   why full-PG counts use [`ExtFloat`](super::extfloat::ExtFloat) instead (F1 §4).
//! - Domain: finite values and `±∞` (`+∞ ⊕ x = +∞`); `NaN` inputs are not supported.
//!
//! # Complexity
//! O(1) per operation (one `exp2` and one `ln_1p`), 8 bytes per element.
//!
//! # References
//! [\[Moh02\]](crate::references#moh02), [\[GM08\]](crate::references#gm08),
//! [\[Goo99\]](crate::references#goo99).
//!
//! ```
//! use dag::semiring::{Semiring, logsum::Log2Sum};
//! let s = Log2Sum::new();
//! assert_eq!(s.plus(3.0, 3.0), 4.0);                       // 8 + 8 = 16
//! assert_eq!(s.plus(s.zero(), s.zero()), f64::NEG_INFINITY); // never NaN
//! ```

use super::Semiring;

/// log2-sum-exp: `⊕ = log2(2^a + 2^b)`, `⊗ = +`, `0̄ = −∞`, `1̄ = 0`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log2Sum;

impl Log2Sum {
    /// The instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Semiring for Log2Sum {
    type Elem = f64;
    fn name(&self) -> &'static str {
        "log2sum"
    }
    /// `−∞`.
    #[inline(always)]
    fn zero(&self) -> f64 {
        f64::NEG_INFINITY
    }
    /// `0`.
    #[inline(always)]
    fn one(&self) -> f64 {
        0.0
    }
    /// `log2(2^a + 2^b)`; `−∞` returns the other operand bitwise.
    #[inline]
    fn plus(&self, a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY {
            return b;
        }
        if b == f64::NEG_INFINITY {
            return a;
        }
        let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
        if hi == f64::INFINITY {
            return hi;
        }
        hi + (lo - hi).exp2().ln_1p() / core::f64::consts::LN_2
    }
    /// `a + b`; `−∞` absorbs; `+0.0` returns the other operand bitwise.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_values() {
        let s = Log2Sum::new();
        assert_eq!(s.plus(f64::INFINITY, f64::INFINITY), f64::INFINITY);
        assert_eq!(s.plus(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(s.times(f64::NEG_INFINITY, f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(s.times(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(s.plus(1e6, 0.0), 1e6);
    }
}
