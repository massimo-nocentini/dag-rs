//! Extended-exponent floats: magnitudes of astronomically large path counts.
//!
//! # Algorithm
//! An [`Ext`] is `m · 2^e` with a normalised `f64` mantissa `m ∈ [1, 2)` and an `i64` exponent,
//! or the canonical zero `{m: 0, e: 0}`. [`ExtFloat`] is the counting semiring
//! `(ℝ≥0, +, ×, 0, 1)` over these values:
//! - `a ⊗ b`: `m = a.m · b.m ∈ [1, 4)` (one rounding), `e = a.e + b.e`, then one exact halving
//!   if `m ≥ 2`;
//! - `a ⊕ b`: order the operands by `(e, m)`, align the smaller one by the exact power of two
//!   `2^−d` (`d = e_hi − e_lo`), add (one rounding), renormalise by an exact halving. When
//!   `d > 64` the smaller operand is below half an ulp of the larger one and the larger operand
//!   is returned, which is what round-to-nearest would give.
//!
//! # Invariants
//! - **Exact identities**: `0 ⊕ x = x ⊕ 0 = x` and `1 ⊗ x = x ⊗ 1 = x` bitwise,
//!   `0 ⊗ x = x ⊗ 0 = 0`; `⊕` and `⊗` are exactly commutative (operands are ordered
//!   deterministically before the one rounded operation).
//! - Every result is normalised (`m ∈ [1, 2)` or the canonical zero), so derived `PartialEq` is
//!   value equality.
//! - Accuracy: each operation has relative error at most `u = 2⁻⁵³` (one rounding of a value in
//!   `[1, 4)`; alignment and renormalisation are exact), independently of the magnitude. The DP
//!   kernels fold the `k_v` predecessors of a node sequentially (in ascending order), and a sequential
//!   fold of non-negative terms adds at most `(k_v − 1) · u` relative error, plus one `⊗` rounding
//!   per arc, to first order ([\[Hig93\]](crate::references#hig93)); since every term is
//!   non-negative the errors do not cancel catastrophically, so a value has relative error at
//!   most `Σ_v (k_v + 1) · u` over the nodes `v` of its longest dependency chain (`D` nodes).
//!   The `log₂ k` form applies only to the pairwise-summed totals of the path counts ([`crate::paths`]). On the PG the
//!   in-degree histogram is dominated by `k_v ≤ 3` (maximum 2·10⁴, on few nodes), so along
//!   the depth `D = 3.56·10⁶` the bound is about `4D · u ≈ 1.6·10⁻⁹` in practice (F1 §4), not a
//!   worst case. Associativity and distributivity hold up to these roundings, not bitwise.
//! - Domain: non-negative values only (path counts, probabilities); the exponent must stay well
//!   inside `i64` (`|e| < 2⁶²`; real path counts have `e < 2³²`). Exponent overflow is a panic in
//!   checked builds.
//!
//! # Complexity
//! O(1) per operation, 16 bytes per element (`Dump` type `"extf"`: `f64` mantissa then `i64`
//! exponent, little endian).
//!
//! # PG fast path
//! Full-PG path counts use `Product<ExtFloat, M61>` in one pass per direction: the
//! log2 of a count reaches millions, far outside `f64`, while `ExtFloat` keeps 53 significant bits
//! at every magnitude. `Log2Sum` is the alternative but loses absolute resolution in the log
//! (≈ 10⁻⁹ at log values ≈ 10⁷).
//!
//! # References
//! Semiring counting: [\[Leh77\]](crate::references#leh77), [\[Moh02\]](crate::references#moh02);
//! floating-point summation error: [\[Hig93\]](crate::references#hig93).
//!
//! ```
//! use dag::semiring::{Semiring, extfloat::{Ext, ExtFloat}};
//! let s = ExtFloat::new();
//! let big = Ext::from_u128(1 << 100).scale_pow2(1000); // 2^1100: beyond f64
//! let x = s.times(big, Ext::from_u64(3));
//! assert!((x.log2() - (1100.0 + 3f64.log2())).abs() < 1e-12);
//! assert_eq!(s.plus(s.zero(), x), x);
//! ```

use super::Semiring;

/// `2^k` as an `f64`, for `k ∈ [−1022, 1023]` (a normal power of two, exact).
#[inline(always)]
const fn pow2(k: i64) -> f64 {
    debug_assert!(k >= -1022 && k <= 1023);
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// An extended-exponent float `m · 2^e` with `m ∈ [1, 2)`, or the canonical zero
/// `{m: 0.0, e: 0}` (stored as `"extf"`, 16 B). Non-negative values only.
///
/// Build values with the constructors ([`Ext::from_f64`], [`Ext::from_u64`],
/// [`Ext::from_u128`], [`Ext::ZERO`], [`Ext::ONE`]); a hand-built `Ext` must be normalised
/// (see [`Ext::is_normalized`]) for the semiring operations to be correct.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ext {
    /// Mantissa, in `[1, 2)` (or `0.0` for zero).
    pub m: f64,
    /// Binary exponent (`0` for zero).
    pub e: i64,
}

impl Ext {
    /// The value 0.
    pub const ZERO: Ext = Ext { m: 0.0, e: 0 };
    /// The value 1.
    pub const ONE: Ext = Ext { m: 1.0, e: 0 };

    /// Converts a finite non-negative `f64` exactly (subnormals included).
    ///
    /// # Panics
    /// If `x` is negative, infinite or NaN.
    pub fn from_f64(x: f64) -> Self {
        assert!(
            x.is_finite() && x >= 0.0,
            "Ext::from_f64 needs a finite non-negative value, got {x}"
        );
        if x == 0.0 {
            return Self::ZERO;
        }
        let bits = x.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i64;
        if biased == 0 {
            // Subnormal: scale into the normal range exactly, then undo the scale.
            let y = Self::from_f64(x * pow2(64));
            return Ext {
                m: y.m,
                e: y.e - 64,
            };
        }
        Ext {
            m: f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52)),
            e: biased - 1023,
        }
    }

    /// Converts `x` (exactly when `x < 2⁵³`, otherwise rounded to nearest).
    pub fn from_u64(x: u64) -> Self {
        Self::from_f64(x as f64)
    }

    /// Converts `x` (exactly when `x < 2⁵³`, otherwise rounded to nearest).
    pub fn from_u128(x: u128) -> Self {
        Self::from_f64(x as f64)
    }

    /// `self · 2^k`, exact.
    #[inline]
    pub const fn scale_pow2(self, k: i64) -> Self {
        if self.m == 0.0 {
            self
        } else {
            Ext {
                m: self.m,
                e: self.e + k,
            }
        }
    }

    /// `log2` of the value (`−∞` for zero). Absolute error ≤ 1 ulp of the result plus the
    /// mantissa's `log2` error (≈ 2⁻⁵²).
    pub fn log2(&self) -> f64 {
        if self.m == 0.0 {
            f64::NEG_INFINITY
        } else {
            self.e as f64 + self.m.log2()
        }
    }

    /// The nearest `f64` (one rounding, `+∞` above `f64::MAX`, `0` below the subnormals).
    pub fn to_f64(&self) -> f64 {
        if self.m == 0.0 || self.e < -1080 {
            0.0
        } else if self.e > 1023 {
            f64::INFINITY
        } else if self.e >= -1022 {
            self.m * pow2(self.e)
        } else {
            // m · 2^(e+200) is normal and exact; the final scaling rounds once.
            self.m * pow2(self.e + 200) * pow2(-200)
        }
    }

    /// Whether the value is normalised: the canonical zero, or `m ∈ [1, 2)`.
    pub fn is_normalized(&self) -> bool {
        (self.m == 0.0 && self.e == 0 && self.m.is_sign_positive()) || (1.0..2.0).contains(&self.m)
    }

    /// Whether the value is zero.
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.m == 0.0
    }
}

impl crate::io::Dump for Ext {
    const DTYPE: &'static str = "extf";
    const BYTES: usize = 16;
    #[inline(always)]
    fn put(&self, out: &mut [u8]) {
        out[..8].copy_from_slice(&self.m.to_le_bytes());
        out[8..16].copy_from_slice(&self.e.to_le_bytes());
    }
    #[inline(always)]
    fn get(inp: &[u8]) -> Self {
        Ext {
            m: f64::from_le_bytes(inp[..8].try_into().expect("8 bytes")),
            e: i64::from_le_bytes(inp[8..16].try_into().expect("8 bytes")),
        }
    }
}

/// Counting `(+, ×)` over extended-exponent floats (see the [module documentation](self)).
#[derive(Clone, Copy, Debug, Default)]
pub struct ExtFloat;

impl ExtFloat {
    /// The instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Semiring for ExtFloat {
    type Elem = Ext;
    fn name(&self) -> &'static str {
        "extfloat"
    }
    /// [`Ext::ZERO`].
    #[inline(always)]
    fn zero(&self) -> Ext {
        Ext::ZERO
    }
    /// [`Ext::ONE`].
    #[inline(always)]
    fn one(&self) -> Ext {
        Ext::ONE
    }
    /// Aligned addition with one rounding; a zero operand returns the other one bitwise.
    #[inline]
    fn plus(&self, a: Ext, b: Ext) -> Ext {
        if a.m == 0.0 {
            return b;
        }
        if b.m == 0.0 {
            return a;
        }
        let (hi, lo) = if a.e > b.e || (a.e == b.e && a.m >= b.m) {
            (a, b)
        } else {
            (b, a)
        };
        let d = hi.e.abs_diff(lo.e);
        if d > 64 {
            return hi;
        }
        let s = hi.m + lo.m * pow2(-(d as i64));
        if s >= 2.0 {
            Ext {
                m: s * 0.5,
                e: hi.e + 1,
            }
        } else {
            Ext { m: s, e: hi.e }
        }
    }
    /// Mantissa product with one rounding, exponent sum; zero annihilates.
    #[inline]
    fn times(&self, a: Ext, b: Ext) -> Ext {
        if a.m == 0.0 || b.m == 0.0 {
            return Ext::ZERO;
        }
        let p = a.m * b.m;
        let e = a.e + b.e;
        if p >= 2.0 {
            Ext {
                m: p * 0.5,
                e: e + 1,
            }
        } else {
            Ext { m: p, e }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_round_trips() {
        for x in [
            1.0,
            1.5,
            3.0,
            0.1,
            1e300,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            1e-310,
        ] {
            let y = Ext::from_f64(x);
            assert!(y.is_normalized(), "{x}");
            assert_eq!(y.to_f64(), x, "{x}");
        }
        assert_eq!(Ext::from_f64(0.0), Ext::ZERO);
        assert_eq!(Ext::from_u64(1), Ext::ONE);
        assert_eq!(Ext::ONE.scale_pow2(2000).to_f64(), f64::INFINITY);
        assert_eq!(Ext::ONE.scale_pow2(-2000).to_f64(), 0.0);
    }

    #[test]
    fn far_operands_are_absorbed() {
        let s = ExtFloat::new();
        let a = Ext::ONE.scale_pow2(100);
        assert_eq!(s.plus(a, Ext::ONE), a);
        assert_eq!(s.plus(Ext::ONE, a), a);
        // 2^53 + 1 rounds to 2^53 (ties to even), 2^53 + 2 is exact.
        let b = Ext::ONE.scale_pow2(53);
        assert_eq!(s.plus(b, Ext::ONE), b);
        assert_eq!(s.plus(b, Ext::from_u64(2)), Ext::from_u64((1 << 53) + 2));
    }
}
