//! Saturating counting semirings: exact path counts, capped at the type maximum.
//!
//! # Algorithm
//! [`SatU64`] and [`SatU128`] are `([0, M], ⊕, ⊗, 0, 1)` with `M = T::MAX`,
//! `a ⊕ b = min(a + b, M)` and `a ⊗ b = min(a · b, M)`, implemented with the hardware
//! `saturating_add` / `saturating_mul` (one add or multiply plus an overflow flag).
//!
//! # Invariants
//! The map `h(x) = min(x, M)` is a semiring homomorphism from `(ℕ, +, ×, 0, 1)` onto this
//! structure: the congruence that identifies all values `≥ M` is compatible with `+` and `×`
//! (if `x ≥ M` then `x + y ≥ M`, and `x · y ≥ M` unless `y = 0`, in which case both sides are 0).
//! The quotient is therefore a semiring and every DP over it returns exactly
//! `min(true count, M)`: the value is **exact when `< M`**, and `M` means "**at least** `M`"
//! (F1 §4). All semiring laws hold exactly, including at the cap; `0 ⊗ M = 0` because `0` is
//! absorbing in ℕ.
//!
//! # Complexity
//! O(1) per operation; elements are 8 or 16 bytes.
//!
//! # PG fast path
//! On the Payment Graph path counts overflow `u128` on most nodes (their log2 reaches millions),
//! so the full-graph counts use [`ExtFloat`](super::extfloat::ExtFloat) and
//! [`M61`](super::modular::M61); these instances serve windows, small graphs, and the
//! "saturated nodes" statistic.
//!
//! # References
//! Counting paths as a semiring DP: [\[Leh77\]](crate::references#leh77),
//! [\[Moh02\]](crate::references#moh02), [\[GM08\]](crate::references#gm08).
//!
//! ```
//! use dag::semiring::{Semiring, counting::SatU64};
//! let s = SatU64::new();
//! assert_eq!(s.plus(u64::MAX - 1, 5), u64::MAX); // saturates: "≥ u64::MAX"
//! assert_eq!(s.times(s.zero(), u64::MAX), 0);    // 0 still annihilates the cap
//! ```

use super::Semiring;

/// Saturating `u64` counting `(+, ×)`; `u64::MAX` means "≥ `u64::MAX`".
///
/// Exact: the semiring is the quotient of `(ℕ, +, ×)` by "all values ≥ `u64::MAX` are equal"
/// (see the [module documentation](self)).
#[derive(Clone, Copy, Debug, Default)]
pub struct SatU64;

impl SatU64 {
    /// The instance.
    pub const fn new() -> Self {
        Self
    }

    /// Whether `x` is the saturation value (the true count is `≥ u64::MAX`).
    #[inline(always)]
    pub const fn is_saturated(x: u64) -> bool {
        x == u64::MAX
    }
}

impl Semiring for SatU64 {
    type Elem = u64;
    fn name(&self) -> &'static str {
        "sat64"
    }
    /// `0`.
    #[inline(always)]
    fn zero(&self) -> u64 {
        0
    }
    /// `1`.
    #[inline(always)]
    fn one(&self) -> u64 {
        1
    }
    /// `min(a + b, u64::MAX)`.
    #[inline(always)]
    fn plus(&self, a: u64, b: u64) -> u64 {
        a.saturating_add(b)
    }
    /// `min(a · b, u64::MAX)`.
    #[inline(always)]
    fn times(&self, a: u64, b: u64) -> u64 {
        a.saturating_mul(b)
    }
}

/// Saturating `u128` counting `(+, ×)`; `u128::MAX` means "≥ `u128::MAX`".
///
/// Exact in the same sense as [`SatU64`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SatU128;

impl SatU128 {
    /// The instance.
    pub const fn new() -> Self {
        Self
    }

    /// Whether `x` is the saturation value (the true count is `≥ u128::MAX`).
    #[inline(always)]
    pub const fn is_saturated(x: u128) -> bool {
        x == u128::MAX
    }
}

impl Semiring for SatU128 {
    type Elem = u128;
    fn name(&self) -> &'static str {
        "sat128"
    }
    /// `0`.
    #[inline(always)]
    fn zero(&self) -> u128 {
        0
    }
    /// `1`.
    #[inline(always)]
    fn one(&self) -> u128 {
        1
    }
    /// `min(a + b, u128::MAX)`.
    #[inline(always)]
    fn plus(&self, a: u128, b: u128) -> u128 {
        a.saturating_add(b)
    }
    /// `min(a · b, u128::MAX)`.
    #[inline(always)]
    fn times(&self, a: u128, b: u128) -> u128 {
        a.saturating_mul(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturation_edges() {
        let s = SatU64::new();
        assert_eq!(s.plus(u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(s.times(1 << 32, 1 << 32), u64::MAX);
        assert_eq!(s.times((1 << 32) - 1, 1 << 32), u64::MAX - (1 << 32) + 1);
        assert!(SatU64::is_saturated(s.times(u64::MAX, 2)));
        let t = SatU128::new();
        assert_eq!(t.times(u128::MAX, 0), 0);
        assert_eq!(t.plus(u128::MAX - 1, 1), u128::MAX);
    }
}
