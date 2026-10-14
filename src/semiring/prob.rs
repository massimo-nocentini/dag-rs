//! Probability semirings over `f64`: path probability mass and the most probable path.
//!
//! # Algorithm
//! - [`Probability`] is `(ℝ≥0 ∪ {+∞}, +, ×, 0, 1)`: the total weight of all paths (e.g. the mass
//!   reaching a node when every node splits its mass over its out-arcs with `inv-outdeg`
//!   weights).
//! - [`Viterbi`] is `([0, ∞], max, ×, 0, 1)`: the weight of the best path
//!   ([\[Vit67\]](crate::references#vit67)); it is isomorphic to `MinPlus` on `−log p`.
//!   It is selective, so the DP kernels can track the arg-max predecessor.
//!
//! # Invariants
//! - **Exact identities**: `0̄ = +0.0` is special-cased in both operations, so
//!   `0 ⊕ x = x ⊕ 0 = x` bitwise (a plain `0.0 + (−0.0)` would give `+0.0`) and
//!   `0 ⊗ x = x ⊗ 0 = +0.0` bitwise (a plain product gives `−0.0` for negative `x` and `NaN` for
//!   `x = ∞`, so `0 · ∞ = 0` by convention, as in measure theory). `1 ⊗ x = x` bitwise for every
//!   domain value including `−0.0`: a `−0.0` operand is treated as zero but keeps its sign unless
//!   the other operand is `0̄` (`1 ⊗ (−0.0) = −0.0 ⊗ ∞ = −0.0`).
//! - `⊕` is exactly commutative. For `Probability` associativity and distributivity hold up to
//!   rounding (relative error `≤ (k + 2) · u` for a fold of `k` terms,
//!   [\[Hig93\]](crate::references#hig93)); for `Viterbi`, `⊕` is exact and distributivity is
//!   exact on non-negative values (rounding is monotone), while `⊗` is associative up to rounding.
//! - Domain: non-negative values (and `+∞`); `NaN` inputs are not supported.
//!
//! # Complexity
//! O(1) per operation, 8 bytes per element.
//!
//! # References
//! [\[Vit67\]](crate::references#vit67), [\[Moh02\]](crate::references#moh02),
//! [\[GM08\]](crate::references#gm08), [\[Goo99\]](crate::references#goo99).
//!
//! ```
//! use dag::semiring::{Semiring, prob::{Probability, Viterbi}};
//! let (p, v) = (Probability::new(), Viterbi::new());
//! assert_eq!(p.plus(0.25, p.times(0.5, 0.5)), 0.5);
//! assert_eq!(v.plus(0.25, v.times(0.5, 0.75)), 0.375);
//! assert_eq!(p.times(p.zero(), f64::INFINITY), 0.0);
//! ```

use super::Semiring;

/// `x ⊗ y` for both instances. `0̄ = +0.0` annihilates and yields `+0.0`; `−0.0` (which lies in
/// the non-negative domain) behaves as in IEEE arithmetic on non-negative operands, so
/// `1 ⊗ (−0.0) = −0.0` bitwise, and by the same measure-theoretic convention `−0.0 ⊗ ∞ = −0.0`.
#[inline(always)]
fn mul0(a: f64, b: f64) -> f64 {
    if a.to_bits() == 0 || b.to_bits() == 0 {
        0.0
    } else if a == 0.0 || b == 0.0 {
        -0.0
    } else {
        a * b
    }
}

/// Probabilities `(+, ×)` over `f64` (see the [module documentation](self)).
#[derive(Clone, Copy, Debug, Default)]
pub struct Probability;

impl Probability {
    /// The instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Semiring for Probability {
    type Elem = f64;
    fn name(&self) -> &'static str {
        "prob"
    }
    /// `+0.0`.
    #[inline(always)]
    fn zero(&self) -> f64 {
        0.0
    }
    /// `1.0`.
    #[inline(always)]
    fn one(&self) -> f64 {
        1.0
    }
    /// `a + b`; `+0.0` returns the other operand bitwise.
    #[inline(always)]
    fn plus(&self, a: f64, b: f64) -> f64 {
        if a.to_bits() == 0 {
            b
        } else if b.to_bits() == 0 {
            a
        } else {
            a + b
        }
    }
    /// `a · b`; zero annihilates (`0 · ∞ = 0`).
    #[inline(always)]
    fn times(&self, a: f64, b: f64) -> f64 {
        mul0(a, b)
    }
}

/// Viterbi `(max, ×)` over `f64` (see the [module documentation](self)).
#[derive(Clone, Copy, Debug, Default)]
pub struct Viterbi;

impl Viterbi {
    /// The instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Semiring for Viterbi {
    type Elem = f64;
    const IDEMPOTENT: bool = true;
    const SELECTIVE: bool = true;
    fn name(&self) -> &'static str {
        "viterbi"
    }
    /// `+0.0`.
    #[inline(always)]
    fn zero(&self) -> f64 {
        0.0
    }
    /// `1.0`.
    #[inline(always)]
    fn one(&self) -> f64 {
        1.0
    }
    /// `max(a, b)`; `+0.0` returns the other operand bitwise.
    #[inline(always)]
    fn plus(&self, a: f64, b: f64) -> f64 {
        if a.to_bits() == 0 {
            b
        } else if b.to_bits() == 0 || a >= b {
            a
        } else {
            b
        }
    }
    /// `a · b`; zero annihilates (`0 · ∞ = 0`).
    #[inline(always)]
    fn times(&self, a: f64, b: f64) -> f64 {
        mul0(a, b)
    }
}
