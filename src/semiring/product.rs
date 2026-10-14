//! Componentwise product of two semirings: several DPs in one sweep.
//!
//! # Algorithm
//! [`Product<A, B>`](Product) is `A × B` with every operation applied componentwise; 0̄ and 1̄
//! are the pairs of the components' identities ([\[GM08\]](crate::references#gm08)). One DP
//! over the product computes both component DPs in a single pass over the graph, halving the
//! decoding cost (e.g. `Product<ExtFloat, M61>` for the magnitude and the exact residue of path
//! counts, see [`crate::paths`]; `Product<M61, Goldilocks>` for CRT-exact counts below ≈ 2¹²⁵).
//!
//! # Invariants
//! - Every semiring law holds componentwise, hence with the same exactness as in the
//!   components; the exact identities of [`Semiring`] carry over bitwise.
//! - `IDEMPOTENT = A::IDEMPOTENT && B::IDEMPOTENT`.
//! - **`SELECTIVE = false` always**: even when both components are selective,
//!   `(a₁, a₂) ⊕ (b₁, b₂)` may be `(a₁, b₂)`, which is neither operand. The DP kernels therefore never track a `pred` for products.
//! - Elements are stored as the pair `Dump` (`"pair"`, `A::BYTES + B::BYTES`).
//!
//! # Complexity
//! The sum of the components' costs.
//!
//! ```
//! use dag::semiring::{Semiring, MaxPlus, MinPlus, product::Product};
//! let s = Product::new(MaxPlus::<u32>::new(), MinPlus::<u32>::new());
//! assert_eq!(s.plus((3, 3), (5, 5)), (5, 3)); // longest and shortest at once
//! assert!(!<Product<MaxPlus<u32>, MinPlus<u32>> as Semiring>::SELECTIVE);
//! ```

use super::Semiring;

/// The componentwise product of two semirings (`SELECTIVE = false` always).
#[derive(Clone, Copy, Debug, Default)]
pub struct Product<A, B>(pub A, pub B);

impl<A, B> Product<A, B> {
    /// The product of `a` and `b`.
    pub const fn new(a: A, b: B) -> Self {
        Self(a, b)
    }
}

impl<A: Semiring, B: Semiring> Semiring for Product<A, B> {
    type Elem = (A::Elem, B::Elem);
    const IDEMPOTENT: bool = A::IDEMPOTENT && B::IDEMPOTENT;
    const SELECTIVE: bool = false;
    fn name(&self) -> &'static str {
        "product"
    }
    /// `(0̄_A, 0̄_B)`.
    #[inline(always)]
    fn zero(&self) -> Self::Elem {
        (self.0.zero(), self.1.zero())
    }
    /// `(1̄_A, 1̄_B)`.
    #[inline(always)]
    fn one(&self) -> Self::Elem {
        (self.0.one(), self.1.one())
    }
    /// Componentwise `⊕`.
    #[inline(always)]
    fn plus(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        (self.0.plus(a.0, b.0), self.1.plus(a.1, b.1))
    }
    /// Componentwise `⊗`.
    #[inline(always)]
    fn times(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        (self.0.times(a.0, b.0), self.1.times(a.1, b.1))
    }
}
