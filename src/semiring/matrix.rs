//! 2×2 matrices over a semiring: a non-commutative semiring for tests and doc examples.
//!
//! # Algorithm
//! [`Matrix2<S>`](Matrix2) is the semiring of 2×2 matrices over `S`: `⊕` is elementwise,
//! `(A ⊗ B)[i][j] = (A[i][0] ⊗ B[0][j]) ⊕ (A[i][1] ⊗ B[1][j])`, 0̄ is the all-0̄ matrix and 1̄
//! the identity (1̄ on the diagonal, 0̄ elsewhere) ([\[GM08\]](crate::references#gm08)). Its `⊗`
//! is **not commutative**, so a DP kernel that multiplies in the wrong order (for instance
//! `α(u) ⊗ F(u)` instead of `F(u) ⊗ α(u)`, or the backward fold on the wrong side) gives a
//! different result. The forward/backward duality `⊕_v F(v) ⊗ c(v) = ⊕_u b(u) ⊗ B(u)` checked
//! with this semiring therefore pins the operand order of the DP kernels ([`crate::dp`]).
//!
//! # Invariants
//! - The semiring laws hold whenever they hold for `S` (matrices over a semiring form a
//!   semiring). With `S = MinPlus<i64>` (the instance used in tests) they are exact as long as
//!   no saturation occurs (entries `|x| ≤ 2⁶⁰`).
//! - The exact identities of [`Semiring`] hold bitwise whenever they hold bitwise in `S`:
//!   `(1̄ ⊗ X)[i][j] = (1̄ ⊗ X[i][j]) ⊕ (0̄ ⊗ X[k][j]) = X[i][j] ⊕ 0̄ = X[i][j]`, and similarly on
//!   the right and for 0̄.
//! - `IDEMPOTENT = S::IDEMPOTENT`; never `SELECTIVE` (elementwise `⊕` mixes operands).
//!
//! # Complexity
//! 8 `⊗` and 4 `⊕` of `S` per matrix product; 4 elements per value. `Dump` is implemented for
//! `Mat2<i64>` (`"m2i64le"`), `Mat2<u32>`, `Mat2<u64>`, `Mat2<f64>` and `Mat2<bool>`.
//!
//! ```
//! use dag::semiring::{Semiring, MinPlus, matrix::{Mat2, Matrix2}};
//! let s = Matrix2::new(MinPlus::<i64>::new());
//! let a = Mat2([[0, 5], [1, 0]]);
//! let b = Mat2([[3, 0], [0, 7]]);
//! assert_ne!(s.times(a, b), s.times(b, a));
//! assert_eq!(s.times(s.one(), a), a);
//! ```

use super::Semiring;

/// A 2×2 matrix (row major).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat2<E>(pub [[E; 2]; 2]);

macro_rules! mat2_dump {
    ($t:ty, $name:literal) => {
        impl crate::io::Dump for Mat2<$t> {
            const DTYPE: &'static str = $name;
            const BYTES: usize = 4 * <$t as crate::io::Dump>::BYTES;
            #[inline]
            fn put(&self, out: &mut [u8]) {
                let b = <$t as crate::io::Dump>::BYTES;
                for (k, x) in self.0.iter().flatten().enumerate() {
                    x.put(&mut out[k * b..(k + 1) * b]);
                }
            }
            #[inline]
            fn get(inp: &[u8]) -> Self {
                let b = <$t as crate::io::Dump>::BYTES;
                let g = |k: usize| <$t as crate::io::Dump>::get(&inp[k * b..(k + 1) * b]);
                Mat2([[g(0), g(1)], [g(2), g(3)]])
            }
        }
    };
}

mat2_dump!(i64, "m2i64le");
mat2_dump!(u64, "m2u64le");
mat2_dump!(u32, "m2u32le");
mat2_dump!(f64, "m2f64le");
mat2_dump!(bool, "m2u8");

/// 2×2 matrices over the semiring `S` (non-commutative `⊗`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Matrix2<S>(pub S);

impl<S> Matrix2<S> {
    /// The matrix semiring over `s`.
    pub const fn new(s: S) -> Self {
        Self(s)
    }
}

impl<S: Semiring> Semiring for Matrix2<S>
where
    Mat2<S::Elem>: crate::io::Dump,
{
    type Elem = Mat2<S::Elem>;
    const IDEMPOTENT: bool = S::IDEMPOTENT;
    const SELECTIVE: bool = false;
    fn name(&self) -> &'static str {
        "matrix2"
    }
    /// The all-0̄ matrix.
    #[inline]
    fn zero(&self) -> Self::Elem {
        let z = self.0.zero();
        Mat2([[z, z], [z, z]])
    }
    /// The identity matrix.
    #[inline]
    fn one(&self) -> Self::Elem {
        let (z, o) = (self.0.zero(), self.0.one());
        Mat2([[o, z], [z, o]])
    }
    /// Elementwise `⊕`.
    #[inline]
    fn plus(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        let s = &self.0;
        Mat2(core::array::from_fn(|i| {
            core::array::from_fn(|j| s.plus(a.0[i][j], b.0[i][j]))
        }))
    }
    /// Matrix product over `S` (each entry folded in ascending `k`).
    #[inline]
    fn times(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        let s = &self.0;
        Mat2(core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                s.plus(s.times(a.0[i][0], b.0[0][j]), s.times(a.0[i][1], b.0[1][j]))
            })
        }))
    }
}
