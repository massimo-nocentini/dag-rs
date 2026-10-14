//! The bottleneck `(max, min)` semiring: widest (maximum-capacity) paths.
//!
//! # Algorithm
//! [`MaxMin<T>`](MaxMin) is `(T, max, min, T::MIN, T::MAX)`: the value of a path is its smallest
//! arc capacity, and a node's value is the best such bottleneck over its paths
//! ([\[GM08\]](crate::references#gm08), [\[Moh02\]](crate::references#moh02)). Instances exist for
//! `u32`, `u64`, `i64` and `f64` (`0̄ = −∞`, `1̄ = +∞` for `f64`).
//!
//! # Invariants
//! - The structure is a distributive lattice, so every semiring law holds **exactly** (also for
//!   `f64`, where `max`/`min` never round). It is idempotent and selective, so the DP kernels can
//!   track the arg-best predecessor.
//! - **Quirk**: `1̄ = T::MAX`, which for `MaxMin<u32>` equals the crate-wide
//!   sentinel [`NONE`](crate::NONE). Here it is a value ("unbounded capacity", e.g. the
//!   bottleneck of a length-0 path), **not** "no value"; `0̄ = 0` ("no path") for the unsigned
//!   instances.
//! - Exact identities bitwise: the `f64` instance special-cases `±∞`, so
//!   `max(−∞, x)` and `min(+∞, x)` return `x` bitwise (including `x = −0.0`); ties are broken by
//!   [`f64::total_cmp`] (`−0.0 < +0.0`), so `⊕` and `⊗` are commutative bitwise.
//!
//! # Complexity
//! O(1) per operation.
//!
//! # References
//! [\[GM08\]](crate::references#gm08), [\[Moh02\]](crate::references#moh02).
//!
//! ```
//! use dag::semiring::{Semiring, bottleneck::MaxMin};
//! let s = MaxMin::<u32>::new();
//! // two paths with capacities (5, 3) and (4, 4): the widest has bottleneck 4
//! assert_eq!(s.plus(s.times(5, 3), s.times(4, 4)), 4);
//! assert_eq!(s.one(), u32::MAX); // "unbounded", a value
//! ```

use super::Semiring;
use core::marker::PhantomData;

/// The `(max, min)` semiring: widest paths. Its 1̄ is `T::MAX` ("unbounded capacity").
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxMin<T>(pub PhantomData<T>);

impl<T> MaxMin<T> {
    /// The instance.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

macro_rules! maxmin_int {
    ($t:ty, $name:literal) => {
        impl Semiring for MaxMin<$t> {
            type Elem = $t;
            const IDEMPOTENT: bool = true;
            const SELECTIVE: bool = true;
            fn name(&self) -> &'static str {
                $name
            }
            /// `T::MIN` ("no path").
            #[inline(always)]
            fn zero(&self) -> $t {
                <$t>::MIN
            }
            /// `T::MAX` ("unbounded capacity"; a value, not a sentinel).
            #[inline(always)]
            fn one(&self) -> $t {
                <$t>::MAX
            }
            /// `max(a, b)`.
            #[inline(always)]
            fn plus(&self, a: $t, b: $t) -> $t {
                a.max(b)
            }
            /// `min(a, b)`.
            #[inline(always)]
            fn times(&self, a: $t, b: $t) -> $t {
                a.min(b)
            }
        }
    };
}

maxmin_int!(u32, "maxmin-u32");
maxmin_int!(u64, "maxmin-u64");
maxmin_int!(i64, "maxmin-i64");

impl Semiring for MaxMin<f64> {
    type Elem = f64;
    const IDEMPOTENT: bool = true;
    const SELECTIVE: bool = true;
    fn name(&self) -> &'static str {
        "maxmin-f64"
    }
    /// `−∞`.
    #[inline(always)]
    fn zero(&self) -> f64 {
        f64::NEG_INFINITY
    }
    /// `+∞`.
    #[inline(always)]
    fn one(&self) -> f64 {
        f64::INFINITY
    }
    /// `max(a, b)` in the [`f64::total_cmp`] order; `−∞` returns the other operand bitwise.
    #[inline(always)]
    fn plus(&self, a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY {
            b
        } else if b == f64::NEG_INFINITY || a.total_cmp(&b).is_ge() {
            a
        } else {
            b
        }
    }
    /// `min(a, b)` in the [`f64::total_cmp`] order; `+∞` returns the other operand bitwise.
    #[inline(always)]
    fn times(&self, a: f64, b: f64) -> f64 {
        if a == f64::INFINITY {
            b
        } else if b == f64::INFINITY || a.total_cmp(&b).is_le() {
            a
        } else {
            b
        }
    }
}
