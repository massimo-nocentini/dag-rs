//! Counting modulo a prime: exact residues of astronomically large path counts.
//!
//! # Algorithm
//! [`ModP<P>`](ModP) is `(ℤ_P, + mod P, × mod P, 0, 1)` over canonical representatives in
//! `[0, P)`. Addition is one add plus one conditional subtraction (with the carry of the 64-bit
//! add, so every `P < 2⁶⁴` works). Multiplication forms the exact 128-bit product and reduces it:
//! - `P = 2⁶¹ − 1` ([`M61`], a Mersenne prime): since `2⁶¹ ≡ 1`, `x mod P` is obtained by folding
//!   `x = (x & P) + (x >> 61)` twice and one conditional subtraction (F1 §4);
//! - `P = 2⁶⁴ − 2³² + 1` ([`GOLDILOCKS_P`], used by [`Goldilocks`] and
//!   `ModP<GOLDILOCKS_P>`): since `2⁶⁴ ≡ 2³² − 1` and `2⁹⁶ ≡ −1`, the 128-bit product
//!   `x_lo + 2⁶⁴(2³² h₁ + h₀)` reduces to `x_lo − h₁ + h₀(2³² − 1)` with two carry fix-ups and
//!   one conditional subtraction;
//! - any other `P`: the generic `u128 % P` (a 128-bit division).
//!
//! The specialisation is a comparison of the const parameter, resolved at compile time.
//!
//! # Invariants
//! `ℤ_P` is a commutative ring, so all semiring laws hold **exactly**. The residue of a DP equals
//! the true count mod `P` because reduction mod `P` is a ring homomorphism `ℕ → ℤ_P`. Inputs
//! must be canonical (`< P`; checked with `debug_assert!`), and outputs always are. Two coprime
//! moduli (`Product<M61, Goldilocks>`, see [`Product`](super::product::Product)) pin down counts
//! below `M61 · Goldilocks ≈ 2¹²⁵` by the CRT and give a 2⁻¹²⁵-strong fingerprint above.
//!
//! # Complexity
//! O(1) per operation: one 64×64→128 multiply and a few adds for the two special primes.
//!
//! # PG fast path
//! `M61` is the exact half of the full-PG path count (`Product<ExtFloat, M61>`, see [`crate::paths`]):
//! forward and backward totals must agree exactly mod `P`.
//!
//! # References
//! Counting in a semiring: [\[Leh77\]](crate::references#leh77),
//! [\[Moh02\]](crate::references#moh02), [\[GM08\]](crate::references#gm08).
//!
//! ```
//! use dag::semiring::{Semiring, modular::{M61, M61_P}};
//! let s = M61::new();
//! assert_eq!(s.plus(M61_P - 1, 2), 1);
//! assert_eq!(s.times(1 << 40, 1 << 40), 1 << 19); // 2⁸⁰ = 2¹⁹ · 2⁶¹ ≡ 2¹⁹
//! ```

use super::Semiring;

/// The Mersenne prime `2⁶¹ − 1`.
pub const M61_P: u64 = (1u64 << 61) - 1;

/// The Goldilocks prime `2⁶⁴ − 2³² + 1`.
pub const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// `2³² − 1 ≡ 2⁶⁴ (mod GOLDILOCKS_P)`.
const EPSILON: u64 = (1u64 << 32) - 1;

/// Counting modulo the prime `P` (`+`, `×` mod `P`) over canonical residues in `[0, P)`.
///
/// `P` must be at least 2 (checked at compile time when the instance is used); primality is
/// the caller's responsibility (with a composite `P` the structure is still a commutative ring,
/// hence a semiring, but CRT arguments need coprime moduli).
#[derive(Clone, Copy, Debug, Default)]
pub struct ModP<const P: u64>;

impl<const P: u64> ModP<P> {
    /// Compile-time guard on the modulus.
    const VALID: () = assert!(P >= 2, "ModP needs a modulus >= 2");

    /// The instance.
    pub const fn new() -> Self {
        Self
    }

    /// The canonical residue of `x`.
    #[inline(always)]
    pub const fn reduce_u128(x: u128) -> u64 {
        if P == M61_P {
            reduce_m61(x)
        } else if P == GOLDILOCKS_P {
            reduce_goldilocks(x)
        } else {
            (x % P as u128) as u64
        }
    }
}

/// Counting modulo the Mersenne prime `2⁶¹ − 1`.
pub type M61 = ModP<M61_P>;

/// Mersenne reduction of `x < 2¹²⁸` modulo `2⁶¹ − 1`.
#[inline(always)]
const fn reduce_m61(x: u128) -> u64 {
    // x < 2^128: the first fold is < 2^61 + 2^67, the second < 2^61 + 2^7, so one subtraction.
    let y = (x & M61_P as u128) + (x >> 61);
    let y = ((y & M61_P as u128) + (y >> 61)) as u64;
    if y >= M61_P { y - M61_P } else { y }
}

/// Reduction of `x < 2¹²⁸` modulo the Goldilocks prime.
#[inline(always)]
const fn reduce_goldilocks(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let h1 = x_hi >> 32;
    let h0 = x_hi & EPSILON;
    // x ≡ x_lo − h1 + h0·ε. A borrow means we computed (value + 2⁶⁴) ≡ value + ε: subtract ε
    // (cannot underflow: on borrow t0 ≥ 2⁶⁴ − 2³² > ε).
    let (mut t0, borrow) = x_lo.overflowing_sub(h1);
    if borrow {
        t0 -= EPSILON;
    }
    let t1 = h0 * EPSILON; // < 2⁶⁴
    // A carry means we dropped 2⁶⁴ ≡ ε: add it back (cannot overflow, the wrapped sum is
    // ≤ (2³² − 1)² − 1).
    let (s, carry) = t0.overflowing_add(t1);
    let s = if carry { s + EPSILON } else { s };
    if s >= GOLDILOCKS_P {
        s - GOLDILOCKS_P
    } else {
        s
    }
}

/// `(a + b) mod p` for canonical `a, b < p`, valid for every `p < 2⁶⁴`.
#[inline(always)]
const fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    let (s, carry) = a.overflowing_add(b);
    if carry || s >= p {
        s.wrapping_sub(p)
    } else {
        s
    }
}

impl<const P: u64> Semiring for ModP<P> {
    type Elem = u64;
    fn name(&self) -> &'static str {
        match P {
            M61_P => "mod61",
            GOLDILOCKS_P => "goldilocks",
            _ => "modp",
        }
    }
    /// `0`.
    #[inline(always)]
    fn zero(&self) -> u64 {
        let () = Self::VALID;
        0
    }
    /// `1`.
    #[inline(always)]
    fn one(&self) -> u64 {
        let () = Self::VALID;
        1
    }
    /// `(a + b) mod P`.
    #[inline(always)]
    fn plus(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < P && b < P, "ModP operands must be canonical");
        add_mod(a, b, P)
    }
    /// `(a · b) mod P` (exact 128-bit product, specialised reduction for `M61` and Goldilocks).
    #[inline(always)]
    fn times(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < P && b < P, "ModP operands must be canonical");
        Self::reduce_u128(a as u128 * b as u128)
    }
}

/// Counting modulo the Goldilocks prime `2⁶⁴ − 2³² + 1`.
///
/// Identical in values to `ModP<GOLDILOCKS_P>`; kept as a named type because it pairs with
/// [`M61`] in `Product<M61, Goldilocks>` (coprime moduli, CRT range ≈ 2¹²⁵).
#[derive(Clone, Copy, Debug, Default)]
pub struct Goldilocks;

impl Goldilocks {
    /// The instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Semiring for Goldilocks {
    type Elem = u64;
    fn name(&self) -> &'static str {
        "goldilocks"
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
    /// `(a + b) mod P`.
    #[inline(always)]
    fn plus(&self, a: u64, b: u64) -> u64 {
        debug_assert!(
            a < GOLDILOCKS_P && b < GOLDILOCKS_P,
            "Goldilocks operands must be canonical"
        );
        add_mod(a, b, GOLDILOCKS_P)
    }
    /// `(a · b) mod P`.
    #[inline(always)]
    fn times(&self, a: u64, b: u64) -> u64 {
        debug_assert!(
            a < GOLDILOCKS_P && b < GOLDILOCKS_P,
            "Goldilocks operands must be canonical"
        );
        reduce_goldilocks(a as u128 * b as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reductions_match_division_at_the_edges() {
        let edges = [
            0u128,
            1,
            M61_P as u128,
            M61_P as u128 + 1,
            (M61_P as u128) * (M61_P as u128),
            GOLDILOCKS_P as u128,
            (GOLDILOCKS_P as u128 - 1) * (GOLDILOCKS_P as u128 - 1),
            u64::MAX as u128,
            1u128 << 64,
            1u128 << 96,
            (1u128 << 96) - 1,
            u128::MAX,
        ];
        for x in edges {
            assert_eq!(reduce_m61(x) as u128, x % M61_P as u128, "m61 {x}");
            assert_eq!(
                reduce_goldilocks(x) as u128,
                x % GOLDILOCKS_P as u128,
                "gold {x}"
            );
        }
    }

    #[test]
    fn names() {
        assert_eq!(M61::new().name(), "mod61");
        assert_eq!(ModP::<GOLDILOCKS_P>::new().name(), "goldilocks");
        assert_eq!(ModP::<7>::new().name(), "modp");
    }
}
