//! Seeded SplitMix64 hasher for HyperLogLog sketches.
//!
//! HyperBall [\[BV13\]](crate::references#bv13) hashes node ids with
//! `BuildHasherDefault<DefaultHasher>`, which is deterministic but neither seedable nor
//! guaranteed stable across Rust releases. Persisted counts need a hash function that is fixed by
//! a recorded seed, and statistics over repetitions need a family of hash functions indexed by
//! the seed. [`SeededBuildHasher`] provides both with the SplitMix64 generator
//! [\[SLF14\]](crate::references#slf14).
//!
//! # Definition
//! With `γ = 0x9E3779B97F4A7C15` (the golden-ratio increment of SplitMix64) and the
//! finalizer `mix64` (Stafford's "variant 13", the output function of SplitMix64), the hasher
//! starts in state `s₀ = mix64(seed)`, and each 64-bit word `x` written to it updates
//!
//! ```text
//! state ← mix64(state + (x + 1)·γ)            (wrapping arithmetic)
//! ```
//!
//! and [`finish`](std::hash::Hasher::finish) returns `state`. Hence the hash of a single node id
//! `x` (all integer writes are widened to `u64`) is the `(x + 1)`-th output of the SplitMix64
//! generator started in state `s₀`. Since `x ↦ s₀ + (x + 1)·γ` is injective modulo 2⁶⁴ (γ is
//! odd) and `mix64` is a bijection, **distinct node ids never collide** under a fixed seed, and
//! the avalanche of `mix64` gives the uniform-looking hash bits HyperLogLog assumes
//! [\[FFGM07\]](crate::references#ffgm07).
//!
//! **Why the seed is mixed.** Without it (`s₀ = seed`), seeds differing by `k·γ` would give
//! shifted copies of one function, `h_{seed+kγ}(x) = h_seed(x + k)`, and the sketches of the
//! id-contiguous cones of a topologically numbered DAG would be correlated across such seeds.
//! With `s₀ = mix64(seed)` two distinct seeds are shifts of each other by `k` only if
//! `mix64(seed') − mix64(seed) ≡ k·γ`, an event of probability `≈ 2K/2⁶⁴` for `|k| ≤ K`, so
//! distinct seeds behave as independent hash functions for all practical purposes. As
//! `mix64(0) = 0`, seed 0 is the plain SplitMix64 stream from state 0 (the reference outputs of
//! `splitmix64.c`).
//!
//! Byte writes (used only by non-integer keys) are packed into little-endian 64-bit words; a
//! final partial word is tagged with its length so that, e.g., `[0]` and `[0, 0]` differ.
//!
//! Complexity: O(1) per word, three multiplications.

use std::hash::{BuildHasher, Hasher};

/// The SplitMix64 increment `γ = ⌊2⁶⁴/φ⌋`, made odd.
pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The SplitMix64 output function (Stafford's mix variant 13), a bijection of `u64`.
///
/// ```
/// use dag::reach::hash::{GAMMA, mix64};
/// // the first output of SplitMix64 seeded with 0
/// assert_eq!(mix64(GAMMA), 0xE220_A839_7B1D_CDAF);
/// ```
#[inline(always)]
pub const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A [`BuildHasher`] whose hashers are SplitMix64 started in state `mix64(seed)` (see the module
/// documentation). Use it as the hasher of the HyperLogLog logic of
/// [`sketch`](crate::reach::sketch); record `seed` with the outputs.
///
/// ```
/// use std::hash::BuildHasher;
/// use dag::reach::hash::SeededBuildHasher;
/// let h = SeededBuildHasher { seed: 42 };
/// assert_eq!(h.hash_one(7usize), h.hash_one(7u64));
/// assert_ne!(h.hash_one(7usize), SeededBuildHasher { seed: 43 }.hash_one(7usize));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct SeededBuildHasher {
    /// The seed; every hasher starts in state `mix64(seed)`.
    pub seed: u64,
}

impl SeededBuildHasher {
    /// A builder with the given seed.
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The initial state `mix64(seed)` of every hasher.
    #[inline(always)]
    pub const fn state(&self) -> u64 {
        mix64(self.seed)
    }

    /// The hash of a single node id: the `(x + 1)`-th SplitMix64 output from state
    /// [`state()`](Self::state).
    #[inline(always)]
    pub const fn hash_u64(&self, x: u64) -> u64 {
        mix64(
            self.state()
                .wrapping_add(x.wrapping_add(1).wrapping_mul(GAMMA)),
        )
    }
}

impl BuildHasher for SeededBuildHasher {
    type Hasher = SplitMix64Hasher;
    #[inline(always)]
    fn build_hasher(&self) -> SplitMix64Hasher {
        SplitMix64Hasher {
            state: self.state(),
        }
    }
}

/// The [`Hasher`] of [`SeededBuildHasher`]: SplitMix64 steps over the written words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64Hasher {
    state: u64,
}

impl Hasher for SplitMix64Hasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.state
    }

    #[inline(always)]
    fn write_u64(&mut self, x: u64) {
        self.state = mix64(
            self.state
                .wrapping_add(x.wrapping_add(1).wrapping_mul(GAMMA)),
        );
    }

    #[inline(always)]
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }

    #[inline(always)]
    fn write_u32(&mut self, x: u32) {
        self.write_u64(u64::from(x));
    }

    #[inline(always)]
    fn write_u16(&mut self, x: u16) {
        self.write_u64(u64::from(x));
    }

    #[inline(always)]
    fn write_u8(&mut self, x: u8) {
        self.write_u64(u64::from(x));
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut words = bytes.chunks_exact(8);
        for w in &mut words {
            self.write_u64(u64::from_le_bytes(w.try_into().expect("8-byte chunk")));
        }
        let rest = words.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            // the length tag lives in the top byte, which a 1..=7-byte tail never fills
            self.write_u64(u64::from_le_bytes(buf) | ((rest.len() as u64) << 56));
        }
    }
}
