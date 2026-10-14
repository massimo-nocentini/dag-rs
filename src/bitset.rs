//! `DirtyBitset`: a range-offset bitset with O(touched words) reset.
//!
//! Searches that run many times over overlapping id ranges (the reachability fallback, TR
//! searches, antichain checks) must never allocate or clear an n-sized array per query or per
//! thread. A `DirtyBitset` covers an id range `[base, base + len)`, remembers which 64-bit words
//! it touched, and [`reset`](DirtyBitset::reset) clears only those words (growing the storage
//! when a larger range is requested). Memory: `len/8` bytes plus 4 bytes per touched word.

/// Bitset over an id range `[base, base + len)` with O(touched words) reset. Capacity grows on
/// demand.
#[derive(Clone, Debug, Default)]
pub struct DirtyBitset {
    base: usize,
    len: usize,
    words: Vec<u64>,
    dirty: Vec<u32>,
}

impl DirtyBitset {
    /// An empty bitset over the empty range.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the set (O(touched words)) and makes it cover `[base, base + len)`.
    pub fn reset(&mut self, base: usize, len: usize) {
        for &w in &self.dirty {
            self.words[w as usize] = 0;
        }
        self.dirty.clear();
        let nw = len.div_ceil(64);
        assert!(nw <= u32::MAX as usize + 1, "DirtyBitset range too large");
        if self.words.len() < nw {
            self.words.resize(nw, 0);
        }
        self.base = base;
        self.len = len;
    }

    /// Inserts `x` (which must lie in the range); returns whether it was newly inserted.
    #[inline]
    pub fn insert(&mut self, x: usize) -> bool {
        let i = self.index(x);
        let (w, b) = (i / 64, 1u64 << (i % 64));
        let word = &mut self.words[w];
        if *word & b != 0 {
            return false;
        }
        if *word == 0 {
            self.dirty.push(w as u32);
        }
        *word |= b;
        true
    }

    /// Whether `x` is in the set (`false` outside the range).
    #[inline]
    pub fn contains(&self, x: usize) -> bool {
        if x < self.base || x - self.base >= self.len {
            return false;
        }
        let i = x - self.base;
        self.words[i / 64] & (1u64 << (i % 64)) != 0
    }

    /// The set elements `≥ x`, ascending.
    pub fn iter_from(&self, x: usize) -> impl Iterator<Item = usize> + '_ {
        let start = x.max(self.base) - self.base;
        let nw = self.len.div_ceil(64);
        let first_w = start / 64;
        let base = self.base;
        let len = self.len;
        (first_w..nw)
            .flat_map(move |w| {
                let mut word = self.words[w];
                if w == first_w {
                    word &= u64::MAX.checked_shl((start % 64) as u32).unwrap_or(0);
                }
                BitIter(word).map(move |b| base + w * 64 + b)
            })
            .take_while(move |&v| v < base + len)
    }

    /// Bytes of heap memory used.
    pub fn bytes(&self) -> usize {
        self.words.capacity() * 8 + self.dirty.capacity() * 4
    }

    /// First id of the range.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Length of the range.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn index(&self, x: usize) -> usize {
        assert!(
            x >= self.base && x - self.base < self.len,
            "DirtyBitset: {x} outside [{}, {})",
            self.base,
            self.base + self.len
        );
        x - self.base
    }
}

/// Iterator over the set bits of a word, ascending.
struct BitIter(u64);

impl Iterator for BitIter {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let b = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(b)
    }
}
