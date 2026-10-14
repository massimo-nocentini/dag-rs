//! Reachability on topologically numbered DAGs.
//!
//! - Approximate ancestor/descendant counts with HyperLogLog sketches in one topological pass
//!   per direction (`hash`, `sketch`; [\[FFGM07\]](crate::references#ffgm07),
//!   [\[QKT16\]](crate::references#qkt16), [\[BRV11\]](crate::references#brv11),
//!   [\[Coh97\]](crate::references#coh97)).
//! - The exact reachability index "ChronoChain": chronological order, level and hi/lo filters,
//!   chain labels, and a bounded exact fallback (pruned DFS with a visit budget, then a window
//!   scan over a [`DirtyBitset`](crate::bitset::DirtyBitset)) (`dagview`, `labels`, `index`,
//!   `bench`; [\[Jag90\]](crate::references#jag90), [\[YCZ10\]](crate::references#ycz10),
//!   [\[HST22\]](crate::references#hst22)).
//!
//! On the Payment Graph both run on the block quotient Q of the
//! [`BlockIndex`](crate::blocks::BlockIndex) and need no BV decoding.

pub mod bench;
pub mod dagview;
pub mod hash;
pub mod index;
pub mod labels;
pub mod sketch;
