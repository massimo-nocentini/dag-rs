//! Approximate ancestor/descendant counts with HyperLogLog sketches.
//!
//! For every node `v` of a topologically numbered DAG (or of a window of it) this module
//! estimates `|anc*(v)|`, the number of **proper** ancestors, and `|desc*(v)|`, the number of
//! proper descendants, with one pass per direction.
//!
//! # Algorithm
//! Counting `|anc(v)|` exactly for every `v` is as hard as the transitive closure, and ancestor
//! sets overlap, so counts cannot be summed over predecessors. Cohen's size-estimation framework
//! [\[Coh97\]](crate::references#coh97) propagates a **mergeable sketch** instead: any sketch
//! with `S(A ∪ B) = S(A) ⊕ S(B)` works. HyperLogLog [\[FFGM07\]](crate::references#ffgm07) keeps
//! `m = 2^p` registers of 5 bits with relative standard deviation `rsd ≈ 1.04/√m`, and its merge
//! is a register-wise maximum (broadword, [\[BRV11\]](crate::references#brv11)). HyperBall
//! [\[BV13\]](crate::references#bv13) iterates this union once per distance; on a DAG the closure
//! is reached in **one** pass in topological order:
//!
//! ```text
//! S_anc(v)  = ⊕_{u ∈ pred(v)} ( {u} ⊕ S_anc(u) )      ascending ids
//! S_desc(u) = ⊕_{v ∈ succ(u)} ( {v} ⊕ S_desc(v) )     descending ids
//! ```
//!
//! **Lemma (exactness of propagation, F3 Lemma 4).** With a fixed hash function, the recurrence
//! evaluated in topological order yields `S_anc(v) = HLL(anc*(v))` (resp. `HLL(desc*(v))`),
//! because the register-wise maximum is associative, commutative and idempotent. Hence:
//! - the sketch of every node is **bit-identical** to the sketch obtained by inserting the exact
//!   set (the test oracle), and the per-node error is that of a single HyperLogLog counter;
//! - errors of different nodes are correlated (they share the hash); aggregate statistics should
//!   average over independent seeds of [`SeededBuildHasher`](crate::reach::hash::SeededBuildHasher).
//!
//! The estimator is card-est-array's default: LogLog-β [\[QKT16\]](crate::references#qkt16)
//! whenever some register is zero (it needs no small-range correction), the raw HyperLogLog
//! estimate otherwise; the empty sketch estimates exactly 0. See also the engineering notes of
//! [\[HNH13\]](crate::references#hnh13).
//!
//! Outputs are **proper** counts as `f32`: `anc = 0` exactly for sources and `desc = 0`
//! exactly for sinks. [`reflexive_sketch_estimates`] gives the reflexive variant
//! (`|{v} ∪ desc*(v)|`), which equals HyperBall's `reachable_nodes()` bit for bit when both use
//! the same hasher, `num_elements` and `log2m`.
//!
//! # Generic mode ([`counts_generic`])
//! One array of `n_w` sketches per direction (`n_w = |window|`, `2^p·5/8` bytes each), refused
//! by [`MemBudget`] when too large. Ancestors: forward push over `g` with the
//! [`sweep_forward`] engine; after visiting `u`, `{u}` is added **in place** to `S[u]` (no
//! longer needed as a proper sketch) and `S[u]` is merged into every successor
//! (`split_at_mut`, since `u < v`, so no scratch copy is needed). Descendants: pull over `g`
//! with [`sweep_backward`]. The two directions run one after the other, so the peak memory is
//! one array.
//!
//! # PG fast path ([`counts_blocks`])
//! On a biclique digraph with [`BlockIndex`] `(create, spend, first, len)`, all outputs of a block
//! `B` share their proper ancestors `A(B)`, and all inputs of `X` share their proper descendants
//! `D(X) ⊇ O(X)`. Both passes are array scans, with no BV decoding:
//!
//! ```text
//! ANCESTORS (u ascending)                        DESCENDANTS (w descending)
//! B = create[u]                                  X = spend[w]; B = create[w]
//! if u == first(B): close A(B), estimate it      if B ≠ ⊥: open acc[B] lazily
//! X = spend[u]                                     acc[B] ⊕= {w}; acc[B] ⊕= D(X)
//! if X ≠ ⊥: open acc[X] lazily                     if w == first(B): close D(B) = acc[B]
//!   acc[X] ⊕= {u}; acc[X] ⊕= A(B)                if X ≠ ⊥: refcnt[X] −= 1, free D(X) at 0
//! if u == last(B): free A(B)
//! ```
//!
//! Sketches live in a slab: a pool of 8 MiB pages (allocated with [`huge_vec`], hence advised
//! for transparent huge pages, D11) with a free list and `slot_of[block]`; the live set
//! is at most `LB(t) + 1` in either pass, where `LB(t)` is the number of blocks with an input
//! `≤ t` and first output `> t` (5.14·10⁶ at most on the 2022 PG, 3.3 GB per pass at `p = 10`;
//! both passes run concurrently, so 6.6 GB). A
//! per-slot memo skips merging the same source sketch twice in a row (idempotent, so the result
//! is unchanged), and merges use [`merge_max5`], a branch-free broadword maximum on 320-bit
//! groups. Closed sketches are copied into a buffer of [`ESTIMATE_BATCH`] sketches; **two
//! buffers alternate**: a full buffer is estimated by a parallel task spawned inside a
//! `rayon::in_place_scope` while the pass fills the other one. The ancestor and descendant
//! passes run concurrently (`rayon::join`); progress is reported by the calling thread from an
//! atomic counter, so no logger crosses threads.
//!
//! Windows `[lo, hi)`: the blocks of the window are `[blo, bhi)` with `blo =`
//! [`block_bound`](BlockIndex::block_bound)`(lo)`, `bhi = block_bound(hi)`, i.e. the blocks whose
//! first output lies in the window. Pushes to blocks `≥ bhi` are skipped, a block straddling `lo`
//! has no input in the window (its outputs are sources of the induced subgraph), the descendant
//! reference counts count only inputs inside the window, and a block straddling `hi` is freed at
//! its last output inside the window: no slot leaks ([`PassStats::live_at_end`] is 0). Because
//! the window is convex, the counts equal those of the induced subgraph.
//!
//! # Complexity
//! Time `O((n_w + m_w)·2^p/64)` word operations per direction for the generic mode, and
//! `O(n_w·2^p/64)` for the block mode (one merge and one insertion per node with a spender or a
//! block), plus `O(2^p)` per estimate, done in parallel. Memory: generic `n_w·2^p·5/8` bytes;
//! blocks, in two phases. **Pass phase** (both passes run concurrently, so their footprints add
//! up), per pass: `LB_max·2^p·5/8` bytes of sketches (plus less than one 8 MiB page),
//! `4·nb_w` bytes of `slot_of` and `4·nb_w` bytes of per-block `f32` outputs, plus `4·nb_w` bytes
//! of `refcnt` for the descendant pass and `2·`[`ESTIMATE_BATCH`]`·2^p·5/8` bytes of estimation
//! buffers per pass. **Expansion phase** (after the passes, whose slabs, `slot_of` and
//! `refcnt` are dropped), per direction: the per-block outputs and `4·n_w` bytes of per-node
//! outputs. The peak is the larger of the two. On the full 2022 PG at `p = 10`
//! (`n = 2.18·10⁹`, `nb = 7.78·10⁸`, `LB_max = 5.14·10⁶`) the pass phase takes
//! `2·(3.3 + 3.1 + 3.1) + 3.1 ≈ 22` GB and the expansion phase `2·(3.1 + 8.7) ≈ 24` GB, so the
//! peak is about 24 GB besides the block index.
//!
//! # Invariants checked by the tests
//! Bit identity with the sketches of the brute-force sets (both modes, windows), blocks ==
//! generic, HyperBall equality, register monotonicity along arcs, no slot leaks, and accuracy
//! within `4·rsd` for ≥ 95 % of the samples over 30 independent seeds.

use std::hash::BuildHasher;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

use anyhow::Result;
use card_est_array::impls::{HyperLogLog, HyperLogLogBuilder};
use card_est_array::traits::{EstimationLogic, SliceEstimationLogic};
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::blocks::BlockIndex;
use crate::graph::ChunkedGraph;
use crate::io::{MemBudget, NONE, huge_vec};
use crate::sweep::{SweepConf, sweep_backward, sweep_forward};
use crate::window::{Which, Window};

/// The HyperLogLog logic of every pass: node ids (`usize`) as elements, 5-bit registers packed
/// in `usize` words, LogLog-β estimator (card-est-array's default).
pub type Logic<H> = HyperLogLog<usize, H, usize, true>;

/// Number of closed sketches per estimation buffer (two buffers alternate).
pub const ESTIMATE_BATCH: usize = 1 << 16;

/// Sketches estimated per parallel task (few large tasks keep the rayon wake-ups cheap).
const ESTIMATE_GRAIN: usize = 2048;

/// Smallest accepted `log2m`: `2^p·5` bits must be a multiple of 64 with `usize` words.
pub const MIN_LOG2M: u32 = 6;

/// Largest accepted `log2m` (LogLog-β is defined up to 18).
pub const MAX_LOG2M: u32 = 18;

/// Nodes between two updates of the shared progress counter in the block passes.
const PROGRESS_STEP: usize = 1 << 16;

/// Target size of a slab page: 8 MiB, i.e. four 2 MiB huge pages, so that
/// [`huge_vec`]'s `MADV_HUGEPAGE` covers most of each page (D11).
const PAGE_BYTES: usize = 8 << 20;

/// Default number of slots per slab page for sketches of `words` words: [`PAGE_BYTES`]
/// rounded down to whole sketches (at least one).
fn default_page_slots(words: usize) -> usize {
    (PAGE_BYTES / (words * std::mem::size_of::<usize>())).max(1)
}

/// Builds the logic with `HyperLogLogBuilder::new(num_elements).log2_num_regs(log2m)
/// .build_hasher(hasher).build::<usize>()` (word type `usize`).
///
/// `num_elements` must be positive (it fixes the register size, 5 bits up to 2³² elements);
/// `log2m` must lie in [`MIN_LOG2M`]`..=`[`MAX_LOG2M`].
///
/// ```
/// use dag::reach::{hash::SeededBuildHasher, sketch};
/// use card_est_array::traits::SliceEstimationLogic;
/// let logic = sketch::logic(1000, 10, SeededBuildHasher::new(1)).unwrap();
/// assert_eq!(logic.backend_len(), 1024 * 5 / 64);
/// assert!(sketch::logic(1000, 5, SeededBuildHasher::new(1)).is_err());
/// ```
pub fn logic<H: BuildHasher + Clone>(
    num_elements: usize,
    log2m: u32,
    hasher: H,
) -> Result<Logic<H>> {
    anyhow::ensure!(
        num_elements > 0,
        "HyperLogLog needs a positive number of elements"
    );
    anyhow::ensure!(
        (MIN_LOG2M..=MAX_LOG2M).contains(&log2m),
        "log2m = {log2m} is outside {MIN_LOG2M}..={MAX_LOG2M} (5-bit registers in 64-bit words need 2^log2m·5 to be a multiple of 64)"
    );
    let logic = HyperLogLogBuilder::new(num_elements)
        .log2_num_regs(log2m)
        .build_hasher(hasher)
        .build::<usize>()
        .map_err(|e| anyhow::anyhow!("cannot build the HyperLogLog logic: {e}"))?;
    // the passes merge with `merge_max5`, which needs 5-bit registers in groups of 5 words
    let (words, regs) = (logic.backend_len(), 1usize << log2m);
    anyhow::ensure!(
        words * 64 == regs * 5 && words % 5 == 0,
        "unsupported HyperLogLog layout: {words} words for {regs} registers (5-bit registers expected)"
    );
    Ok(logic)
}

/// The relative standard deviation of one counter with `2^log2m` registers.
pub fn rsd(log2m: u32) -> f64 {
    HyperLogLog::rel_std(log2m)
}

/// Bytes of one sketch with `2^log2m` registers of 5 bits.
pub fn sketch_bytes(log2m: u32) -> u64 {
    (1u64 << log2m) * 5 / 8
}

/// Configuration of [`counts_generic`] and [`counts_blocks`].
#[derive(Clone, Debug)]
pub struct SketchConf<H> {
    /// log₂ of the number of registers (≥ 6 with `usize` words; default 10, rsd 3.25 %).
    pub log2m: u32,
    /// The hash function, e.g. [`SeededBuildHasher`](crate::reach::hash::SeededBuildHasher).
    pub hasher: H,
    /// The node window.
    pub window: Window,
    /// The sweep engine configuration (generic mode).
    pub sweep: SweepConf,
}

/// Measurements of one pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct PassStats {
    /// Sketches closed (handed to the visitor).
    pub closed: usize,
    /// Largest number of simultaneously live sketches.
    pub peak_live: usize,
    /// Live sketches when the pass ended (0: nothing leaked).
    pub live_at_end: usize,
    /// Sketch merges performed.
    pub merges: u64,
    /// Merges skipped because the same source had just been merged into the same target.
    pub merges_skipped: u64,
}

/// Report of one direction of [`Counts`].
#[derive(Clone, Debug, serde::Serialize)]
pub struct PassReport {
    /// `"anc"` or `"desc"`.
    pub name: &'static str,
    /// Wall time of the pass in seconds (including the overlapped estimation).
    pub secs: f64,
    /// Pass measurements.
    pub stats: PassStats,
}

/// Estimated proper ancestor/descendant counts of a window.
#[derive(Clone, Debug)]
pub struct Counts {
    /// The node window; per-node arrays are indexed by `v − window.lo`.
    pub window: Window,
    /// `[n_w]` estimated `|anc*(v)|` (0 for sources), if requested.
    pub anc: Option<Vec<f32>>,
    /// `[n_w]` estimated `|desc*(v)|` (0 for sinks), if requested.
    pub desc: Option<Vec<f32>>,
    /// Block mode: `[nb_w]` estimated `|A(X)|` for blocks `X ∈ blocks`, indexed by `X − blocks.start`.
    pub blk_anc: Option<Vec<f32>>,
    /// Block mode: `[nb_w]` estimated `|D(X)|` (outputs included).
    pub blk_desc: Option<Vec<f32>>,
    /// Largest number of simultaneously live sketches over the passes (generic mode: the
    /// window size, since every node keeps its sketch).
    pub peak_live: usize,
    /// Block mode: the block window `[blo, bhi)` (blocks whose first output lies in the window).
    pub blocks: Option<Range<usize>>,
    /// One report per pass that ran.
    pub passes: Vec<PassReport>,
}

impl Counts {
    fn empty(window: Window, which: Which, block_range: Option<Range<usize>>) -> Self {
        let blocks = block_range.is_some();
        let some = |b: bool| b.then(Vec::new);
        Self {
            window,
            anc: some(which.forward()),
            desc: some(which.backward()),
            blk_anc: some(blocks && which.forward()),
            blk_desc: some(blocks && which.backward()),
            peak_live: 0,
            blocks: block_range,
            passes: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Register-wise maximum
// ---------------------------------------------------------------------------------------------

const _: () = assert!(usize::BITS == 64, "sketch words are 64 bits");

/// Mask of bit `bit` of each of the 64 registers of 5 bits in a 320-bit group.
const fn group_mask(bit: usize) -> [usize; 5] {
    let mut m = [0usize; 5];
    let mut r = 0;
    while r < 64 {
        let pos = r * 5 + bit;
        m[pos / 64] |= 1 << (pos % 64);
        r += 1;
    }
    m
}

/// Highest bit of every register of a group.
const H5: [usize; 5] = group_mask(4);
/// Lowest bit of every register of a group.
const L5: [usize; 5] = group_mask(0);

/// Register-wise maximum `dst ← max(dst, src)` of sketches with 5-bit registers packed in 64-bit
/// words (the layout of card-est-array's `HyperLogLog` with `usize` words), the merge of
/// HyperLogLog [\[FFGM07\]](crate::references#ffgm07).
///
/// It uses the broadword comparison of [\[BRV11\]](crate::references#brv11), as card-est-array
/// does, but on independent 320-bit groups of 64 whole registers (5 words), with branch-free
/// borrows: no borrow ever leaves a register, so the groups need no carry chain between them
/// and the compiler can overlap them. The result is the register-wise maximum, hence
/// bit-identical to `MergeEstimationLogic::merge` (checked by the tests on random words).
///
/// Panics if the lengths differ or are not a multiple of 5.
///
/// ```
/// use dag::reach::sketch::merge_max5;
/// let mut x = [0b11111usize, 0, 0, 0, 1 << 59];
/// let y = [0b00001usize << 5, 0, 0, 0, 3 << 59];
/// merge_max5(&mut x, &y);
/// assert_eq!(x, [0b11111 | 0b00001 << 5, 0, 0, 0, 3 << 59]);
/// ```
#[inline]
pub fn merge_max5(dst: &mut [usize], src: &[usize]) {
    assert!(
        dst.len() == src.len() && dst.len() % 5 == 0,
        "bad sketch lengths"
    );
    for (x, y) in dst.chunks_exact_mut(5).zip(src.chunks_exact(5)) {
        // acc = (y | H) − (x & !H): per register, the high bit of acc says y_low ≥ x_low
        let mut acc = [0usize; 5];
        let mut borrow = false;
        for i in 0..5 {
            let (d, b1) = (y[i] | H5[i]).overflowing_sub(x[i] & !H5[i]);
            let (d, b2) = d.overflowing_sub(usize::from(borrow));
            acc[i] = d;
            borrow = b1 | b2;
        }
        // z: high bit of each register set iff y > x
        let mut z = [0usize; 5];
        for i in 0..5 {
            z[i] = ((acc[i] | (y[i] ^ x[i])) ^ (y[i] | !x[i])) & H5[i];
        }
        // t = (z >> 4) | H, then mask = ((t − L) | H) ^ z: all ones in the registers where y > x
        let mut borrow = false;
        for i in 0..5 {
            let next = if i < 4 { z[i + 1] << 60 } else { 0 };
            let t = (z[i] >> 4) | next | H5[i];
            let (d, b1) = t.overflowing_sub(L5[i]);
            let (d, b2) = d.overflowing_sub(usize::from(borrow));
            borrow = b1 | b2;
            let mask = (d | H5[i]) ^ z[i];
            x[i] ^= (x[i] ^ y[i]) & mask;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Slab of sketches
// ---------------------------------------------------------------------------------------------

/// A pool of equally sized sketches in pages of `page_slots` slots (allocated with
/// [`huge_vec`]), with a LIFO free list and a per-slot memo of the last merged source.
struct Slab {
    words: usize,
    page_slots: usize,
    pages: Vec<Vec<usize>>,
    tag: Vec<u32>,
    free: Vec<u32>,
    next: usize,
    live: usize,
    peak: usize,
}

impl Slab {
    fn new(words: usize, page_slots: usize) -> Self {
        assert!(page_slots > 0, "a slab page holds at least one slot");
        Self {
            words,
            page_slots,
            pages: Vec::new(),
            tag: Vec::new(),
            free: Vec::new(),
            next: 0,
            live: 0,
            peak: 0,
        }
    }

    /// A zeroed slot with an empty memo.
    fn alloc(&mut self) -> u32 {
        let s = match self.free.pop() {
            Some(s) => {
                self.get_mut(s).fill(0);
                s
            }
            None => {
                if self.next == self.pages.len() * self.page_slots {
                    self.pages
                        .push(huge_vec(self.page_slots * self.words, 0usize));
                    self.tag.resize(self.pages.len() * self.page_slots, NONE);
                }
                let s = u32::try_from(self.next).expect("more than 2^32 live sketches");
                self.next += 1;
                s
            }
        };
        self.tag[s as usize] = NONE;
        self.live += 1;
        self.peak = self.peak.max(self.live);
        s
    }

    fn release(&mut self, s: u32) {
        self.free.push(s);
        self.live -= 1;
    }

    #[inline(always)]
    fn get(&self, s: u32) -> &[usize] {
        let (p, o) = (s as usize / self.page_slots, s as usize % self.page_slots);
        &self.pages[p][o * self.words..(o + 1) * self.words]
    }

    #[inline(always)]
    fn get_mut(&mut self, s: u32) -> &mut [usize] {
        let (p, o) = (s as usize / self.page_slots, s as usize % self.page_slots);
        &mut self.pages[p][o * self.words..(o + 1) * self.words]
    }

    /// `(slot d mutably, slot s)` for `d ≠ s`.
    #[inline(always)]
    fn pair(&mut self, d: u32, s: u32) -> (&mut [usize], &[usize]) {
        debug_assert_ne!(d, s);
        let (w, slots) = (self.words, self.page_slots);
        let (pd, od) = (d as usize / slots, d as usize % slots);
        let (ps, os) = (s as usize / slots, s as usize % slots);
        if pd == ps {
            let page = &mut self.pages[pd];
            if od < os {
                let (l, r) = page.split_at_mut(os * w);
                (&mut l[od * w..(od + 1) * w], &r[..w])
            } else {
                let (l, r) = page.split_at_mut(od * w);
                (&mut r[..w], &l[os * w..(os + 1) * w])
            }
        } else if pd < ps {
            let (l, r) = self.pages.split_at_mut(ps);
            (
                &mut l[pd][od * w..(od + 1) * w],
                &r[0][os * w..(os + 1) * w],
            )
        } else {
            let (l, r) = self.pages.split_at_mut(pd);
            (
                &mut r[0][od * w..(od + 1) * w],
                &l[ps][os * w..(os + 1) * w],
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Overlapped estimation
// ---------------------------------------------------------------------------------------------

/// Receives closed sketches in output order and estimates them in parallel batches, two buffers
/// alternating (see the module documentation).
struct Collector<'s, 'scope, L> {
    scope: &'s rayon::Scope<'scope>,
    logic: &'scope L,
    words: usize,
    batch: usize,
    ascending: bool,
    /// Output positions not yet assigned to a batch.
    out: &'scope mut [f32],
    /// Position expected next (ascending: from 0 up; descending: from len − 1 down).
    expected: usize,
    buf: Vec<usize>,
    count: usize,
    spare: Option<Vec<usize>>,
    tx: mpsc::Sender<Vec<usize>>,
    rx: mpsc::Receiver<Vec<usize>>,
}

impl<'s, 'scope, L> Collector<'s, 'scope, L>
where
    L: EstimationLogic<Backend = [usize]> + Sync,
{
    fn push(&mut self, pos: usize, sketch: &[usize]) {
        assert_eq!(
            pos, self.expected,
            "sketches must be closed in output order"
        );
        if self.ascending {
            self.expected += 1;
        } else {
            self.expected = self.expected.wrapping_sub(1);
        }
        self.buf.extend_from_slice(sketch);
        self.count += 1;
        if self.count == self.batch {
            self.flush();
        }
    }

    fn flush(&mut self) {
        let k = self.count;
        if k == 0 {
            return;
        }
        let out = std::mem::take(&mut self.out);
        let chunk = if self.ascending {
            let (chunk, rest) = out.split_at_mut(k);
            self.out = rest;
            chunk
        } else {
            let len = out.len();
            let (rest, chunk) = out.split_at_mut(len - k);
            self.out = rest;
            chunk
        };
        let next = self.next_buffer();
        let buf = std::mem::replace(&mut self.buf, next);
        self.count = 0;
        let (logic, words, ascending, tx) =
            (self.logic, self.words, self.ascending, self.tx.clone());
        self.scope.spawn(move |_| {
            chunk
                .par_iter_mut()
                .enumerate()
                .with_min_len(ESTIMATE_GRAIN)
                .for_each(|(j, o)| {
                    let b = if ascending { j } else { k - 1 - j };
                    *o = logic.estimate(&buf[b * words..(b + 1) * words]) as f32;
                });
            // the receiver may be gone at the very end of the pass
            let _ = tx.send(buf);
        });
    }

    /// The spare buffer, or the buffer of the previous batch once its estimation is over. While
    /// waiting, the thread runs other rayon jobs (possibly that very estimation), so this never
    /// deadlocks, even with one thread.
    fn next_buffer(&mut self) -> Vec<usize> {
        let mut b = match self.spare.take() {
            Some(b) => b,
            None => loop {
                match self.rx.try_recv() {
                    Ok(b) => break b,
                    Err(mpsc::TryRecvError::Empty) => {
                        if rayon::yield_now() != Some(rayon::Yield::Executed) {
                            std::thread::yield_now();
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        unreachable!("the collector owns a sender")
                    }
                }
            },
        };
        b.clear();
        b
    }

    /// Estimates the last partial batch. Positions never visited (a pass that failed) keep
    /// their initial value.
    fn finish(mut self) {
        self.flush();
    }
}

/// Runs `pass` with a visitor `(pos, proper sketch)` that estimates the sketches into `out`
/// (positions visited in ascending or descending order, each exactly once), overlapping the
/// estimation of one batch with the filling of the next.
fn estimate_streaming<L, R>(
    logic: &L,
    out: &mut [f32],
    ascending: bool,
    batch: usize,
    pass: impl FnOnce(&mut (dyn FnMut(usize, &[usize]) + Send)) -> R,
) -> R
where
    L: SliceEstimationLogic<usize> + Sync,
{
    let words = logic.backend_len();
    let batch = batch.max(1);
    let len = out.len();
    // in place: the pass (which may hold a non-Send progress logger) runs on this thread
    rayon::in_place_scope(|s| {
        let (tx, rx) = mpsc::channel();
        let mut col = Collector {
            scope: s,
            logic,
            words,
            batch,
            ascending,
            out,
            expected: if ascending { 0 } else { len.wrapping_sub(1) },
            buf: Vec::with_capacity(batch.min(len.max(1)) * words),
            count: 0,
            spare: Some(Vec::with_capacity(batch.min(len.max(1)) * words)),
            tx,
            rx,
        };
        let r = pass(&mut |pos, sk| col.push(pos, sk));
        col.finish();
        r
    })
}

// ---------------------------------------------------------------------------------------------
// Block passes (PG fast path)
// ---------------------------------------------------------------------------------------------

/// The block window `[blo, bhi)` of a node window: the blocks whose first output lies in it.
pub fn block_window(blocks: &BlockIndex, w: Window) -> Range<usize> {
    blocks.block_bound(w.lo)..blocks.block_bound(w.hi)
}

/// Ancestor pass over the blocks of window `w` (ids ascending).
///
/// Calls `visit(X − blo, sketch of A(X))` for every block `X` of the window, in ascending
/// order, when `A(X)` (the proper ancestors in `w` of the outputs of `X`) is final. Adds
/// `PROGRESS_STEP`-sized increments of visited nodes to `progress`.
pub fn anc_pass_blocks<H, V>(
    blocks: &BlockIndex,
    w: Window,
    logic: &Logic<H>,
    progress: &AtomicUsize,
    visit: V,
) -> PassStats
where
    H: BuildHasher + Clone,
    V: FnMut(usize, &[usize]),
{
    let slots = default_page_slots(logic.backend_len());
    anc_pass_blocks_paged(blocks, w, logic, progress, slots, visit)
}

/// [`anc_pass_blocks`] with `page_slots` slots per slab page (tests use tiny pages so that
/// cross-page merges and page growth run against the oracle).
#[doc(hidden)]
pub fn anc_pass_blocks_paged<H, V>(
    blocks: &BlockIndex,
    w: Window,
    logic: &Logic<H>,
    progress: &AtomicUsize,
    page_slots: usize,
    mut visit: V,
) -> PassStats
where
    H: BuildHasher + Clone,
    V: FnMut(usize, &[usize]),
{
    let Range {
        start: blo,
        end: bhi,
    } = block_window(blocks, w);
    let words = logic.backend_len();
    let zero = vec![0usize; words];
    let mut slab = Slab::new(words, page_slots);
    let mut slot_of = huge_vec(bhi - blo, NONE);
    let mut st = PassStats::default();
    let (create, spend, first, len) = (&blocks.create, &blocks.spend, &blocks.first, &blocks.len);
    for u in w.lo..w.hi {
        let b = create[u];
        // a block with first output < lo has no input in the window
        let (bi, bslot) = if b != NONE && b as usize >= blo {
            let bi = b as usize - blo;
            (bi, slot_of[bi])
        } else {
            (usize::MAX, NONE)
        };
        if bi != usize::MAX && u == first[b as usize] as usize {
            // every input of b is < first(b) = u, so A(b) is final
            visit(
                bi,
                if bslot == NONE {
                    &zero
                } else {
                    slab.get(bslot)
                },
            );
            st.closed += 1;
        }
        let x = spend[u];
        if x != NONE && (x as usize) < bhi {
            let xi = x as usize - blo;
            let mut xs = slot_of[xi];
            if xs == NONE {
                xs = slab.alloc();
                slot_of[xi] = xs;
            }
            if bslot != NONE {
                if slab.tag[xs as usize] != b {
                    slab.tag[xs as usize] = b;
                    let (d, s) = slab.pair(xs, bslot);
                    merge_max5(d, s);
                    st.merges += 1;
                } else {
                    st.merges_skipped += 1;
                }
            }
            logic.add(slab.get_mut(xs), u);
        }
        if bslot != NONE {
            let last = first[b as usize] as usize + len[b as usize] as usize - 1;
            if u == last.min(w.hi - 1) {
                slab.release(bslot);
                slot_of[bi] = NONE;
            }
        }
        if (u - w.lo + 1) % PROGRESS_STEP == 0 {
            progress.fetch_add(PROGRESS_STEP, Ordering::Relaxed);
        }
    }
    progress.fetch_add(w.len() % PROGRESS_STEP, Ordering::Relaxed);
    st.peak_live = slab.peak;
    st.live_at_end = slab.live;
    st
}

/// The descendant reference counts of the blocks `[blo, bhi)` of `w`: the number of inputs of
/// each block inside the window (`nin` when `lo = 0`, else a parallel histogram of `spend`).
fn desc_refcounts(blocks: &BlockIndex, w: Window, bw: &Range<usize>) -> Vec<u32> {
    if w.lo == 0 {
        // every input of a block of the window precedes its first output < hi
        let mut refcnt = huge_vec(bw.len(), 0u32);
        refcnt
            .par_iter_mut()
            .zip(blocks.nin[bw.clone()].par_iter())
            .with_min_len(1 << 16)
            .for_each(|(r, &k)| *r = k);
        return refcnt;
    }
    let mut refcnt = huge_vec(bw.len(), 0u32);
    let cnt = crate::io::as_atomic_u32(&mut refcnt);
    blocks.spend[w.lo..w.hi]
        .par_iter()
        .with_min_len(1 << 16)
        .for_each(|&x| {
            if x != NONE && (x as usize) < bw.end {
                cnt[x as usize - bw.start].fetch_add(1, Ordering::Relaxed);
            }
        });
    refcnt
}

/// Descendant pass over the blocks of window `w` (ids descending).
///
/// Calls `visit(X − blo, sketch of D(X))` for every block `X` of the window, in descending
/// order, when `D(X)` (the outputs of `X` in `w` and their descendants in `w`) is final.
pub fn desc_pass_blocks<H, V>(
    blocks: &BlockIndex,
    w: Window,
    logic: &Logic<H>,
    progress: &AtomicUsize,
    visit: V,
) -> PassStats
where
    H: BuildHasher + Clone,
    V: FnMut(usize, &[usize]),
{
    let slots = default_page_slots(logic.backend_len());
    desc_pass_blocks_paged(blocks, w, logic, progress, slots, visit)
}

/// [`desc_pass_blocks`] with `page_slots` slots per slab page (see
/// [`anc_pass_blocks_paged`]).
#[doc(hidden)]
pub fn desc_pass_blocks_paged<H, V>(
    blocks: &BlockIndex,
    w: Window,
    logic: &Logic<H>,
    progress: &AtomicUsize,
    page_slots: usize,
    mut visit: V,
) -> PassStats
where
    H: BuildHasher + Clone,
    V: FnMut(usize, &[usize]),
{
    let bw = block_window(blocks, w);
    let blo = bw.start;
    let bhi = bw.end;
    let words = logic.backend_len();
    let mut refcnt = desc_refcounts(blocks, w, &bw);
    let mut slab = Slab::new(words, page_slots);
    let mut slot_of = huge_vec(bhi - blo, NONE);
    let mut st = PassStats::default();
    let (create, spend, first) = (&blocks.create, &blocks.spend, &blocks.first);
    for (k, wn) in (w.lo..w.hi).rev().enumerate() {
        let x = spend[wn];
        let (xi, xs) = if x != NONE && (x as usize) < bhi {
            // first(x) > wn ≥ lo, so x ≥ blo and D(x) was closed earlier and is still alive
            let xi = x as usize - blo;
            (xi, slot_of[xi])
        } else {
            (usize::MAX, NONE)
        };
        debug_assert!(xi == usize::MAX || xs != NONE, "D(X) freed too early");
        let b = create[wn];
        if b != NONE && b as usize >= blo {
            let bi = b as usize - blo;
            let mut bs = slot_of[bi];
            if bs == NONE {
                bs = slab.alloc();
                slot_of[bi] = bs;
            }
            if xs != NONE {
                if slab.tag[bs as usize] != x {
                    slab.tag[bs as usize] = x;
                    let (d, s) = slab.pair(bs, xs);
                    merge_max5(d, s);
                    st.merges += 1;
                } else {
                    st.merges_skipped += 1;
                }
            }
            logic.add(slab.get_mut(bs), wn);
            if wn == first[b as usize] as usize {
                visit(bi, slab.get(bs));
                st.closed += 1;
                if refcnt[bi] == 0 {
                    slab.release(bs);
                    slot_of[bi] = NONE;
                }
            }
        }
        if xs != NONE {
            refcnt[xi] -= 1;
            if refcnt[xi] == 0 {
                slab.release(xs);
                slot_of[xi] = NONE;
            }
        }
        if (k + 1) % PROGRESS_STEP == 0 {
            progress.fetch_add(PROGRESS_STEP, Ordering::Relaxed);
        }
    }
    progress.fetch_add(w.len() % PROGRESS_STEP, Ordering::Relaxed);
    st.peak_live = slab.peak;
    st.live_at_end = slab.live;
    st
}

/// Runs `work` as a job of the **current** rayon pool (the pool installed by the caller, or the
/// global one) while the calling thread forwards the progress `counter` to `pl`, so the logger
/// never crosses threads. While waiting, the calling thread runs pending rayon jobs (possibly
/// `work` itself, e.g. in a one-thread pool), so this never deadlocks. Panics of `work` are
/// propagated.
fn with_progress<R: Send>(
    pl: &mut impl ProgressLog,
    counter: &AtomicUsize,
    work: impl FnOnce() -> R + Send,
) -> R {
    let me = std::thread::current();
    let done = AtomicBool::new(false);
    let result = Mutex::new(None);
    rayon::in_place_scope(|s| {
        let (done, result) = (&done, &result);
        s.spawn(move |_| {
            // set the flag even if `work` panics (the scope then re-raises the panic)
            struct Signal<'a>(&'a AtomicBool, std::thread::Thread);
            impl Drop for Signal<'_> {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Release);
                    self.1.unpark();
                }
            }
            let _signal = Signal(done, me);
            let r = work();
            *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
        });
        let mut seen = 0;
        loop {
            let finished = done.load(Ordering::Acquire);
            let now = counter.load(Ordering::Relaxed);
            if now > seen {
                pl.update_with_count(now - seen);
                seen = now;
            }
            if finished {
                break;
            }
            if rayon::yield_now() != Some(rayon::Yield::Executed) {
                std::thread::park_timeout(Duration::from_millis(250));
            }
        }
    });
    result
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .expect("the work job completed without panicking")
}

/// Expands per-block estimates to the nodes of `w`: `out[v − lo] = blk[key(v) − blo]` when
/// `key(v)` is a block of the window, else 0.
fn expand(keys: &[u32], w: Window, bw: &Range<usize>, blk: &[f32]) -> Vec<f32> {
    let mut out = huge_vec(w.len(), 0f32);
    out.par_iter_mut()
        .zip(keys[w.lo..w.hi].par_iter())
        .with_min_len(1 << 16)
        .for_each(|(o, &x)| {
            if x != NONE && bw.contains(&(x as usize)) {
                *o = blk[x as usize - bw.start];
            }
        });
    out
}

/// Block-mode counts (the PG fast path; see the module documentation).
///
/// Returns per-block (`blk_anc`, `blk_desc`, indexed by `X − blo`) and per-node (`anc`,
/// `desc`, indexed by `v − lo`) estimates for the requested directions; the two passes run
/// concurrently. The caller is responsible for the memory budget (see the module
/// documentation for the footprint).
///
/// ```
/// use dag::blocks::BlockIndex;
/// use dag::reach::{hash::SeededBuildHasher, sketch::{SketchConf, counts_blocks}};
/// use dag::{Which, Window};
/// // two coinbase outputs 0, 1 spent together by a transaction with outputs 2, 3
/// let g = dag::testkit::vec_graph(4, [(0, 2), (0, 3), (1, 2), (1, 3)]);
/// let (idx, _) = BlockIndex::from_vec_graph(&g).unwrap();
/// let conf = SketchConf { log2m: 10, hasher: SeededBuildHasher::new(0), window: Window::full(4),
///                         sweep: Default::default() };
/// let c = counts_blocks(&idx, Which::Both, &conf, &mut Option::<dsi_progress_logger::ProgressLogger>::None).unwrap();
/// let (anc, desc) = (c.anc.unwrap(), c.desc.unwrap());
/// assert_eq!((anc[0], anc[1], desc[2], desc[3]), (0.0, 0.0, 0.0, 0.0)); // sources, sinks
/// assert!((anc[2] - 2.0).abs() < 0.01 && (desc[0] - 2.0).abs() < 0.01); // ≈ 2
/// assert_eq!(c.blk_anc.unwrap(), [anc[2]]); // one block, outputs 2..4
/// ```
pub fn counts_blocks<H: BuildHasher + Clone + Send + Sync>(
    blocks: &BlockIndex,
    which: Which,
    conf: &SketchConf<H>,
    pl: &mut impl ProgressLog,
) -> Result<Counts> {
    counts_blocks_batched(blocks, which, conf, ESTIMATE_BATCH, pl)
}

/// [`counts_blocks`] with an explicit estimation batch size (tests exercise small batches).
#[doc(hidden)]
pub fn counts_blocks_batched<H: BuildHasher + Clone + Send + Sync>(
    blocks: &BlockIndex,
    which: Which,
    conf: &SketchConf<H>,
    batch: usize,
    pl: &mut impl ProgressLog,
) -> Result<Counts> {
    counts_blocks_paged(blocks, which, conf, batch, None, pl)
}

/// [`counts_blocks_batched`] with `page_slots` slots per slab page (`None`: 8 MiB pages).
#[doc(hidden)]
pub fn counts_blocks_paged<H: BuildHasher + Clone + Send + Sync>(
    blocks: &BlockIndex,
    which: Which,
    conf: &SketchConf<H>,
    batch: usize,
    page_slots: Option<usize>,
    pl: &mut impl ProgressLog,
) -> Result<Counts> {
    let w = conf.window;
    anyhow::ensure!(
        w.lo <= w.hi && w.hi <= blocks.n,
        crate::error::DagError::BadWindow {
            lo: w.lo,
            hi: w.hi,
            n: blocks.n
        }
    );
    if blocks.n == 0 || w.is_empty() || !(which.forward() || which.backward()) {
        let bw = if blocks.n == 0 {
            0..0
        } else {
            block_window(blocks, w)
        };
        return Ok(Counts::empty(w, which, Some(bw)));
    }
    let logic = logic(blocks.n, conf.log2m, conf.hasher.clone())?;
    let page_slots = page_slots.unwrap_or_else(|| default_page_slots(logic.backend_len()));
    let bw = block_window(blocks, w);
    let nbw = bw.len();
    let mut blk_anc = which.forward().then(|| huge_vec(nbw, 0f32));
    let mut blk_desc = which.backward().then(|| huge_vec(nbw, 0f32));
    let counter = AtomicUsize::new(0);
    let passes = usize::from(which.forward()) + usize::from(which.backward());
    pl.item_name("node")
        .expected_updates(Some(w.len() * passes));
    pl.start(format!(
        "HyperLogLog block passes ({which:?}) over nodes {w} and blocks {}..{}, log2m = {}",
        bw.start, bw.end, conf.log2m
    ));
    let logic_ref = &logic;
    let counter_ref = &counter;
    let anc_job = |out: &mut [f32]| {
        let t = Instant::now();
        let stats = estimate_streaming(logic_ref, out, true, batch, |visit| {
            anc_pass_blocks_paged(blocks, w, logic_ref, counter_ref, page_slots, visit)
        });
        PassReport {
            name: "anc",
            secs: t.elapsed().as_secs_f64(),
            stats,
        }
    };
    let desc_job = |out: &mut [f32]| {
        let t = Instant::now();
        let stats = estimate_streaming(logic_ref, out, false, batch, |visit| {
            desc_pass_blocks_paged(blocks, w, logic_ref, counter_ref, page_slots, visit)
        });
        PassReport {
            name: "desc",
            secs: t.elapsed().as_secs_f64(),
            stats,
        }
    };
    let reports: Vec<PassReport> =
        with_progress(pl, &counter, || match (&mut blk_anc, &mut blk_desc) {
            (Some(a), Some(d)) => {
                let (ra, rd) = rayon::join(|| anc_job(a), || desc_job(d));
                vec![ra, rd]
            }
            (Some(a), None) => vec![anc_job(a)],
            (None, Some(d)) => vec![desc_job(d)],
            (None, None) => Vec::new(),
        });
    pl.done();
    for r in &reports {
        anyhow::ensure!(
            r.stats.live_at_end == 0 && r.stats.closed == nbw,
            "{} pass: {} sketches leaked, {} of {nbw} blocks closed",
            r.name,
            r.stats.live_at_end,
            r.stats.closed
        );
    }
    let anc = blk_anc
        .as_deref()
        .map(|b| expand(&blocks.create, w, &bw, b));
    let desc = blk_desc
        .as_deref()
        .map(|b| expand(&blocks.spend, w, &bw, b));
    Ok(Counts {
        window: w,
        anc,
        desc,
        blk_anc,
        blk_desc,
        peak_live: reports.iter().map(|r| r.stats.peak_live).max().unwrap_or(0),
        blocks: Some(bw),
        passes: reports,
    })
}

// ---------------------------------------------------------------------------------------------
// Generic passes
// ---------------------------------------------------------------------------------------------

/// Generic ancestor pass (forward push with [`sweep_forward`]).
///
/// Calls `visit(v − lo, sketch of anc*(v) ∩ w)` for every node of `w` in ascending order. Holds
/// `|w|` sketches. Fails with [`DagError::NotTopological`](crate::DagError::NotTopological) if
/// the identity is not a topological order of the window (cycles and self-loops included).
pub fn anc_pass_generic<G, H, V>(
    g: &ChunkedGraph<G>,
    w: Window,
    logic: &Logic<H>,
    sweep: &SweepConf,
    pl: &mut impl ProgressLog,
    mut visit: V,
) -> Result<PassStats>
where
    G: RandomAccessGraph + Sync,
    H: BuildHasher + Clone + Send + Sync,
    V: FnMut(usize, &[usize]) + Send,
{
    let words = logic.backend_len();
    let mut s = huge_vec(w.len() * words, 0usize);
    let mut st = PassStats {
        peak_live: w.len(),
        ..Default::default()
    };
    let lo = w.lo;
    pl.item_name("node").expected_updates(Some(w.len()));
    pl.start(format!("HyperLogLog ancestor push over nodes {w}"));
    sweep_forward(g, w, sweep, pl, |chunk| {
        for (u, succ) in chunk.iter() {
            let i = (u - lo) * words;
            visit(u - lo, &s[i..i + words]);
            logic.add(&mut s[i..i + words], u);
            for &v in succ {
                let j = (v as usize - lo) * words;
                let (a, b) = s.split_at_mut(j);
                merge_max5(&mut b[..words], &a[i..i + words]);
            }
            st.merges += succ.len() as u64;
        }
        Ok(())
    })?;
    pl.done();
    st.closed = w.len();
    Ok(st)
}

/// Generic descendant pass (pull with [`sweep_backward`]).
///
/// Calls `visit(u − lo, sketch of desc*(u) ∩ w)` for every node of `w` in descending order.
/// Holds `|w|` sketches. Fails like [`anc_pass_generic`].
pub fn desc_pass_generic<G, H, V>(
    g: &ChunkedGraph<G>,
    w: Window,
    logic: &Logic<H>,
    sweep: &SweepConf,
    pl: &mut impl ProgressLog,
    mut visit: V,
) -> Result<PassStats>
where
    G: RandomAccessGraph + Sync,
    H: BuildHasher + Clone + Send + Sync,
    V: FnMut(usize, &[usize]) + Send,
{
    let words = logic.backend_len();
    let mut s = huge_vec(w.len() * words, 0usize);
    let mut st = PassStats {
        peak_live: w.len(),
        ..Default::default()
    };
    let lo = w.lo;
    pl.item_name("node").expected_updates(Some(w.len()));
    pl.start(format!("HyperLogLog descendant pull over nodes {w}"));
    sweep_backward(g, w, sweep, pl, |chunk| {
        for (u, succ) in chunk.iter_rev() {
            let i = (u - lo) * words;
            for &v in succ {
                let j = (v as usize - lo) * words;
                let (a, b) = s.split_at_mut(j);
                merge_max5(&mut a[i..i + words], &b[..words]);
            }
            st.merges += succ.len() as u64;
            visit(u - lo, &s[i..i + words]);
            logic.add(&mut s[i..i + words], u);
        }
        Ok(())
    })?;
    pl.done();
    st.closed = w.len();
    Ok(st)
}

/// Generic-mode counts: [`anc_pass_generic`] then [`desc_pass_generic`] (one sketch array at a
/// time), each reserving `|w|·2^p·5/8` bytes in `budget` (released afterwards), plus the `f32`
/// outputs.
pub fn counts_generic<G, H>(
    g: &ChunkedGraph<G>,
    which: Which,
    conf: &SketchConf<H>,
    budget: &mut MemBudget,
    pl: &mut impl ProgressLog,
) -> Result<Counts>
where
    G: RandomAccessGraph + Sync,
    H: BuildHasher + Clone + Send + Sync,
{
    counts_generic_batched(g, which, conf, ESTIMATE_BATCH, budget, pl)
}

/// [`counts_generic`] with an explicit estimation batch size (tests exercise small batches).
#[doc(hidden)]
pub fn counts_generic_batched<G, H>(
    g: &ChunkedGraph<G>,
    which: Which,
    conf: &SketchConf<H>,
    batch: usize,
    budget: &mut MemBudget,
    pl: &mut impl ProgressLog,
) -> Result<Counts>
where
    G: RandomAccessGraph + Sync,
    H: BuildHasher + Clone + Send + Sync,
{
    let n = g.num_nodes();
    let w = conf.window;
    anyhow::ensure!(
        w.lo <= w.hi && w.hi <= n,
        crate::error::DagError::BadWindow {
            lo: w.lo,
            hi: w.hi,
            n
        }
    );
    if n == 0 || w.is_empty() || !(which.forward() || which.backward()) {
        return Ok(Counts::empty(w, which, None));
    }
    let logic = logic(n, conf.log2m, conf.hasher.clone())?;
    let sketches = w.len() as u64 * logic.backend_len() as u64 * 8;
    let outputs = 4 * w.len() as u64 * (u64::from(which.forward()) + u64::from(which.backward()));
    budget.need(outputs, "reach-counts f32 outputs")?;
    let mut passes = Vec::new();
    let mut anc = None;
    let mut desc = None;
    for fwd in [true, false] {
        if (fwd && !which.forward()) || (!fwd && !which.backward()) {
            continue;
        }
        let r = budget
            .need(sketches, "reach-counts sketch array")
            .and_then(|()| {
                let r = generic_direction(g, w, &logic, &conf.sweep, batch, fwd, pl);
                budget.release(sketches);
                r
            });
        // on failure nothing stays reserved; on success the returned outputs do
        let (out, report) = r.inspect_err(|_| budget.release(outputs))?;
        passes.push(report);
        *(if fwd { &mut anc } else { &mut desc }) = Some(out);
    }
    Ok(Counts {
        window: w,
        anc,
        desc,
        blk_anc: None,
        blk_desc: None,
        peak_live: w.len(),
        blocks: None,
        passes,
    })
}

/// One generic direction with overlapped estimation.
fn generic_direction<G, H>(
    g: &ChunkedGraph<G>,
    w: Window,
    logic: &Logic<H>,
    sweep: &SweepConf,
    batch: usize,
    fwd: bool,
    pl: &mut impl ProgressLog,
) -> Result<(Vec<f32>, PassReport)>
where
    G: RandomAccessGraph + Sync,
    H: BuildHasher + Clone + Send + Sync,
{
    let t = Instant::now();
    let mut out = huge_vec(w.len(), 0f32);
    let stats = estimate_streaming(logic, &mut out, fwd, batch, |visit| {
        if fwd {
            anc_pass_generic(g, w, logic, sweep, pl, visit)
        } else {
            desc_pass_generic(g, w, logic, sweep, pl, visit)
        }
    })?;
    let report = PassReport {
        name: if fwd { "anc" } else { "desc" },
        secs: t.elapsed().as_secs_f64(),
        stats,
    };
    Ok((out, report))
}

/// Reflexive descendant estimates `|{v} ∪ desc*(v)|` of every node of `g` (full graph), with
/// `num_elements = n`, `2^log2m` registers and `hasher`.
///
/// With `BuildHasherDefault<DefaultHasher>` this is **bit-identical** to HyperBall's
/// `reachable_nodes()` run to stabilization on a DAG (a test oracle; HyperBall needs one
/// iteration per distance, 3.56·10⁶ on the PG, so it is never run there).
pub fn reflexive_sketch_estimates<G, H>(
    g: &ChunkedGraph<G>,
    log2m: u32,
    hasher: H,
    sweep: &SweepConf,
) -> Result<Vec<f32>>
where
    G: RandomAccessGraph + Sync,
    H: BuildHasher + Clone + Send + Sync,
{
    let n = g.num_nodes();
    if n == 0 {
        return Ok(Vec::new());
    }
    let logic = logic(n, log2m, hasher)?;
    let mut out = vec![0f32; n];
    let mut tmp = vec![0usize; logic.backend_len()];
    desc_pass_generic(
        g,
        Window::full(n),
        &logic,
        sweep,
        &mut Option::<ProgressLogger>::None,
        |v, sk| {
            tmp.copy_from_slice(sk);
            logic.add(&mut tmp, v);
            out[v] = logic.estimate(&tmp) as f32;
        },
    )?;
    Ok(out)
}
