//! Decode-free biclique kernels over a [`BlockIndex`] (F1 §6.5, the PG fast path).
//!
//! # Algorithm
//! On a line digraph every arc set is a disjoint union of bicliques `I(B) × O(B)` with
//! `O(B)` an id interval and `max I(B) < min O(B)` (P1–P3 of [`crate::blocks`]); all outputs of
//! a block share the predecessor set `I(B)`. With separable weights `w(u, v) = α(u) ⊗ β(v)`,
//! right distributivity factors the forward fold through one accumulator per block, and left
//! distributivity factors the backward fold through one sum per block:
//! ```text
//! forward_blocks:   slots[blo..bhi] = { acc: 0̄, best_in: NONE }      (best_in = NONE ⟺ untouched)
//!   for v in lo..hi:
//!       B = create[v]; hp = B ≠ NONE ∧ B ≥ blo ∧ slots[B].best_in ≠ NONE
//!       val[v] = hp ? (slots[B].acc ⊗ β(v)) ⊕ b(v) : b(v)
//!       pred[v] = hp ∧ ¬better(b(v), slots[B].acc ⊗ β(v)) ? slots[B].best_in : NONE
//!       X = spend[v]
//!       if X ≠ NONE ∧ X < bhi:  y = val[v] ⊗ α(v)
//!           first push: slots[X] = { 0̄ ⊕ y, v };  else: best_in = v if better(y, acc); acc ⊕= y
//!
//! backward_blocks:  bsum[blo..bhi]
//!   for v in (lo..hi).rev():
//!       X = spend[v]; hasX = X ≠ NONE ∧ X < bhi
//!       val[v] = hasX ? (α(v) ⊗ bsum[X]) ⊕ c(v) : c(v)
//!       B = create[v]
//!       if B ≠ NONE ∧ v == first[B]:          (every output of B inside the window is final)
//!           acc = 0̄; for x in v..min(first[B] + len[B], hi) ascending: acc ⊕= β(x) ⊗ val[x]
//!           bsum[B] = acc                     (argbest x kept in bpred[B])
//! ```
//! Here `blo..bhi` are the blocks whose first output lies in the window
//! ([`BlockIndex::block_bound`]). Every input `u` of `B` satisfies `u < first[B] ≤ v` (P3), so
//! the accumulator of `B` is final when the forward sweep reaches its outputs; a block with
//! `first[B] < lo` has all its inputs before the window, so its window outputs are window
//! sources. Backward, the successors of `v` are the outputs of `spend[v]`, all `> v`, folded
//! once per block in ascending order. Both kernels therefore evaluate exactly the canonical
//! folds of [`crate::dp`] in the same order as push and pull: the results are bit-identical
//! for every semiring (floats included).
//!
//! # Complexity
//! Forward: one ⊗ per node and one ⊗ + ⊕ per non-sink, one random access per node (the slot
//! of `spend[v]`). Backward: one ⊗ per node plus one ⊗ + ⊕ per block output. No BV decoding.
//! Memory: `|E|·n_W` values plus one [`Slot`] (forward) or one element (backward) per window
//! block, `+ 4·n_W` (and `4` per block backward) with predecessors.
//!
//! References: [\[Moh02\]](crate::references#moh02), [\[Kah62\]](crate::references#kah62).

use anyhow::{Result, ensure};
use dsi_progress_logger::prelude::*;

use super::solve::{
    LOG_STEP, block_range, check_init, check_window, final_pred, init_term, pred_tracking,
    start_log,
};
use super::{ArcWeights, DpConf, DpResult, Init};
use crate::blocks::BlockIndex;
use crate::io::{NONE, huge_vec};
use crate::semiring::Semiring;

/// The per-block record of [`forward_blocks`]: the accumulator `⊕_{u ∈ I(B)} val[u] ⊗ α(u)` and
/// the argbest input (`NONE` while no input has pushed). Packed so that one cache line holds
/// both.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Slot<E> {
    /// The accumulator.
    pub acc: E,
    /// The first pushing input, replaced by strictly better ones (selective semirings);
    /// [`NONE`] means "no input of the window has pushed".
    pub best_in: u32,
}

/// Size in bytes of a [`Slot<E>`].
pub fn slot_bytes<E>() -> usize {
    std::mem::size_of::<Slot<E>>()
}

fn check_blocks<S: Semiring, W: ArcWeights<S>>(b: &BlockIndex, conf: &DpConf) -> Result<()> {
    ensure!(
        W::SEPARABLE,
        "the blocks kernels need separable arc weights"
    );
    check_window(conf.window, b.n)
}

/// Forward DP over a [`BlockIndex`] (see the module documentation). Needs separable weights.
pub fn forward_blocks<S, W>(
    s: &S,
    b: &BlockIndex,
    w: &W,
    init: &Init<S::Elem>,
    conf: &DpConf,
    pl: &mut impl ProgressLog,
) -> Result<DpResult<S::Elem>>
where
    S: Semiring,
    W: ArcWeights<S>,
{
    check_blocks::<S, W>(b, conf)?;
    let win = conf.window;
    check_init(init, win)?;
    let track = pred_tracking(s, conf.pred)?;
    let (lo, hi) = (win.lo, win.hi);
    let blocks = block_range(b, win);
    let blo = blocks.start;
    let mut val = huge_vec(win.len(), s.zero());
    let mut pred = track.then(|| huge_vec(win.len(), NONE));
    let mut slots = huge_vec(
        blocks.len(),
        Slot {
            acc: s.zero(),
            best_in: NONE,
        },
    );
    let create: &[u32] = &b.create;
    let spend: &[u32] = &b.spend;
    let bhi = blocks.end as u32;
    let blo32 = blo as u32;

    start_log(pl, "forward blocks", win);
    let mut start = lo;
    while start < hi {
        let end = hi.min(start + LOG_STEP);
        for v in start..end {
            let i = v - lo;
            let bb = create[v];
            let (hp, slot) = if bb != NONE && bb >= blo32 {
                let sl = slots[(bb - blo32) as usize];
                (sl.best_in != NONE, sl)
            } else {
                (
                    false,
                    Slot {
                        acc: s.zero(),
                        best_in: NONE,
                    },
                )
            };
            let bv = init_term(s, init, lo, v, hp);
            let x = if hp {
                let t = s.times(slot.acc, w.beta(s, v));
                if let Some(p) = pred.as_mut() {
                    p[i] = final_pred(s, true, t, bv, slot.best_in);
                }
                s.plus(t, bv)
            } else {
                bv
            };
            val[i] = x;
            let xb = spend[v];
            if xb != NONE && xb < bhi {
                debug_assert!(xb >= blo32, "spend[{v}] = {xb} lies before the window");
                let y = s.times(x, w.alpha(s, v));
                let sl = &mut slots[(xb - blo32) as usize];
                if sl.best_in == NONE || (track && s.better(y, sl.acc)) {
                    sl.best_in = v as u32;
                }
                sl.acc = s.plus(sl.acc, y);
            }
        }
        pl.update_with_count(end - start);
        start = end;
    }
    pl.done();
    Ok(DpResult {
        window: win,
        val,
        pred,
    })
}

/// Backward DP over a [`BlockIndex`] (see the module documentation). Needs separable weights.
pub fn backward_blocks<S, W>(
    s: &S,
    b: &BlockIndex,
    w: &W,
    init: &Init<S::Elem>,
    conf: &DpConf,
    pl: &mut impl ProgressLog,
) -> Result<DpResult<S::Elem>>
where
    S: Semiring,
    W: ArcWeights<S>,
{
    check_blocks::<S, W>(b, conf)?;
    let win = conf.window;
    check_init(init, win)?;
    let track = pred_tracking(s, conf.pred)?;
    let (lo, hi) = (win.lo, win.hi);
    let blocks = block_range(b, win);
    let blo = blocks.start;
    let mut val = huge_vec(win.len(), s.zero());
    let mut pred = track.then(|| huge_vec(win.len(), NONE));
    let mut bsum = huge_vec(blocks.len(), s.zero());
    let mut bpred = track.then(|| huge_vec(blocks.len(), NONE));
    let create: &[u32] = &b.create;
    let spend: &[u32] = &b.spend;
    let first: &[u32] = &b.first;
    let len: &[u32] = &b.len;
    let bhi = blocks.end as u32;

    start_log(pl, "backward blocks", win);
    let mut end = hi;
    while end > lo {
        let start = lo.max(end.saturating_sub(LOG_STEP));
        for v in (start..end).rev() {
            let i = v - lo;
            let xb = spend[v];
            let has = xb != NONE && xb < bhi;
            let c = init_term(s, init, lo, v, has);
            val[i] = if has {
                let x = xb as usize - blo;
                let t = s.times(w.alpha(s, v), bsum[x]);
                if let (Some(p), Some(bp)) = (pred.as_mut(), bpred.as_ref()) {
                    p[i] = final_pred(s, true, t, c, bp[x]);
                }
                s.plus(t, c)
            } else {
                c
            };
            let bb = create[v];
            if bb != NONE && first[bb as usize] as usize == v {
                let bb = bb as usize;
                let stop = hi.min(v + len[bb] as usize);
                let mut acc = s.zero();
                let mut best = NONE;
                for x in v..stop {
                    let y = s.times(w.beta(s, x), val[x - lo]);
                    if track && (best == NONE || s.better(y, acc)) {
                        best = x as u32;
                    }
                    acc = s.plus(acc, y);
                }
                bsum[bb - blo] = acc;
                if let Some(bp) = bpred.as_mut() {
                    bp[bb - blo] = best;
                }
            }
        }
        pl.update_with_count(end - start);
        end = start;
    }
    pl.done();
    Ok(DpResult {
        window: win,
        val,
        pred,
    })
}
