//! Pull kernels: forward over the transpose (F1 §6.2), backward over the graph (F1 §6.3).
//!
//! # Algorithm
//! Both kernels write each value exactly once, folding the already final values of the
//! neighbours in ascending id order (the canonical order of [`crate::dp`]):
//! ```text
//! forward_pull  (Gᵀ, ascending v):   acc = 0̄; for u in pred(v) ∩ W ascending: acc ⊕= val[u] ⊗ α(u)
//!                                    val[v] = has_pred ? (acc ⊗ β(v)) ⊕ b(v) : b(v)
//! backward_pull (G, descending u):   acc = 0̄; for v in succ(u) ∩ W ascending: acc ⊕= β(v) ⊗ val[v]
//!                                    val[u] = has_succ ? (α(u) ⊗ acc) ⊕ c(u) : c(u)
//! ```
//! (with [`Arc`](super::Arc) weights the terms are `val[u] ⊗ w(u, v)`, resp. `w(u, v) ⊗ val[v]`,
//! and no outer factor is applied). They are the reference kernels: push and the block kernels
//! are tested bit for bit against them.
//!
//! `backward_pull` runs on [`sweep_backward`] (chunks decoded forward in parallel, consumed in
//! reverse). `forward_pull` runs on [`sweep_forward_preds`], the ordered decode of the
//! **transpose** (predecessors `< lo` are dropped and every predecessor must be smaller than its
//! node, else [`DagError::NotTopological`](crate::error::DagError::NotTopological)).
//!
//! # Complexity
//! `O(n_W + m_W)` semiring operations plus the parallel decode; memory `|E|·n_W` (`+ 4·n_W` with
//! predecessors) plus the chunk buffers.
//!
//! References: [\[Kah62\]](crate::references#kah62), [\[Moh02\]](crate::references#moh02),
//! [\[BV04\]](crate::references#bv04) (the BV graphs read by the decoders).

use anyhow::Result;
use dsi_progress_logger::prelude::*;
use webgraph::traits::RandomAccessGraph;

use super::solve::{check_init, check_window, final_pred, init_term, pred_tracking, start_log};
use super::{ArcWeights, DpConf, DpResult, Init};
use crate::graph::ChunkedGraph;
use crate::io::{NONE, huge_vec};
use crate::semiring::Semiring;
use crate::sweep::{sweep_backward, sweep_forward_preds};

/// Forward DP by pulling over the transpose `gt` (see the module documentation).
///
/// Fails with [`DagError::NotTopological`](crate::error::DagError::NotTopological) if a
/// predecessor is not smaller than its node.
pub fn forward_pull<S, W, G>(
    s: &S,
    gt: &ChunkedGraph<G>,
    w: &W,
    init: &Init<S::Elem>,
    conf: &DpConf,
    pl: &mut impl ProgressLog,
) -> Result<DpResult<S::Elem>>
where
    S: Semiring,
    W: ArcWeights<S>,
    G: RandomAccessGraph + Sync,
{
    let win = conf.window;
    check_window(win, gt.num_nodes())?;
    check_init(init, win)?;
    let track = pred_tracking(s, conf.pred)?;
    let lo = win.lo;
    let n_w = win.len();
    let mut val = huge_vec(n_w, s.zero());
    let mut pred = track.then(|| huge_vec(n_w, NONE));

    start_log(pl, "forward pull", win);
    {
        let (val, pred) = (&mut val, &mut pred);
        sweep_forward_preds(gt, win, &conf.sweep, pl, move |chunk| {
            for (v, preds) in chunk.iter() {
                let mut acc = s.zero();
                let mut best = NONE;
                for &u in preds {
                    let u = u as usize;
                    let y = if W::SEPARABLE {
                        s.times(val[u - lo], w.alpha(s, u))
                    } else {
                        s.times(val[u - lo], w.arc(s, u, v))
                    };
                    if track && (best == NONE || s.better(y, acc)) {
                        best = u as u32;
                    }
                    acc = s.plus(acc, y);
                }
                let hp = !preds.is_empty();
                let b = init_term(s, init, lo, v, hp);
                let i = v - lo;
                val[i] = if hp {
                    let t = if W::SEPARABLE {
                        s.times(acc, w.beta(s, v))
                    } else {
                        acc
                    };
                    if let Some(p) = pred.as_mut() {
                        p[i] = final_pred(s, true, t, b, best);
                    }
                    s.plus(t, b)
                } else {
                    b
                };
            }
            Ok(())
        })?;
    }
    pl.done();
    Ok(DpResult {
        window: win,
        val,
        pred,
    })
}

/// Backward DP by pulling over `g` in reverse id order (see the module documentation).
///
/// Fails with [`DagError::NotTopological`](crate::error::DagError::NotTopological) if an arc of
/// the window goes backwards.
pub fn backward_pull<S, W, G>(
    s: &S,
    g: &ChunkedGraph<G>,
    w: &W,
    init: &Init<S::Elem>,
    conf: &DpConf,
    pl: &mut impl ProgressLog,
) -> Result<DpResult<S::Elem>>
where
    S: Semiring,
    W: ArcWeights<S>,
    G: RandomAccessGraph + Sync,
{
    let win = conf.window;
    check_window(win, g.num_nodes())?;
    check_init(init, win)?;
    let track = pred_tracking(s, conf.pred)?;
    let lo = win.lo;
    let n_w = win.len();
    let mut val = huge_vec(n_w, s.zero());
    let mut pred = track.then(|| huge_vec(n_w, NONE));

    start_log(pl, "backward pull", win);
    {
        let (val, pred) = (&mut val, &mut pred);
        sweep_backward(g, win, &conf.sweep, pl, move |chunk| {
            for (u, succ) in chunk.iter_rev() {
                let mut acc = s.zero();
                let mut best = NONE;
                for &v in succ {
                    let v = v as usize;
                    let y = if W::SEPARABLE {
                        s.times(w.beta(s, v), val[v - lo])
                    } else {
                        s.times(w.arc(s, u, v), val[v - lo])
                    };
                    if track && (best == NONE || s.better(y, acc)) {
                        best = v as u32;
                    }
                    acc = s.plus(acc, y);
                }
                let hs = !succ.is_empty();
                let c = init_term(s, init, lo, u, hs);
                let i = u - lo;
                val[i] = if hs {
                    let t = if W::SEPARABLE {
                        s.times(w.alpha(s, u), acc)
                    } else {
                        acc
                    };
                    if let Some(p) = pred.as_mut() {
                        p[i] = final_pred(s, true, t, c, best);
                    }
                    s.plus(t, c)
                } else {
                    c
                };
            }
            Ok(())
        })?;
    }
    pl.done();
    Ok(DpResult {
        window: win,
        val,
        pred,
    })
}
