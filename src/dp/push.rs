//! Forward push kernel over the graph (F1 §6.1 with the canonical order of [`crate::dp`]).
//!
//! # Algorithm
//! One [`sweep_forward`] over the window. `val[v]` holds the un-β'd accumulator
//! `⊕_{u ∈ pred(v), u < current} val[u] ⊗ α(u)` until the sweep reaches `v`; then it is
//! finalised and pushed to the successors:
//! ```text
//! val[·] = 0̄; has_pred = ∅
//! for u in lo..hi (ascending, parallel decode):
//!     hp = has_pred[u]
//!     val[u] = hp ? (val[u] ⊗ β(u)) ⊕ b(u) : b(u)          (Arc: hp ? val[u] ⊕ b(u) : b(u))
//!     y = val[u] ⊗ α(u)
//!     for v in succ(u) ∩ W ascending: val[v] ⊕= y           (Arc: y = val[u] ⊗ w(u, v))
//!                                     has_pred[v] = 1
//! ```
//! Every predecessor of `v` is smaller than `v`, so `val[v]` is final when it is reached, and
//! the pushes into `v` arrive in ascending predecessor order: exactly the canonical fold.
//! `has_pred` is set independently of the pushed values, so a predecessor carrying 0̄ still
//! makes `v` a non-source for [`Init::Sources`].
//!
//! With predecessor tracking (selective semirings) `pred[v]` holds the first pushing node and
//! is replaced only on a strictly [`better`](crate::semiring::Semiring::better) term, which
//! gives the smallest argbest (the tie-break of `analysis_path`).
//!
//! # Complexity
//! `O(n_W + m_W)` semiring operations (one ⊗ per node for separable weights, one per arc for
//! [`Arc`](super::Arc) weights) plus the parallel decode. Memory: `|E|·n_W + n_W/8` bytes
//! (`+ 4·n_W` with predecessors) plus the sweep buffers.
//!
//! References: [\[Kah62\]](crate::references#kah62) (topological pass),
//! [\[Moh02\]](crate::references#moh02) (semiring framework).

use anyhow::Result;
use dsi_progress_logger::prelude::*;
use webgraph::traits::RandomAccessGraph;

use super::solve::{check_init, check_window, final_pred, init_term, pred_tracking, start_log};
use super::{ArcWeights, DpConf, DpResult, Init};
use crate::graph::ChunkedGraph;
use crate::io::{NONE, huge_vec};
use crate::semiring::Semiring;
use crate::sweep::sweep_forward;

/// Forward DP by pushing along the arcs of `g` (see the module documentation).
///
/// Fails with [`DagError::NotTopological`](crate::DagError::NotTopological) if an arc of the
/// window goes backwards.
pub fn forward_push<S, W, G>(
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
    let mut has = huge_vec(n_w.div_ceil(64), 0u64);
    let mut pred = track.then(|| huge_vec(n_w, NONE));

    start_log(pl, "forward push", win);
    {
        let (val, has, pred) = (&mut val, &mut has, &mut pred);
        sweep_forward(g, win, &conf.sweep, pl, move |chunk| {
            for (u, succ) in chunk.iter() {
                let i = u - lo;
                let hp = (has[i >> 6] >> (i & 63)) & 1 != 0;
                let b = init_term(s, init, lo, u, hp);
                let x = if hp {
                    let t = if W::SEPARABLE {
                        s.times(val[i], w.beta(s, u))
                    } else {
                        val[i]
                    };
                    if let Some(p) = pred.as_mut() {
                        p[i] = final_pred(s, true, t, b, p[i]);
                    }
                    s.plus(t, b)
                } else {
                    b
                };
                val[i] = x;
                if succ.is_empty() {
                    continue;
                }
                let xa = if W::SEPARABLE {
                    s.times(x, w.alpha(s, u))
                } else {
                    x
                };
                for &v in succ {
                    let v = v as usize;
                    let j = v - lo;
                    let y = if W::SEPARABLE {
                        xa
                    } else {
                        s.times(x, w.arc(s, u, v))
                    };
                    let bit = 1u64 << (j & 63);
                    let first = has[j >> 6] & bit == 0;
                    has[j >> 6] |= bit;
                    if let Some(p) = pred.as_mut() {
                        if first || s.better(y, val[j]) {
                            p[j] = u as u32;
                        }
                    }
                    val[j] = s.plus(val[j], y);
                }
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
