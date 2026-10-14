//! Parallel certification of DP fixpoints (F1 §7.6).
//!
//! # Algorithm
//! On a DAG the fixpoint equations of a DP have a unique solution, so a vector is **the**
//! solution iff it satisfies them at every node. [`certify`] re-evaluates every equation in
//! parallel ([`par_chunks`], no ordering constraint) with the canonical fold of [`crate::dp`],
//! reading the stored values of the neighbours:
//! ```text
//! forward  (reads Gᵀ):  x(v) = has_pred ? (⊕_{u ∈ pred_W(v)} x(u) ⊗ α(u)) ⊗ β(v) ⊕ b(v) : b(v)
//! backward (reads G):   x(u) = has_succ ? α(u) ⊗ (⊕_{v ∈ succ_W(u)} β(v) ⊗ x(v)) ⊕ c(u) : c(u)
//! ```
//! and compares the result with the stored value:
//! - [`Tolerance::Exact`]: identical bytes (exact semirings and selective float semirings, for
//!   which every kernel is bit-reproducible);
//! - [`Tolerance::PerTerm(c)`](Tolerance::PerTerm): relative error at most `c·(k+2)·u` with
//!   `k` the number of terms of the equation and `u = 2⁻⁵³` (non-selective float semirings; the
//!   local error of a recursive sum of `k` terms is at most `(k−1)·u` to first order
//!   [\[Hig93\]](crate::references#hig93)).
//!
//! With predecessors it also checks, for every node, that the stored predecessor `p` is a
//! neighbour inside the window whose term realises the value (`term(p) ⊕ b(v)` equals the
//! stored value bit for bit), that no neighbour is strictly better and that the initial term
//! is not strictly better; `NONE` is valid iff there is no neighbour or the initial term is
//! strictly better than the fold. Any optimal predecessor is accepted (the canonical tie-break
//! is a property of the kernels, tested separately).
//!
//! Certification never uses the [`BlockIndex`](crate::blocks::BlockIndex): it shares no data
//! path with the block kernels.
//!
//! # Complexity
//! `O(n_W + m_W)` semiring operations, spread over the rayon pool; no extra memory besides the
//! decoders' buffers.

use anyhow::{Result, ensure};
use dsi_progress_logger::prelude::*;
use webgraph::traits::RandomAccessGraph;

use super::solve::{check_init, check_window, init_term};
use super::{ArcWeights, DpResult, Init};
use crate::graph::ChunkedGraph;
use crate::io::{Dump, NONE};
use crate::semiring::Semiring;
use crate::semiring::extfloat::Ext;
use crate::semiring::matrix::Mat2;
use crate::sweep::{NodeSuccIter, par_chunks};
use crate::window::Dir;

/// Unit roundoff of `f64`, `2⁻⁵³`.
pub const UNIT_ROUNDOFF: f64 = 1.0 / 9_007_199_254_740_992.0;

/// How stored and recomputed values are compared.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub enum Tolerance {
    /// Identical bytes.
    Exact,
    /// Relative error at most `c·(k+2)·u` for an equation of `k` terms (`u = 2⁻⁵³`); the
    /// default `c` is 4.
    PerTerm(f64),
}

impl Tolerance {
    /// `PerTerm(4.0)`, the default for non-selective float semirings.
    pub const DEFAULT_PER_TERM: Self = Self::PerTerm(4.0);
}

/// Result of [`certify`].
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct CertReport {
    /// Equations checked (window nodes).
    pub checked: u64,
    /// Nodes whose stored value violates its equation.
    pub violations: u64,
    /// Nodes whose stored predecessor is not valid.
    pub pred_violations: u64,
    /// Largest relative difference observed between a stored and a recomputed value.
    pub max_rel_err: f64,
    /// Smallest node (absolute id) with a value or predecessor violation.
    pub first_violation: Option<usize>,
}

impl CertReport {
    /// Whether no violation was found.
    pub fn ok(&self) -> bool {
        self.violations == 0 && self.pred_violations == 0
    }

    fn merge(self, o: Self) -> Self {
        Self {
            checked: self.checked + o.checked,
            violations: self.violations + o.violations,
            pred_violations: self.pred_violations + o.pred_violations,
            max_rel_err: self.max_rel_err.max(o.max_rel_err),
            first_violation: match (self.first_violation, o.first_violation) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
        }
    }
}

/// A numeric distance between two semiring elements, used by [`certify`] for
/// [`Tolerance::PerTerm`] and for [`CertReport::max_rel_err`].
///
/// `rel_err(a, b)` is 0 when the values are equal and otherwise `|a − b| / max(|a|, |b|)`
/// (infinite when they differ and one of them is infinite or NaN; for integers `|a − b|` is
/// computed exactly, so different integers always have a positive error). Implemented for every
/// element type of the crate's semirings.
///
/// [`within`](Self::within) decides acceptance under [`Tolerance::PerTerm`]: floating-point
/// types accept a relative error up to the bound, *exact* types (integers, `bool`, and the exact
/// components of tuples and matrices, e.g. the `M61` count of a `Product<ExtFloat, M61>`)
/// accept only equality, whatever the bound (the per-term rounding-error model of
/// [\[Hig93\]](crate::references#hig93) applies to rounded arithmetic only).
pub trait RelErr {
    /// The relative difference between `self` and `other`.
    fn rel_err(&self, other: &Self) -> f64;

    /// Whether `other` is within relative error `tol` of `self`: `rel_err ≤ tol` (not NaN) for
    /// rounded types, equality for exact types.
    #[inline]
    fn within(&self, other: &Self, tol: f64) -> bool {
        let e = self.rel_err(other);
        !e.is_nan() && e <= tol
    }
}

fn rel_f64(a: f64, b: f64) -> f64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        0.0
    } else if !a.is_finite() || !b.is_finite() {
        f64::INFINITY
    } else {
        (a - b).abs() / a.abs().max(b.abs())
    }
}

macro_rules! rel_err_int {
    ($($t:ty: $mag:ident),*) => {$(
        impl RelErr for $t {
            /// Exact: `|a − b|` is computed without rounding (≥ 1 when `a ≠ b`), so the result
            /// is positive for any two different values, even above 2⁵³.
            #[inline]
            fn rel_err(&self, other: &Self) -> f64 {
                if self == other {
                    0.0
                } else {
                    let d = self.abs_diff(*other);
                    let m = $mag(*self).max($mag(*other));
                    d as f64 / m as f64
                }
            }

            #[inline]
            fn within(&self, other: &Self, _tol: f64) -> bool {
                self == other
            }
        }
    )*};
}
#[inline]
fn id<T>(x: T) -> T {
    x
}
rel_err_int!(u8: id, u32: id, u64: id, u128: id, i64: i64_mag);

#[inline]
fn i64_mag(x: i64) -> u64 {
    x.unsigned_abs()
}

impl RelErr for f64 {
    #[inline]
    fn rel_err(&self, other: &Self) -> f64 {
        rel_f64(*self, *other)
    }
}

impl RelErr for f32 {
    #[inline]
    fn rel_err(&self, other: &Self) -> f64 {
        rel_f64(*self as f64, *other as f64)
    }
}

impl RelErr for bool {
    #[inline]
    fn rel_err(&self, other: &Self) -> f64 {
        if self == other { 0.0 } else { 1.0 }
    }

    #[inline]
    fn within(&self, other: &Self, _tol: f64) -> bool {
        self == other
    }
}

/// Componentwise: the error is the larger one, and each component must be
/// [`within`](RelErr::within) the bound on its own terms.
impl<A: RelErr, B: RelErr> RelErr for (A, B) {
    #[inline]
    fn rel_err(&self, other: &Self) -> f64 {
        self.0.rel_err(&other.0).max(self.1.rel_err(&other.1))
    }

    #[inline]
    fn within(&self, other: &Self, tol: f64) -> bool {
        self.0.within(&other.0, tol) && self.1.within(&other.1, tol)
    }
}

impl<E: RelErr> RelErr for Mat2<E> {
    fn rel_err(&self, other: &Self) -> f64 {
        let mut r = 0.0f64;
        for i in 0..2 {
            for j in 0..2 {
                r = r.max(self.0[i][j].rel_err(&other.0[i][j]));
            }
        }
        r
    }

    fn within(&self, other: &Self, tol: f64) -> bool {
        (0..2).all(|i| (0..2).all(|j| self.0[i][j].within(&other.0[i][j], tol)))
    }
}

/// Extended-exponent floats `m·2^e`: compared through `m₁·2^(e₁−e₂)` against `m₂`, using only
/// the public fields (a zero mantissa is zero).
impl RelErr for Ext {
    fn rel_err(&self, other: &Self) -> f64 {
        if self.m == other.m && (self.e == other.e || self.m == 0.0) {
            return 0.0;
        }
        if self.m == 0.0 || other.m == 0.0 {
            return 1.0;
        }
        let d = self.e.saturating_sub(other.e);
        if d.abs() > 1000 {
            return 1.0;
        }
        rel_f64(self.m * (d as f64).exp2(), other.m)
    }
}

/// Whether two elements have identical bytes (their [`Dump`] encodings).
pub fn bit_eq<E: Dump>(a: &E, b: &E) -> bool {
    let mut x = [0u8; 64];
    let mut y = [0u8; 64];
    if E::BYTES <= 64 {
        a.put(&mut x[..E::BYTES]);
        b.put(&mut y[..E::BYTES]);
        x[..E::BYTES] == y[..E::BYTES]
    } else {
        let mut x = vec![0u8; E::BYTES];
        let mut y = vec![0u8; E::BYTES];
        a.put(&mut x);
        b.put(&mut y);
        x == y
    }
}

/// Certifies `res` against the fixpoint equations of the DP `dir` (see the module
/// documentation). Forward equations read the transpose (`g_or_gt` = Gᵀ), backward equations
/// the graph (`g_or_gt` = G).
///
/// Fails (rather than reporting violations) if `res` does not match the window or the graph.
#[allow(clippy::too_many_arguments)]
pub fn certify<S, W, G>(
    s: &S,
    g_or_gt: &ChunkedGraph<G>,
    dir: Dir,
    w: &W,
    init: &Init<S::Elem>,
    res: &DpResult<S::Elem>,
    tol: Tolerance,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<CertReport>
where
    S: Semiring,
    S::Elem: RelErr,
    W: ArcWeights<S>,
    G: RandomAccessGraph + Sync,
{
    let win = res.window;
    check_window(win, g_or_gt.num_nodes())?;
    check_init(init, win)?;
    ensure!(
        res.val.len() == win.len(),
        "{} values for a window of {} nodes",
        res.val.len(),
        win.len()
    );
    if let Some(p) = &res.pred {
        ensure!(
            p.len() == win.len(),
            "{} predecessors for a window of {} nodes",
            p.len(),
            win.len()
        );
    }
    let lo = win.lo;
    let val = &res.val;
    let pred = res.pred.as_deref();
    let allowed = |k: usize| match tol {
        Tolerance::Exact => 0.0,
        Tolerance::PerTerm(c) => c * (k as f64 + 2.0) * UNIT_ROUNDOFF,
    };
    // the term of neighbour `x` in the equation of `v`, as folded by the kernels
    let term = |v: usize, x: usize| -> S::Elem {
        match dir {
            Dir::Forward if W::SEPARABLE => s.times(val[x - lo], w.alpha(s, x)),
            Dir::Forward => s.times(val[x - lo], w.arc(s, x, v)),
            Dir::Backward if W::SEPARABLE => s.times(w.beta(s, x), val[x - lo]),
            Dir::Backward => s.times(w.arc(s, v, x), val[x - lo]),
        }
    };
    // the outer factor applied to a fold (β(v) forward, α(u) backward)
    let outer = |v: usize, acc: S::Elem| -> S::Elem {
        match dir {
            _ if !W::SEPARABLE => acc,
            Dir::Forward => s.times(acc, w.beta(s, v)),
            Dir::Backward => s.times(w.alpha(s, v), acc),
        }
    };

    pl.item_name("node")
        .display_memory(true)
        .expected_updates(Some(win.len()));
    pl.start(format!("certifying {dir:?} values over {win}"));
    let rep = par_chunks(
        g_or_gt,
        win,
        pl,
        |it: &mut dyn NodeSuccIter| -> Result<CertReport> {
            let mut r = CertReport::default();
            while let Some((v, nbrs)) = it.next_node() {
                // forward lists are predecessors (Gᵀ): drop those before the window
                let nbrs = match dir {
                    Dir::Forward => &nbrs[nbrs.partition_point(|&u| (u as usize) < lo)..],
                    Dir::Backward => nbrs,
                };
                let mut acc = s.zero();
                for &x in nbrs {
                    acc = s.plus(acc, term(v, x as usize));
                }
                let hp = !nbrs.is_empty();
                let t = if hp { outer(v, acc) } else { acc };
                let b = init_term(s, init, lo, v, hp);
                let x = if hp { s.plus(t, b) } else { b };
                let stored = val[v - lo];
                let e = x.rel_err(&stored);
                r.checked += 1;
                r.max_rel_err = r.max_rel_err.max(e);
                let mut bad = match tol {
                    Tolerance::Exact => !bit_eq(&x, &stored),
                    Tolerance::PerTerm(_) => !stored.within(&x, allowed(nbrs.len() + 1)),
                };
                if bad {
                    r.violations += 1;
                }
                if let Some(p) = pred {
                    let ok = match p[v - lo] {
                        NONE => !hp || s.better(b, t),
                        p => {
                            let p = p as usize;
                            nbrs.binary_search(&(p as u32)).is_ok() && {
                                let tp = outer(v, term(v, p));
                                !s.better(t, tp)
                                    && !s.better(b, tp)
                                    && bit_eq(&s.plus(tp, b), &stored)
                            }
                        }
                    };
                    if !ok {
                        r.pred_violations += 1;
                        bad = true;
                    }
                }
                if bad && r.first_violation.is_none() {
                    r.first_violation = Some(v);
                }
            }
            Ok(r)
        },
        CertReport::merge,
    )?;
    pl.done();
    Ok(rep)
}
