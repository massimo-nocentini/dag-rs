//! Path counting: the number of source-to-node paths `F(v)` and node-to-sink paths `B(u)` of a
//! window, with exact residues, extended-exponent magnitudes and saturating integers.
//!
//! # Definitions
//! For a window `W = [lo, hi)` of a DAG whose identity is a topological order (ids are
//! chronological on the PG, so `W` is convex), a *window source* has no predecessor in `W` and a
//! *window sink* no successor in `W`. Then
//! ```text
//! F(v) = #paths inside W from a window source to v      (forward,  Init::Sources(1̄), weights 1̄)
//! B(u) = #paths inside W from u to a window sink         (backward, Init::Sources(1̄), weights 1̄)
//! through(v) = F(v) · B(v)                               (source-to-sink paths through v)
//! total = Σ_{window sinks v} F(v) = Σ_{window sources u} B(u)   (source-to-sink paths of W)
//! ```
//! Both directions are one semiring DP of [`crate::dp`] in the counting semiring `(ℕ, +, ×)`
//! ([\[Moh02\]](crate::references#moh02), [\[Leh77\]](crate::references#leh77)) evaluated in id
//! order ([\[Kah62\]](crate::references#kah62)); the totals identity is the duality
//! `⊕_v F(v)⊗c(v) = ⊕_u b(u)⊗B(u)` with `b`, `c` the source and sink indicators, a direct
//! consequence of distributivity ([\[GM08\]](crate::references#gm08)).
//!
//! # Representations ([`CountKind`])
//! Counts grow exponentially with depth (the PG has depth 3.56·10⁶), so no fixed-width integer
//! holds them. Four representations are offered:
//!
//! | Kind | Semiring | Output | Meaning |
//! |---|---|---|---|
//! | `extfloat` | [`ExtFloat`] | `log2.f64le` | magnitude, `log2 F(v)` (relative error below, see *Accuracy*) |
//! | `mod61` | [`M61`] | `mod61.u64le` | exact residue `F(v) mod (2⁶¹ − 1)` |
//! | `sat64` | [`SatU64`] | `sat64.u64le` | exact below `u64::MAX`, `u64::MAX` = "≥ u64::MAX" |
//! | `sat128` | [`SatU128`] | `sat128.u128le` | exact below `u128::MAX`, `u128::MAX` = "≥ u128::MAX" |
//!
//! `extfloat` and `mod61` are computed in **one pass per direction** with
//! `Product<ExtFloat, M61>` (24 B per node); each saturating kind is one more pass. The forward
//! and backward directions run concurrently (`rayon::join`) when the memory budget allows,
//! otherwise one after the other ([`count_paths_budgeted`]).
//!
//! # Checks
//! - **Duality** ([`Duality`]): the two totals must be equal: exactly in `M61` and in the
//!   saturating kinds (unless saturated), and within relative [`EXT_DUALITY_TOL`] = 10⁻⁶ in
//!   ExtFloat. The tolerance is **empirical**, not a proven bound: the a-priori bound of
//!   *Accuracy* below can exceed it on the full PG (≈ 1.5·10⁻⁶), while the observed relative
//!   difference is ≈ 4·10⁻¹³ on the window `0..10⁸`. A failure therefore means either a defect
//!   or an adversarial accumulation of rounding errors; the observed difference is recorded in
//!   [`Duality::extfloat_rel_diff`] so that the report can quote it.
//! - Every ExtFloat count is ≥ 1 (every window node has a window source above it and a window
//!   sink below it), so `log2` values are ≥ 0 and finite.
//!
//! # Accuracy of ExtFloat
//! Let `u = 2⁻⁵³`. Each ⊕ rounds once, ⊗ by 1̄ is exact, and the kernels fold the `k`
//! predecessors of a node one after the other (recursive summation, D10), which has the
//! first-order error bound `(k − 1)·u` relative to the exact sum of the (already perturbed)
//! terms ([\[Hig93\]](crate::references#hig93)). Hence, writing `indeg_W(w)` for the in-degree
//! of `w` inside the window, the relative error of `F(v)` is at most, to first order,
//! ```text
//! ε(v) ≤ u · max over paths P ending at v of Σ_{w ∈ P} (indeg_W(w) − 1)⁺
//!      ≤ u · min( depth(v)·(Δ⁻ − 1),  Σ_{w ∈ W} (indeg_W(w) − 1)⁺ = m_W − n_W + sources_W )
//! ```
//! (symmetrically with out-degrees for `B`). The depth form `D·(Δ⁻ − 1)·u ≈ 7.9·10⁻⁶` is weak
//! on the PG (`D = 3.56·10⁶`, `Δ⁻ = 20 000`); the global form gives ≈ 7.2·10⁻⁷ forward and
//! ≈ 7.3·10⁻⁷ backward on the full PG. Each total adds the error `⌈log2 n_W⌉·u` of its
//! deterministic pairwise sum, so the first-order bound on the duality difference is the sum
//! of the two directions' bounds, ≈ 1.5·10⁻⁶ (hence the tolerance above is empirical). These
//! bounds are worst cases that need every rounding in the same direction; random-walk
//! behaviour gives errors of order `√k·u`. The exponent is an `i64`, so there is no overflow.
//!
//! # Complexity and memory
//! `O(n_W + m_W)` semiring operations per direction and kind with the generic kernels, `O(n_W)`
//! with the block kernels. Memory per direction: 24 B/node for the product pass (plus one
//! `Slot<(Ext, u64)>` of 32 B per block of the window with the forward block kernel), then
//! 8 B/node for the `log2` array while the residues are compacted in place into the product
//! buffer (peak 32 B/node, then 16 B/node retained); 8 B/node per further `log2`/`mod61`/`sat64`
//! output and 16 B/node for `sat128`; the endpoint bitsets take `n_W/4` bytes. See
//! [`direction_bytes`].
//!
//! # PG fast path
//! With a [`BlockIndex`] both passes run the decode-free block kernels of
//! [`crate::dp::blocks`], and the window sources and sinks come from `create`/`spend` scans
//! ([`Endpoints::compute`]), so no BV list is decoded at all. All outputs of a transaction have
//! the same `F` (same predecessor set, F1 §6.5).
//!
//! # Example
//! ```
//! use dag::dp::{DpConf, DpInput};
//! use dag::paths::{CountKind, count_paths};
//! use dag::{Which, Window, testkit};
//!
//! // 0 → {1, 2} → 3: two source-to-sink paths
//! let g = testkit::vec_graph(4, [(0, 1), (0, 2), (1, 3), (2, 3)]);
//! let (cg, cgt) = (testkit::chunked(&g, 2), testkit::chunked(&testkit::transpose(&g), 2));
//! let conf = DpConf::new(Window::full(4));
//! let c = count_paths(&DpInput::with_transpose(&cg, &cgt), Which::Both,
//!                     &[CountKind::ExtFloat, CountKind::M61, CountKind::SatU64], &conf,
//!                     &mut Option::<dsi_progress_logger::ProgressLogger>::None).unwrap();
//! assert_eq!(c.fwd_mod61.as_deref(), Some(&[1, 1, 1, 2][..]));
//! assert_eq!(c.bwd_sat64.as_deref(), Some(&[2, 1, 1, 1][..]));
//! assert_eq!(c.total_mod61, 2);
//! assert_eq!(c.total_log2, 1.0);
//! assert!(c.summary.duality.unwrap().ok);
//! ```

use std::time::Instant;

use anyhow::{Result, ensure};
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::traits::RandomAccessGraph;

use crate::blocks::BlockIndex;
use crate::cpm::backward_kernel;
use crate::dp::solve::{estimate_bytes, resolve_kernel, solve, sweep_buffer_bytes};
use crate::dp::{DpConf, DpInput, Init, Kernel, One};
use crate::graph::ChunkedGraph;
use crate::io::{MemBudget, NONE, as_atomic_u64, huge_vec};
use crate::semiring::Semiring;
use crate::semiring::counting::{SatU64, SatU128};
use crate::semiring::extfloat::{Ext, ExtFloat};
use crate::semiring::modular::M61;
use crate::semiring::product::Product;
use crate::sweep::NodeSuccIter;
use crate::window::{Dir, Which, Window};

/// A path-count representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, clap::ValueEnum, serde::Serialize)]
pub enum CountKind {
    /// Extended-exponent float magnitudes ([`ExtFloat`]), written as `log2` values.
    #[value(name = "extfloat")]
    ExtFloat,
    /// Exact residues modulo 2⁶¹ − 1 ([`M61`]).
    #[value(name = "mod61")]
    M61,
    /// Saturating u64 counts ([`SatU64`]).
    #[value(name = "sat64")]
    SatU64,
    /// Saturating u128 counts ([`SatU128`]).
    #[value(name = "sat128")]
    SatU128,
}

/// Relative tolerance of the ExtFloat duality check: an empirical threshold, not a proven
/// bound (see *Checks* and *Accuracy of ExtFloat* in the module documentation).
pub const EXT_DUALITY_TOL: f64 = 1e-6;

/// Ranges shorter than this are summed sequentially by the pairwise reductions.
const PAR_MIN: usize = 1 << 16;

/// Number of bins of [`DirSummary::log2_hist`].
pub const HIST_BINS: usize = 64;

/// The requested kinds as flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Kinds {
    ext: bool,
    m61: bool,
    sat64: bool,
    sat128: bool,
}

impl Kinds {
    fn new(kinds: &[CountKind]) -> Self {
        let has = |k| kinds.contains(&k);
        Self {
            ext: has(CountKind::ExtFloat),
            m61: has(CountKind::M61),
            sat64: has(CountKind::SatU64),
            sat128: has(CountKind::SatU128),
        }
    }

    fn list(self) -> Vec<CountKind> {
        [
            (self.ext, CountKind::ExtFloat),
            (self.m61, CountKind::M61),
            (self.sat64, CountKind::SatU64),
            (self.sat128, CountKind::SatU128),
        ]
        .into_iter()
        .filter_map(|(b, k)| b.then_some(k))
        .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Window sources and sinks
// ---------------------------------------------------------------------------------------------

/// The window sources (no predecessor in the window) and sinks (no successor in the window), as
/// two bitsets over `v − lo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoints {
    window: Window,
    has_pred: Vec<u64>,
    has_succ: Vec<u64>,
}

impl Endpoints {
    /// Bytes of the two bitsets of a window of `n_w` nodes.
    pub fn bytes(n_w: usize) -> u64 {
        2 * 8 * n_w.div_ceil(64) as u64
    }

    /// Computes the endpoints of `w` from the cheapest input: the [`BlockIndex`] (no decoding),
    /// else one parallel scan of the forward graph, else of the transpose.
    ///
    /// With blocks, `v` has a window successor iff `spend[v] < bhi` (the block bound of `hi`),
    /// and a window predecessor iff some input `u ∈ [lo, hi)` spends `create[v]` (one parallel
    /// scan of `spend[lo..hi)` marks those blocks).
    pub fn compute<G: RandomAccessGraph + Sync>(
        input: &DpInput<G>,
        w: Window,
        pl: &mut impl ConcurrentProgressLog,
    ) -> Result<Self> {
        ensure!(
            w.lo <= w.hi && w.hi <= input.n,
            "invalid window {w} for {} nodes",
            input.n
        );
        if let Some(b) = input.blocks {
            Ok(Self::from_blocks(b, w))
        } else if let Some(g) = input.g {
            Self::from_graph(g, w, false, pl)
        } else if let Some(gt) = input.gt {
            Self::from_graph(gt, w, true, pl)
        } else {
            anyhow::bail!("path counting needs a graph, a transpose or a BlockIndex")
        }
    }

    /// Endpoints from a block index (see [`compute`](Self::compute)).
    pub fn from_blocks(b: &BlockIndex, w: Window) -> Self {
        let (lo, hi) = (w.lo, w.hi);
        let words = w.len().div_ceil(64);
        let blo = b.block_bound(lo);
        let bhi = b.block_bound(hi);
        // blocks of [blo, bhi) with at least one input in the window
        let mut fed = vec![0u64; (bhi - blo).div_ceil(64)];
        {
            let fed = as_atomic_u64(&mut fed);
            (lo..hi)
                .into_par_iter()
                .with_min_len(PAR_MIN)
                .for_each(|u| {
                    let x = b.spend[u];
                    // first[x] > u ≥ lo (P3), hence x ≥ blo
                    if x != NONE && (blo..bhi).contains(&(x as usize)) {
                        let k = x as usize - blo;
                        fed[k / 64].fetch_or(1 << (k % 64), std::sync::atomic::Ordering::Relaxed);
                    }
                });
        }
        let mut has_pred = vec![0u64; words];
        let mut has_succ = vec![0u64; words];
        has_pred
            .par_iter_mut()
            .zip(has_succ.par_iter_mut())
            .enumerate()
            .for_each(|(k, (p, s))| {
                let start = lo + 64 * k;
                for v in start..(start + 64).min(hi) {
                    let bit = 1u64 << (v - start);
                    let x = b.spend[v];
                    if x != NONE && (x as usize) < bhi {
                        *s |= bit;
                    }
                    let c = b.create[v];
                    if c != NONE && (c as usize) >= blo && (c as usize) < bhi {
                        let k = c as usize - blo;
                        if fed[k / 64] & (1 << (k % 64)) != 0 {
                            *p |= bit;
                        }
                    }
                }
            });
        Self {
            window: w,
            has_pred,
            has_succ,
        }
    }

    /// Endpoints from one parallel scan of `g` (`reversed = false`: the forward graph;
    /// `reversed = true`: the transpose, whose lists are predecessors).
    pub fn from_graph<G: RandomAccessGraph + Sync>(
        g: &ChunkedGraph<G>,
        w: Window,
        reversed: bool,
        pl: &mut impl ConcurrentProgressLog,
    ) -> Result<Self> {
        let words = w.len().div_ceil(64);
        let mut has_pred = vec![0u64; words];
        let mut has_succ = vec![0u64; words];
        {
            let (tails, heads) = if reversed {
                (as_atomic_u64(&mut has_pred), as_atomic_u64(&mut has_succ))
            } else {
                (as_atomic_u64(&mut has_succ), as_atomic_u64(&mut has_pred))
            };
            let lo = w.lo;
            let set = |bits: &[std::sync::atomic::AtomicU64], v: usize| {
                let i = v - lo;
                bits[i / 64].fetch_or(1 << (i % 64), std::sync::atomic::Ordering::Relaxed);
            };
            pl.item_name("node")
                .display_memory(true)
                .expected_updates(Some(w.len()));
            pl.start(format!("window endpoints over {w}"));
            // lists are already restricted to `< hi` by `par_chunks`
            par_chunks_dyn(g, w, pl, &|it: &mut dyn NodeSuccIter| {
                while let Some((x, list)) = it.next_node() {
                    let mut any = false;
                    for &y in list.iter().filter(|&&y| y as usize >= lo) {
                        any = true;
                        set(heads, y as usize);
                    }
                    if any {
                        set(tails, x);
                    }
                }
                Ok(())
            })?;
            pl.done();
        }
        Ok(Self {
            window: w,
            has_pred,
            has_succ,
        })
    }

    /// The window.
    pub fn window(&self) -> Window {
        self.window
    }

    #[inline(always)]
    fn bit(bits: &[u64], i: usize) -> bool {
        bits[i / 64] & (1 << (i % 64)) != 0
    }

    /// Whether `v` (absolute id, in the window) is a window source.
    #[inline]
    pub fn is_source(&self, v: usize) -> bool {
        !Self::bit(&self.has_pred, v - self.window.lo)
    }

    /// Whether `v` (absolute id, in the window) is a window sink.
    #[inline]
    pub fn is_sink(&self, v: usize) -> bool {
        !Self::bit(&self.has_succ, v - self.window.lo)
    }

    /// Whether `v` is the initial node of the DP `dir`: a source (forward) or a sink
    /// (backward).
    #[inline]
    fn is_start(&self, dir: Dir, v: usize) -> bool {
        match dir {
            Dir::Forward => self.is_source(v),
            Dir::Backward => self.is_sink(v),
        }
    }

    /// Whether `v` is the final node of the DP `dir`, where its total is collected: a sink
    /// (forward) or a source (backward).
    #[inline]
    pub fn is_end(&self, dir: Dir, v: usize) -> bool {
        match dir {
            Dir::Forward => self.is_sink(v),
            Dir::Backward => self.is_source(v),
        }
    }

    fn count(&self, bits: &[u64]) -> u64 {
        let ones: u64 = bits.par_iter().map(|x| x.count_ones() as u64).sum();
        self.window.len() as u64 - ones
    }

    /// Number of window sources.
    pub fn sources(&self) -> u64 {
        self.count(&self.has_pred)
    }

    /// Number of window sinks.
    pub fn sinks(&self) -> u64 {
        self.count(&self.has_succ)
    }
}

/// [`crate::sweep::par_chunks`] with a unit result, through a trait object (keeps the closure
/// type out of the generic signature).
fn par_chunks_dyn<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    pl: &mut impl ConcurrentProgressLog,
    map: &(dyn Fn(&mut dyn NodeSuccIter) -> Result<()> + Sync),
) -> Result<()> {
    crate::sweep::par_chunks(g, w, pl, map, |(), ()| ())
}

// ---------------------------------------------------------------------------------------------
// Deterministic reductions
// ---------------------------------------------------------------------------------------------

/// Pairwise reduction `op(leaf(lo..mid), leaf(mid..hi))` with fixed split points: the result
/// does not depend on the thread count, and a floating-point sum has error `⌈log2 n⌉·u`
/// ([\[Hig93\]](crate::references#hig93)).
pub fn pairwise<T, L, O>(lo: usize, hi: usize, zero: T, leaf: &L, op: &O) -> T
where
    T: Copy + Send + Sync,
    L: Fn(usize) -> T + Sync,
    O: Fn(T, T) -> T + Sync,
{
    match hi.saturating_sub(lo) {
        0 => zero,
        1 => leaf(lo),
        len => {
            let mid = lo + len / 2;
            let (a, b) = if len >= PAR_MIN {
                rayon::join(
                    || pairwise(lo, mid, zero, leaf, op),
                    || pairwise(mid, hi, zero, leaf, op),
                )
            } else {
                (
                    pairwise(lo, mid, zero, leaf, op),
                    pairwise(mid, hi, zero, leaf, op),
                )
            };
            op(a, b)
        }
    }
}

/// `|a − b| / max(a, b)` of two non-negative extended floats, computed without overflow
/// (0 when both are zero, 1 when exactly one is).
pub fn ext_rel_diff(a: Ext, b: Ext) -> f64 {
    if a.is_zero() && b.is_zero() {
        return 0.0;
    }
    if a.is_zero() || b.is_zero() {
        return 1.0;
    }
    let (hi, lo) = if (a.e, a.m) >= (b.e, b.m) {
        (a, b)
    } else {
        (b, a)
    };
    let d = hi.e - lo.e;
    if d > 64 {
        return 1.0;
    }
    (hi.m - lo.m * (-(d as f64)).exp2()) / hi.m
}

/// The histogram bin of a `log2` count: 0 for a count of 1 (`x < 1`), else `⌊log2 x⌋ + 1`
/// (counts in `[2^(2^(k−1)), 2^(2^k))` share bin `k`), capped at [`HIST_BINS`]` − 1`.
#[inline]
pub fn hist_bin(x: f64) -> usize {
    if x < 1.0 {
        0
    } else {
        (x.log2().floor() as usize + 1).min(HIST_BINS - 1)
    }
}

// ---------------------------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------------------------

/// Statistics of one direction.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct DirSummary {
    /// The direction.
    pub dir: Option<Dir>,
    /// The kernel actually used.
    pub kernel: Option<Kernel>,
    /// Nodes where the total is collected: window sinks (forward) or window sources (backward).
    pub endpoints: u64,
    /// Nodes where the DP starts with 1: window sources (forward) or sinks (backward).
    pub starts: u64,
    /// `log2` of the total (ExtFloat, pairwise sum over the endpoints).
    pub total_log2: Option<f64>,
    /// The total modulo 2⁶¹ − 1.
    pub total_mod61: Option<u64>,
    /// The saturating u64 total (`u64::MAX` = saturated).
    pub total_sat64: Option<u64>,
    /// The saturating u128 total in decimal (`u128::MAX` = saturated).
    pub total_sat128: Option<String>,
    /// Largest `log2` count (ExtFloat).
    pub max_log2: Option<f64>,
    /// Smallest node (absolute id) with the largest count.
    pub argmax: Option<usize>,
    /// Mean of the `log2` counts over the window (pairwise sum).
    pub mean_log2: Option<f64>,
    /// Nodes whose count is at least 2⁶⁴ according to ExtFloat (`log2 ≥ 64`).
    pub above_2_64: Option<u64>,
    /// Nodes whose count is at least 2¹²⁸ according to ExtFloat (`log2 ≥ 128`).
    pub above_2_128: Option<u64>,
    /// Histogram of the ExtFloat `log2` counts by [`hist_bin`] (bin 0: count 1; bin k ≥ 1:
    /// `log2` in `[2^(k−1), 2^k)`); trailing empty bins are dropped.
    pub log2_hist: Vec<u64>,
    /// Saturated SatU64 counts.
    pub saturated_sat64: Option<u64>,
    /// Saturated SatU128 counts.
    pub saturated_sat128: Option<u64>,
    /// Wall time of the direction (all passes), in seconds.
    pub elapsed_s: f64,
}

/// The totals check `Σ_{sinks} F = Σ_{sources} B` (only when both directions are computed).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct Duality {
    /// Equality of the M61 totals (exact).
    pub mod61_equal: Option<bool>,
    /// Equality of the SatU64 totals (exact; `None` if a total saturated).
    pub sat64_equal: Option<bool>,
    /// Equality of the SatU128 totals (exact; `None` if a total saturated).
    pub sat128_equal: Option<bool>,
    /// Relative difference of the ExtFloat totals.
    pub extfloat_rel_diff: Option<f64>,
    /// The ExtFloat tolerance ([`EXT_DUALITY_TOL`]).
    pub extfloat_tol: f64,
    /// Whether every available check passed.
    pub ok: bool,
}

/// The totals of one direction that enter the [`Duality`] check (`None`: kind not computed).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DirTotals {
    /// The ExtFloat total.
    pub ext: Option<Ext>,
    /// The M61 total.
    pub mod61: Option<u64>,
    /// The SatU64 total (`u64::MAX` = saturated).
    pub sat64: Option<u64>,
    /// The SatU128 total (`u128::MAX` = saturated).
    pub sat128: Option<u128>,
}

impl Duality {
    /// Compares the forward totals `f` (`Σ_{sinks} F`) with the backward totals `b`
    /// (`Σ_{sources} B`): exact equality in M61, exact equality in each saturating kind unless
    /// one of the two totals saturated (then `None`, not checked), and [`ext_rel_diff`] ≤
    /// [`EXT_DUALITY_TOL`] in ExtFloat. A kind is checked only when both directions computed
    /// it; [`ok`](Self::ok) holds when no check failed. The identity is the semiring duality
    /// `⊕_v F(v)⊗c(v) = ⊕_u b(u)⊗B(u)` ([\[GM08\]](crate::references#gm08)).
    ///
    /// ```
    /// use dag::paths::{DirTotals, Duality};
    /// let f = DirTotals { mod61: Some(5), sat64: Some(5), ..Default::default() };
    /// assert!(Duality::check(&f, &f).ok);
    /// let b = DirTotals { mod61: Some(6), ..f };
    /// assert_eq!(Duality::check(&f, &b).mod61_equal, Some(false));
    /// assert!(!Duality::check(&f, &b).ok);
    /// ```
    pub fn check(f: &DirTotals, b: &DirTotals) -> Self {
        let mod61_equal = f.mod61.zip(b.mod61).map(|(a, b)| a == b);
        let sat64_equal = f
            .sat64
            .zip(b.sat64)
            .filter(|&(a, b)| a != u64::MAX && b != u64::MAX)
            .map(|(a, b)| a == b);
        let sat128_equal = f
            .sat128
            .zip(b.sat128)
            .filter(|&(a, b)| a != u128::MAX && b != u128::MAX)
            .map(|(a, b)| a == b);
        let extfloat_rel_diff = f.ext.zip(b.ext).map(|(a, b)| ext_rel_diff(a, b));
        let ok = [
            mod61_equal,
            sat64_equal,
            sat128_equal,
            extfloat_rel_diff.map(|r| r <= EXT_DUALITY_TOL),
        ]
        .into_iter()
        .all(|c| c != Some(false));
        Self {
            mod61_equal,
            sat64_equal,
            sat128_equal,
            extfloat_rel_diff,
            extfloat_tol: EXT_DUALITY_TOL,
            ok,
        }
    }
}

/// Statistics of the "through" counts `F(v)·B(v)` (both directions with ExtFloat).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct ThroughSummary {
    /// Largest `log2 (F·B)`.
    pub max_log2: f64,
    /// Smallest node (absolute id) attaining it.
    pub argmax: usize,
    /// Mean of `log2 (F·B)` over the window.
    pub mean_log2: f64,
    /// Fraction of all source-to-sink paths through the argmax node, `F·B / total`, from the
    /// ExtFloat values (1 means that every path of the window crosses it; rounding can push
    /// it slightly above 1).
    pub max_share: f64,
}

/// Everything [`count_paths`] reports besides the per-node arrays (serialized into the JSON
/// sidecar of `dag count-paths`).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct PathSummary {
    /// The window.
    pub window: Option<Window>,
    /// The kinds computed.
    pub kinds: Vec<CountKind>,
    /// Window sources.
    pub sources: u64,
    /// Window sinks.
    pub sinks: u64,
    /// Forward statistics.
    pub forward: Option<DirSummary>,
    /// Backward statistics.
    pub backward: Option<DirSummary>,
    /// The totals check.
    pub duality: Option<Duality>,
    /// Through-count statistics (non-empty window, both directions, ExtFloat).
    pub through: Option<ThroughSummary>,
    /// Whether the two directions ran concurrently.
    pub concurrent: bool,
    /// See [`PathCounts::saturated_nodes`].
    pub saturated_nodes: u64,
}

/// Per-node path counts of a window (all arrays indexed by `v − lo`) and their totals.
#[derive(Clone, Debug, PartialEq)]
pub struct PathCounts {
    /// The window.
    pub window: Window,
    /// `log2 F(v)` (ExtFloat).
    pub fwd_log2: Option<Vec<f64>>,
    /// `F(v) mod (2⁶¹ − 1)`.
    pub fwd_mod61: Option<Vec<u64>>,
    /// `log2 B(v)` (ExtFloat).
    pub bwd_log2: Option<Vec<f64>>,
    /// `B(v) mod (2⁶¹ − 1)`.
    pub bwd_mod61: Option<Vec<u64>>,
    /// Saturating `F(v)` (`u64::MAX` = saturated).
    pub fwd_sat64: Option<Vec<u64>>,
    /// Saturating `B(v)` (`u64::MAX` = saturated).
    pub bwd_sat64: Option<Vec<u64>>,
    /// Saturating `F(v)` (`u128::MAX` = saturated).
    pub fwd_sat128: Option<Vec<u128>>,
    /// Saturating `B(v)` (`u128::MAX` = saturated).
    pub bwd_sat128: Option<Vec<u128>>,
    /// `log2` of the number of source-to-sink paths of the window (forward total if computed,
    /// else backward); `NaN` without ExtFloat.
    pub total_log2: f64,
    /// The number of source-to-sink paths modulo 2⁶¹ − 1 (forward total if computed, else
    /// backward); 0 without M61.
    pub total_mod61: u64,
    /// Window nodes with at least one saturated count (any requested saturating kind, any
    /// direction).
    pub saturated_nodes: u64,
    /// Totals, checks and statistics.
    pub summary: PathSummary,
}

/// The output of one direction.
#[derive(Default)]
struct DirOut {
    log2: Option<Vec<f64>>,
    mod61: Option<Vec<u64>>,
    sat64: Option<Vec<u64>>,
    sat128: Option<Vec<u128>>,
    total_ext: Option<Ext>,
    total_sat128: Option<u128>,
    summary: DirSummary,
}

// ---------------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------------

/// Estimated peak heap bytes of one direction over a window of `n_w` nodes with `n_blocks`
/// blocks, for the resolved `kernel` and `sweep_bytes` of sweep buffers per generic pass: the
/// output arrays plus the largest DP pass ([`estimate_bytes`]), plus, for the product pass, the
/// two arrays it is split into.
pub fn direction_bytes(
    n_w: usize,
    dir: Dir,
    kernel: Kernel,
    kinds: &[CountKind],
    n_blocks: usize,
    sweep_bytes: u64,
) -> u64 {
    let k = Kinds::new(kinds);
    let n = n_w as u64;
    let dp = |e_bytes: u64| -> u64 {
        // estimate_bytes depends on the element type only through its size
        match e_bytes {
            24 => estimate_bytes::<(Ext, u64)>(n_w, dir, kernel, false, n_blocks, sweep_bytes),
            16 => estimate_bytes::<u128>(n_w, dir, kernel, false, n_blocks, sweep_bytes),
            _ => estimate_bytes::<u64>(n_w, dir, kernel, false, n_blocks, sweep_bytes),
        }
    };
    let mut outputs = 0u64;
    let mut peak = 0u64;
    match (k.ext, k.m61) {
        // the product array, then the log2 array; the residues are compacted in place into
        // the product buffer (`split_product`)
        (true, true) => {
            peak = dp(24) + 8 * n;
            outputs += 16 * n;
        }
        // the Ext array, then the log2 array
        (true, false) => {
            peak = dp(16) + 8 * n;
            outputs += 8 * n;
        }
        (false, true) => {
            peak = dp(8);
            outputs += 8 * n;
        }
        (false, false) => {}
    }
    if k.sat64 {
        peak = peak.max(outputs + dp(8));
        outputs += 8 * n;
    }
    if k.sat128 {
        peak = peak.max(outputs + dp(16));
        outputs += 16 * n;
    }
    debug_assert_eq!(outputs, output_bytes(n_w, kinds));
    peak.max(outputs)
}

/// Bytes of the per-node output arrays of one direction over `n_w` nodes: 8 B per node for
/// each of `log2`, `mod61`, `sat64`, and 16 B for `sat128`. They stay allocated after the
/// direction ends; the rest of [`direction_bytes`] is transient.
pub fn output_bytes(n_w: usize, kinds: &[CountKind]) -> u64 {
    let k = Kinds::new(kinds);
    let per = 8 * (k.ext as u64 + k.m61 as u64 + k.sat64 as u64) + 16 * k.sat128 as u64;
    per * n_w as u64
}

// ---------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------

/// Counts paths in the directions `which` with the representations `kinds` over
/// `conf.window` (unlimited memory budget). See [`count_paths_budgeted`].
pub fn count_paths<G: RandomAccessGraph + Sync>(
    input: &DpInput<G>,
    which: Which,
    kinds: &[CountKind],
    conf: &DpConf,
    pl: &mut impl ProgressLog,
) -> Result<PathCounts> {
    count_paths_budgeted(input, which, kinds, conf, &mut MemBudget::unlimited(), pl)
}

/// Counts paths in the directions `which` with the representations `kinds` over
/// `conf.window`, reserving memory in `budget`.
///
/// `conf.kernel` selects the DP kernel (`Push` runs the backward direction with `Pull`, as
/// there is no backward push kernel); `conf.pred` must be false. When both directions are
/// requested they run concurrently if `budget` admits both peak estimates
/// ([`direction_bytes`]), else one after the other. After each direction its transient memory
/// is returned to `budget` and only its outputs ([`output_bytes`]) stay reserved. Fails
/// if `kinds` is empty, if the inputs needed by the kernels are missing, if the graph is not
/// in topological order, or with [`DagError::OverBudget`](crate::DagError::OverBudget).
///
/// The duality check is reported in [`PathSummary::duality`], not raised as an error.
pub fn count_paths_budgeted<G: RandomAccessGraph + Sync>(
    input: &DpInput<G>,
    which: Which,
    kinds: &[CountKind],
    conf: &DpConf,
    budget: &mut MemBudget,
    pl: &mut impl ProgressLog,
) -> Result<PathCounts> {
    let k = Kinds::new(kinds);
    ensure!(k != Kinds::default(), "no path-count kind requested");
    ensure!(!conf.pred, "path counting does not track predecessors");
    let w = conf.window;
    ensure!(
        w.lo <= w.hi && w.hi <= input.n,
        "invalid window {w} for {} nodes",
        input.n
    );
    let (fwd, bwd) = match which {
        Which::Forward => (true, false),
        Which::Backward => (false, true),
        Which::Both => (true, true),
    };
    let fconf = conf.clone();
    let bconf = DpConf {
        kernel: backward_kernel(conf.kernel),
        ..conf.clone()
    };
    // resolve (and validate) the kernels before any work
    let kf = fwd
        .then(|| resolve_kernel(input, Dir::Forward, true, fconf.kernel))
        .transpose()?;
    let kb = bwd
        .then(|| resolve_kernel(input, Dir::Backward, true, bconf.kernel))
        .transpose()?;
    let n_blocks = input
        .blocks
        .map_or(0, |b| b.block_bound(w.hi) - b.block_bound(w.lo));
    let sweep_bytes = |gr: Option<&ChunkedGraph<G>>| {
        gr.map_or(0, |g| {
            sweep_buffer_bytes(&conf.sweep, g.conf.chunk_arcs, g.conf.max_chunk_nodes)
        })
    };
    let fb = kf.map_or(0, |k| {
        direction_bytes(
            w.len(),
            Dir::Forward,
            k,
            kinds,
            n_blocks,
            sweep_bytes(input.gt.or(input.g)),
        )
    });
    let bb = kb.map_or(0, |k| {
        direction_bytes(
            w.len(),
            Dir::Backward,
            k,
            kinds,
            n_blocks,
            sweep_bytes(input.g),
        )
    });

    budget.need(Endpoints::bytes(w.len()), "path-count endpoints")?;
    let ends = match input.blocks {
        // decode-free: no logger needed
        Some(b) => Endpoints::from_blocks(b, w),
        None => Endpoints::compute(input, w, &mut pl.concurrent())?,
    };
    let mut summary = PathSummary {
        window: Some(w),
        kinds: k.list(),
        sources: ends.sources(),
        sinks: ends.sinks(),
        ..Default::default()
    };
    log::info!(
        "count-paths over {w}: {} window sources, {} window sinks, kinds {:?}",
        summary.sources,
        summary.sinks,
        summary.kinds
    );

    let out_bytes = output_bytes(w.len(), kinds);
    let concurrent = fwd
        && bwd
        && budget
            .need(fb + bb, "path counts (both directions)")
            .is_ok();
    let (fo, bo) = if concurrent {
        log::info!("count-paths: forward ∥ backward");
        let mut pf = pl.concurrent();
        let mut pb = pl.concurrent();
        pf.log_target("dag::paths::fwd");
        pb.log_target("dag::paths::bwd");
        let (f, b) = rayon::join(
            || count_dir(input, Dir::Forward, k, &fconf, &ends, &mut pf),
            || count_dir(input, Dir::Backward, k, &bconf, &ends, &mut pb),
        );
        budget.release(fb + bb - 2 * out_bytes);
        (Some(f?), Some(b?))
    } else {
        if fwd && bwd {
            log::info!("count-paths: the two directions do not fit together, running them in turn");
        }
        let mut run = |on: bool, dir: Dir, bytes: u64, c: &DpConf| -> Result<Option<DirOut>> {
            if !on {
                return Ok(None);
            }
            budget.need(bytes, &format!("{dir:?} path counts"))?;
            let r = count_dir(input, dir, k, c, &ends, pl)?;
            budget.release(bytes - out_bytes);
            Ok(Some(r))
        };
        let f = run(fwd, Dir::Forward, fb, &fconf)?;
        let b = run(bwd, Dir::Backward, bb, &bconf)?;
        (f, b)
    };
    summary.concurrent = concurrent;
    Ok(assemble(w, k, summary, fo, bo))
}

/// Combines the two directions: totals, duality, through statistics, saturated nodes.
fn assemble(
    w: Window,
    k: Kinds,
    mut summary: PathSummary,
    fo: Option<DirOut>,
    bo: Option<DirOut>,
) -> PathCounts {
    let first = fo.as_ref().or(bo.as_ref());
    let total_log2 = first.and_then(|d| d.summary.total_log2).unwrap_or(f64::NAN);
    let total_mod61 = first.and_then(|d| d.summary.total_mod61).unwrap_or(0);
    let total_ext = first.and_then(|d| d.total_ext);

    if let (Some(f), Some(b)) = (&fo, &bo) {
        let totals = |d: &DirOut| DirTotals {
            ext: d.total_ext,
            mod61: d.summary.total_mod61,
            sat64: d.summary.total_sat64,
            sat128: d.total_sat128,
        };
        let d = Duality::check(&totals(f), &totals(b));
        if d.ok {
            log::info!("count-paths: duality ok ({d:?})");
        } else {
            log::warn!("count-paths: duality FAILED ({d:?})");
        }
        summary.duality = Some(d);
        if let (Some(fl), Some(bl), Some(t), true) = (&f.log2, &b.log2, total_ext, !w.is_empty()) {
            summary.through = Some(through_summary(w, fl, bl, t));
        }
    }

    let n_w = w.len();
    let sat_at = |i: usize| -> bool {
        let s64 = |d: &Option<DirOut>| {
            d.as_ref()
                .and_then(|d| d.sat64.as_ref())
                .is_some_and(|v| v[i] == u64::MAX)
        };
        let s128 = |d: &Option<DirOut>| {
            d.as_ref()
                .and_then(|d| d.sat128.as_ref())
                .is_some_and(|v| v[i] == u128::MAX)
        };
        s64(&fo) || s64(&bo) || s128(&fo) || s128(&bo)
    };
    let saturated_nodes = if k.sat64 || k.sat128 {
        (0..n_w)
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .filter(|&i| sat_at(i))
            .count() as u64
    } else {
        0
    };
    summary.saturated_nodes = saturated_nodes;
    let (mut fo, mut bo) = (fo.unwrap_or_default(), bo.unwrap_or_default());
    summary.forward = fo
        .summary
        .dir
        .is_some()
        .then(|| std::mem::take(&mut fo.summary));
    summary.backward = bo
        .summary
        .dir
        .is_some()
        .then(|| std::mem::take(&mut bo.summary));
    PathCounts {
        window: w,
        fwd_log2: fo.log2,
        fwd_mod61: fo.mod61,
        bwd_log2: bo.log2,
        bwd_mod61: bo.mod61,
        fwd_sat64: fo.sat64,
        bwd_sat64: bo.sat64,
        fwd_sat128: fo.sat128,
        bwd_sat128: bo.sat128,
        total_log2,
        total_mod61,
        saturated_nodes,
        summary,
    }
}

fn through_summary(w: Window, f: &[f64], b: &[f64], total: Ext) -> ThroughSummary {
    let t = |i: usize| f[i] + b[i];
    let (max_log2, i) = argmax(f.len(), &t);
    let mean = pairwise(0, f.len(), 0.0, &t, &|a, b| a + b) / f.len().max(1) as f64;
    ThroughSummary {
        max_log2,
        argmax: i + w.lo,
        mean_log2: mean,
        max_share: (max_log2 - total.log2()).exp2(),
    }
}

/// `(max, smallest argmax)` of `x(0..n)` (deterministic); `(−∞, 0)` for `n = 0`.
fn argmax(n: usize, x: &(impl Fn(usize) -> f64 + Sync)) -> (f64, usize) {
    pairwise(0, n, (f64::NEG_INFINITY, 0), &|i| (x(i), i), &|a: (
        f64,
        usize,
    ),
                                                             b: (
        f64,
        usize,
    )| {
        if b.0 > a.0 || (b.0 == a.0 && b.1 < a.1) {
            b
        } else {
            a
        }
    })
}

/// One DP pass `dir` of semiring `s` with unit weights and `Init::Sources(1̄)`.
fn pass<S: Semiring, G: RandomAccessGraph + Sync>(
    s: &S,
    input: &DpInput<G>,
    dir: Dir,
    conf: &DpConf,
    pl: &mut impl ProgressLog,
) -> Result<Vec<S::Elem>> {
    Ok(solve(s, input, dir, &One, &Init::Sources(s.one()), conf, pl)?.val)
}

/// Splits the product array into the `log2` magnitudes (a new 8 B/node array) and the M61
/// residues, which are moved into the front of the product buffer itself: `into_iter().map()
/// .collect()` from 24 B to 8 B elements of the same alignment reuses the allocation (std's
/// in-place collection), and `shrink_to_fit` then returns the tail (`realloc`, which shrinks
/// large mappings in place). The peak is thus 32 B/node instead of 40 B/node.
fn split_product(v: Vec<(Ext, u64)>) -> (Vec<f64>, Vec<u64>) {
    let mut l = huge_vec(v.len(), 0.0f64);
    l.par_iter_mut()
        .zip(v.par_iter())
        .with_min_len(PAR_MIN)
        .for_each(|(l, x)| *l = x.0.log2());
    let mut m: Vec<u64> = v.into_iter().map(|x| x.1).collect();
    m.shrink_to_fit();
    (l, m)
}

/// All passes of one direction.
fn count_dir<G: RandomAccessGraph + Sync>(
    input: &DpInput<G>,
    dir: Dir,
    k: Kinds,
    conf: &DpConf,
    ends: &Endpoints,
    pl: &mut impl ProgressLog,
) -> Result<DirOut> {
    let t0 = Instant::now();
    let w = conf.window;
    let (lo, n_w) = (w.lo, w.len());
    let is_end = |i: usize| ends.is_end(dir, i + lo);
    let mut out = DirOut::default();
    let s = &mut out.summary;
    s.dir = Some(dir);
    s.kernel = Some(resolve_kernel(input, dir, true, conf.kernel)?);
    s.endpoints = (0..n_w)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .filter(|&i| is_end(i))
        .count() as u64;
    s.starts = (0..n_w)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .filter(|&i| ends.is_start(dir, i + lo))
        .count() as u64;

    let ext_total = |x: &(dyn Fn(usize) -> Ext + Sync)| {
        let ef = ExtFloat::new();
        pairwise(
            0,
            n_w,
            Ext::ZERO,
            &|i| if is_end(i) { x(i) } else { Ext::ZERO },
            &|a, b| ef.plus(a, b),
        )
    };
    let m61_total = |x: &(dyn Fn(usize) -> u64 + Sync)| {
        let m = M61::new();
        (0..n_w)
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .filter(|&i| is_end(i))
            .map(x)
            .reduce(|| 0, |a, b| m.plus(a, b))
    };
    let log2_of = |v: &[Ext]| {
        let mut l = huge_vec(v.len(), 0.0f64);
        l.par_iter_mut()
            .zip(v.par_iter())
            .with_min_len(PAR_MIN)
            .for_each(|(l, x)| *l = x.log2());
        l
    };

    match (k.ext, k.m61) {
        (true, true) => {
            let v = pass(&Product(ExtFloat::new(), M61::new()), input, dir, conf, pl)?;
            out.total_ext = Some(ext_total(&|i| v[i].0));
            out.summary.total_mod61 = Some(m61_total(&|i| v[i].1));
            let (l, m) = split_product(v);
            out.log2 = Some(l);
            out.mod61 = Some(m);
        }
        (true, false) => {
            let v = pass(&ExtFloat::new(), input, dir, conf, pl)?;
            out.total_ext = Some(ext_total(&|i| v[i]));
            out.log2 = Some(log2_of(&v));
        }
        (false, true) => {
            let v = pass(&M61::new(), input, dir, conf, pl)?;
            out.summary.total_mod61 = Some(m61_total(&|i| v[i]));
            out.mod61 = Some(v);
        }
        (false, false) => {}
    }
    if k.sat64 {
        let s = SatU64::new();
        let v = pass(&s, input, dir, conf, pl)?;
        out.summary.total_sat64 = Some(
            (0..n_w)
                .into_par_iter()
                .with_min_len(PAR_MIN)
                .filter(|&i| is_end(i))
                .map(|i| v[i])
                .reduce(|| 0, |a, b| s.plus(a, b)),
        );
        out.summary.saturated_sat64 =
            Some(v.par_iter().filter(|&&x| SatU64::is_saturated(x)).count() as u64);
        out.sat64 = Some(v);
    }
    if k.sat128 {
        let s = SatU128::new();
        let v = pass(&s, input, dir, conf, pl)?;
        let t = (0..n_w)
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .filter(|&i| is_end(i))
            .map(|i| v[i])
            .reduce(|| 0, |a, b| s.plus(a, b));
        out.total_sat128 = Some(t);
        out.summary.total_sat128 = Some(t.to_string());
        out.summary.saturated_sat128 =
            Some(v.par_iter().filter(|&&x| SatU128::is_saturated(x)).count() as u64);
        out.sat128 = Some(v);
    }

    if let Some(l) = &out.log2 {
        let s = &mut out.summary;
        s.total_log2 = out.total_ext.map(|t| t.log2());
        if n_w > 0 {
            let (mx, i) = argmax(n_w, &|i| l[i]);
            s.max_log2 = Some(mx);
            s.argmax = Some(i + lo);
            s.mean_log2 = Some(pairwise(0, n_w, 0.0, &|i| l[i], &|a, b| a + b) / n_w as f64);
        }
        s.above_2_64 = Some(l.par_iter().filter(|&&x| x >= 64.0).count() as u64);
        s.above_2_128 = Some(l.par_iter().filter(|&&x| x >= 128.0).count() as u64);
        let mut h = l
            .par_chunks(PAR_MIN)
            .map(|c| {
                let mut h = [0u64; HIST_BINS];
                for &x in c {
                    h[hist_bin(x)] += 1;
                }
                h
            })
            .reduce(
                || [0u64; HIST_BINS],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                    a
                },
            )
            .to_vec();
        while h.last() == Some(&0) {
            h.pop();
        }
        s.log2_hist = h;
    }
    out.summary.elapsed_s = t0.elapsed().as_secs_f64();
    log::info!(
        "count-paths {dir:?}: total log2 {:?}, mod61 {:?}, sat64 {:?}, max log2 {:?} at {:?} ({:.1} s)",
        out.summary.total_log2,
        out.summary.total_mod61,
        out.summary.total_sat64,
        out.summary.max_log2,
        out.summary.argmax,
        out.summary.elapsed_s
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The residues must be compacted into the product buffer (no third allocation), else the
    /// 32 B/node peak of [`direction_bytes`] is wrong.
    #[test]
    fn split_product_reuses_the_product_buffer() {
        let n = 100_000;
        let v: Vec<(Ext, u64)> = (0..n)
            .map(|i| (Ext::from_u64(i as u64 + 1), i as u64))
            .collect();
        let p = v.as_ptr() as usize;
        let (l, m) = split_product(v);
        assert_eq!(m.as_ptr() as usize, p, "in-place collection did not happen");
        assert_eq!(m.capacity(), n);
        assert!(m.iter().enumerate().all(|(i, &x)| x == i as u64));
        assert!(
            l.iter()
                .enumerate()
                .all(|(i, &x)| x == Ext::from_u64(i as u64 + 1).log2())
        );
    }
}
