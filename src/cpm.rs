//! Critical path method: earliest times, heights, slack, critical nodes and arcs (F1 §8).
//!
//! # Algorithm
//! Nodes are events and arcs activities of length `w(u, v) ≥ 0` (activity-on-node durations
//! map to separable weights `α(u) = dur(u)`, `β = 0`). The classical CPM/PERT quantities
//! [\[KW59\]](crate::references#kw59), [\[MRCF59\]](crate::references#mrcf59) are two
//! max-plus DPs, run concurrently (`rayon::join`) with [`solve`]:
//! ```text
//! E = forward  MaxPlus, Init::Sources(0)      earliest time = longest path from a source
//! H = backward MaxPlus, Init::Sources(0)      height = longest path to a sink
//! T = max E (checked: = max H)                project length
//! slack(v) = T − E(v) − H(v) ≥ 0              latest time L(v) = T − H(v)
//! critical node: slack = 0                    (lies on some longest path)
//! critical arc:  slack(u) = slack(v) = 0 ∧ E(u) ⊗ w(u, v) = E(v)
//! ```
//! Using `H` instead of a min-plus pass for `L` gives the same result with integers only. The
//! layer profile groups the critical nodes by `E`; with unit weights every longest path has
//! exactly one node per E-layer `0..=T`, so a layer of width 1 is a node on **every** longest
//! path ([`CpmSummary::width1_layers`], reported only for unit weights).
//!
//! # Complexity
//! Two DP passes (each `O(n_W + m_W)`, or `O(n_W)` with the block kernels), a parallel scan for
//! the summary (filling a histogram of the critical nodes' `E` values, `4·(T + 1)` bytes) and
//! one optional parallel arc scan for the critical arcs. Memory: `2·|T|·n_W` (+ `4·n_W` with predecessors, + `|T|·n_W` for
//! [`CpmResult::slacks`]).
//!
//! # PG fast path
//! With a [`BlockIndex`](crate::blocks::BlockIndex) and unit weights both passes are block
//! kernels and no BV decoding happens (`Kernel::Auto`).

use std::ops::Sub;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, ensure};
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::traits::RandomAccessGraph;

use crate::dp::path::path_to;
use crate::dp::solve::solve;
use crate::dp::{ArcWeights, DpConf, DpInput, Init, Kernel};
use crate::graph::ChunkedGraph;
use crate::io::Dump;
use crate::semiring::{MaxPlus, Semiring};
use crate::summary::{LOG2_BINS, log2_bin};
use crate::sweep::{NodeSuccIter, par_chunks};
use crate::window::{Dir, Window};

/// Integer times of the CPM (`u32`, `u64`). The max-plus sentinel `T::MAX` (−∞) never occurs
/// in CPM values (every node has a finite earliest time).
pub trait CpmInt:
    Copy + Ord + Send + Sync + Dump + Sub<Output = Self> + Into<u64> + 'static
{
}

impl CpmInt for u32 {}
impl CpmInt for u64 {}

/// Output of [`cpm`]: per-node arrays indexed by `v − window.lo`.
#[derive(Clone, Debug, PartialEq)]
pub struct CpmResult<T> {
    /// The window.
    pub window: Window,
    /// Earliest times (longest path from a window source).
    pub e: Vec<T>,
    /// Heights (longest path to a window sink).
    pub h: Vec<T>,
    /// Project length `max E = max H`.
    pub t: T,
    /// Argbest predecessors of the forward pass (absolute ids, [`NONE`](crate::NONE) at
    /// sources), when `conf.pred` was set. It feeds [`CpmSummary::longest_path`].
    pub pred: Option<Vec<u32>>,
}

/// Aggregate CPM measures ([`CpmResult::summary`]).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CpmSummary {
    /// Project length.
    pub t: u64,
    /// Nodes with slack 0.
    pub critical_nodes: u64,
    /// Critical arcs, when counted ([`CpmResult::critical_arcs`]).
    pub critical_arcs: Option<u64>,
    /// E-layers with exactly one critical node (unit weights only).
    pub width1_layers: Option<u64>,
    /// Largest number of critical nodes sharing one `E` value.
    pub max_critical_width: u64,
    /// The smallest `E` value whose layer has `max_critical_width` critical nodes.
    pub max_critical_layer: u64,
    /// The smallest and largest critical node of that layer (absolute ids; `None` for an empty
    /// window).
    pub max_critical_layer_nodes: Option<(usize, usize)>,
    /// Log2 histogram of the critical layer widths over the `T + 1` layers `E = 0..=T`
    /// ([`crate::summary`] bins: bin 0 counts the layers without a critical node, bin `k ≥ 1`
    /// the layers with `[2^(k−1), 2^k)` critical nodes; trailing empty bins omitted).
    pub critical_width_log2_hist: Vec<u64>,
    /// The smallest node with `E = T` (absolute id; the end of a longest path).
    pub longest_path_end: usize,
    /// One longest path from a source to `longest_path_end` (absolute ids), when predecessors
    /// were tracked.
    pub longest_path: Option<Vec<u32>>,
}

/// Runs the CPM over `conf.window` (see the module documentation).
///
/// The kernels are chosen as by [`solve`]: both passes resolve `conf.kernel` independently,
/// except that [`Kernel::Push`] is used for E only and H runs with [`Kernel::Pull`] (there is
/// no backward push kernel; the backward pull reads the forward graph, as push does), see
/// [`backward_kernel`]. `conf.pred` tracks the argbest predecessors of the forward pass only.
/// Fails if a pass fails or if `max E ≠ max H`.
///
/// ```
/// use dag::cpm::cpm;
/// use dag::dp::{Const, DpConf, DpInput};
/// use dag::{Window, testkit};
///
/// let g = testkit::chunked(&testkit::diamonds(2), 3);   // 0→{1,2}→3→{4,5}→6
/// let r = cpm::<u32, _, _>(&DpInput::graph(&g), &Const(1u32), &DpConf::new(Window::full(7)),
///                          &mut Option::<dsi_progress_logger::ProgressLogger>::None).unwrap();
/// assert_eq!(r.t, 4);
/// assert_eq!(r.summary(true).critical_nodes, 7);
/// ```
pub fn cpm<T, W, G>(
    input: &DpInput<G>,
    w: &W,
    conf: &DpConf,
    pl: &mut impl ProgressLog,
) -> Result<CpmResult<T>>
where
    T: CpmInt,
    MaxPlus<T>: Semiring<Elem = T>,
    W: ArcWeights<MaxPlus<T>>,
    G: RandomAccessGraph + Sync,
{
    let s = MaxPlus::<T>::new();
    let init = Init::Sources(s.one());
    let hconf = DpConf {
        pred: false,
        kernel: backward_kernel(conf.kernel),
        ..conf.clone()
    };
    let mut pe = pl.concurrent();
    let mut ph = pl.concurrent();
    pe.log_target("dag::cpm::E");
    ph.log_target("dag::cpm::H");
    log::info!("cpm: E ∥ H over {}", conf.window);
    let (e, h) = rayon::join(
        || solve(&s, input, Dir::Forward, w, &init, conf, &mut pe),
        || solve(&s, input, Dir::Backward, w, &init, &hconf, &mut ph),
    );
    let (e, h) = (e?, h?);
    let max = |v: &[T]| v.par_iter().copied().filter(|&x| x != s.zero()).max();
    let (te, th) = (max(&e.val), max(&h.val));
    ensure!(
        te == th,
        "max E = {:?} differs from max H = {:?}",
        te.map(Into::<u64>::into),
        th.map(Into::<u64>::into)
    );
    let t = te.unwrap_or(s.one());
    if t.into() == s.zero().into() - 1 {
        log::warn!("cpm: T = {} is saturated (use 64-bit times)", t.into());
    }
    Ok(CpmResult {
        window: conf.window,
        e: e.val,
        h: h.val,
        t,
        pred: e.pred,
    })
}

/// The kernel of the H (backward) pass of [`cpm`] for the requested `kernel`: [`Kernel::Push`]
/// becomes [`Kernel::Pull`], every other kernel is kept.
///
/// ```
/// use dag::cpm::backward_kernel;
/// use dag::dp::Kernel;
///
/// assert_eq!(backward_kernel(Kernel::Push), Kernel::Pull);
/// assert_eq!(backward_kernel(Kernel::Blocks), Kernel::Blocks);
/// ```
pub fn backward_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Push => Kernel::Pull,
        k => k,
    }
}

impl<T: CpmInt> CpmResult<T> {
    #[inline]
    fn slack_at(&self, i: usize) -> Option<T> {
        let r = self.t - self.e[i];
        (self.h[i] <= r).then(|| r - self.h[i])
    }

    /// `slack(v) = T − E(v) − H(v)` of the node `v` (absolute id).
    ///
    /// # Panics
    /// If `v` is not in the window or `E(v) + H(v) > T` (inconsistent arrays).
    pub fn slack(&self, v: usize) -> T {
        assert!(self.window.contains(v), "{v} is not in {}", self.window);
        let i = v - self.window.lo;
        self.slack_at(i)
            .unwrap_or_else(|| panic!("E({v}) + H({v}) exceeds T"))
    }

    /// The slack of every window node, indexed by `v − lo` (`|T|·n_W` bytes, allocated with
    /// [`io::huge_vec`](crate::io::huge_vec) and filled in parallel).
    ///
    /// # Panics
    /// If some `E(v) + H(v) > T`.
    pub fn slacks(&self) -> Vec<T> {
        let mut sl = crate::io::huge_vec(self.e.len(), self.t);
        sl.par_iter_mut()
            .with_min_len(1 << 16)
            .enumerate()
            .for_each(|(i, x)| {
                *x = self
                    .slack_at(i)
                    .unwrap_or_else(|| panic!("E + H exceeds T at node {}", i + self.window.lo));
            });
        sl
    }

    /// Whether the window node `v` (absolute id) has slack 0.
    #[inline]
    pub fn is_critical(&self, v: usize) -> bool {
        let i = v - self.window.lo;
        self.t - self.e[i] == self.h[i]
    }

    /// Whether [`summary`](Self::summary) profiles the critical layers with a histogram over
    /// `E ∈ [0, T]` (`4·(T + 1)` bytes) rather than by sorting the critical nodes' `E` values
    /// (up to `8·n_W` bytes): when `T < 2²⁰` or the histogram is no larger than the sort.
    fn histogram_layers(&self) -> bool {
        let layers = self.t.into().saturating_add(1);
        layers < 1 << 20 || layers.saturating_mul(4) <= 8 * self.e.len() as u64
    }

    /// Bytes allocated by [`summary`](Self::summary) (the layer profile), for
    /// [`MemBudget::need`](crate::io::MemBudget::need) before calling it: `4·(T + 1)` for the
    /// histogram, `8·n_W` (an upper bound) for the sort fallback.
    pub fn summary_bytes(&self) -> u64 {
        if self.histogram_layers() {
            4 * (self.t.into() + 1)
        } else {
            8 * self.e.len() as u64
        }
    }

    /// Aggregate measures (no graph needed); `critical_arcs` is left `None`, and
    /// `width1_layers` is computed only for `unit_weights`.
    ///
    /// The layer profile (critical nodes per `E` value) is the parallel histogram of
    /// `T + 1` atomic `u32` counters filled by a parallel scan that coalesces runs
    /// of equal `E` before each `fetch_add` (on the PG, `T ≈ 3.6·10⁶`, so 14 MB). Counts fit in
    /// `u32` because a window has fewer than 2³² nodes. With general weights and `T` larger
    /// than `2·n_W` it falls back to a parallel sort of the critical `E` values; see
    /// [`summary_bytes`](Self::summary_bytes).
    pub fn summary(&self, unit_weights: bool) -> CpmSummary {
        let lo = self.window.lo;
        let critical = |i: usize| self.t - self.e[i] == self.h[i];
        let (critical_nodes, max_width, width1, layer, widths) = if self.histogram_layers() {
            let hist: Vec<AtomicU32> = (0..=self.t.into())
                .into_par_iter()
                .map(|_| AtomicU32::new(0))
                .collect();
            let flush = |run: Option<(u64, u32)>| {
                if let Some((e, c)) = run {
                    hist[e as usize].fetch_add(c, Ordering::Relaxed);
                }
            };
            let critical_nodes: u64 = (0..self.e.len())
                .into_par_iter()
                .with_min_len(1 << 16)
                .fold(
                    || (0u64, None::<(u64, u32)>),
                    |(n, run), i| {
                        if !critical(i) {
                            return (n, run);
                        }
                        let e: u64 = self.e[i].into();
                        match run {
                            Some((r, c)) if r == e => (n + 1, Some((r, c + 1))),
                            _ => {
                                flush(run);
                                (n + 1, Some((e, 1)))
                            }
                        }
                    },
                )
                .map(|(n, run)| {
                    flush(run);
                    n
                })
                .sum();
            let (max_width, width1) = hist
                .par_iter()
                .map(|c| {
                    let c = c.load(Ordering::Relaxed);
                    (u64::from(c), u64::from(c == 1))
                })
                .reduce(|| (0, 0), |a, b| (a.0.max(b.0), a.1 + b.1));
            let layer = hist
                .par_iter()
                .position_first(|c| u64::from(c.load(Ordering::Relaxed)) == max_width)
                .unwrap_or(0) as u64;
            let widths = hist
                .par_iter()
                .fold(
                    || [0u64; LOG2_BINS],
                    |mut h, c| {
                        h[log2_bin(f64::from(c.load(Ordering::Relaxed)))] += 1;
                        h
                    },
                )
                .reduce(
                    || [0u64; LOG2_BINS],
                    |mut a, b| {
                        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                        a
                    },
                );
            (critical_nodes, max_width, width1, layer, widths)
        } else {
            let mut layers: Vec<u64> = (0..self.e.len())
                .into_par_iter()
                .filter(|&i| critical(i))
                .map(|i| self.e[i].into())
                .collect();
            layers.par_sort_unstable();
            let mut max_width = 0u64;
            let mut width1 = 0u64;
            let mut layer = 0u64;
            let mut widths = [0u64; LOG2_BINS];
            let mut nonempty = 0u64;
            for run in layers.chunk_by(|a, b| a == b) {
                if run.len() as u64 > max_width {
                    (max_width, layer) = (run.len() as u64, run[0]);
                }
                width1 += u64::from(run.len() == 1);
                widths[log2_bin(run.len() as f64)] += 1;
                nonempty += 1;
            }
            widths[0] = self.t.into() + 1 - nonempty;
            (layers.len() as u64, max_width, width1, layer, widths)
        };
        let last = widths.iter().rposition(|&c| c > 0).map_or(0, |p| p + 1);
        // the node range of the widest layer (one parallel scan)
        let in_layer = |i: usize| critical(i) && self.e[i].into() == layer;
        let layer_nodes = (max_width > 0).then(|| {
            let first = (0..self.e.len())
                .into_par_iter()
                .position_first(in_layer)
                .unwrap_or(0);
            let last = (0..self.e.len())
                .into_par_iter()
                .position_last(in_layer)
                .unwrap_or(0);
            (first + lo, last + lo)
        });
        let end = self
            .e
            .par_iter()
            .position_first(|&x| x == self.t)
            .map_or(lo, |i| i + lo);
        let longest_path = self.pred.as_ref().filter(|_| !self.e.is_empty()).map(|p| {
            let mut path = path_to(p, self.window, end);
            path.reverse();
            path
        });
        CpmSummary {
            t: self.t.into(),
            critical_nodes,
            critical_arcs: None,
            width1_layers: unit_weights.then_some(width1),
            max_critical_width: max_width,
            max_critical_layer: layer,
            max_critical_layer_nodes: layer_nodes,
            critical_width_log2_hist: widths[..last].to_vec(),
            longest_path_end: end,
            longest_path,
        }
    }

    /// Counts the critical arcs with one parallel scan of the forward graph `g` (the weights
    /// must be those of the [`cpm`] run).
    pub fn critical_arcs<W, G>(
        &self,
        g: &ChunkedGraph<G>,
        w: &W,
        pl: &mut impl ConcurrentProgressLog,
    ) -> Result<u64>
    where
        MaxPlus<T>: Semiring<Elem = T>,
        W: ArcWeights<MaxPlus<T>>,
        G: RandomAccessGraph + Sync,
    {
        let s = MaxPlus::<T>::new();
        let lo = self.window.lo;
        pl.item_name("node")
            .expected_updates(Some(self.window.len()));
        pl.start(format!("counting critical arcs over {}", self.window));
        let c = par_chunks(
            g,
            self.window,
            pl,
            |it: &mut dyn NodeSuccIter| -> Result<u64> {
                let mut c = 0u64;
                while let Some((u, succ)) = it.next_node() {
                    if !self.is_critical(u) {
                        continue;
                    }
                    let eu = self.e[u - lo];
                    let eua = if W::SEPARABLE {
                        s.times(eu, w.alpha(&s, u))
                    } else {
                        eu
                    };
                    for &v in succ {
                        let v = v as usize;
                        let reach = if W::SEPARABLE {
                            s.times(eua, w.beta(&s, v))
                        } else {
                            s.times(eu, w.arc(&s, u, v))
                        };
                        c += u64::from(self.is_critical(v) && reach == self.e[v - lo]);
                    }
                }
                Ok(c)
            },
            |a, b| a + b,
        )?;
        pl.done();
        Ok(c)
    }
}
