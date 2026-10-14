//! Kernel selection and the DP entry point [`solve`].
//!
//! # Algorithm
//! A semiring DP on a DAG whose identity is a topological order is one pass in id order
//! (forward) or reverse id order (backward) [\[Kah62\]](crate::references#kah62),
//! [\[Leh77\]](crate::references#leh77), [\[Moh02\]](crate::references#moh02). [`solve`] checks
//! the inputs, resolves [`Kernel::Auto`] and dispatches to one of five kernels:
//!
//! | Direction | Kernel | Reads | Module |
//! |---|---|---|---|
//! | forward | `Push` | graph | [`push`](super::push) |
//! | forward | `Pull` | transpose | [`pull`](super::pull) |
//! | backward | `Pull` | graph (reverse sweep) | [`pull`](super::pull) |
//! | forward / backward | `Blocks` | [`BlockIndex`], separable weights | [`blocks`](super::blocks) |
//!
//! `Auto` resolution: `Blocks` if a block index is given and the weights are separable; else,
//! forward, `Pull` if the transpose is given, else `Push`; backward, `Pull`.
//!
//! Every kernel follows the canonical evaluation order of [`crate::dp`], so all of them return
//! bit-identical values (and predecessors) for every semiring with [`One`](super::One),
//! [`Const`](super::Const) and [`Node`](super::Node) weights; with [`Arc`](super::Arc) weights
//! push and pull agree.
//!
//! # Complexity
//! `O(n_W + m_W)` semiring operations for push and pull (one ⊗ per arc), `O(n_W)` ⊗ plus one ⊕
//! per non-sink for the block kernels (F1 §6.5). Memory: `|E|·n_W` for the values, `4·n_W` for
//! the predecessors, plus `n_W/8` (push), or one record per block of the window (blocks).
//!
//! # PG fast path
//! On the Payment Graph the block kernels read only `create`, `spend` and (backward) `first`,
//! `len`: no BV decoding at all.

use std::ops::Range;

use anyhow::{Result, bail, ensure};
use dsi_progress_logger::prelude::*;
use webgraph::traits::RandomAccessGraph;

use super::blocks::{backward_blocks, forward_blocks};
use super::pull::{backward_pull, forward_pull};
use super::push::forward_push;
use super::{ArcWeights, DpConf, DpInput, DpResult, Init, Kernel};
use crate::blocks::BlockIndex;
use crate::error::DagError;
use crate::io::NONE;
use crate::semiring::Semiring;
use crate::sweep::SweepConf;
use crate::window::{Dir, Window};

/// Nodes between two progress updates of the sequential block kernels.
pub(crate) const LOG_STEP: usize = 1 << 22;

/// Solves the DP `dir` over `conf.window` with weights `w` and initial terms `init`.
///
/// See the module documentation for the kernel choice and [`crate::dp`] for the canonical
/// evaluation order. Fails if the inputs required by the (resolved) kernel are missing, if the
/// window does not fit the graph, if `Init::Dense` does not have one term per window node, if
/// predecessors are requested for a non-selective semiring, or if the graph is not in
/// topological order ([`DagError::NotTopological`]).
///
/// ```
/// use dag::dp::{Const, DpConf, DpInput, Init, solve::solve};
/// use dag::semiring::MaxPlus;
/// use dag::{Dir, Window, testkit};
///
/// let g = testkit::chunked(&testkit::chain(5), 2);
/// let s = MaxPlus::<u32>::new();
/// let conf = DpConf::new(Window::full(5));
/// let r = solve(&s, &DpInput::graph(&g), Dir::Forward, &Const(1u32), &Init::Sources(0),
///               &conf, &mut Option::<dsi_progress_logger::ProgressLogger>::None).unwrap();
/// assert_eq!(r.val, [0, 1, 2, 3, 4]);
/// ```
pub fn solve<S, W, G>(
    s: &S,
    input: &DpInput<G>,
    dir: Dir,
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
    check_window(conf.window, input.n)?;
    let kernel = resolve_kernel(input, dir, W::SEPARABLE, conf.kernel)?;
    log::info!(
        "dp: {} {dir:?} over {} with the {kernel:?} kernel",
        s.name(),
        conf.window
    );
    // resolve_kernel guarantees that the inputs used below are present
    match (dir, kernel) {
        (Dir::Forward, Kernel::Push) => forward_push(s, need(input.g, "graph")?, w, init, conf, pl),
        (Dir::Forward, Kernel::Pull) => {
            forward_pull(s, need(input.gt, "transpose")?, w, init, conf, pl)
        }
        (Dir::Forward, _) => forward_blocks(s, need(input.blocks, "blocks")?, w, init, conf, pl),
        (Dir::Backward, Kernel::Pull) => {
            backward_pull(s, need(input.g, "graph")?, w, init, conf, pl)
        }
        (Dir::Backward, _) => backward_blocks(s, need(input.blocks, "blocks")?, w, init, conf, pl),
    }
}

fn need<T>(x: Option<T>, what: &str) -> Result<T> {
    x.ok_or_else(|| anyhow::anyhow!("internal error: resolved kernel needs the {what}"))
}

/// Resolves `requested` for `input` and `dir` (see the module documentation) and checks that
/// the inputs it needs are present. `separable` is [`ArcWeights::SEPARABLE`].
///
/// Never returns [`Kernel::Auto`]; backward runs never return [`Kernel::Push`] (there is no
/// backward push kernel: use `Pull` or `Blocks`).
pub fn resolve_kernel<G>(
    input: &DpInput<G>,
    dir: Dir,
    separable: bool,
    requested: Kernel,
) -> Result<Kernel> {
    let blocks_ok = input.blocks.is_some() && separable;
    let k = match requested {
        Kernel::Auto if blocks_ok => Kernel::Blocks,
        Kernel::Auto => match dir {
            Dir::Forward if input.gt.is_some() => Kernel::Pull,
            Dir::Forward => Kernel::Push,
            Dir::Backward => Kernel::Pull,
        },
        k => k,
    };
    match (dir, k) {
        (Dir::Forward, Kernel::Push) => {
            ensure!(input.g.is_some(), "the push kernel needs the graph")
        }
        (Dir::Forward, Kernel::Pull) => ensure!(
            input.gt.is_some(),
            "the forward pull kernel needs the transpose"
        ),
        (Dir::Backward, Kernel::Pull) => ensure!(
            input.g.is_some(),
            "the backward pull kernel needs the graph"
        ),
        (Dir::Backward, Kernel::Push) => {
            bail!("there is no backward push kernel: use --kernel pull or blocks")
        }
        (_, Kernel::Blocks) => {
            ensure!(
                input.blocks.is_some(),
                "the blocks kernel needs a BlockIndex"
            );
            ensure!(separable, "the blocks kernel needs separable arc weights");
        }
        (_, Kernel::Auto) => unreachable!("Auto is resolved above"),
    }
    if let Some(b) = input.blocks {
        ensure!(
            b.n == input.n,
            "the BlockIndex has {} nodes, the input {}",
            b.n,
            input.n
        );
    }
    Ok(k)
}

/// Estimated heap bytes of a DP run with elements of type `E` over a window of `n_w` nodes:
/// values, predecessors (`pred`), the push `has_pred` bitset, the block records of the
/// `n_blocks` blocks of the window (block kernels) and `sweep_bytes` of sweep buffers (generic
/// kernels; see [`sweep_buffer_bytes`]).
pub fn estimate_bytes<E>(
    n_w: usize,
    dir: Dir,
    kernel: Kernel,
    pred: bool,
    n_blocks: usize,
    sweep_bytes: u64,
) -> u64 {
    let n_w = n_w as u64;
    let nb = n_blocks as u64;
    let e = std::mem::size_of::<E>() as u64;
    let p = if pred { 4 } else { 0 };
    let mut b = (e + p) * n_w;
    match kernel {
        Kernel::Blocks => {
            b += match dir {
                Dir::Forward => super::blocks::slot_bytes::<E>() as u64 * nb,
                Dir::Backward => (e + p) * nb,
            };
        }
        _ => {
            if kernel == Kernel::Push {
                b += n_w.div_ceil(8);
            }
            b += sweep_bytes;
        }
    }
    b
}

/// Peak bytes of the sweep engine's chunk buffers: `2 × decoders × (4·chunk_arcs +
/// 4·max_chunk_nodes)`.
pub fn sweep_buffer_bytes(sweep: &SweepConf, chunk_arcs: u64, max_chunk_nodes: usize) -> u64 {
    2 * sweep.decoders as u64 * (4 * chunk_arcs + 4 * max_chunk_nodes as u64)
}

// ---------------------------------------------------------------------------------------------
// Helpers shared by the kernels and the certifier
// ---------------------------------------------------------------------------------------------

/// Fails with [`DagError::BadWindow`] unless `w ⊆ [0, n)`.
pub(crate) fn check_window(w: Window, n: usize) -> Result<()> {
    if w.lo > w.hi || w.hi > n {
        return Err(DagError::BadWindow {
            lo: w.lo,
            hi: w.hi,
            n,
        }
        .into());
    }
    Ok(())
}

/// Checks that `Init::Dense` has one term per window node.
pub(crate) fn check_init<E>(init: &Init<E>, w: Window) -> Result<()> {
    if let Init::Dense(d) = init {
        ensure!(
            d.len() == w.len(),
            "Init::Dense has {} terms, the window {w} has {} nodes",
            d.len(),
            w.len()
        );
    }
    Ok(())
}

/// Whether predecessors are tracked; fails if they are requested for a non-selective semiring.
pub(crate) fn pred_tracking<S: Semiring>(s: &S, requested: bool) -> Result<bool> {
    ensure!(
        !requested || S::SELECTIVE,
        "predecessors need a selective semiring ({} is not)",
        s.name()
    );
    Ok(requested)
}

/// The initial term `b(v)` (forward) or `c(v)` (backward) of node `v`: `Sources(e)` gives `e`
/// iff `!has_pred` (no predecessor, resp. successor, in the window), else 0̄.
#[inline(always)]
pub(crate) fn init_term<S: Semiring>(
    s: &S,
    init: &Init<S::Elem>,
    lo: usize,
    v: usize,
    has_pred: bool,
) -> S::Elem {
    match init {
        Init::Sources(e) => {
            if has_pred {
                s.zero()
            } else {
                *e
            }
        }
        Init::All(e) => *e,
        Init::Dense(d) => d[v - lo],
    }
}

/// The canonical predecessor of a node: `best` unless there is no predecessor (`!has_pred`) or
/// the initial term `b` is strictly better than the fold `t`.
#[inline(always)]
pub(crate) fn final_pred<S: Semiring>(
    s: &S,
    has_pred: bool,
    t: S::Elem,
    b: S::Elem,
    best: u32,
) -> u32 {
    if !has_pred || s.better(b, t) {
        NONE
    } else {
        best
    }
}

/// Starts a kernel's progress logger.
pub(crate) fn start_log(pl: &mut impl ProgressLog, what: &str, w: Window) {
    pl.item_name("node")
        .display_memory(true)
        .expected_updates(Some(w.len()));
    pl.start(format!("{what} over {w}"));
}

/// The block range `[blo, bhi)` of the blocks whose first output lies in `w`.
pub(crate) fn block_range(b: &BlockIndex, w: Window) -> Range<usize> {
    b.block_bound(w.lo)..b.block_bound(w.hi)
}
