//! Semiring dynamic programming over a topologically numbered DAG: the shared types.
//!
//! This module freezes the input/output types of every DP kernel; the kernels live in the
//! submodules (`solve`, `push`, `pull`, `blocks`, `certify`, `path`).
//!
//! # Canonical evaluation order (binding for every kernel)
//! Write `t ⊕= x` for `t = plus(t, x)`.
//! ```text
//! forward, separable:  acc = 0̄; for u in pred_W(v) ascending: acc ⊕= val[u] ⊗ α(u)
//!                      val[v] = has_pred ? (acc ⊗ β(v)) ⊕ b(v) : b(v)
//! forward, Arc:        acc ⊕= val[u] ⊗ w(u,v);   val[v] = has_pred ? acc ⊕ b(v) : b(v)
//! backward, separable: acc = 0̄; for v in succ_W(u) ascending: acc ⊕= β(v) ⊗ val[v]
//!                      val[u] = has_succ ? (α(u) ⊗ acc) ⊕ c(u) : c(u)
//! backward, Arc:       acc ⊕= w(u,v) ⊗ val[v];   val[u] = has_succ ? acc ⊕ c(u) : c(u)
//! ```
//! - Init: `Sources(e)` gives `b(v) = e` iff `¬has_pred`, else 0̄; `All(e)` gives `b(v) = e`;
//!   `Dense(d)` gives `b(v) = d[v − lo]`. `has_pred` means "has a predecessor inside the window",
//!   independently of the values pushed (a predecessor carrying 0̄ still counts).
//! - `pred` (only when `S::SELECTIVE`): the smallest `u` whose term equals the fold result,
//!   maintained by replacing only on strict [`better`](crate::semiring::Semiring::better);
//!   [`NONE`](crate::NONE) when the init term is strictly better or there is no predecessor.
//!
//! Consequence: push, pull and blocks kernels are bit-identical for every semiring (floats
//! included) with [`One`], [`Const`] and [`Node`] weights; with [`Arc`] weights only push == pull.
//! The algebraic framework is [\[Moh02\]](crate::references#moh02); the topological pass is
//! [\[Kah62\]](crate::references#kah62), and the biclique (block) kernel follows F1 §6.5.

use crate::blocks::BlockIndex;
use crate::graph::{BvRa, ChunkedGraph};
use crate::semiring::Semiring;
use crate::sweep::SweepConf;
use crate::window::Window;

pub mod blocks;
pub mod certify;
pub mod path;
pub mod pull;
pub mod push;
pub mod solve;

/// Arc weights. Separable weights are `w(u, v) = alpha(u) ⊗ beta(v)`.
pub trait ArcWeights<S: Semiring>: Sync {
    /// Whether the weights are separable (then `alpha`/`beta` are used and `arc` is their
    /// product).
    const SEPARABLE: bool;
    /// Tail factor `α(u)` (absolute id); used iff `SEPARABLE`.
    fn alpha(&self, s: &S, u: usize) -> S::Elem;
    /// Head factor `β(v)` (absolute id); used iff `SEPARABLE`.
    fn beta(&self, s: &S, v: usize) -> S::Elem;
    /// Weight of the arc `(u, v)`; default `alpha(u) ⊗ beta(v)`.
    #[inline]
    fn arc(&self, s: &S, u: usize, v: usize) -> S::Elem {
        s.times(self.alpha(s, u), self.beta(s, v))
    }
}

/// Every arc weighs 1̄ (counting, Boolean, probabilities, Viterbi, bottleneck).
#[derive(Clone, Copy, Debug, Default)]
pub struct One;

impl<S: Semiring> ArcWeights<S> for One {
    const SEPARABLE: bool = true;
    #[inline(always)]
    fn alpha(&self, s: &S, _u: usize) -> S::Elem {
        s.one()
    }
    #[inline(always)]
    fn beta(&self, s: &S, _v: usize) -> S::Elem {
        s.one()
    }
    #[inline(always)]
    fn arc(&self, s: &S, _u: usize, _v: usize) -> S::Elem {
        s.one()
    }
}

/// Every arc weighs `e`: `alpha = e`, `beta = 1̄`. The tropical "unit length" is `Const(1)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Const<E>(pub E);

impl<S: Semiring> ArcWeights<S> for Const<S::Elem> {
    const SEPARABLE: bool = true;
    #[inline(always)]
    fn alpha(&self, _s: &S, _u: usize) -> S::Elem {
        self.0
    }
    #[inline(always)]
    fn beta(&self, s: &S, _v: usize) -> S::Elem {
        s.one()
    }
}

/// Node-separable weights `w(u, v) = alpha(u) ⊗ beta(v)` given by closures over absolute ids
/// (monomorphized).
#[derive(Clone, Copy, Debug, Default)]
pub struct Node<A, B> {
    /// `α(u)`.
    pub alpha: A,
    /// `β(v)`.
    pub beta: B,
}

impl<S, A, B> ArcWeights<S> for Node<A, B>
where
    S: Semiring,
    A: Fn(usize) -> S::Elem + Sync,
    B: Fn(usize) -> S::Elem + Sync,
{
    const SEPARABLE: bool = true;
    #[inline(always)]
    fn alpha(&self, _s: &S, u: usize) -> S::Elem {
        (self.alpha)(u)
    }
    #[inline(always)]
    fn beta(&self, _s: &S, v: usize) -> S::Elem {
        (self.beta)(v)
    }
}

/// General arc weights `w(u, v)` given by a closure over absolute ids (not separable).
#[derive(Clone, Copy, Debug, Default)]
pub struct Arc<C>(pub C);

impl<S, C> ArcWeights<S> for Arc<C>
where
    S: Semiring,
    C: Fn(usize, usize) -> S::Elem + Sync,
{
    const SEPARABLE: bool = false;
    /// Not used for non-separable weights; returns 1̄.
    #[inline(always)]
    fn alpha(&self, s: &S, _u: usize) -> S::Elem {
        s.one()
    }
    /// Not used for non-separable weights; returns 1̄.
    #[inline(always)]
    fn beta(&self, s: &S, _v: usize) -> S::Elem {
        s.one()
    }
    #[inline(always)]
    fn arc(&self, _s: &S, u: usize, v: usize) -> S::Elem {
        (self.0)(u, v)
    }
}

/// Initial terms of the DP. Forward `Sources` = indegree 0 in the window; backward `Sources` =
/// outdegree 0 in the window.
#[derive(Clone, Debug, PartialEq)]
pub enum Init<E> {
    /// `e` on window sources (resp. sinks), 0̄ elsewhere.
    Sources(E),
    /// `e` on every node.
    All(E),
    /// One term per window node, indexed by `v − lo`.
    Dense(Box<[E]>),
}

/// Which DP kernel to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
pub enum Kernel {
    /// Forward: `Blocks` if blocks are given and the weights are separable, else `Pull` if the
    /// transpose is given, else `Push`. Backward: `Blocks` under the same condition, else `Pull`.
    Auto,
    /// Forward push over the graph.
    Push,
    /// Pull over the transpose (forward) or the graph (backward).
    Pull,
    /// The decode-free biclique kernel over a [`BlockIndex`].
    Blocks,
}

/// Configuration of a DP run.
#[derive(Clone, Debug)]
pub struct DpConf {
    /// The window.
    pub window: Window,
    /// The kernel.
    pub kernel: Kernel,
    /// Whether to track argbest predecessors (selective semirings only).
    pub pred: bool,
    /// Sweep configuration.
    pub sweep: SweepConf,
}

impl DpConf {
    /// `Auto` kernel, no pred, default sweep, over `window`.
    pub fn new(window: Window) -> Self {
        Self {
            window,
            kernel: Kernel::Auto,
            pred: false,
            sweep: SweepConf::default(),
        }
    }
}

/// The graphs a DP may read: forward graph, transpose and/or block index.
pub struct DpInput<'a, G = BvRa> {
    /// The forward graph.
    pub g: Option<&'a ChunkedGraph<G>>,
    /// The transpose.
    pub gt: Option<&'a ChunkedGraph<G>>,
    /// The block index.
    pub blocks: Option<&'a BlockIndex>,
    /// Number of nodes.
    pub n: usize,
}

impl<'a> DpInput<'a, BvRa> {
    /// Only a block index (no graph).
    pub fn blocks(b: &'a BlockIndex) -> Self {
        Self {
            g: None,
            gt: None,
            blocks: Some(b),
            n: b.n,
        }
    }
}

impl<'a, G: webgraph::traits::RandomAccessGraph + Sync> DpInput<'a, G> {
    /// Only the forward graph.
    pub fn graph(g: &'a ChunkedGraph<G>) -> Self {
        Self {
            g: Some(g),
            gt: None,
            blocks: None,
            n: g.num_nodes(),
        }
    }

    /// The forward graph and its transpose.
    pub fn with_transpose(g: &'a ChunkedGraph<G>, gt: &'a ChunkedGraph<G>) -> Self {
        Self {
            g: Some(g),
            gt: Some(gt),
            blocks: None,
            n: g.num_nodes(),
        }
    }

    /// Adds a block index.
    pub fn and_blocks(self, b: &'a BlockIndex) -> Self {
        Self {
            blocks: Some(b),
            ..self
        }
    }
}

/// Output of a DP: values indexed by `v − lo`, predecessors as absolute ids ([`NONE`](crate::NONE)
/// if none).
#[derive(Clone, Debug, PartialEq)]
pub struct DpResult<E> {
    /// The window.
    pub window: Window,
    /// `val[v − lo]`.
    pub val: Vec<E>,
    /// `pred[v − lo]` (absolute ids), when requested.
    pub pred: Option<Vec<u32>>,
}
