//! Time windows and pass directions.
//!
//! On the Payment Graph node ids are chronological and topological, so a half-open id range
//! `[lo, hi)` is a *time window*. A window is **convex**: every path between two window nodes
//! stays inside the window, because ids strictly increase along paths. Reachability, dominance by
//! window nodes, transitive reduction, width and DP values over window-internal paths are
//! therefore exact statements about the induced subgraph
//! ([\[AGU72\]](crate::references#agu72) for reductions of induced subgraphs).
//!
//! Conventions shared by every subcommand:
//! - per-node outputs of windowed runs are indexed by `v − lo`;
//! - node ids stored as **values** (pred, idom, chains, antichains) are **absolute** ids.

use crate::error::DagError;

/// A half-open node range `[lo, hi)`: a time window on the PG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Window {
    /// First node of the window.
    pub lo: usize,
    /// One past the last node of the window.
    pub hi: usize,
}

impl Window {
    /// The window covering all `n` nodes.
    pub fn full(n: usize) -> Self {
        Self { lo: 0, hi: n }
    }

    /// Creates the window `[lo, hi)` after checking `lo <= hi <= n`.
    pub fn new(lo: usize, hi: usize, n: usize) -> anyhow::Result<Self> {
        if lo > hi || hi > n {
            return Err(DagError::BadWindow { lo, hi, n }.into());
        }
        Ok(Self { lo, hi })
    }

    /// Parses `"a..b"`, `"a.."`, `"..b"`, `".."` or `""` (the full range).
    ///
    /// Underscores are accepted as digit separators (`0..100_000_000`). Returns
    /// [`DagError::BadWindow`] if `a > b` or `b > n`.
    ///
    /// ```
    /// use dag::Window;
    /// assert_eq!(Window::parse("2..5", 10).unwrap(), Window { lo: 2, hi: 5 });
    /// assert_eq!(Window::parse("3..", 10).unwrap(), Window { lo: 3, hi: 10 });
    /// assert_eq!(Window::parse("", 10).unwrap(), Window::full(10));
    /// assert!(Window::parse("5..2", 10).is_err());
    /// ```
    pub fn parse(s: &str, n: usize) -> anyhow::Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Self::full(n));
        }
        let Some((a, b)) = s.split_once("..") else {
            anyhow::bail!("invalid window {s:?}: expected a..b, a.., ..b or ..");
        };
        let num = |t: &str, default: usize| -> anyhow::Result<usize> {
            let t = t.trim().replace('_', "");
            if t.is_empty() {
                Ok(default)
            } else {
                t.parse::<usize>()
                    .map_err(|e| anyhow::anyhow!("invalid window bound {t:?} in {s:?}: {e}"))
            }
        };
        let lo = num(a, 0)?;
        let hi = num(b, n)?;
        Self::new(lo, hi, n)
    }

    /// Number of nodes in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.hi - self.lo
    }

    /// Whether the window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hi == self.lo
    }

    /// Whether `v` lies in the window.
    #[inline]
    pub fn contains(&self, v: usize) -> bool {
        self.lo <= v && v < self.hi
    }

    /// Whether this is the full window of a graph with `n` nodes.
    #[inline]
    pub fn is_full(&self, n: usize) -> bool {
        self.lo == 0 && self.hi == n
    }
}

impl std::fmt::Display for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.lo, self.hi)
    }
}

/// Direction of a pass over a topologically numbered graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, clap::ValueEnum)]
pub enum Dir {
    /// Ascending ids (sources first).
    #[value(alias = "fwd")]
    Forward,
    /// Descending ids (sinks first).
    #[value(alias = "bwd")]
    Backward,
}

/// Which side(s) of a two-sided computation to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, clap::ValueEnum)]
pub enum Which {
    /// The forward side: ancestors, or the forward DP.
    #[value(alias = "fwd", alias = "anc")]
    Forward,
    /// The backward side: descendants, or the backward DP.
    #[value(alias = "bwd", alias = "desc")]
    Backward,
    /// Both sides.
    Both,
}

impl Which {
    /// Whether the forward side is requested.
    pub fn forward(self) -> bool {
        matches!(self, Which::Forward | Which::Both)
    }
    /// Whether the backward side is requested.
    pub fn backward(self) -> bool {
        matches!(self, Which::Backward | Which::Both)
    }
}
