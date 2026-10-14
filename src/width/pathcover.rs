//! Minimum vertex-disjoint path covers (MPC_d): Hopcroft–Karp and the PG closed form.
//!
//! # Generic ([\[BG77\]](crate::references#bg77), [\[HK73\]](crate::references#hk73))
//! A maximum matching `M` of the split graph gives a path cover with `n − |M|` paths:
//! `next[u] = mate_l[u]`, and the paths start at the nodes with `mate_r[v] = NONE`.
//! [`min_path_cover_hk`] runs [`hopcroft_karp`] on a [`Csr`].
//!
//! # PG closed form (corrected for windows)
//! On a line digraph the split graph is a disjoint union of complete bicliques
//! `I(B) × O(B)`, so `ν = Σ_B min(i_B, o_B)`. For a window `W = [lo, hi)`:
//! ```text
//! MPC_d(W) = |W| − Σ_B min(i_B^W, o_B^W)
//!   i_B^W = #{u ∈ W : spend[u] = B}          (only blocks with first[B] ∈ [lo, hi) qualify:
//!   o_B^W = |[first[B], first[B] + len[B]) ∩ W|  inputs precede first[B] and are ≥ lo)
//! ```
//! Sources and blocks that straddle `lo` contribute through `|W|` only. One parallel scan of
//! `spend[lo..hi)` builds the per-window histogram `i^W` over the blocks
//! `[block_bound(lo), block_bound(hi))` (4 B/block), then one parallel scan of those blocks
//! sums the minima. Full PG: `2,181,021,971 − 1,016,828,057 = 1,164,193,914`.
//!
//! Explicit cover: at block `B` with window inputs `x_1 < … < x_k` (ascending), set
//! `next[x_j] = first[B] + j − 1` for `j ≤ min(i_B^W, o_B^W)` (one sequential scan of the window
//! with a per-block counter). Every output receives at most one predecessor and every input at
//! most one successor, so this is a matching of size `ν`.
//!
//! # Checks
//! [`check_path_cover`] verifies a cover against a [`Csr`]: every `next` link is an arc, no node
//! has two predecessors, `starts` are exactly the nodes without a predecessor, and the paths
//! from `starts` visit every node exactly once.

use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, ensure};
use rayon::prelude::*;

use super::matching::hopcroft_karp;
use crate::blocks::BlockIndex;
use crate::csr::Csr;
use crate::io::{NONE, huge_vec};
use crate::window::Window;

/// A vertex-disjoint path cover.
///
/// For a cover of a [`Csr`] the ids are those of the `Csr`. For a cover from
/// [`min_path_cover_blocks`] `next` is indexed by `v − lo` and holds absolute ids, and `starts`
/// holds absolute ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathCover {
    /// Successor of each node on its path; [`NONE`] at the end of a path.
    pub next: Vec<u32>,
    /// First node of each path, ascending.
    pub starts: Vec<u32>,
}

impl PathCover {
    /// Number of paths.
    pub fn num_paths(&self) -> usize {
        self.starts.len()
    }
}

/// Totals of the closed-form path cover of a window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct LineCoverStats {
    /// Nodes in the window.
    pub n: u64,
    /// Size of a maximum matching of the split graph, `Σ_B min(i_B^W, o_B^W)`.
    pub matching: u64,
    /// `n − matching`.
    pub mpc_d: u64,
}

/// Minimum path cover of `g` from a Hopcroft–Karp maximum matching: `n − ν` paths.
///
/// ```
/// use dag::csr::Csr;
/// use dag::width::min_path_cover_hk;
/// let g = Csr::from_arcs(4, [(0, 1), (0, 2), (1, 3), (2, 3)]); // a diamond
/// let pc = min_path_cover_hk(&g);
/// assert_eq!(pc.num_paths(), 2);
/// ```
pub fn min_path_cover_hk(g: &Csr) -> PathCover {
    let m = hopcroft_karp(g);
    let starts = (0..g.n as u32)
        .filter(|&v| m.mate_r[v as usize] == NONE)
        .collect();
    PathCover {
        next: m.mate_l,
        starts,
    }
}

/// Window input histogram `i^W` over the blocks `[blo, bhi)`: `hist[B − blo] = #{u ∈ W :
/// spend[u] = B}` (blocks `≥ bhi` start after the window and are not counted).
pub(crate) fn input_histogram(blocks: &BlockIndex, w: Window) -> (usize, usize, Vec<u32>) {
    let blo = blocks.block_bound(w.lo);
    let bhi = blocks.block_bound(w.hi);
    let mut hist = huge_vec(bhi - blo, 0u32);
    {
        let h: &[AtomicU32] = crate::io::as_atomic_u32(&mut hist);
        blocks.spend[w.lo..w.hi]
            .par_iter()
            .with_min_len(1 << 16)
            .for_each(|&b| {
                let b = b as usize;
                if b != NONE as usize && b < bhi {
                    debug_assert!(b >= blo);
                    h[b - blo].fetch_add(1, Ordering::Relaxed);
                }
            });
    }
    (blo, bhi, hist)
}

/// `o_B^W = |O(B) ∩ [lo, hi)|` for a block with `first[B] ∈ [lo, hi)`.
#[inline]
pub(crate) fn outputs_in(blocks: &BlockIndex, b: usize, w: Window) -> u32 {
    let r = blocks.outputs(b);
    (r.end.min(w.hi) - r.start) as u32
}

/// MPC_d of the window `w` by the closed form (see the module documentation), and, if
/// `explicit`, the cover itself (`next` indexed by `v − lo`, absolute ids as values).
///
/// Memory: 4 B per block of the window, plus, with `explicit`, 5 B/node (`next` and a
/// predecessor flag) and 4 B per path (`starts`, at most 4 B/node).
pub fn min_path_cover_blocks(
    blocks: &BlockIndex,
    w: Window,
    explicit: bool,
) -> (LineCoverStats, Option<PathCover>) {
    let (blo, bhi, hist) = input_histogram(blocks, w);
    let matching: u64 = (blo..bhi)
        .into_par_iter()
        .with_min_len(1 << 16)
        .map(|b| hist[b - blo].min(outputs_in(blocks, b, w)) as u64)
        .sum();
    let n = w.len() as u64;
    let stats = LineCoverStats {
        n,
        matching,
        mpc_d: n - matching,
    };
    if !explicit {
        return (stats, None);
    }
    let mut next = huge_vec(w.len(), NONE);
    let mut has_pred = huge_vec(w.len(), false);
    // per-block counter of the inputs seen so far (reuses the histogram's storage)
    let mut seen = hist;
    seen.fill(0);
    for (i, &b) in blocks.spend[w.lo..w.hi].iter().enumerate() {
        let b = b as usize;
        if b == NONE as usize || b >= bhi {
            continue;
        }
        let j = seen[b - blo];
        if j < outputs_in(blocks, b, w) {
            let v = blocks.first[b] + j;
            next[i] = v;
            has_pred[v as usize - w.lo] = true;
            seen[b - blo] = j + 1;
        }
    }
    let starts = (0..w.len())
        .filter(|&i| !has_pred[i])
        .map(|i| (w.lo + i) as u32)
        .collect();
    (stats, Some(PathCover { next, starts }))
}

/// Verifies that `pc` is a vertex-disjoint path cover of `g` (ids of `g`) and returns its
/// number of paths. See the module documentation for the checks. Memory: 1 B/node.
pub fn check_path_cover(g: &Csr, pc: &PathCover) -> Result<usize> {
    let n = g.n;
    ensure!(
        pc.next.len() == n,
        "next has {} entries, the graph {n} nodes",
        pc.next.len()
    );
    let mut has_pred = huge_vec(n, false);
    for u in 0..n {
        let v = pc.next[u];
        if v == NONE {
            continue;
        }
        ensure!(
            (v as usize) < n && g.succ(u).binary_search(&v).is_ok(),
            "next[{u}] = {v} is not an arc"
        );
        ensure!(!has_pred[v as usize], "node {v} has two predecessors");
        has_pred[v as usize] = true;
    }
    // compared as streams: no 4 B/node copy of the expected starts
    ensure!(
        (0..n as u32)
            .filter(|&v| !has_pred[v as usize])
            .eq(pc.starts.iter().copied()),
        "starts are not the nodes without a predecessor"
    );
    let mut visited = 0usize;
    for &s in &pc.starts {
        let mut x = s;
        while x != NONE {
            visited += 1;
            ensure!(visited <= n, "the paths visit more than {n} nodes");
            x = pc.next[x as usize];
        }
    }
    ensure!(visited == n, "the paths visit {visited} of {n} nodes");
    Ok(pc.starts.len())
}
