//! `Csr`: a compact in-memory graph for windows (matching, flow, transitive reduction).
//!
//! Node ids are relative to the window (`v − lo`); successor lists are sorted and deduplicated
//! (inputs with duplicate arcs are deduplicated, T19). Offsets are `u64` because window arc
//! counts can exceed `u32`. Memory: `8(n + 1) + 4m` bytes.

use anyhow::Result;
use dsi_progress_logger::prelude::*;
use webgraph::prelude::*;

use crate::error::DagError;
use crate::graph::ChunkedGraph;
use crate::sweep::par_chunks;
use crate::window::Window;

/// Compressed sparse rows: `succ[offsets[u]..offsets[u + 1]]` are the successors of `u`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Csr {
    /// Number of nodes.
    pub n: usize,
    /// `n + 1` offsets into `succ`.
    pub offsets: Box<[u64]>,
    /// Concatenated successor lists (sorted, deduplicated).
    pub succ: Box<[u32]>,
}

impl Csr {
    /// Successors of `u`.
    #[inline]
    pub fn succ(&self, u: usize) -> &[u32] {
        &self.succ[self.offsets[u] as usize..self.offsets[u + 1] as usize]
    }

    /// Number of arcs.
    #[inline]
    pub fn num_arcs(&self) -> usize {
        self.succ.len()
    }

    /// Outdegree of `u`.
    #[inline]
    pub fn outdegree(&self, u: usize) -> usize {
        (self.offsets[u + 1] - self.offsets[u]) as usize
    }

    /// The subgraph induced by the window `w`, with ids shifted by `−w.lo` (one parallel scan).
    pub fn from_window<G: RandomAccessGraph + Sync>(
        g: &ChunkedGraph<G>,
        w: Window,
        pl: &mut impl ConcurrentProgressLog,
    ) -> Result<Self> {
        let lo = w.lo;
        // Per chunk: (start, degrees, successors); chunks are folded in index order by
        // collecting into a list ordered by start.
        let mut parts: Vec<(usize, Vec<u32>, Vec<u32>)> = par_chunks(
            g,
            w,
            pl,
            |it| {
                let mut start = usize::MAX;
                let mut deg = Vec::new();
                let mut succ = Vec::new();
                while let Some((u, s)) = it.next_node() {
                    start = start.min(u);
                    let before = succ.len();
                    succ.extend(
                        s.iter()
                            .filter(|&&v| v as usize >= lo)
                            .map(|&v| v - lo as u32),
                    );
                    deg.push((succ.len() - before) as u32);
                }
                Ok(vec![(start, deg, succ)])
            },
            |mut a, mut b| {
                a.append(&mut b);
                a
            },
        )?;
        parts.sort_unstable_by_key(|p| p.0);
        let n = w.len();
        let m: usize = parts.iter().map(|p| p.2.len()).sum();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut succ = Vec::with_capacity(m);
        offsets.push(0u64);
        for (_, deg, s) in parts {
            for d in deg {
                let last = *offsets.last().unwrap();
                offsets.push(last + d as u64);
            }
            succ.extend_from_slice(&s);
        }
        anyhow::ensure!(
            offsets.len() == n + 1,
            "window decode produced {} nodes, expected {n}",
            offsets.len() - 1
        );
        Ok(Self {
            n,
            offsets: offsets.into(),
            succ: succ.into(),
        })
    }

    /// Builds a graph with `n` nodes from arcs (sorted and deduplicated; panics on ids `≥ n`).
    pub fn from_arcs(n: usize, arcs: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut arcs: Vec<(u32, u32)> = arcs
            .into_iter()
            .map(|(u, v)| {
                assert!(u < n && v < n, "arc ({u}, {v}) out of range for {n} nodes");
                (u as u32, v as u32)
            })
            .collect();
        arcs.sort_unstable();
        arcs.dedup();
        let mut offsets = vec![0u64; n + 1];
        for &(u, _) in &arcs {
            offsets[u as usize + 1] += 1;
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let succ: Box<[u32]> = arcs.into_iter().map(|(_, v)| v).collect();
        Self {
            n,
            offsets: offsets.into(),
            succ,
        }
    }

    /// Converts a [`VecGraph`].
    pub fn from_vec_graph(g: &VecGraph) -> Self {
        let n = g.num_nodes();
        Self::from_arcs(n, (0..n).flat_map(|u| g.successors(u).map(move |v| (u, v))))
    }

    /// Converts to a [`VecGraph`] (with all `n` nodes, including trailing isolated ones).
    pub fn to_vec_graph(&self) -> VecGraph {
        let mut g = VecGraph::empty(self.n);
        g.add_arcs((0..self.n).flat_map(|u| self.succ(u).iter().map(move |&v| (u, v as usize))));
        g
    }

    /// The transpose (predecessor lists, sorted).
    pub fn transpose(&self) -> Csr {
        let mut offsets = vec![0u64; self.n + 1];
        for &v in self.succ.iter() {
            offsets[v as usize + 1] += 1;
        }
        for i in 0..self.n {
            offsets[i + 1] += offsets[i];
        }
        let mut pos: Vec<u64> = offsets[..self.n].to_vec();
        let mut succ = vec![0u32; self.succ.len()];
        for u in 0..self.n {
            for &v in self.succ(u) {
                succ[pos[v as usize] as usize] = u as u32;
                pos[v as usize] += 1;
            }
        }
        Csr {
            n: self.n,
            offsets: offsets.into(),
            succ: succ.into(),
        }
    }

    /// Fails with [`DagError::NotTopological`] at the smallest offending `u` if some arc has
    /// `v ≤ u`.
    pub fn check_topological(&self) -> Result<()> {
        for u in 0..self.n {
            if let Some(&v) = self.succ(u).first() {
                if v as usize <= u {
                    return Err(DagError::NotTopological { u, v: v as usize }.into());
                }
            }
        }
        Ok(())
    }
}
