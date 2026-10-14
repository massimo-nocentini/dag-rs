//! Graph loading and arc-balanced chunking.
//!
//! A [`ChunkedGraph`] is a random-access graph plus **cut points**: node ids that split the
//! graph into chunks of about [`ChunkConf::chunk_arcs`] arcs (and at most
//! [`ChunkConf::max_chunk_nodes`] nodes). Chunks are the unit of every sweep
//! ([`crate::sweep`]); chunk geometry lives only here.
//!
//! Cut rule (identical for both constructors): with `cum(u)` = number of arcs leaving nodes
//! `< u`, node `u` is a cut point iff it is the smallest node with `cum(u) ≥ k·chunk_arcs` for
//! some `k ≥ 1`; then every chunk longer than `max_chunk_nodes` is split evenly.
//! [`ChunkedGraph::with_dcf`] reads `cum` from the on-disk degree cumulative function (`.dcf`,
//! an Elias–Fano list built by `webgraph build dcf`, [\[BV04\]](crate::references#bv04));
//! [`ChunkedGraph::new`] computes it from outdegrees in one parallel pass.
//!
//! The only `unsafe` code is the ε-serde memory mapping (and storing) of the DCF.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dsi_bitstream::prelude::BE;
use epserde::deser::{Deserialize, Flags};
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::error::DagError;
use crate::window::Window;

/// The random-access BV graph type used for on-disk graphs (memory-mapped graph and Elias–Fano
/// offsets, big endian, dynamic code dispatch).
pub type BvRa = BvGraph<DynCodesDecoderFactory<BE, MmapHelper<u32>, EF>>;

/// Largest supported number of nodes: ids are `u32` and `u32::MAX` is [`NONE`](crate::NONE).
pub const MAX_NODES: usize = u32::MAX as usize - 1;

/// Loads a BV graph for random access. Needs `<basename>.ef`.
pub fn load_ra(basename: &Path) -> Result<BvRa> {
    let ef = basename.with_extension(EF_EXTENSION);
    let graph = basename.with_extension(GRAPH_EXTENSION);
    anyhow::ensure!(graph.exists(), "graph file {} not found", graph.display());
    anyhow::ensure!(
        ef.exists(),
        "{} not found: random access needs the Elias-Fano offsets (run `webgraph build ef {}`)",
        ef.display(),
        basename.display()
    );
    BvGraph::with_basename(basename)
        .load()
        .with_context(|| format!("cannot load {}", basename.display()))
}

/// Reads the number of nodes and arcs from `<basename>.properties` (the `nodes=` and `arcs=`
/// keys written by every BV compressor, [\[BV04\]](crate::references#bv04)) without loading
/// the graph.
pub fn read_properties(basename: &Path) -> Result<(usize, u64)> {
    let path = basename.with_extension(PROPERTIES_EXTENSION);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let get = |key: &str| -> Result<u64> {
        let v = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .find(|(k, _)| k.trim() == key)
            .with_context(|| format!("{}: no `{key}=` entry", path.display()))?
            .1;
        v.trim()
            .parse()
            .with_context(|| format!("{}: invalid `{key}={v}`", path.display()))
    };
    Ok((get("nodes")? as usize, get("arcs")?))
}

/// Returns `<b>-t` if `<b>-t.graph` exists.
pub fn transpose_basename(basename: &Path) -> Option<PathBuf> {
    let mut s = basename.as_os_str().to_owned();
    s.push("-t");
    let t = PathBuf::from(s);
    t.with_extension(GRAPH_EXTENSION).exists().then_some(t)
}

/// Chunk geometry of a [`ChunkedGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkConf {
    /// Target number of arcs per chunk (default 2²⁴).
    pub chunk_arcs: u64,
    /// Maximum number of nodes per chunk (default 2²²).
    pub max_chunk_nodes: usize,
}

impl Default for ChunkConf {
    fn default() -> Self {
        Self {
            chunk_arcs: 1 << 24,
            max_chunk_nodes: 1 << 22,
        }
    }
}

/// A random-access graph plus arc-balanced chunk cut points (the unit of every sweep).
pub struct ChunkedGraph<G> {
    /// The graph.
    pub graph: G,
    /// Cut points: strictly increasing, `cut[0] = 0`, last = `n`.
    pub cut: Box<[usize]>,
    /// Number of arcs of the graph.
    pub num_arcs: u64,
    /// The geometry the cut points were computed with.
    pub conf: ChunkConf,
}

impl<G: RandomAccessGraph + Sync> ChunkedGraph<G> {
    /// Computes the cut points from the outdegrees (one parallel pass, no `.dcf` needed).
    pub fn new(graph: G, conf: ChunkConf) -> Result<Self> {
        let n = graph.num_nodes();
        check_nodes(n)?;
        let chunk_arcs = conf.chunk_arcs.max(1);
        const BLOCK: usize = 1 << 16;
        let nblocks = n.div_ceil(BLOCK);
        // Pass 1: arcs per block of nodes.
        let sums: Vec<u64> = (0..nblocks)
            .into_par_iter()
            .map(|b| {
                let (s, e) = (b * BLOCK, ((b + 1) * BLOCK).min(n));
                (s..e).map(|u| graph.outdegree(u) as u64).sum()
            })
            .collect();
        let mut base = Vec::with_capacity(nblocks + 1);
        let mut acc = 0u64;
        for s in &sums {
            base.push(acc);
            acc += s;
        }
        let num_arcs = acc;
        // Pass 2: cut points inside each block.
        let cuts: Vec<Vec<usize>> = (0..nblocks)
            .into_par_iter()
            .map(|b| {
                let (s, e) = (b * BLOCK, ((b + 1) * BLOCK).min(n));
                let mut out = Vec::new();
                let mut cum = base[b];
                // prev = cum(u - 1); for u = 0 there is no previous node
                let mut prev = if s > 0 {
                    cum - graph.outdegree(s - 1) as u64
                } else {
                    0
                };
                for u in s..e {
                    if u > 0 && cum / chunk_arcs > prev / chunk_arcs {
                        out.push(u);
                    }
                    prev = cum;
                    cum += graph.outdegree(u) as u64;
                }
                out
            })
            .collect();
        let mut cut = vec![0usize];
        cut.extend(cuts.into_iter().flatten());
        cut.push(n);
        Ok(Self {
            cut: finish_cuts(cut, n, conf.max_chunk_nodes),
            graph,
            num_arcs,
            conf,
        })
    }

    /// Computes the cut points from the on-disk degree cumulative function `dcf_path`
    /// (Elias–Fano, `webgraph build dcf`). Falls back to [`new`](Self::new) with a warning if
    /// the DCF does not match the graph.
    pub fn with_dcf(graph: G, dcf_path: &Path, conf: ChunkConf) -> Result<Self> {
        use sux::traits::{IndexedSeq, Succ};
        let n = graph.num_nodes();
        check_nodes(n)?;
        let chunk_arcs = conf.chunk_arcs.max(1);
        // SAFETY: ε-serde maps the file and reinterprets it as the serialized DCF type; the
        // file is a DCF written by `webgraph build dcf` (or by `store_dcf`), and the mapping is
        // read-only and dropped at the end of this function.
        let case = unsafe { DCF::mmap(dcf_path, Flags::empty()) }
            .with_context(|| format!("cannot map {}", dcf_path.display()))?;
        let dcf = case.uncase();
        if dcf.len() != n + 1 {
            log::warn!(
                "{}: {} entries for {} nodes; computing cut points from outdegrees",
                dcf_path.display(),
                dcf.len(),
                n
            );
            drop(case);
            return Self::new(graph, conf);
        }
        let num_arcs = dcf.get(n);
        if let Some(m) = graph.get_num_arcs() {
            if m != num_arcs {
                log::warn!(
                    "{}: {num_arcs} arcs, graph has {m}; computing cut points from outdegrees",
                    dcf_path.display()
                );
                drop(case);
                return Self::new(graph, conf);
            }
        }
        let mut cut = vec![0usize];
        let mut k = 1u64;
        while k.saturating_mul(chunk_arcs) <= num_arcs {
            let target = k * chunk_arcs;
            let (idx, _) = dcf.succ(target).expect("target <= dcf[n]");
            cut.push(idx.min(n));
            k += 1;
        }
        cut.push(n);
        drop(case);
        Ok(Self {
            cut: finish_cuts(cut, n, conf.max_chunk_nodes),
            graph,
            num_arcs,
            conf,
        })
    }

    /// Number of nodes of the graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    /// Cut points restricted to `w`, with `w.lo` and `w.hi` included (a single `[lo]` for an
    /// empty window).
    pub fn cut_in(&self, w: Window) -> Vec<usize> {
        let mut out = vec![w.lo];
        let first = self.cut.partition_point(|&c| c <= w.lo);
        out.extend(self.cut[first..].iter().copied().take_while(|&c| c < w.hi));
        if w.hi > w.lo {
            out.push(w.hi);
        }
        out
    }

    /// The chunks `[s, e)` of `w`, in ascending order.
    pub fn chunks_in(&self, w: Window) -> Vec<(usize, usize)> {
        self.cut_in(w).windows(2).map(|p| (p[0], p[1])).collect()
    }
}

/// Sorts, deduplicates and splits chunks longer than `max_nodes`.
fn finish_cuts(mut cut: Vec<usize>, n: usize, max_nodes: usize) -> Box<[usize]> {
    cut.sort_unstable();
    cut.dedup();
    let max_nodes = max_nodes.max(1);
    let mut out = Vec::with_capacity(cut.len());
    out.push(0);
    for p in cut.windows(2) {
        let (s, e) = (p[0], p[1]);
        let len = e - s;
        if len > max_nodes {
            let parts = len.div_ceil(max_nodes);
            for i in 1..parts {
                out.push(s + len * i / parts);
            }
        }
        out.push(e);
    }
    out.dedup();
    if n == 0 {
        out.truncate(1);
    }
    out.into_boxed_slice()
}

/// Checks `n <= u32::MAX - 1`.
pub fn check_nodes(n: usize) -> Result<()> {
    if n > MAX_NODES {
        return Err(DagError::TooManyNodes { n, max: MAX_NODES }.into());
    }
    Ok(())
}

/// Opens `<basename>` for random access and chunks it, using `<basename>.dcf` when present.
/// Fails with [`DagError::TooManyNodes`] if `n > u32::MAX − 1`.
pub fn open(basename: &Path, conf: ChunkConf) -> Result<ChunkedGraph<BvRa>> {
    let g = load_ra(basename)?;
    check_nodes(g.num_nodes())?;
    let dcf = basename.with_extension(DEG_CUMUL_EXTENSION);
    if dcf.exists() {
        ChunkedGraph::with_dcf(g, &dcf, conf)
    } else {
        log::info!(
            "{} not found: computing cut points from outdegrees",
            dcf.display()
        );
        ChunkedGraph::new(g, conf)
    }
}

/// Builds the degree cumulative function of `graph` and stores it at `path` (as
/// `webgraph build dcf` does).
pub fn store_dcf<G: SequentialGraph>(graph: &G, path: &Path) -> Result<()> {
    let dcf = graph.build_dcf();
    let file =
        std::fs::File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    let mut w = std::io::BufWriter::new(file);
    // SAFETY: ε-serde serialization of a fully built, owned DCF value; the unsafety only
    // concerns readers that map the file, which use the same DCF type.
    unsafe { epserde::ser::Serialize::serialize(&dcf, &mut w) }
        .with_context(|| format!("cannot serialize {}", path.display()))?;
    std::io::Write::flush(&mut w)?;
    Ok(())
}
