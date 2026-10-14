//! Topological order: the identity check and relabelling.
//!
//! **Every kernel of the crate assumes that the identity is a topological order** (`u < v` for
//! every arc). This holds on the Payment Graph, whose ids are chronological. Graphs whose ids are
//! not topological are relabelled once:
//!
//! ```text
//! order = webgraph_algo::top_sort(g)         // reverse DFS postorder: position -> node [Kah62, CLRS22]
//! rank  = order⁻¹                            // old -> new
//! check rank[u] < rank[v] for every arc      // mandatory: top_sort does not detect cycles
//! permute_par(g, rank), transpose, compress, .ef, .dcf
//! check_identity(output)                     // fails loudly on a wrong permutation direction
//! ```
//!
//! Results of later passes come back in the new ids; [`map_back`] maps per-node arrays to the old
//! ids with the stored `order` (new → old). Topological sorting is [\[Kah62\]](crate::references#kah62);
//! the DFS-based variant is in [\[CLRS22\]](crate::references#clrs22).
//!
//! Complexity: [`check_identity`] is one parallel O(n + m) scan (3–6 s on the PG);
//! [`topo_order`] is a sequential DFS; [`relabel_bv`] is dominated by external sorting, whose
//! buffers the CLI sizes from `--max-mem` with [`sort_memory`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dsi_bitstream::prelude::BE;
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::graphs::par_graphs::ParGraph;
use webgraph::prelude::*;

use crate::error::DagError;
use crate::graph::{self, ChunkConf, ChunkedGraph};
use crate::io::{self, Dump, MemBudget};
use crate::sweep::par_chunks;
use crate::window::Window;

/// Result of [`check_identity`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TopoReport {
    /// Number of nodes.
    pub n: usize,
    /// Number of arcs counted by the scan of the forward graph.
    pub m: u64,
    /// Whether `u < v` for every arc (of the transpose too, when given).
    pub identity_topological: bool,
    /// The violating arc `(u, v)` with the smallest `u` (and smallest `v` for that `u`), in
    /// forward orientation.
    pub first_violation: Option<(usize, usize)>,
    /// Nodes without successors.
    pub sinks: usize,
    /// Nodes without predecessors (counted on the transpose, when given).
    pub sources: Option<usize>,
}

#[derive(Default)]
struct ScanAcc {
    first: Option<(usize, usize)>,
    zero: usize,
    arcs: u64,
}

fn merge_first(a: Option<(usize, usize)>, b: Option<(usize, usize)>) -> Option<(usize, usize)> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Scans `g` in parallel (and `gt`, the transpose, when given): checks `u < v` for every arc,
/// counts arcs, sinks and (with `gt`) sources.
///
/// On the transpose an arc `(v, u)` must have `u < v`; a violation there is reported as the
/// forward arc `(u, v)`. The scan also checks that `gt` has as many nodes and arcs as `g`.
pub fn check_identity<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    gt: Option<&ChunkedGraph<G>>,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<TopoReport> {
    let n = g.num_nodes();
    pl.item_name("node").expected_updates(Some(n));
    pl.start("checking that the identity is a topological order");
    let fwd = par_chunks(
        g,
        Window::full(n),
        pl,
        |it| {
            let mut acc = ScanAcc::default();
            while let Some((u, s)) = it.next_node() {
                if s.is_empty() {
                    acc.zero += 1;
                } else if acc.first.is_none() && s[0] as usize <= u {
                    acc.first = Some((u, s[0] as usize));
                }
                acc.arcs += s.len() as u64;
            }
            Ok(acc)
        },
        |a, b| ScanAcc {
            first: merge_first(a.first, b.first),
            zero: a.zero + b.zero,
            arcs: a.arcs + b.arcs,
        },
    )?;
    pl.done();
    let mut report = TopoReport {
        n,
        m: fwd.arcs,
        identity_topological: fwd.first.is_none(),
        first_violation: fwd.first,
        sinks: fwd.zero,
        sources: None,
    };
    if let Some(gt) = gt {
        anyhow::ensure!(
            gt.num_nodes() == n,
            "transpose has {} nodes, graph has {n}",
            gt.num_nodes()
        );
        pl.expected_updates(Some(n));
        pl.start("scanning the transpose");
        let bwd = par_chunks(
            gt,
            Window::full(n),
            pl,
            |it| {
                let mut acc = ScanAcc::default();
                while let Some((v, p)) = it.next_node() {
                    if p.is_empty() {
                        acc.zero += 1;
                    }
                    // the predecessors p of v must all be < v; report the forward arc (u, v)
                    // with the smallest u, i.e. the first p >= v ... among violations of v
                    if let Some(&u) = p.iter().find(|&&u| u as usize >= v) {
                        acc.first = merge_first(acc.first, Some((u as usize, v)));
                    }
                    acc.arcs += p.len() as u64;
                }
                Ok(acc)
            },
            |a, b| ScanAcc {
                first: merge_first(a.first, b.first),
                zero: a.zero + b.zero,
                arcs: a.arcs + b.arcs,
            },
        )?;
        pl.done();
        anyhow::ensure!(
            bwd.arcs == fwd.arcs,
            "transpose has {} arcs, graph has {}",
            bwd.arcs,
            fwd.arcs
        );
        report.sources = Some(bwd.zero);
        if bwd.first.is_some() {
            report.identity_topological = false;
            report.first_violation = merge_first(report.first_violation, bwd.first);
        }
    }
    Ok(report)
}

/// A topological order and its inverse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopoOrder {
    /// New → old: `order[position] = node`.
    pub order: Box<[usize]>,
    /// Old → new: `rank[node] = position`.
    pub rank: Box<[usize]>,
}

/// Computes a topological order with `webgraph_algo::top_sort` (DFS) and **verifies** it:
/// fails with [`DagError::Cycle`] (an arc lying on a cycle) if some arc has
/// `rank[u] >= rank[v]`.
///
/// The verification is mandatory because `top_sort` silently returns a DFS order on cyclic
/// graphs. In a reverse DFS postorder the only arcs going backward are DFS back arcs, which lie
/// on cycles, so the reported arc is on a cycle.
pub fn topo_order<G: RandomAccessGraph>(g: &G, pl: &mut impl ProgressLog) -> Result<TopoOrder> {
    let order = webgraph_algo::top_sort(g, pl);
    let n = order.len();
    let mut rank = vec![usize::MAX; n].into_boxed_slice();
    for (pos, &node) in order.iter().enumerate() {
        rank[node] = pos;
    }
    for u in 0..n {
        for v in g.successors(u) {
            if rank[u] >= rank[v] {
                return Err(DagError::Cycle { u, v }.into());
            }
        }
    }
    Ok(TopoOrder { order, rank })
}

/// Relabels a small in-memory graph: node `u` becomes `rank[u]`.
pub fn relabel_vec(g: &VecGraph, rank: &[usize]) -> VecGraph {
    let n = g.num_nodes();
    assert_eq!(
        rank.len(),
        n,
        "rank has {} entries for {n} nodes",
        rank.len()
    );
    let mut out = VecGraph::empty(n);
    let mut arcs = Vec::with_capacity(g.num_arcs() as usize);
    for u in 0..n {
        for v in g.successors(u) {
            arcs.push((rank[u], rank[v]));
        }
    }
    arcs.sort_unstable();
    out.add_arcs(arcs);
    out
}

/// Builds the `.ef` and `.dcf` of a freshly compressed BV graph.
pub(crate) fn finish_bv(basename: &Path, n: usize) -> Result<()> {
    store_ef_with_data(
        n,
        basename.with_extension(GRAPH_EXTENSION),
        basename.with_extension(OFFSETS_EXTENSION),
        basename.with_extension(EF_EXTENSION),
        &mut Option::<ProgressLogger>::None,
    )
    .with_context(|| format!("cannot build {}.ef", basename.display()))?;
    let g = graph::load_ra(basename)?;
    graph::store_dcf(&g, &basename.with_extension(DEG_CUMUL_EXTENSION))?;
    Ok(())
}

/// `<b>-t`.
pub(crate) fn t_basename(b: &Path) -> PathBuf {
    let mut s = b.as_os_str().to_owned();
    s.push("-t");
    PathBuf::from(s)
}

/// Compresses the transpose of the BV graph `basename` into `<basename>-t` (+ `.ef`, `.dcf`),
/// sorting with at most `mem` bytes of buffers.
pub(crate) fn write_transpose(
    basename: &Path,
    tmp: &Path,
    mem: MemoryUsage,
    pl: &mut impl ProgressLog,
) -> Result<()> {
    // `finish_bv` has just written the `.dcf`, so `open` reads the cut points from it
    let cg = graph::open(basename, ChunkConf::default())?;
    let n = cg.num_nodes();
    let pg = ParGraph::with_cutpoints(cg.graph, cg.cut.into_vec());
    let sorted = webgraph::transform::transpose_par(&pg, mem, pl)?;
    let tb = t_basename(basename);
    BvCompConf::new(&tb)
        .tmp_dir(tmp)
        .par_comp::<BE, _>(sorted)?;
    finish_bv(&tb, n)
}

/// Lower bound of the external-sort buffer reserved by [`sort_memory`] (1 GB).
pub const SORT_MEM_MIN: u64 = 1_000_000_000;
/// Upper bound of the external-sort buffer reserved by [`sort_memory`] (32 GB).
pub const SORT_MEM_MAX: u64 = 32_000_000_000;

/// Reserves the buffer of an external sort (`permute_par`, `transpose_par`, `par_sort_pairs`)
/// in `budget` and returns it as an explicit [`MemoryUsage`].
///
/// The size is half of what the budget still has available, clamped to
/// [[`SORT_MEM_MIN`], [`SORT_MEM_MAX`]]; fails with
/// [`DagError::OverBudget`] if not even [`SORT_MEM_MIN`] fits. An explicit size replaces
/// webgraph's `MemoryUsage::default()`, which is sized on the machine's total RAM
/// (≈ 96 GB on a 503 GiB host) and would ignore `--max-mem`.
pub fn sort_memory(budget: &mut MemBudget, what: &str) -> Result<MemoryUsage> {
    let bytes = (budget.available() / 2).clamp(SORT_MEM_MIN, SORT_MEM_MAX);
    budget.need(bytes, what)?;
    Ok(MemoryUsage::MemorySize(bytes as usize))
}

/// [`relabel_bv`] with webgraph's default sort memory (`MemoryUsage::default()`).
///
/// Writes `out` (BV + `.ef` + `.dcf`), its transpose `out-t` (same), `out.order.u32le` (new →
/// old) and `out.rank.u32le` (old → new); `tmp` holds the external-sort batches. Finally runs
/// [`check_identity`] on the output and fails if it is not topological (which catches an
/// inverted permutation).
pub fn relabel_bv(
    basename: &Path,
    out: &Path,
    rank: &[usize],
    tmp: &Path,
    pl: &mut impl ProgressLog,
) -> Result<()> {
    relabel_bv_with(basename, out, rank, tmp, MemoryUsage::default(), pl)
}

/// Relabels the BV graph `basename` into `out` with the **old → new** permutation `rank`,
/// sorting with at most `mem` bytes of buffers (see [`sort_memory`]).
///
/// Outputs as in [`relabel_bv`]. Heap besides the sort buffers: `4n` bytes (the inverse
/// permutation as `u32`); `rank` is passed to `permute_par` without copying and
/// `out.rank.u32le` is streamed.
pub fn relabel_bv_with(
    basename: &Path,
    out: &Path,
    rank: &[usize],
    tmp: &Path,
    mem: MemoryUsage,
    pl: &mut impl ProgressLog,
) -> Result<()> {
    let cg = graph::open(basename, ChunkConf::default())?;
    let n = cg.num_nodes();
    anyhow::ensure!(
        rank.len() == n,
        "rank has {} entries for {n} nodes",
        rank.len()
    );
    // the inverse permutation, which also checks that rank is a permutation
    let mut order = io::huge_vec(n, u32::MAX);
    for (old, &new) in rank.iter().enumerate() {
        anyhow::ensure!(
            new < n && order[new] == u32::MAX,
            "rank is not a permutation (value {new} at {old})"
        );
        order[new] = old as u32;
    }
    std::fs::create_dir_all(tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
    {
        let pg = ParGraph::with_cutpoints(cg.graph, cg.cut.into_vec());
        let sorted = webgraph::transform::permute_par(&pg, &rank, mem, &mut *pl)?;
        BvCompConf::new(out)
            .tmp_dir(tmp)
            .par_comp::<BE, _>(sorted)?;
    }
    finish_bv(out, n)?;
    write_transpose(out, tmp, mem, pl)?;
    io::write_array(&io::suffixed(out, "order.u32le"), &order)?;
    drop(order);
    let mut w = io::array_writer::<u32>(&io::suffixed(out, "rank.u32le"))?;
    for c in rank.chunks(1 << 20) {
        let c32: Vec<u32> = c.iter().map(|&r| r as u32).collect();
        w.extend(&c32)?;
    }
    w.finish()?;
    // post-check
    let cg = graph::open(out, ChunkConf::default())?;
    let rep = check_identity(
        &cg,
        None,
        &mut Option::<ConcurrentWrapper<ProgressLogger>>::None,
    )?;
    if let Some((u, v)) = rep.first_violation {
        return Err(
            anyhow::Error::new(DagError::NotTopological { u, v }).context(format!(
                "relabelled graph {} is not topological: was `order` passed instead of `rank`?",
                out.display()
            )),
        );
    }
    Ok(())
}

/// Maps a per-node array from new ids back to old ids: `out[order[i]] = values[i]`.
///
/// Heap: `n·size_of::<T>()` for the values, `4n` for `order`, `n·size_of::<T>()` for the
/// output and `n/8` for the permutation check (a bitset of the targets already written).
pub fn map_back<T: Dump>(values: &Path, order: &Path, out: &Path) -> Result<()> {
    let vals = io::read_array::<T>(values)?;
    let ord = io::read_array::<u32>(order)?;
    anyhow::ensure!(
        vals.len() == ord.len(),
        "{} has {} values, {} has {}",
        values.display(),
        vals.len(),
        order.display(),
        ord.len()
    );
    let n = vals.len();
    let Some(&fill) = vals.first() else {
        return io::write_array::<T>(out, &[]);
    };
    let mut seen = io::huge_vec(n.div_ceil(64), 0u64);
    let mut res = io::huge_vec(n, fill);
    for (&v, &o) in vals.iter().zip(&ord) {
        let o = o as usize;
        anyhow::ensure!(
            o < n && seen[o / 64] & (1 << (o % 64)) == 0,
            "{} is not a permutation",
            order.display()
        );
        seen[o / 64] |= 1 << (o % 64);
        res[o] = v;
    }
    io::write_array(out, &res)
}

/// Number of pieces each chunk is split into by [`write_induced`].
pub const INDUCED_SPLIT: usize = 16;

/// Writes the subgraph induced by the window `w` of `g` as the BV graph `out` (ids shifted by
/// `−w.lo`, + `.ef` + `.dcf`) and its transpose `out-t`; `tmp` holds the sort batches.
///
/// Because a window of a topologically numbered graph is convex, the induced subgraph keeps
/// every path between window nodes. Returns the number of nodes and arcs written.
///
/// Heap: the sort buffers (`mem`, see [`sort_memory`]) plus, per rayon thread, the arcs of the
/// piece being decoded at `16 B` per arc. Each chunk is decoded in [`INDUCED_SPLIT`] pieces of
/// equal node count, so a piece holds `chunk_arcs / INDUCED_SPLIT` arcs on average (and never
/// more than its chunk).
pub fn write_induced<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    out: &Path,
    tmp: &Path,
    mem: MemoryUsage,
    pl: &mut impl ProgressLog,
) -> Result<(usize, u64)> {
    use lender::prelude::*;
    let n = w.len();
    std::fs::create_dir_all(tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
    let graph = &g.graph;
    let (lo, hi) = (w.lo, w.hi);
    let pieces: Vec<(usize, usize)> = g
        .chunks_in(w)
        .into_iter()
        .flat_map(|(s, e)| {
            let step = (e - s).div_ceil(INDUCED_SPLIT).max(1);
            (s..e).step_by(step).map(move |a| (a, (a + step).min(e)))
        })
        .collect();
    let pairs = pieces.into_par_iter().flat_map_iter(move |(s, e)| {
        let mut arcs = Vec::new();
        let mut it = graph.iter_from(s).take(e - s);
        while let Some((u, succ)) = it.next() {
            arcs.extend(
                succ.into_iter()
                    .filter(|&v| v >= lo && v < hi)
                    .map(|v| (u - lo, v - lo)),
            );
        }
        arcs
    });
    let sorted = ParSortedGraph::config()
        .memory_usage(mem)
        .progress_logger(&mut *pl)
        .par_sort_pairs(n, pairs)?;
    BvCompConf::new(out)
        .tmp_dir(tmp)
        .par_comp::<BE, _>(sorted)?;
    finish_bv(out, n)?;
    let m = graph::load_ra(out)?.num_arcs();
    write_transpose(out, tmp, mem, pl)?;
    Ok((n, m))
}
