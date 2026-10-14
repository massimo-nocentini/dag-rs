//! Exact transitive reduction: the bitset algorithm A, the search algorithm C, and the exact
//! re-check of the arcs that fail the H3 certificate.
//!
//! For a DAG the transitive reduction `G⁻` is the unique minimal subgraph with the same
//! reachability ([\[AGU72\]](crate::references#agu72)): an arc `(u, v)` is kept iff there is
//! no path `u ⇝ v` of length `≥ 2`. Both algorithms below assume that the identity is a
//! topological order and that successor lists are ascending (true for [`Csr`]); they are the
//! output-sensitive, topological-order schemes of [\[GK79\]](crate::references#gk79),
//! [\[Sim88\]](crate::references#sim88) and [\[HMR93\]](crate::references#hmr93).
//!
//! # Algorithm A: triangular bitsets ([`transitive_reduction_bitset`])
//! ```text
//! for u = n−1 down to 0:
//!     R = ∅                                  // a bitset over ids > u
//!     for v in succ(u), ascending:           // the order is essential
//!         if v ∈ R: (u, v) is redundant
//!         else:     keep (u, v); R |= reach[v] ∪ {v}
//!         remaining_indeg[v] −= 1; free reach[v] when it reaches 0
//!     reach[u] = R                           // kept only if u has predecessors
//! ```
//! Invariant: when `v` is examined, `R` is the union of `{c} ∪ reach(c)` over the children
//! `c < v` examined so far. A path `u → c ⇝ v` has `c < v`; either `c` was kept (so
//! `v ∈ reach(c) ⊆ R`) or `c` was redundant, i.e. `c ∈ reach(c')` for a kept `c' < c`, and
//! then `reach(c) ⊆ reach(c')` ⊆ R. Rows are word-aligned (row `u` starts at word
//! `⌊(u + 1)/64⌋`), so `R |= reach[v]` is a plain word-wise OR. Time O(n + m + m⁻·n/64),
//! memory at most `n²/16` bytes; rows are freed as soon as their last predecessor is done.
//! Used for windows of at most 2¹⁶ nodes (≤ 256 MiB).
//!
//! # Algorithm C: pruned searches ([`transitive_reduction_search`])
//! In parallel over `u` (rayon), with a per-thread [`DirtyBitset`] over `(u, maxsucc(u)]`:
//! visit the children in increasing order; a marked child is redundant, an unmarked one is
//! kept and a DFS from it marks every node `≤ maxsucc(u)` it reaches. Nodes beyond the last
//! child can never witness redundancy, so the search is pruned there. Worst case O(n·m) time,
//! O(maxsucc(u) − u) bits per thread, near-linear when successor ranges are short.
//! [`redundant_arcs_search`] caps the thread count by the memory budget and returns the
//! redundant arcs straight from the per-arc keep flags.
//!
//! # Exact re-check of H3 failures ([`redundant_arcs_of`], [`exact_redundant`])
//! For a node `u` with H3 threshold `t_u` (see [`super::certificate`]), the candidates are all
//! `v ∈ succ(u)` with `v ≥ t_u`. One search from `succ(succ(u))`, restricted to ids
//! `≤ max candidate`, marks every node reachable from `u` by a path of length `≥ 2`; `(u, v)`
//! is redundant iff `v` is marked. **Completeness**: if `(u, v)` is redundant via
//! `u → x₁ → x₂ ⇝ v`, then `F[x₁] ≤ x₂ ≤ v ≤ L[u]`, so H3 fails at `u` and
//! `t_u ≤ F[x₁] ≤ v`: `v` is a candidate. Arcs below `t_u` are never redundant for the same
//! reason. [`exact_redundant`] runs these searches in parallel with a thread count capped by the
//! memory budget.

use anyhow::Result;
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::bitset::DirtyBitset;
use crate::csr::Csr;
use crate::io::{self, MemBudget};
use crate::window::Window;

/// Whether the identity is a topological order of `g` (`u < v` for every arc).
fn assert_topological(g: &Csr) {
    if let Err(e) = g.check_topological() {
        panic!("transitive reduction needs a topologically numbered DAG: {e}");
    }
}

/// Most nodes in one range of [`arc_ranges`] (bounds the work of a rayon task).
const RANGE_MAX_NODES: usize = 1 << 14;

/// Splits the nodes of `g` into consecutive ranges `[s, e)` of about `m / (64·threads)` arcs
/// (at least 1024) and at most [`RANGE_MAX_NODES`] nodes each, as rayon work units.
fn arc_ranges(g: &Csr) -> Vec<(usize, usize)> {
    let parts = 64 * rayon::current_num_threads() as u64;
    let target = (g.num_arcs() as u64 / parts).max(1024);
    let mut out = Vec::new();
    let mut s = 0;
    while s < g.n {
        let goal = g.offsets[s] + target;
        let e = (s + 1 + g.offsets[s + 1..=g.n].partition_point(|&o| o < goal))
            .min(g.n)
            .min(s + RANGE_MAX_NODES);
        out.push((s, e));
        s = e;
    }
    out
}

/// Splits `keep` (one flag per arc of `g`) into the disjoint arc slices of `ranges`.
fn split_mask<'a>(
    g: &Csr,
    ranges: &[(usize, usize)],
    mut keep: &'a mut [bool],
) -> Vec<(usize, usize, &'a mut [bool])> {
    let mut out = Vec::with_capacity(ranges.len());
    for &(s, e) in ranges {
        let (head, tail) = keep.split_at_mut((g.offsets[e] - g.offsets[s]) as usize);
        out.push((s, e, head));
        keep = tail;
    }
    out
}

/// Nodes between two progress-logger updates of the sequential bitset reduction (§11.4 rule 4:
/// update per chunk, never per node).
const UPDATE_NODES: usize = 1 << 14;

/// H3-failing nodes per rayon task (and per progress-logger update) of [`exact_redundant`].
const FAIL_CHUNK: usize = 256;

/// In-place exclusive prefix sums of `counts[1..]` (`counts[0]` must be 0), parallel over the
/// node ranges `ranges`: per-range totals, a sequential scan over the ranges, then a parallel
/// scan inside each range.
fn prefix_sums(counts: &mut [u64], ranges: &[(usize, usize)]) {
    let totals: Vec<u64> = ranges
        .par_iter()
        .map(|&(s, e)| counts[s + 1..=e].iter().sum())
        .collect();
    let mut acc = 0;
    let starts: Vec<u64> = totals
        .iter()
        .map(|&t| {
            let a = acc;
            acc += t;
            a
        })
        .collect();
    let mut rest = &mut counts[1..];
    let mut parts = Vec::with_capacity(ranges.len());
    for (&(s, e), &a) in ranges.iter().zip(&starts) {
        let (head, tail) = rest.split_at_mut(e - s);
        parts.push((a, head));
        rest = tail;
    }
    parts.into_par_iter().for_each(|(mut a, part)| {
        for c in part {
            a += *c;
            *c = a;
        }
    });
}

/// The subgraph of `g` made of the arcs `i` with `keep[i]`, written straight into a flat
/// [`Csr`]: kept degrees and their (parallel) prefix sums, then a parallel copy into disjoint
/// slices of the successor array. No per-node allocation (§11.4 rule 6); heap
/// `8(n + 1) + 4m⁻` bytes.
fn csr_from_mask(g: &Csr, keep: &[bool], ranges: &[(usize, usize)]) -> Csr {
    let n = g.n;
    let mut offsets = io::huge_vec(n + 1, 0u64);
    offsets[1..].par_iter_mut().enumerate().for_each(|(u, o)| {
        let (a, b) = (g.offsets[u] as usize, g.offsets[u + 1] as usize);
        *o = keep[a..b].iter().filter(|&&k| k).count() as u64;
    });
    prefix_sums(&mut offsets, ranges);
    let mut succ = io::huge_vec(offsets[n] as usize, 0u32);
    let mut rest = &mut succ[..];
    let mut parts = Vec::with_capacity(ranges.len());
    for &(s, e) in ranges {
        let (head, tail) = rest.split_at_mut((offsets[e] - offsets[s]) as usize);
        parts.push((s, e, head));
        rest = tail;
    }
    parts.into_par_iter().for_each(|(s, e, out)| {
        let (a, b) = (g.offsets[s] as usize, g.offsets[e] as usize);
        let kept = g.succ[a..b]
            .iter()
            .zip(&keep[a..b])
            .filter(|&(_, &k)| k)
            .map(|(&v, _)| v);
        for (slot, v) in out.iter_mut().zip(kept) {
            *slot = v;
        }
    });
    Csr {
        n,
        offsets: offsets.into_boxed_slice(),
        succ: succ.into_boxed_slice(),
    }
}

/// The arcs `i` of `g` with `!keep[i]`, as `(u + lo, v + lo)`, sorted: a parallel count per
/// node range, then a parallel fill of disjoint slices of one exact-size vector (`8` bytes per
/// dropped arc, no other allocation).
fn dropped_arcs(g: &Csr, keep: &[bool], ranges: &[(usize, usize)], lo: usize) -> Vec<(u32, u32)> {
    let counts: Vec<usize> = ranges
        .par_iter()
        .map(|&(s, e)| {
            let (a, b) = (g.offsets[s] as usize, g.offsets[e] as usize);
            keep[a..b].iter().filter(|&&k| !k).count()
        })
        .collect();
    let mut out = io::huge_vec(counts.iter().sum(), (0u32, 0u32));
    let mut rest = &mut out[..];
    let mut parts = Vec::with_capacity(ranges.len());
    for (&(s, e), &c) in ranges.iter().zip(&counts) {
        let (head, tail) = rest.split_at_mut(c);
        parts.push((s, e, head));
        rest = tail;
    }
    parts.into_par_iter().for_each(|(s, e, part)| {
        let mut slots = part.iter_mut();
        for u in s..e {
            let a = g.offsets[u] as usize;
            for (&v, _) in g.succ(u).iter().zip(&keep[a..]).filter(|&(_, &k)| !k) {
                *slots.next().expect("dropped-arc count") = ((u + lo) as u32, v + lo as u32);
            }
        }
    });
    out
}

/// Heap bytes of the output of [`transitive_reduction_bitset`] and
/// [`transitive_reduction_search`] for an input with `n` nodes and `m` arcs, in the worst case
/// `m⁻ = m`: the per-arc keep flags (`m` bytes) and the flat output [`Csr`]
/// (`8(n + 1) + 4m⁻` bytes). The input is not included.
pub fn reduction_output_bytes(n: usize, m: usize) -> u64 {
    8 * (n as u64 + 1) + 5 * m as u64
}

/// Upper bound on the working heap of [`transitive_reduction_bitset`] for `n` nodes, in bytes:
/// the triangular rows (`≤ n²/16` plus one word and a 24-byte vector header per row) and the
/// remaining in-degrees. The output is bounded separately by [`reduction_output_bytes`].
pub fn bitset_bytes(n: usize) -> u64 {
    let n = n as u64;
    let words = n.div_ceil(64);
    // Σ_u (words − ⌊(u + 1)/64⌋) ≤ n·words/2 + n words, 8 bytes each; headers; in-degrees
    4 * words * n + 8 * n + 24 * n + 4 * n
}

/// The transitive reduction by triangular bitsets (algorithm A of the module documentation).
///
/// Time O(n + m + m⁻·n/64), memory at most [`bitset_bytes`]`(n)` plus
/// [`reduction_output_bytes`]; meant for `n ≤ 2¹⁶`.
///
/// # Panics
/// If the identity is not a topological order of `g`.
pub fn transitive_reduction_bitset(g: &Csr) -> Csr {
    transitive_reduction_bitset_logged(g, &mut Option::<ProgressLogger>::None)
}

/// [`transitive_reduction_bitset`] with a progress logger, updated once every 2¹⁴ nodes.
///
/// # Panics
/// If the identity is not a topological order of `g`.
pub fn transitive_reduction_bitset_logged(g: &Csr, pl: &mut impl ProgressLog) -> Csr {
    let keep = keep_mask_bitset(g, pl);
    let tr = csr_from_mask(g, &keep, &arc_ranges(g));
    pl.done();
    tr
}

/// The keep flags (one per arc of `g`) of algorithm A; starts `pl` but does not stop it.
fn keep_mask_bitset(g: &Csr, pl: &mut impl ProgressLog) -> Vec<bool> {
    assert_topological(g);
    let n = g.n;
    let nw = n.div_ceil(64);
    pl.item_name("node").expected_updates(Some(n));
    pl.start("tr: triangular-bitset reduction");
    let mut rem = vec![0u32; n];
    for &v in g.succ.iter() {
        rem[v as usize] += 1;
    }
    let mut keep = io::huge_vec(g.num_arcs(), false);
    let mut rows: Vec<Vec<u64>> = vec![Vec::new(); n];
    let mut pending = 0;
    for u in (0..n).rev() {
        let bu = (u + 1) / 64;
        let mut r = vec![0u64; nw - bu];
        let base = g.offsets[u] as usize;
        for (i, &v) in g.succ(u).iter().enumerate() {
            let v = v as usize;
            let (wv, bv) = (v / 64 - bu, 1u64 << (v % 64));
            if r[wv] & bv == 0 {
                keep[base + i] = true;
                let row = &rows[v];
                let off = (v + 1) / 64 - bu;
                for (x, &y) in r[off..].iter_mut().zip(row) {
                    *x |= y;
                }
                r[wv] |= bv;
            }
            rem[v] -= 1;
            if rem[v] == 0 {
                rows[v] = Vec::new();
            }
        }
        if rem[u] > 0 {
            rows[u] = r;
        }
        pending += 1;
        if pending == UPDATE_NODES {
            pl.update_with_count(pending);
            pending = 0;
        }
    }
    pl.update_with_count(pending);
    keep
}

/// Marks in `bits` every node `≤ maxc` reachable from the nodes already on `stack` (each of
/// which is already marked), by a DFS over `succ`; marked nodes `< maxc` are expanded once.
#[inline]
fn mark_reachable<I: IntoIterator<Item = usize>>(
    succ: impl Fn(usize) -> I,
    maxc: usize,
    bits: &mut DirtyBitset,
    stack: &mut Vec<u32>,
) {
    while let Some(y) = stack.pop() {
        for z in succ(y as usize).into_iter().take_while(|&z| z <= maxc) {
            if bits.insert(z) && z < maxc {
                stack.push(z as u32);
            }
        }
    }
}

/// The transitive reduction by pruned searches (algorithm C of the module documentation),
/// parallel over arc-balanced node ranges with one [`DirtyBitset`] per rayon task, on the
/// current rayon pool.
///
/// Memory: [`search_bytes`]`(`[`search_span`]`(g))` per thread plus
/// [`reduction_output_bytes`]; the kept arcs are flagged in a per-arc mask and copied once into
/// the flat output. [`redundant_arcs_search`] is the budgeted variant that caps the threads.
///
/// # Panics
/// If the identity is not a topological order of `g`.
pub fn transitive_reduction_search(g: &Csr) -> Csr {
    transitive_reduction_search_logged(g, &mut Option::<ConcurrentWrapper<ProgressLogger>>::None)
}

/// [`transitive_reduction_search`] with a concurrent progress logger, updated once per range
/// of nodes.
///
/// # Panics
/// If the identity is not a topological order of `g`.
pub fn transitive_reduction_search_logged(g: &Csr, pl: &mut impl ConcurrentProgressLog) -> Csr {
    let ranges = arc_ranges(g);
    let keep = keep_mask_search(g, &ranges, pl);
    let tr = csr_from_mask(g, &keep, &ranges);
    pl.done();
    tr
}

/// The keep flags (one per arc of `g`) of algorithm C over `ranges` (from [`arc_ranges`]);
/// starts `pl` but does not stop it.
fn keep_mask_search(
    g: &Csr,
    ranges: &[(usize, usize)],
    pl: &mut impl ConcurrentProgressLog,
) -> Vec<bool> {
    assert_topological(g);
    pl.item_name("node").expected_updates(Some(g.n));
    pl.start("tr: search reduction");
    let mut keep = io::huge_vec(g.num_arcs(), false);
    split_mask(g, ranges, &mut keep)
        .into_par_iter()
        .for_each_with(
            (pl.clone(), DirtyBitset::new(), Vec::<u32>::new()),
            |(pl, bits, stack), (s, e, mask)| {
                let first = g.offsets[s];
                for u in s..e {
                    let succ = g.succ(u);
                    let base = (g.offsets[u] - first) as usize;
                    let mask = &mut mask[base..base + succ.len()];
                    let Some(&maxc) = succ.last() else {
                        continue;
                    };
                    if succ.len() == 1 {
                        mask[0] = true;
                        continue;
                    }
                    let maxc = maxc as usize;
                    bits.reset(u + 1, maxc - u);
                    for (k, &v) in mask.iter_mut().zip(succ) {
                        let v = v as usize;
                        if bits.contains(v) {
                            continue;
                        }
                        *k = true;
                        bits.insert(v);
                        if v < maxc {
                            stack.clear();
                            stack.push(v as u32);
                            mark_reachable(
                                |x| g.succ(x).iter().map(|&y| y as usize),
                                maxc,
                                bits,
                                stack,
                            );
                        }
                    }
                }
                pl.update_with_count(e - s);
            },
        );
    keep
}

/// The largest successor span `max (maxsucc(u) − u)` of `g` (0 if `g` has no arcs), computed in
/// parallel: the id range one search of algorithm C covers.
pub fn search_span(g: &Csr) -> usize {
    (0..g.n)
        .into_par_iter()
        .filter_map(|u| g.succ(u).last().map(|&v| v as usize - u))
        .max()
        .unwrap_or(0)
}

/// Result of [`redundant_arcs_bitset`] and [`redundant_arcs_search`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InMemoryOutcome {
    /// Redundant arcs `(u + lo, v + lo)`, sorted.
    pub redundant: Vec<(u32, u32)>,
    /// Threads used (1 for the sequential bitset algorithm).
    pub threads: usize,
}

/// The redundant arcs of `g` (a window loaded as a [`Csr`], ids shifted by `−lo`) by
/// triangular bitsets (algorithm A), as absolute ids `(u + lo, v + lo)`, sorted.
///
/// Reserves in `budget` the working memory [`bitset_bytes`]`(n)`, the keep flags (`m` bytes)
/// and the redundant-arc list (8 bytes per arc, once its size is known); everything is released
/// on return, so the caller accounts for the returned list. Fails with
/// [`DagError::OverBudget`](crate::DagError::OverBudget) without allocating if it does not fit.
///
/// # Panics
/// If the identity is not a topological order of `g`.
pub fn redundant_arcs_bitset(
    g: &Csr,
    lo: usize,
    budget: &mut MemBudget,
    pl: &mut impl ProgressLog,
) -> Result<InMemoryOutcome> {
    let work = bitset_bytes(g.n) + g.num_arcs() as u64;
    budget.need(work, "tr: triangular bitsets and keep flags")?;
    let keep = keep_mask_bitset(g, pl);
    pl.done();
    let res = collect_dropped(g, &keep, &arc_ranges(g), lo, budget);
    budget.release(work);
    Ok(InMemoryOutcome {
        redundant: res?,
        threads: 1,
    })
}

/// The redundant arcs of `g` (a window loaded as a [`Csr`], ids shifted by `−lo`) by pruned
/// searches (algorithm C), as absolute ids `(u + lo, v + lo)`, sorted.
///
/// Reserves the keep flags (`m` bytes), then as many search threads as fit in `budget` at
/// [`search_bytes`]`(`[`search_span`]`(g))` each (at most the rayon pool size; the searches run
/// in a smaller pool when fewer fit), then the redundant-arc list (8 bytes per arc); all are
/// released on return. Fails with [`DagError::OverBudget`](crate::DagError::OverBudget) only
/// if not even one thread fits.
///
/// # Panics
/// If the identity is not a topological order of `g`.
pub fn redundant_arcs_search(
    g: &Csr,
    lo: usize,
    budget: &mut MemBudget,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<InMemoryOutcome> {
    let mask = g.num_arcs() as u64;
    budget.need(mask, "tr: keep flags (1 B/arc)")?;
    let per = search_bytes(search_span(g));
    let threads = match reserve_threads(per, budget, "tr: search reduction (per-thread bitsets)") {
        Ok(t) => t,
        Err(e) => {
            budget.release(mask);
            return Err(e);
        }
    };
    log::info!("tr: search reduction with {threads} threads ({per} B per thread)");
    let mut tpl = pl.clone();
    let keep = install(threads, move || {
        let ranges = arc_ranges(g);
        let keep = keep_mask_search(g, &ranges, &mut tpl);
        tpl.done();
        keep
    });
    budget.release(per * threads as u64);
    let res = keep.and_then(|keep| collect_dropped(g, &keep, &arc_ranges(g), lo, budget));
    budget.release(mask);
    Ok(InMemoryOutcome {
        redundant: res?,
        threads,
    })
}

/// [`dropped_arcs`] with the list (8 bytes per arc) reserved in `budget` while it is built and
/// released on return.
fn collect_dropped(
    g: &Csr,
    keep: &[bool],
    ranges: &[(usize, usize)],
    lo: usize,
    budget: &mut MemBudget,
) -> Result<Vec<(u32, u32)>> {
    let bytes = 8 * keep.par_iter().filter(|&&k| !k).count() as u64;
    budget.need(bytes, "tr: redundant-arc list (8 B/arc)")?;
    let red = dropped_arcs(g, keep, ranges, lo);
    budget.release(bytes);
    Ok(red)
}

/// Reserves `per` bytes per thread for the largest thread count that fits `budget` (between 1
/// and the current rayon pool size) and returns it; fails with
/// [`DagError::OverBudget`](crate::DagError::OverBudget) if not even one thread fits.
fn reserve_threads(per: u64, budget: &mut MemBudget, what: &str) -> Result<usize> {
    let pool = rayon::current_num_threads();
    let threads = ((budget.available() / per.max(1)) as usize).clamp(1, pool);
    budget.need(per * threads as u64, what)?;
    Ok(threads)
}

/// Runs `f` on a dedicated pool of `threads` threads when that is fewer than the current pool,
/// otherwise on the current pool.
fn install<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> Result<R> {
    if threads < rayon::current_num_threads() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| anyhow::Error::new(e).context("cannot build the search pool"))?;
        Ok(pool.install(f))
    } else {
        Ok(f())
    }
}

/// The redundant arcs `(u, v)` with `v ≥ from`, as the sorted list of their heads `v`, for the
/// whole graph (see [`redundant_arcs_in`]).
pub fn redundant_arcs_of<G: RandomAccessGraph>(
    g: &G,
    u: usize,
    from: usize,
    bits: &mut DirtyBitset,
    stack: &mut Vec<u32>,
) -> Vec<u32> {
    redundant_arcs_in(g, u, from, g.num_nodes(), bits, stack).1
}
/// The arcs `(u, v)` of the subgraph induced by `[lo, hi)` (for any `lo ≤ u`) with `v ≥ from`
/// that are redundant, i.e. implied by a path of length `≥ 2`. Returns the number of candidate
/// arcs examined (`v ∈ succ(u)`, `from ≤ v < hi`) and the sorted heads of the redundant ones.
///
/// One search from `succ(succ(u))` restricted to ids `≤` the largest candidate, with `bits`
/// reset over `(u, max candidate]`; stops early once every candidate is marked. Because ids are
/// topological, paths between window nodes stay in the window, so the result is the same for
/// the induced subgraph and for the whole graph. `g`'s successor lists must be ascending (BV
/// graphs, `VecGraph`).
pub fn redundant_arcs_in<G: RandomAccessGraph>(
    g: &G,
    u: usize,
    from: usize,
    hi: usize,
    bits: &mut DirtyBitset,
    stack: &mut Vec<u32>,
) -> (u64, Vec<u32>) {
    let s: Vec<u32> = g
        .successors(u)
        .into_iter()
        .take_while(|&v| v < hi)
        .map(|v| v as u32)
        .collect();
    let first = s.partition_point(|&v| (v as usize) < from);
    let cands = &s[first..];
    let Some(&maxc) = cands.last() else {
        return (0, Vec::new());
    };
    let maxc = maxc as usize;
    bits.reset(u + 1, maxc - u);
    stack.clear();
    let is_cand = |y: usize| y >= from && cands.binary_search(&(y as u32)).is_ok();
    let mut left = cands.len();
    'outer: for &x in &s {
        let x = x as usize;
        if x >= maxc {
            break;
        }
        for y in g.successors(x).into_iter().take_while(|&y| y <= maxc) {
            if bits.insert(y) {
                if is_cand(y) {
                    left -= 1;
                }
                if y < maxc {
                    stack.push(y as u32);
                }
            }
        }
        // expand, counting the candidates reached
        while let Some(y) = stack.pop() {
            for z in g
                .successors(y as usize)
                .into_iter()
                .take_while(|&z| z <= maxc)
            {
                if bits.insert(z) {
                    if is_cand(z) {
                        left -= 1;
                    }
                    if z < maxc {
                        stack.push(z as u32);
                    }
                }
            }
            if left == 0 {
                break 'outer;
            }
        }
        if left == 0 {
            break;
        }
    }
    let red = cands
        .iter()
        .copied()
        .filter(|&v| bits.contains(v as usize))
        .collect();
    (cands.len() as u64, red)
}

/// Result of [`exact_redundant`].
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ExactOutcome {
    /// Candidate arcs examined (`v ∈ succ_W(u)`, `v ≥ t_u`, over all H3-failing `u`).
    pub checked_arcs: u64,
    /// Threads used (capped by the memory budget).
    pub threads: usize,
    /// Redundant arcs `(u, v)` (absolute ids), sorted.
    #[serde(skip)]
    pub redundant: Vec<(u32, u32)>,
}

/// DFS-stack entries reserved per search thread by [`search_bytes`] (2²⁰ ids, 4 MiB).
pub const STACK_RESERVE_IDS: usize = 1 << 20;

/// Heap bytes reserved for one search over a span of `span` ids: the bitset (`span/8`), its
/// dirty-word list (at most one `u32` per word, `span/16`) and a DFS stack of
/// `min(span, `[`STACK_RESERVE_IDS`]`)` ids.
///
/// The bitset and the dirty list are exact worst cases. The stack is not: it holds at most one
/// entry per marked node (`≤ span`), but only the pending frontier of a depth-first search is
/// ever on it, which in practice is orders of magnitude smaller than the span (on the
/// Payment-Graph window `0..10⁸`, whose span is close to 10⁸, the whole `tr reduce --algo
/// search` run peaked at 7 GB of anonymous memory with 112 threads). Reserving `4·span` per thread would refuse such runs
/// for no reason, so the stack is reserved up to [`STACK_RESERVE_IDS`] and grows on demand
/// beyond it, like the [`DirtyBitset`] storage (a search thread is sized by its bitset,
/// `(maxsucc(u) − u)/8` bytes).
pub fn search_bytes(span: usize) -> u64 {
    let s = span as u64;
    s / 8 + s / 16 + 4 * span.min(STACK_RESERVE_IDS) as u64 + 64
}

/// Exact re-check of the H3 failures `(u, t_u)` of the window `w` (see the module
/// documentation): runs [`redundant_arcs_in`] for each failing node in parallel, in tasks of
/// 256 failing nodes; `pl` is updated once per task.
///
/// `l` is the `maxsucc` array of the window (indexed by `u − lo`). The thread count is the
/// largest that fits `budget` at [`search_bytes`]`(max (L[u] − u))` per thread (at most the
/// rayon pool size); the reservation is released on return. Fails with
/// [`DagError::OverBudget`](crate::DagError::OverBudget) if not even one thread fits.
///
/// The returned list of redundant arcs (8 bytes per arc, `O(m)` on an arbitrary DAG, empty on
/// the Payment Graph) is materialized before its size is known and is **not** reserved here:
/// the caller owns it and accounts for it (the `tr` command reserves it before sorting).
pub fn exact_redundant<G: RandomAccessGraph + Sync>(
    g: &G,
    w: Window,
    failures: &[(u32, u32)],
    l: &[u32],
    budget: &mut MemBudget,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<ExactOutcome> {
    if failures.is_empty() {
        return Ok(ExactOutcome {
            threads: 0,
            ..Default::default()
        });
    }
    let span = failures
        .par_iter()
        .map(|&(u, _)| (l[u as usize - w.lo] - u) as usize)
        .max()
        .unwrap_or(0);
    let per = search_bytes(span);
    let threads = reserve_threads(per, budget, "tr: exact searches (per-thread bitsets)")?;
    pl.item_name("node").expected_updates(Some(failures.len()));
    pl.start(format!(
        "tr: exact check of {} H3-failing nodes ({threads} threads)",
        failures.len()
    ));
    let tpl = pl.clone();
    let run = move || -> (u64, Vec<(u32, u32)>) {
        failures
            .par_chunks(FAIL_CHUNK)
            .map_with(
                (tpl, DirtyBitset::new(), Vec::<u32>::new()),
                |(pl, bits, stack), chunk| {
                    let mut checked = 0;
                    let mut red = Vec::new();
                    for &(u, t) in chunk {
                        let (c, r) =
                            redundant_arcs_in(g, u as usize, t as usize, w.hi, bits, stack);
                        checked += c;
                        red.extend(r.into_iter().map(|v| (u, v)));
                    }
                    pl.update_with_count(chunk.len());
                    (checked, red)
                },
            )
            .reduce(
                || (0, Vec::new()),
                |mut a, mut b| {
                    a.1.append(&mut b.1);
                    (a.0 + b.0, a.1)
                },
            )
    };
    let res = install(threads, run);
    budget.release(per * threads as u64);
    pl.done();
    let (checked_arcs, mut redundant) = res?;
    redundant.par_sort_unstable();
    log::info!(
        "tr: exact check of {} nodes ({checked_arcs} candidate arcs, {threads} threads): {} redundant arcs",
        failures.len(),
        redundant.len()
    );
    Ok(ExactOutcome {
        checked_arcs,
        threads,
        redundant,
    })
}
