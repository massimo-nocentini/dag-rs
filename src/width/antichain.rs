//! Frontier antichains, the exact closure width on the BlockIndex, antichain certificates and
//! Fulkerson's small-graph oracle.
//!
//! # Frontier (a lower bound on the width, corrected for windows)
//! For a window `W = [lo, hi)` and a cut `c ∈ [lo, hi]`,
//! `A_c = {u ∈ [lo, c) : minsucc_W(u) ≥ c or u has no successor in W}` is an antichain: on a
//! path `u ⇝ v` with `u < v < c` the first arc leaves `u` towards an id `≥ c` and ids
//! increase, so `v ≥ c`. With `cnt[x] = #{u ∈ W : minsucc_W(u) = x}`,
//! ```text
//! |A_c| = (c − lo) − Σ_{x < c} cnt[x]          frontier = max_c |A_c| (smallest c on ties)
//! ```
//! [`frontier_generic`] builds `cnt` with one parallel decode of the window (4 B/node) and
//! scans it; [`frontier_blocks`] uses `cnt[first[B]] = #{u ∈ W : spend[u] = B, B < bhi}`, the
//! per-window input histogram (**not** `nin`), and only evaluates the cuts `c = first[B]` and
//! `c = hi`: between two block starts `|A_c|` grows by one per node, so the smallest maximizing
//! cut ends an increasing run, i.e. it is `hi` or a block start with `cnt > 0`. On
//! the PG `|A_c|` at a transaction boundary is the number of outputs created and not spent
//! before `c` (the graph's UTXO count). With `materialize`, `A_c*` is collected; both functions
//! leave the verification to the caller (the `frontier` subcommand checks `|A_c*| == size` and,
//! on blocks, runs [`check_antichain_blocks`], exiting with status 1 on a failure), with a
//! `debug_assert` of the size in debug builds.
//!
//! # Exact width of a PG window: maximum-weight closure ([\[Pic76\]](crate::references#pic76))
//! For a line digraph `L(TG)` of an acyclic transaction graph, `width = max_S Σ_{T ∈ S}
//! (o_T − i_T)` over the down-sets `S` of transactions, and `δ⁺(S*)` (outputs created in `S*`
//! and spent outside it, or unspent) is a maximum antichain (after
//! [\[HN60\]](crate::references#hn60)). For a window, from the BlockIndex only:
//! ```text
//! vertices  blocks B with first[B] ∈ [lo, hi)       weight w_B = o_B^W − i_B^W
//! constant  window nodes with create = NONE or a block starting before lo (always in S*)
//! edges     s → B (cap w_B) if w_B > 0;  B → t (cap −w_B) if w_B < 0
//!           B → create[u] (∞) for every window input u of B with first[create[u]] ≥ lo
//! width     = constant + Σ_{w_B > 0} w_B − maxflow        S* = residual side of s
//! antichain = window nodes created in S* (or source-like) whose spender is not in S*,
//!             is NONE, or starts at or after hi
//! ```
//! The result is **self-certified**: [`check_antichain_blocks`] verifies in one forward scan
//! of the window that no member of `A` reaches another (a block is *reached* iff one of its
//! inputs is in `A` or is a reached output; a member that is an output of a reached block
//! fails), and the caller checks `|A| == width`. Since `A` is an antichain, `width ≥ |A|`, and
//! the closure value is an upper bound by the theorem, so equality certifies both (the CLI also
//! cross-checks against [`width_min_flow`](super::flow::width_min_flow) on windows of at most
//! 10⁶ nodes). Network: one vertex per block of the window, at most one ∞ edge per
//! (block, creating block) pair after deduplication.
//!
//! # Generic certificates and oracle
//! [`check_antichain`] marks, in id order, every node reached from `A` in one pass over a
//! [`Csr`] with topological ids; [`check_width_certificate`] adds the chain checks.
//! [`width_fulkerson`] is Fulkerson's reduction ([\[Ful56\]](crate::references#ful56)):
//! bitset transitive closure, [`hopcroft_karp`] on its split graph, and the antichain
//! `{v : v⁺ ∉ C, v⁻ ∉ C}` from the Kőnig cover `C` ([\[Kon31\]](crate::references#kon31));
//! it needs `n²/8` bytes and is meant for small graphs (tests, `n ≤ 2¹⁵`).

use anyhow::{Result, bail, ensure};
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use sux::traits::BitVecOps;
use webgraph::prelude::*;

use super::flow::{Chains, Dinic, FlowAlgo, INF, WidthResult, check_chains};
use super::matching::{hopcroft_karp, konig_cover};
use super::pathcover::{input_histogram, outputs_in};
use crate::blocks::BlockIndex;
use crate::csr::Csr;
use crate::error::DagError;
use crate::graph::ChunkedGraph;
use crate::io::{NONE, as_atomic_u32, huge_vec};
use crate::sweep::par_chunks;
use crate::window::Window;

/// The largest prefix antichain of a window.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Frontier {
    /// `max_c |A_c|`.
    pub size: u64,
    /// The maximizing cut `c` (absolute id; the smallest one on ties, as in the reference
    /// scan of `pg-t`).
    pub cut: usize,
    /// The members of `A_c` (absolute ids, ascending), when materialized.
    pub members: Option<Vec<u32>>,
}

/// Whether `cand` beats `best` as `(size, cut)`: a larger size, or the same size at a smaller
/// cut.
#[inline]
fn better(cand: (i64, usize), best: (i64, usize)) -> bool {
    cand.0 > best.0 || (cand.0 == best.0 && cand.1 < best.1)
}

/// Frontier of the window `w` of any DAG with topological ids (see the module
/// documentation): one parallel decode of the window plus a sequential scan of a 4 B/node
/// histogram, and with `materialize` a second parallel decode of `[lo, c*)` (the caller
/// verifies `members.len() == size`).
///
/// Fails with [`DagError::NotTopological`] if an arc goes backwards.
pub fn frontier_generic<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    materialize: bool,
    pl: &mut impl ConcurrentProgressLog,
) -> Result<Frontier> {
    let (lo, hi) = (w.lo, w.hi);
    let mut cnt = huge_vec(w.len(), 0u32);
    {
        let c = as_atomic_u32(&mut cnt);
        pl.item_name("node").expected_updates(Some(w.len()));
        pl.start(format!("frontier: first successors in {w}"));
        par_chunks(
            g,
            w,
            pl,
            |it| {
                while let Some((u, s)) = it.next_node() {
                    if let Some(&x) = s.first() {
                        let x = x as usize;
                        if x <= u {
                            return Err(DagError::NotTopological { u, v: x }.into());
                        }
                        c[x - lo].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                Ok(())
            },
            |_, _| (),
        )?;
        pl.done();
    }
    let mut best = (0i64, lo);
    let mut alive = 0i64;
    for c in lo + 1..=hi {
        alive += 1 - cnt[c - 1 - lo] as i64;
        if better((alive, c), best) {
            best = (alive, c);
        }
    }
    drop(cnt);
    let (size, cut) = (best.0 as u64, best.1);
    let members = if materialize {
        pl.expected_updates(Some(cut - lo));
        pl.start(format!("frontier: materializing A_c for c = {cut}"));
        let mut a: Vec<u32> = par_chunks(
            g,
            Window { lo, hi: cut },
            pl,
            |it| {
                let mut v = Vec::new();
                while let Some((u, s)) = it.next_node() {
                    if s.is_empty() {
                        v.push(u as u32);
                    }
                }
                Ok(v)
            },
            |mut a, mut b| {
                a.append(&mut b);
                a
            },
        )?;
        pl.done();
        a.par_sort_unstable();
        debug_assert_eq!(a.len() as u64, size, "frontier: materialized size");
        Some(a)
    } else {
        None
    };
    Ok(Frontier { size, cut, members })
}

/// Frontier of the window `w` of a line digraph from its [`BlockIndex`] (see the module
/// documentation): one parallel scan of `spend[lo..hi)` and of the window's blocks, no BV
/// decoding. Memory: 4 B per block of the window (plus the members with `materialize`; the
/// caller verifies them, e.g. with [`check_antichain_blocks`] and `members.len() == size`).
pub fn frontier_blocks(blocks: &BlockIndex, w: Window, materialize: bool) -> Frontier {
    let (blo, bhi, hist) = input_histogram(blocks, w);
    let lo = w.lo;
    // per range of blocks: (Σ hist, best (alive relative to the range start, cut))
    const STEP: usize = 1 << 20;
    let parts: Vec<(i64, Option<(i64, usize)>)> = (blo..bhi)
        .step_by(STEP)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|b0| {
            let mut sum = 0i64;
            let mut best: Option<(i64, usize)> = None;
            for b in b0..(b0 + STEP).min(bhi) {
                let c = blocks.first[b] as usize;
                let cand = ((c - lo) as i64 - sum, c);
                if best.is_none_or(|x| better(cand, x)) {
                    best = Some(cand);
                }
                sum += hist[b - blo] as i64;
            }
            (sum, best)
        })
        .collect();
    let mut best = (0i64, lo);
    let mut before = 0i64;
    for (sum, local) in parts {
        if let Some((a, c)) = local {
            let cand = (a - before, c);
            if better(cand, best) {
                best = cand;
            }
        }
        before += sum;
    }
    let at_hi = ((w.hi - lo) as i64 - before, w.hi);
    if better(at_hi, best) {
        best = at_hi;
    }
    drop(hist);
    let (size, cut) = (best.0 as u64, best.1);
    let members = materialize.then(|| {
        let a: Vec<u32> = (lo..cut)
            .into_par_iter()
            .with_min_len(1 << 16)
            .filter(|&u| match blocks.spend[u] {
                NONE => true,
                b => blocks.first[b as usize] as usize >= cut,
            })
            .map(|u| u as u32)
            .collect();
        debug_assert_eq!(a.len() as u64, size, "frontier: materialized size");
        a
    });
    Frontier { size, cut, members }
}

/// The frontier curve `c ↦ |A_c|` over one range of cuts `[a, b]` (absolute ids,
/// `lo ≤ a ≤ b ≤ hi`), see [`frontier_curve_blocks`]. On the PG `|A_c|` is the number of
/// outputs created before `c` and not spent before `c` (the UTXO set at the cut).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CutRange {
    /// Label of the range (e.g. the half-year of a bucket).
    pub label: String,
    /// First cut `a`.
    pub lo: usize,
    /// Last cut `b` (inclusive).
    pub hi: usize,
    /// `|A_a|`.
    pub at_lo: u64,
    /// `|A_b|`.
    pub at_hi: u64,
    /// `max_{a ≤ c ≤ b} |A_c|`.
    pub max: u64,
    /// The smallest maximizing cut.
    pub argmax: usize,
}

/// The ranges of cuts `[max(lo, x), min(hi, y)]` of the node ranges `(label, x, y)` that
/// intersect the window `[lo, hi]` (as cut sets, so a range ending at `lo` is kept).
pub fn cut_ranges(w: Window, ranges: &[(String, usize, usize)]) -> Vec<(String, usize, usize)> {
    ranges
        .iter()
        .filter(|(_, a, b)| *a <= w.hi && *b >= w.lo && a <= b)
        .map(|(l, a, b)| (l.clone(), (*a).max(w.lo), (*b).min(w.hi)))
        .collect()
}

/// [`CutRange`] statistics of the frontier curve of the window `w` of a line digraph for each
/// range of cuts (clipped to the window with [`cut_ranges`]), from its [`BlockIndex`]: the
/// window input histogram of [`frontier_blocks`], prefix sums per 2²⁰ blocks, then one
/// sequential scan per range (ranges in parallel). As in [`frontier_blocks`], the maximum over
/// `[a, b]` is attained at `a`, `b` or a block start in `(a, b]`. Time `O(n_W + nb_W)` plus
/// `O(2²⁰)` per range; memory 4 B per block of the window.
pub fn frontier_curve_blocks(
    blocks: &BlockIndex,
    w: Window,
    ranges: &[(String, usize, usize)],
) -> Vec<CutRange> {
    let ranges = cut_ranges(w, ranges);
    if ranges.is_empty() {
        return Vec::new();
    }
    let (blo, bhi, hist) = input_histogram(blocks, w);
    const STEP: usize = 1 << 20;
    let group: Vec<i64> = hist
        .par_chunks(STEP)
        .map(|c| c.iter().map(|&x| i64::from(x)).sum())
        .collect();
    // Σ hist over the window blocks [blo, x)
    let prefix = |x: usize| -> i64 {
        let i = x - blo;
        group[..i / STEP].iter().sum::<i64>()
            + hist[i / STEP * STEP..i]
                .iter()
                .map(|&x| i64::from(x))
                .sum::<i64>()
    };
    // the first window block whose first output is ≥ c
    let first_ge = |c: usize| blo + blocks.first[blo..bhi].partition_point(|&f| (f as usize) < c);
    let lo = w.lo;
    ranges
        .into_par_iter()
        .map(|(label, a, b)| {
            let mut x = first_ge(a);
            // S(c) = Σ hist of the window blocks starting before c; |A_c| = (c − lo) − S(c)
            let mut sum = prefix(x);
            let alive = |c: usize, sum: i64| (c - lo) as i64 - sum;
            let at_lo = alive(a, sum);
            let mut best = (at_lo, a);
            // a block starting exactly at a counts for every later cut
            while x < bhi && (blocks.first[x] as usize) <= a {
                sum += i64::from(hist[x - blo]);
                x += 1;
            }
            // candidates: the block starts in (a, b], each with S(first[x]) = sum_at
            let mut sum_at = sum;
            while x < bhi && (blocks.first[x] as usize) <= b {
                let c = blocks.first[x] as usize;
                let v = alive(c, sum_at);
                if better((v, c), best) {
                    best = (v, c);
                }
                sum_at += i64::from(hist[x - blo]);
                x += 1;
            }
            // sum_at covers the blocks with first ≤ b; |A_b| excludes one starting exactly at b
            let at_b = x > blo && blocks.first[x - 1] as usize == b;
            let at_hi = alive(
                b,
                sum_at
                    - if at_b {
                        i64::from(hist[x - 1 - blo])
                    } else {
                        0
                    },
            );
            if better((at_hi, b), best) {
                best = (at_hi, b);
            }
            CutRange {
                label,
                lo: a,
                hi: b,
                at_lo: at_lo as u64,
                at_hi: at_hi as u64,
                max: best.0 as u64,
                argmax: best.1,
            }
        })
        .collect()
}

/// `|A_c|` at every cut of `cuts` (sorted ascending, each in `[lo, hi]`) of the window `w` of a
/// line digraph, from its [`BlockIndex`]: one sequential merge of the cuts with the window's
/// blocks after the input histogram of [`frontier_blocks`]. Time `O(n_W + nb_W + #cuts)`,
/// memory 4 B per block of the window. On the PG, with the first node of every blockchain block
/// as cuts, this is the UTXO-set size at each block.
///
/// Fails if the cuts are not sorted or leave `[lo, hi]`.
pub fn frontier_at_cuts_blocks(blocks: &BlockIndex, w: Window, cuts: &[usize]) -> Result<Vec<u64>> {
    check_cuts(w, cuts)?;
    let (blo, bhi, hist) = input_histogram(blocks, w);
    let mut out = Vec::with_capacity(cuts.len());
    let (mut x, mut sum) = (blo, 0i64);
    for &c in cuts {
        // S(c): the window blocks starting before c
        while x < bhi && (blocks.first[x] as usize) < c {
            sum += i64::from(hist[x - blo]);
            x += 1;
        }
        out.push(((c - w.lo) as i64 - sum) as u64);
    }
    Ok(out)
}

/// `|A_c|` at every cut of `cuts` (sorted, in `[lo, hi]`) of the window `w` of any DAG with
/// topological ids (one parallel decode of the window and one sequential scan; 4 B per window
/// node).
pub fn frontier_at_cuts_generic<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    cuts: &[usize],
    pl: &mut impl ConcurrentProgressLog,
) -> Result<Vec<u64>> {
    check_cuts(w, cuts)?;
    let lo = w.lo;
    let mut cnt = huge_vec(w.len(), 0u32);
    {
        let c = as_atomic_u32(&mut cnt);
        pl.item_name("node").expected_updates(Some(w.len()));
        pl.start(format!("frontier cuts: first successors in {w}"));
        par_chunks(
            g,
            w,
            pl,
            |it| {
                while let Some((u, s)) = it.next_node() {
                    if let Some(&x) = s.first() {
                        let x = x as usize;
                        if x <= u {
                            return Err(DagError::NotTopological { u, v: x }.into());
                        }
                        c[x - lo].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                Ok(())
            },
            |_, _| (),
        )?;
        pl.done();
    }
    let mut out = Vec::with_capacity(cuts.len());
    let (mut c0, mut sum) = (lo, 0i64);
    for &c in cuts {
        while c0 < c {
            sum += i64::from(cnt[c0 - lo]);
            c0 += 1;
        }
        out.push(((c - lo) as i64 - sum) as u64);
    }
    Ok(out)
}

fn check_cuts(w: Window, cuts: &[usize]) -> Result<()> {
    if let Some(p) = cuts.windows(2).position(|p| p[0] > p[1]) {
        bail!(
            "cuts are not sorted: {} > {} at position {p}",
            cuts[p],
            cuts[p + 1]
        );
    }
    if let Some(&c) = cuts.iter().find(|&&c| c < w.lo || c > w.hi) {
        bail!("cut {c} lies outside the window {w} (cuts range over [lo, hi])");
    }
    Ok(())
}

/// [`CutRange`] statistics of the frontier curve of the window `w` of any DAG with topological
/// ids (one parallel decode of the window, as [`frontier_generic`], and one sequential scan of
/// all the cuts; the ranges may overlap and come in any order). Memory: 4 B per window node.
///
/// Fails with [`DagError::NotTopological`] if an arc goes backwards.
pub fn frontier_curve_generic<G: RandomAccessGraph + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    ranges: &[(String, usize, usize)],
    pl: &mut impl ConcurrentProgressLog,
) -> Result<Vec<CutRange>> {
    let ranges = cut_ranges(w, ranges);
    if ranges.is_empty() {
        return Ok(Vec::new());
    }
    let lo = w.lo;
    let mut cnt = huge_vec(w.len(), 0u32);
    {
        let c = as_atomic_u32(&mut cnt);
        pl.item_name("node").expected_updates(Some(w.len()));
        pl.start(format!("frontier curve: first successors in {w}"));
        par_chunks(
            g,
            w,
            pl,
            |it| {
                while let Some((u, s)) = it.next_node() {
                    if let Some(&x) = s.first() {
                        let x = x as usize;
                        if x <= u {
                            return Err(DagError::NotTopological { u, v: x }.into());
                        }
                        c[x - lo].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                Ok(())
            },
            |_, _| (),
        )?;
        pl.done();
    }
    // alive(c) for c = lo..=hi, updating every range containing c
    let mut out: Vec<CutRange> = ranges
        .iter()
        .map(|(l, a, b)| CutRange {
            label: l.clone(),
            lo: *a,
            hi: *b,
            at_lo: 0,
            at_hi: 0,
            max: 0,
            argmax: *a,
        })
        .collect();
    // ranges become active at their first cut and inactive after their last one
    let mut order: Vec<usize> = (0..out.len()).collect();
    order.sort_by_key(|&j| out[j].lo);
    let (mut next, mut active) = (0, Vec::<usize>::new());
    let mut alive = 0i64;
    for c in lo..=w.hi {
        if c > lo {
            alive += 1 - i64::from(cnt[c - 1 - lo]);
        }
        while next < order.len() && out[order[next]].lo == c {
            active.push(order[next]);
            next += 1;
        }
        let v = alive as u64;
        for &j in &active {
            let r = &mut out[j];
            if c == r.lo {
                (r.at_lo, r.max, r.argmax) = (v, v, c);
            } else if v > r.max {
                (r.max, r.argmax) = (v, c);
            }
            if c == r.hi {
                r.at_hi = v;
            }
        }
        active.retain(|&j| out[j].hi > c);
    }
    Ok(out)
}

/// Exact width of the window `w` of a line digraph by the maximum-weight closure network on its
/// blocks (see the module documentation). `chains` is `None`; the antichain holds absolute ids
/// and is checked with [`check_antichain_blocks`] by the caller.
///
/// Measured time on PG prefixes with the default [`FlowAlgo::PushRelabel`] (112 cores, one
/// thread does the flow): `[0, 10⁷)` 4.16 M blocks in 24.6 s; `[0, 10⁸)` 38.4 M blocks and
/// 1.15·10⁸ edges in 1225 s (3.1·10⁹ pushes, 1.1·10⁹ relabels, ≈ 29 relabels per vertex).
/// Ten times the nodes cost ≈ 50 times the time (exponent ≈ 1.7); extrapolated to the full PG
/// (21.8 × the nodes of `[0, 10⁸)`) this is ≈ 2·10⁵ s, i.e. days, which is why the full-graph
/// closure is behind `--experimental-full` and the default full-graph answer is the bracket
/// `[frontier, MPC_d]`.
pub fn width_closure_blocks(blocks: &BlockIndex, w: Window) -> WidthResult {
    width_closure_blocks_with(blocks, w, FlowAlgo::default())
}

/// [`width_closure_blocks`] with an explicit max-flow algorithm (for ablations).
pub fn width_closure_blocks_with(blocks: &BlockIndex, w: Window, algo: FlowAlgo) -> WidthResult {
    let (blo, bhi, hist) = input_histogram(blocks, w);
    let nbw = bhi - blo;
    let block_in_w = |b: u32| b != NONE && (b as usize) >= blo && (b as usize) < bhi;
    // ---- inputs grouped by block (counting sort over the window) ----
    let mut off = huge_vec(nbw + 1, 0u64);
    for i in 0..nbw {
        off[i + 1] = off[i] + hist[i] as u64;
    }
    drop(hist);
    let mut pos: Vec<u64> = huge_vec(nbw, 0u64);
    pos.copy_from_slice(&off[..nbw]);
    let mut creators = huge_vec(off[nbw] as usize, 0u32);
    let mut constant = 0u64;
    for u in w.lo..w.hi {
        let cr = blocks.create[u];
        if !block_in_w(cr) {
            constant += 1;
        }
        let b = blocks.spend[u];
        if b != NONE && (b as usize) < bhi {
            let i = b as usize - blo;
            // the creator of an input, as a window block index, or NONE if source-like
            creators[pos[i] as usize] = if block_in_w(cr) {
                cr - blo as u32
            } else {
                NONE
            };
            pos[i] += 1;
        }
    }
    drop(pos);
    // ---- network ----
    let (s, t) = (nbw, nbw + 1);
    let mut din = Dinic::new(nbw + 2, nbw + creators.len());
    let mut positive = 0u64;
    let mut targets: Vec<u32> = Vec::new();
    for i in 0..nbw {
        let b = blo + i;
        let i_b = off[i + 1] - off[i];
        let wb = outputs_in(blocks, b, w) as i64 - i_b as i64;
        if wb > 0 {
            din.add_edge(s, i, wb);
            positive += wb as u64;
        } else if wb < 0 {
            din.add_edge(i, t, -wb);
        }
        targets.clear();
        targets.extend(
            creators[off[i] as usize..off[i + 1] as usize]
                .iter()
                .copied()
                .filter(|&c| c != NONE),
        );
        targets.sort_unstable();
        targets.dedup();
        for &c in &targets {
            din.add_edge(i, c as usize, INF);
        }
    }
    drop(creators);
    drop(off);
    log::info!(
        "closure network on {nbw} blocks of {w}: constant {constant}, positive weight {positive}"
    );
    din.finalize();
    let cut = algo.run(&mut din, s, t) as u64;
    let width = constant + positive - cut;
    log::info!("closure: max flow {cut}, width {width}; extracting the antichain");
    let side = din.min_cut_side(s);
    drop(din);
    let in_s = |b: u32| -> bool {
        if block_in_w(b) {
            side.get(b as usize - blo)
        } else {
            // source-like creators are always in S*; spenders outside the window never are
            b == NONE || (b as usize) < blo
        }
    };
    let antichain: Vec<u32> = (w.lo..w.hi)
        .into_par_iter()
        .with_min_len(1 << 16)
        .filter(|&v| {
            let created_in_s = in_s(blocks.create[v]);
            let sp = blocks.spend[v];
            let spent_in_s = sp != NONE && block_in_w(sp) && side.get(sp as usize - blo);
            created_in_s && !spent_in_s
        })
        .map(|v| v as u32)
        .collect();
    WidthResult {
        width,
        chains: None,
        antichain,
        method: "closure",
    }
}

/// Verifies that `a` (absolute ids, strictly ascending, inside `w`) is an antichain of the
/// window of a line digraph, in one forward scan of the window (see the module documentation).
/// A member is rejected when it is reached by a path of any length from another member: the
/// *reached* mark is propagated through non-member nodes too. Memory: one byte per window
/// block; nothing per node.
pub fn check_antichain_blocks(blocks: &BlockIndex, w: Window, a: &[u32]) -> Result<()> {
    ensure!(
        a.windows(2).all(|p| p[0] < p[1]),
        "antichain is not strictly ascending"
    );
    if let (Some(&f), Some(&l)) = (a.first(), a.last()) {
        ensure!(
            w.contains(f as usize) && w.contains(l as usize),
            "antichain member outside the window {w}"
        );
    }
    let blo = blocks.block_bound(w.lo);
    let bhi = blocks.block_bound(w.hi);
    let mut reached = huge_vec(bhi - blo, false);
    let mut next = 0usize;
    for u in w.lo..w.hi {
        let member = next < a.len() && a[next] as usize == u;
        if member {
            next += 1;
        }
        let cr = blocks.create[u];
        let u_reached = cr != NONE && (cr as usize) >= blo && reached[cr as usize - blo];
        if member && u_reached {
            bail!("antichain member {u} is reachable from another member");
        }
        if member || u_reached {
            let sp = blocks.spend[u];
            if sp != NONE && (sp as usize) < bhi {
                reached[sp as usize - blo] = true;
            }
        }
    }
    Ok(())
}

/// Verifies that `a` (ids of `g`, strictly ascending) is an antichain of `g`, whose ids must be
/// topological: one pass in id order marks every node reached from `A` by a path of length
/// ≥ 1 and fails if a member is marked.
pub fn check_antichain(g: &Csr, a: &[u32]) -> Result<()> {
    g.check_topological()?;
    let n = g.n;
    ensure!(
        a.windows(2).all(|p| p[0] < p[1]),
        "antichain is not strictly ascending"
    );
    ensure!(
        a.last().is_none_or(|&x| (x as usize) < n),
        "antichain member out of range"
    );
    let mut member = vec![false; n];
    for &x in a {
        member[x as usize] = true;
    }
    let mut reached = vec![false; n];
    for u in 0..n {
        if member[u] && reached[u] {
            bail!("antichain member {u} is reachable from another member");
        }
        if member[u] || reached[u] {
            for &v in g.succ(u) {
                reached[v as usize] = true;
            }
        }
    }
    Ok(())
}

/// Verifies a [`WidthResult`] of `g` (ids of `g`): the antichain is an antichain of size
/// `width`, and, when chains are given, there are `width` non-empty chains that cover every
/// node and whose consecutive members are comparable. Equal sizes prove both optimal (weak
/// duality).
pub fn check_width_certificate(g: &Csr, r: &WidthResult) -> Result<()> {
    ensure!(
        r.antichain.len() as u64 == r.width,
        "antichain has {} nodes, width is {}",
        r.antichain.len(),
        r.width
    );
    check_antichain(g, &r.antichain)?;
    if let Some(c) = &r.chains {
        ensure!(
            c.len() as u64 == r.width,
            "{} chains, width is {}",
            c.len(),
            r.width
        );
        check_chains(g, c)?;
    }
    Ok(())
}

/// Width by Fulkerson's reduction (see the module documentation): `n − ν(S(TC(g)))`, with a
/// minimum chain partition (chains of comparable nodes, not necessarily arcs) and the Kőnig
/// antichain. Needs topological ids and `n²/8` bytes; panics if `n > 2¹⁵`.
pub fn width_fulkerson(g: &Csr) -> WidthResult {
    let n = g.n;
    assert!(n <= 1 << 15, "width_fulkerson: n = {n} > 2^15");
    let words = n.div_ceil(64);
    let mut rows = vec![0u64; n * words];
    for u in (0..n).rev() {
        for &v in g.succ(u) {
            let v = v as usize;
            assert!(v > u, "width_fulkerson: ids are not topological");
            let (head, tail) = rows.split_at_mut(v * words);
            let ru = &mut head[u * words..(u + 1) * words];
            ru[v / 64] |= 1 << (v % 64);
            for (x, y) in ru.iter_mut().zip(&tail[..words]) {
                *x |= *y;
            }
        }
    }
    let tc = Csr::from_arcs(
        n,
        (0..n).flat_map(|u| {
            let row = &rows[u * words..(u + 1) * words];
            (0..n)
                .filter(move |&v| row[v / 64] >> (v % 64) & 1 == 1)
                .map(move |v| (u, v))
        }),
    );
    drop(rows);
    let m = hopcroft_karp(&tc);
    let (cl, cr) = konig_cover(&tc, &m);
    let mut in_l = vec![false; n];
    let mut in_r = vec![false; n];
    for &x in &cl {
        in_l[x as usize] = true;
    }
    for &x in &cr {
        in_r[x as usize] = true;
    }
    let antichain: Vec<u32> = (0..n as u32)
        .filter(|&v| !in_l[v as usize] && !in_r[v as usize])
        .collect();
    let mut nodes = Vec::with_capacity(n);
    let mut offsets = vec![0u64];
    for v in 0..n {
        if m.mate_r[v] != NONE {
            continue;
        }
        let mut x = v as u32;
        while x != NONE {
            nodes.push(x);
            x = m.mate_l[x as usize];
        }
        offsets.push(nodes.len() as u64);
    }
    let width = (n - m.size) as u64;
    debug_assert_eq!(antichain.len() as u64, width);
    WidthResult {
        width,
        chains: Some(Chains { nodes, offsets }),
        antichain,
        method: "fulkerson",
    }
}
