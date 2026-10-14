//! Reachability labels of the ChronoChain index: levels, hi/lo bounds and chain labels.
//!
//! All labels are per DAG node, `u32`, reflexive and always defined:
//!
//! | label | definition | pass |
//! |---|---|---|
//! | `lvl(X)` | longest path (arcs) from a source to `X` | forward |
//! | `lo(X)` | `min{Z : Z ⇝* X}` | forward |
//! | `inn_c(X)` | `1 + max{pos_c(Z) : Z ⇝* X, Z ∈ P_c}`, `0` = −∞ | forward |
//! | `ht(X)` | longest path (arcs) from `X` to a sink | backward |
//! | `hi(X)` | `max{Z : X ⇝* Z}` | backward |
//! | `out_c(X)` | `min{pos_c(Z) : X ⇝* Z, Z ∈ P_c}`, [`NONE`] = +∞ | backward |
//!
//! `P_c` is chain `c`, a path of the DAG stored as its node list (`pos_c(Z)` is the index of
//! `Z` in it). The chain-cover labels go back to [\[Jag90\]](crate::references#jag90) and
//! [\[Sim88\]](crate::references#sim88); the level and hi/lo filters are the topological-level
//! and order-bound observations of O'Reach [\[HST22\]](crate::references#hst22), specialised to
//! the chronological order of the PG.
//!
//! # Chain selection ([`select_chains`])
//! Chain 0 is the longest path (the PG "spine"); chains `1..k` are greedy maximum-new-coverage
//! paths. Each chain costs one forward DP with weight 1 on uncovered nodes and 0 on covered
//! ones:
//! ```text
//! best[·] = 0, pred[·] = NONE
//! for X ascending:  best[X] += [X uncovered]            // best[X] is now final
//!                   for W in succ(X): if best[X] > best[W] or pred[W] = NONE:
//!                                         best[W] = max(best[W], best[X]); pred[W] = X
//! E = smallest argmax best;  gain = best[E]  (stop if gain = 0: everything is covered)
//! chain = backtrack pred from E to a source, then extend E forward along first successors
//!         to a sink (zero gain, but longer chains decide more queries)
//! ```
//! The tie rule (strictly better, else the first predecessor) makes the chain deterministic and
//! every backtrack end at a DAG source. The chains are selected sequentially (data
//! dependence): `k` forward passes, with `best`, `pred` (8 B/node) and a covered bitmap as
//! transient memory.
//!
//! # Fused label passes ([`forward_labels`], [`backward_labels`])
//! One forward push computes `lvl`, `lo` and every `inn_c`; one backward pull computes `ht`,
//! `hi` and every `out_c`. A node's own chain position is found with one cursor per chain
//! (chain nodes are strictly increasing ids), so each pass is O(n + k·m).
//!
//! Memory: `4·4 + 8·k` bytes per DAG node for the labels (48 B with `k = 4`).

use anyhow::Result;
use rayon::prelude::*;

use crate::io::{NONE, huge_vec};
use crate::reach::dagview::SuccDag;
use crate::sweep::DynProgressLog;

/// The selected chains (paths of the DAG, node lists in path order).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ChainSelection {
    /// The chains.
    pub chains: Vec<Vec<u32>>,
    /// Newly covered nodes of each chain at selection time (`chains[0]`: the longest path's
    /// node count).
    pub gains: Vec<u64>,
}

#[inline]
fn get_bit(bits: &[u64], x: usize) -> bool {
    bits[x / 64] >> (x % 64) & 1 != 0
}

#[inline]
fn set_bit(bits: &mut [u64], x: usize) {
    bits[x / 64] |= 1 << (x % 64);
}

/// Selects up to `k` chains: the longest path, then greedy maximum-new-coverage paths (see the
/// module documentation). Stops early when every node is covered.
pub fn select_chains(
    dag: &dyn SuccDag,
    k: usize,
    pl: &mut dyn DynProgressLog,
) -> Result<ChainSelection> {
    let n = dag.num_nodes();
    let mut sel = ChainSelection::default();
    if n == 0 || k == 0 {
        return Ok(sel);
    }
    let mut covered: Vec<u64> = huge_vec(n.div_ceil(64), 0);
    let mut best: Vec<u32> = huge_vec(n, 0);
    let mut pred: Vec<u32> = huge_vec(n, NONE);
    let mut buf = Vec::new();
    for c in 0..k {
        if c > 0 {
            best.par_iter_mut().for_each(|b| *b = 0);
            pred.par_iter_mut().for_each(|p| *p = NONE);
        }
        pl.dyn_expected_updates(Some(n));
        pl.dyn_start(&format!("selecting chain {c}"));
        {
            let (best, pred, covered) = (&mut best, &mut pred, &covered);
            dag.for_each_forward(
                &mut |x, succ| {
                    let v = best[x] + u32::from(!get_bit(covered, x));
                    best[x] = v;
                    for &w in succ {
                        let w = w as usize;
                        if v > best[w] || pred[w] == NONE {
                            best[w] = best[w].max(v);
                            pred[w] = x as u32;
                        }
                    }
                },
                pl,
            )?;
        }
        pl.dyn_done();
        // smallest argmax
        let (end, gain) = best.par_iter().enumerate().map(|(i, &b)| (i, b)).reduce(
            || (usize::MAX, 0),
            |a, b| {
                if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                    b
                } else {
                    a
                }
            },
        );
        if gain == 0 {
            break;
        }
        let mut chain = Vec::new();
        let mut x = end as u32;
        loop {
            chain.push(x);
            match pred[x as usize] {
                NONE => break,
                p => x = p,
            }
        }
        chain.reverse();
        let mut x = end;
        loop {
            dag.succ_into(x, &mut buf);
            match buf.first() {
                Some(&w) => {
                    chain.push(w);
                    x = w as usize;
                }
                None => break,
            }
        }
        for &z in &chain {
            set_bit(&mut covered, z as usize);
        }
        log::info!(
            "chain {c}: {} nodes, {gain} newly covered, ends at {}",
            chain.len(),
            chain.last().copied().unwrap_or(NONE)
        );
        sel.chains.push(chain);
        sel.gains.push(u64::from(gain));
    }
    Ok(sel)
}

/// Labels of the forward pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForwardLabels {
    /// Longest path (arcs) from a source.
    pub lvl: Vec<u32>,
    /// Smallest node reaching the node (reflexive).
    pub lo: Vec<u32>,
    /// Per chain: `1 + max pos` of a chain node reaching the node (reflexive); 0 = −∞.
    pub inn: Vec<Vec<u32>>,
}

/// Labels of the backward pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackwardLabels {
    /// Longest path (arcs) to a sink.
    pub ht: Vec<u32>,
    /// Largest node reachable from the node (reflexive).
    pub hi: Vec<u32>,
    /// Per chain: smallest pos of a chain node reachable from the node (reflexive);
    /// [`NONE`] = +∞.
    pub out: Vec<Vec<u32>>,
}

/// One forward push computing `lvl`, `lo` and `inn_c` for every chain.
pub fn forward_labels(
    dag: &dyn SuccDag,
    chains: &[Vec<u32>],
    pl: &mut dyn DynProgressLog,
) -> Result<ForwardLabels> {
    let n = dag.num_nodes();
    let mut lvl: Vec<u32> = huge_vec(n, 0);
    let mut lo: Vec<u32> = huge_vec(n, NONE);
    let mut inn: Vec<Vec<u32>> = chains.iter().map(|_| huge_vec(n, 0)).collect();
    let mut cursor = vec![0usize; chains.len()];
    pl.dyn_expected_updates(Some(n));
    pl.dyn_start("forward labels (lvl, lo, inn)");
    {
        let (lvl, lo, inn) = (&mut lvl, &mut lo, &mut inn);
        dag.for_each_forward(
            &mut |x, succ| {
                let l = lvl[x] + 1;
                let m = lo[x].min(x as u32);
                lo[x] = m;
                for &w in succ {
                    let w = w as usize;
                    lvl[w] = lvl[w].max(l);
                    lo[w] = lo[w].min(m);
                }
                for (c, chain) in chains.iter().enumerate() {
                    let p = &mut cursor[c];
                    let arr = &mut inn[c];
                    if chain.get(*p) == Some(&(x as u32)) {
                        *p += 1;
                        arr[x] = arr[x].max(*p as u32);
                    }
                    let v = arr[x];
                    if v != 0 {
                        for &w in succ {
                            let w = w as usize;
                            arr[w] = arr[w].max(v);
                        }
                    }
                }
            },
            pl,
        )?;
    }
    pl.dyn_done();
    for (c, chain) in chains.iter().enumerate() {
        anyhow::ensure!(
            cursor[c] == chain.len(),
            "chain {c} is not increasing or has nodes outside the DAG"
        );
    }
    Ok(ForwardLabels { lvl, lo, inn })
}

/// One backward pull computing `ht`, `hi` and `out_c` for every chain.
pub fn backward_labels(
    dag: &dyn SuccDag,
    chains: &[Vec<u32>],
    pl: &mut dyn DynProgressLog,
) -> Result<BackwardLabels> {
    let n = dag.num_nodes();
    let mut ht: Vec<u32> = huge_vec(n, 0);
    let mut hi: Vec<u32> = huge_vec(n, 0);
    let mut out: Vec<Vec<u32>> = chains.iter().map(|_| huge_vec(n, NONE)).collect();
    let mut cursor: Vec<usize> = chains.iter().map(Vec::len).collect();
    pl.dyn_expected_updates(Some(n));
    pl.dyn_start("backward labels (ht, hi, out)");
    {
        let (ht, hi, out) = (&mut ht, &mut hi, &mut out);
        dag.for_each_backward(
            &mut |x, succ| {
                let mut h = 0;
                let mut m = x as u32;
                for &w in succ {
                    let w = w as usize;
                    h = h.max(ht[w] + 1);
                    m = m.max(hi[w]);
                }
                ht[x] = h;
                hi[x] = m;
                for (c, chain) in chains.iter().enumerate() {
                    let p = &mut cursor[c];
                    let arr = &mut out[c];
                    let mut v = NONE;
                    if *p > 0 && chain[*p - 1] == x as u32 {
                        *p -= 1;
                        v = *p as u32;
                    }
                    for &w in succ {
                        v = v.min(arr[w as usize]);
                    }
                    arr[x] = v;
                }
            },
            pl,
        )?;
    }
    pl.dyn_done();
    for (c, p) in cursor.iter().enumerate() {
        anyhow::ensure!(
            *p == 0,
            "chain {c} is not increasing or has nodes outside the DAG"
        );
    }
    Ok(BackwardLabels { ht, hi, out })
}
