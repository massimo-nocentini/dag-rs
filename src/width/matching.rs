//! Hopcroft–Karp maximum matching of the split graph and the Kőnig cover certificate.
//!
//! # Split graph
//! For a [`Csr`] with `n` nodes the bipartite graph has a left copy `u` and a right copy `v` of
//! every node and an edge `(u, v)` per arc `u → v`. A matching is stored as two arrays:
//! `mate_l[u]` (the right partner of left `u`) and `mate_r[v]` (the left partner of right `v`),
//! [`NONE`] when free.
//!
//! # Algorithm ([\[HK73\]](crate::references#hk73))
//! ```text
//! greedy: for u ascending, match u to its first free successor
//! loop (phase):
//!     BFS from every free left u (dist 0) over u → v → mate_r[v]; stop at the layer where a
//!       free right v is first reached (limit); if none: stop
//!     for each free left u: iterative DFS along dist + 1 layers with a per-node arc cursor,
//!       accepting a free right v only from layer limit − 1; augment along the stack
//! ```
//! Each phase finds a maximal set of vertex-disjoint shortest augmenting paths, so there are
//! `O(√n)` phases and the time is `O(m √n)`. Memory: `mate_l`, `mate_r`, `dist` (4 B/node
//! each), the arc cursor (8 B/node), the BFS queue and DFS stack (≤ 4 B/node each, reserved
//! up front); all per-node arrays come from [`huge_vec`] / [`huge_capacity`]. [`konig_cover`]
//! adds 2 B/node of flags, a queue of ≤ 4 B/node and the cover (≤ 4 B/node in all), and
//! [`check_konig`] 2 B/node. Progress is logged at `info` level once per phase.
//!
//! On the PG the greedy start is already optimal: the split graph is a disjoint union of
//! complete bicliques, and the greedy scan matches `min(i_B, o_B)` pairs in each, so one BFS
//! proves optimality.
//!
//! # Certificate ([\[Kon31\]](crate::references#kon31))
//! Let `Z` be the nodes reachable from the free left nodes by alternating paths (any edge from
//! left to right, the matched edge from right to left). Then `C = (L \ Z) ∪ (R ∩ Z)` is a
//! vertex cover with `|C| = |M|`, which proves that `M` is maximum (weak duality).
//! [`konig_cover`] builds it and [`check_konig`] verifies both properties independently.

use anyhow::{Result, ensure};

use crate::csr::Csr;
use crate::io::{NONE, huge_capacity, huge_vec};

/// A matching of the split graph of a [`Csr`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matching {
    /// `mate_l[u]`: the right partner of left node `u`, or [`NONE`].
    pub mate_l: Vec<u32>,
    /// `mate_r[v]`: the left partner of right node `v`, or [`NONE`].
    pub mate_r: Vec<u32>,
    /// Number of matched pairs.
    pub size: usize,
}

/// Maximum matching of the split graph of `g` (greedy start + Hopcroft–Karp phases with an
/// iterative DFS; see the module documentation).
///
/// ```
/// use dag::csr::Csr;
/// use dag::width::hopcroft_karp;
/// // 0 → 1 → 2 and 0 → 2: a path cover with one path, so ν = 2
/// let g = Csr::from_arcs(3, [(0, 1), (1, 2), (0, 2)]);
/// assert_eq!(hopcroft_karp(&g).size, 2);
/// ```
pub fn hopcroft_karp(g: &Csr) -> Matching {
    let n = g.n;
    let mut mate_l = huge_vec(n, NONE);
    let mut mate_r = huge_vec(n, NONE);
    let mut size = 0usize;
    // greedy start
    for (u, ml) in mate_l.iter_mut().enumerate() {
        for &v in g.succ(u) {
            if mate_r[v as usize] == NONE {
                mate_r[v as usize] = u as u32;
                *ml = v;
                size += 1;
                break;
            }
        }
    }
    log::info!("Hopcroft–Karp: {n} nodes, greedy matching {size}");
    let mut dist = huge_vec(n, NONE);
    // each left node enters the BFS queue and the DFS stack at most once per phase
    let mut queue: Vec<u32> = huge_capacity(n);
    let mut cursor: Vec<u64> = huge_vec(n, 0u64);
    let mut stack: Vec<u32> = huge_capacity(n);
    let mut phase = 0u64;
    loop {
        // ---- BFS layering from the free left nodes ----
        queue.clear();
        for u in 0..n {
            if mate_l[u] == NONE && g.outdegree(u) > 0 {
                dist[u] = 0;
                queue.push(u as u32);
            } else {
                dist[u] = NONE;
            }
        }
        let mut limit = NONE;
        let mut head = 0;
        while head < queue.len() {
            let u = queue[head] as usize;
            head += 1;
            let du = dist[u];
            if du >= limit {
                break;
            }
            for &v in g.succ(u) {
                let w = mate_r[v as usize];
                if w == NONE {
                    limit = limit.min(du + 1);
                } else if dist[w as usize] == NONE {
                    dist[w as usize] = du + 1;
                    queue.push(w);
                }
            }
        }
        if limit == NONE {
            log::info!("Hopcroft–Karp: maximum matching {size} after {phase} phases");
            break;
        }
        phase += 1;
        // ---- DFS phase: vertex-disjoint shortest augmenting paths ----
        cursor.copy_from_slice(&g.offsets[..n]);
        for root in 0..n {
            if mate_l[root] != NONE || dist[root] != 0 {
                continue;
            }
            stack.clear();
            stack.push(root as u32);
            while let Some(&top) = stack.last() {
                let u = top as usize;
                let end = g.offsets[u + 1];
                let mut pushed = false;
                let mut found = false;
                while cursor[u] < end {
                    let v = g.succ[cursor[u] as usize] as usize;
                    let w = mate_r[v];
                    if w == NONE {
                        if dist[u] + 1 == limit {
                            found = true;
                            break;
                        }
                    } else if dist[w as usize] == dist[u] + 1 && dist[u] + 1 < limit {
                        stack.push(w);
                        pushed = true;
                        break;
                    }
                    cursor[u] += 1;
                }
                if found {
                    // augment: every stack node takes the right node under its cursor
                    for &x in stack.iter() {
                        let x = x as usize;
                        let v = g.succ[cursor[x] as usize];
                        mate_l[x] = v;
                        mate_r[v as usize] = x as u32;
                        // the node is used for this phase
                        dist[x] = NONE;
                    }
                    size += 1;
                    break;
                }
                if !pushed {
                    // dead end: remove u from the layered graph and advance the parent
                    dist[u] = NONE;
                    stack.pop();
                    if let Some(&p) = stack.last() {
                        cursor[p as usize] += 1;
                    }
                }
            }
        }
        log::info!(
            "Hopcroft–Karp: phase {phase} (augmenting paths of length {limit}), matching {size}"
        );
    }
    Matching {
        mate_l,
        mate_r,
        size,
    }
}

/// The Kőnig vertex cover `C = (L \ Z) ∪ (R ∩ Z)` of a **maximum** matching `m`, as
/// `(left nodes, right nodes)`, both ascending. Its size equals `m.size` exactly when `m` is
/// maximum (see the module documentation).
pub fn konig_cover(g: &Csr, m: &Matching) -> (Vec<u32>, Vec<u32>) {
    let n = g.n;
    let mut zl = huge_vec(n, false);
    let mut zr = huge_vec(n, false);
    // every left node enters the queue at most once
    let mut queue: Vec<u32> = huge_capacity(n);
    queue.extend((0..n as u32).filter(|&u| m.mate_l[u as usize] == NONE));
    for &u in &queue {
        zl[u as usize] = true;
    }
    let mut head = 0;
    while head < queue.len() {
        let u = queue[head] as usize;
        head += 1;
        for &v in g.succ(u) {
            let v = v as usize;
            if zr[v] {
                continue;
            }
            zr[v] = true;
            let w = m.mate_r[v];
            if w != NONE && !zl[w as usize] {
                zl[w as usize] = true;
                queue.push(w);
            }
        }
    }
    let left = (0..n as u32).filter(|&u| !zl[u as usize]).collect();
    let right = (0..n as u32).filter(|&v| zr[v as usize]).collect();
    (left, right)
}

/// Verifies that `m` is a matching of the split graph of `g` (partners are arcs, `mate_l` and
/// `mate_r` are inverse, `size` is right) and that `cover` is a vertex cover of the split graph
/// with `|cover| == m.size`, which certifies that `m` is maximum.
pub fn check_konig(g: &Csr, m: &Matching, cover: &(Vec<u32>, Vec<u32>)) -> Result<()> {
    let n = g.n;
    ensure!(
        m.mate_l.len() == n && m.mate_r.len() == n,
        "matching arrays have the wrong length"
    );
    let mut size = 0usize;
    for u in 0..n {
        let v = m.mate_l[u];
        if v == NONE {
            continue;
        }
        size += 1;
        ensure!(
            g.succ(u).binary_search(&v).is_ok(),
            "matched pair ({u}, {v}) is not an arc"
        );
        ensure!(
            m.mate_r[v as usize] == u as u32,
            "mate_r[{v}] = {} but mate_l[{u}] = {v}",
            m.mate_r[v as usize]
        );
    }
    let size_r = m.mate_r.iter().filter(|&&u| u != NONE).count();
    ensure!(
        size == m.size && size_r == m.size,
        "matching size {} but {size} left and {size_r} right nodes are matched",
        m.size
    );
    let (cl, cr) = cover;
    ensure!(
        cl.len() + cr.len() == m.size,
        "Kőnig cover has {} nodes, the matching {}",
        cl.len() + cr.len(),
        m.size
    );
    let mut in_l = huge_vec(n, false);
    let mut in_r = huge_vec(n, false);
    for &u in cl {
        ensure!((u as usize) < n, "cover node {u} out of range");
        in_l[u as usize] = true;
    }
    for &v in cr {
        ensure!((v as usize) < n, "cover node {v} out of range");
        in_r[v as usize] = true;
    }
    for (u, &covered) in in_l.iter().enumerate() {
        if covered {
            continue;
        }
        for &v in g.succ(u) {
            ensure!(
                in_r[v as usize],
                "arc ({u}, {v}) is not covered by the Kőnig cover"
            );
        }
    }
    Ok(())
}
