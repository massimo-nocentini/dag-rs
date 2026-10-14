//! Acceptance tests of matchings, path covers, frontiers, flows,
//! width and antichain certificates, against brute-force oracles.

use dag::blocks::BlockIndex;
use dag::csr::Csr;
use dag::testkit::{self, MiniPgOpts};
use dag::width::{
    Chains, Dinic, FlowAlgo, WidthResult, check_antichain, check_antichain_blocks, check_konig,
    check_path_cover, check_width_certificate, frontier_blocks, frontier_generic, hopcroft_karp,
    konig_cover, min_path_cover_blocks, min_path_cover_hk, width_closure_blocks,
    width_closure_blocks_with, width_fulkerson, width_min_flow, width_min_flow_with,
};
use dag::{DagError, NONE, Window};
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;

fn no_log() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

fn csr_of(g: &VecGraph) -> Csr {
    Csr::from_vec_graph(g)
}

/// The window `w` of `g` as a `Csr` (ids − lo), through the chunked decoder with tiny chunks.
fn window_csr(g: &VecGraph, w: Window, chunk_arcs: u64) -> Csr {
    let cg = testkit::chunked(g, chunk_arcs);
    Csr::from_window(&cg, w, &mut no_log()).unwrap()
}

/// Random windows of `n` nodes: the full range, empty ones, and random sub-ranges.
fn windows(n: usize, rng: &mut SmallRng, k: usize) -> Vec<Window> {
    let mut ws = vec![
        Window::full(n),
        Window { lo: 0, hi: 0 },
        Window { lo: n, hi: n },
    ];
    for _ in 0..k {
        let a = rng.random_range(0..=n);
        let b = rng.random_range(0..=n);
        ws.push(Window {
            lo: a.min(b),
            hi: a.max(b),
        });
    }
    ws
}

fn mini_pgs() -> Vec<(u64, VecGraph)> {
    let mut v = Vec::new();
    for seed in 0..12u64 {
        let opts = MiniPgOpts {
            zero_output_prob: if seed % 3 == 0 { 0.15 } else { 0.0 },
            coinbase_prob: if seed % 4 == 1 { 0.3 } else { 0.1 },
            max_outputs: 2 + (seed as usize % 4),
            ..Default::default()
        };
        let txs = [10, 40, 150, 400][seed as usize % 4];
        v.push((seed, testkit::mini_pg(txs, seed, &opts).graph));
    }
    v
}

/// Frontier by definition on a `Csr`: for every cut `c`, the nodes `u < c` whose successors are
/// all `≥ c`; the maximum with the smallest `c` on ties.
fn brute_frontier(g: &Csr) -> (u64, usize) {
    let mut best = (0u64, 0usize);
    for c in 0..=g.n {
        let size = (0..c)
            .filter(|&u| g.succ(u).iter().all(|&v| v as usize >= c))
            .count() as u64;
        if size > best.0 {
            best = (size, c);
        }
    }
    best
}

fn to_vec_graph(g: &Csr) -> VecGraph {
    g.to_vec_graph()
}

// -------------------------------------------------------------------------------------------
// Matching and path covers
// -------------------------------------------------------------------------------------------

#[test]
fn hk_matches_brute_force_and_konig() {
    for seed in 0..400u64 {
        let n = (seed % 17) as usize;
        let p = [0.1, 0.25, 0.5, 0.8][(seed / 17 % 4) as usize];
        let g = testkit::random_dag(n, p, seed);
        let csr = csr_of(&g);
        let m = hopcroft_karp(&csr);
        assert_eq!(m.size, testkit::brute_matching(&g), "seed {seed}");
        let cover = konig_cover(&csr, &m);
        check_konig(&csr, &m, &cover).unwrap();
        let pc = min_path_cover_hk(&csr);
        assert_eq!(check_path_cover(&csr, &pc).unwrap(), n - m.size);
    }
}

#[test]
fn hk_large_graphs_are_certified() {
    for seed in 0..20u64 {
        let g = match seed % 4 {
            0 => testkit::random_dag(600, 0.01, seed),
            1 => testkit::layered(&[50, 80, 30, 120, 60, 90], 0.05, seed),
            2 => testkit::ladder(200),
            _ => testkit::random_dag(300, 0.05, seed),
        };
        let csr = csr_of(&g);
        let m = hopcroft_karp(&csr);
        check_konig(&csr, &m, &konig_cover(&csr, &m)).unwrap();
        let pc = min_path_cover_hk(&csr);
        assert_eq!(check_path_cover(&csr, &pc).unwrap(), csr.n - m.size);
    }
}

#[test]
fn konig_rejects_bad_certificates() {
    let g = csr_of(&testkit::random_dag(12, 0.3, 7));
    let m = hopcroft_karp(&g);
    assert!(m.size > 1);
    let (mut l, r) = konig_cover(&g, &m);
    // a non-maximum matching: drop one pair
    let mut small = m.clone();
    let u = small.mate_l.iter().position(|&v| v != NONE).unwrap();
    let v = small.mate_l[u];
    small.mate_l[u] = NONE;
    small.mate_r[v as usize] = NONE;
    small.size -= 1;
    assert!(check_konig(&g, &small, &konig_cover(&g, &small)).is_err());
    // a cover that misses an arc
    if !l.is_empty() {
        l.pop();
        assert!(check_konig(&g, &m, &(l, r)).is_err());
    }
}

/// Regression: each property checked by `check_konig` is
/// exercised on its own, with every other property intact.
#[test]
fn konig_rejects_each_violated_property() {
    use dag::width::Matching;
    // 0 → 1, 2 → 3: M = {(0, 1), (2, 3)} is maximum
    let g = Csr::from_arcs(4, [(0, 1), (2, 3)]);
    let m = Matching {
        mate_l: vec![1, NONE, 3, NONE],
        mate_r: vec![NONE, 0, NONE, 2],
        size: 2,
    };
    check_konig(&g, &m, &(vec![0, 2], vec![])).unwrap();
    check_konig(&g, &m, &(vec![], vec![1, 3])).unwrap();
    check_konig(&g, &m, &konig_cover(&g, &m)).unwrap();
    // (a) a cover of the right size that leaves the arc (2, 3) uncovered
    let e = check_konig(&g, &m, &(vec![0, 1], vec![])).unwrap_err();
    assert!(e.to_string().contains("not covered"), "{e}");
    let e = check_konig(&g, &m, &(vec![0], vec![1])).unwrap_err();
    assert!(e.to_string().contains("not covered"), "{e}");
    // (b) a size field that disagrees with the mate arrays (the cover has the claimed size)
    let wrong = Matching {
        size: 1,
        ..m.clone()
    };
    let e = check_konig(&g, &wrong, &(vec![0], vec![])).unwrap_err();
    assert!(e.to_string().contains("matching size"), "{e}");
    let wrong = Matching {
        size: 3,
        ..m.clone()
    };
    let e = check_konig(&g, &wrong, &(vec![0, 2], vec![1])).unwrap_err();
    assert!(e.to_string().contains("matching size"), "{e}");
    // a right node matched to nobody on the left (left side consistent, size field right)
    let g3 = Csr::from_arcs(4, [(0, 1), (2, 3), (0, 3)]);
    let stray = Matching {
        mate_l: vec![1, NONE, NONE, NONE],
        mate_r: vec![NONE, 0, NONE, 2],
        size: 1,
    };
    let e = check_konig(&g3, &stray, &(vec![0], vec![])).unwrap_err();
    assert!(e.to_string().contains("matching size"), "{e}");
    // (c) mate_l and mate_r that are not inverse
    let crossed = Matching {
        mate_l: vec![1, NONE, 3, NONE],
        mate_r: vec![NONE, 2, NONE, 0],
        size: 2,
    };
    let e = check_konig(&g, &crossed, &(vec![0, 2], vec![])).unwrap_err();
    assert!(e.to_string().contains("mate_r"), "{e}");
    // (d) a matched pair that is not an arc
    let fake = Matching {
        mate_l: vec![3, NONE, 1, NONE],
        mate_r: vec![NONE, 2, NONE, 0],
        size: 2,
    };
    let e = check_konig(&g, &fake, &(vec![0, 2], vec![])).unwrap_err();
    assert!(e.to_string().contains("not an arc"), "{e}");
    // wrong lengths and out-of-range cover nodes
    let short = Matching {
        mate_l: vec![1, NONE, 3],
        ..m.clone()
    };
    assert!(check_konig(&g, &short, &(vec![0, 2], vec![])).is_err());
    assert!(check_konig(&g, &m, &(vec![0, 9], vec![])).is_err());
}

/// Regression: each property checked by `check_path_cover` is
/// exercised on its own.
#[test]
fn path_cover_check_rejects_each_violated_property() {
    use dag::width::PathCover;
    // 0 → 1 → 2, 0 → 2, 3 → 2
    let g = Csr::from_arcs(4, [(0, 1), (1, 2), (0, 2), (3, 2)]);
    let good = PathCover {
        next: vec![1, 2, NONE, NONE],
        starts: vec![0, 3],
    };
    assert_eq!(check_path_cover(&g, &good).unwrap(), 2);
    // a link that is not an arc (2 is a sink; 1 → 3 does not exist)
    let bad = PathCover {
        next: vec![1, 3, NONE, NONE],
        starts: vec![0, 2],
    };
    let e = check_path_cover(&g, &bad).unwrap_err();
    assert!(e.to_string().contains("not an arc"), "{e}");
    // a link out of range
    let bad = PathCover {
        next: vec![1, 7, NONE, NONE],
        starts: vec![0, 3],
    };
    assert!(check_path_cover(&g, &bad).is_err());
    // a node with two predecessors (both links are arcs)
    let bad = PathCover {
        next: vec![2, 2, NONE, NONE],
        starts: vec![0, 1, 3],
    };
    let e = check_path_cover(&g, &bad).unwrap_err();
    assert!(e.to_string().contains("two predecessors"), "{e}");
    let bad = PathCover {
        next: vec![1, 2, NONE, 2],
        starts: vec![0, 3],
    };
    let e = check_path_cover(&g, &bad).unwrap_err();
    assert!(e.to_string().contains("two predecessors"), "{e}");
    // starts that are not the nodes without a predecessor: missing, extra, unsorted
    for starts in [vec![0], vec![0, 2, 3], vec![3, 0], vec![0, 1, 3]] {
        let bad = PathCover {
            next: good.next.clone(),
            starts,
        };
        let e = check_path_cover(&g, &bad).unwrap_err();
        assert!(e.to_string().contains("starts"), "{e}");
    }
    // paths that miss nodes: only possible on a cycle, which a Csr does not forbid
    let cyc = Csr::from_arcs(3, [(0, 1), (1, 0)]);
    let bad = PathCover {
        next: vec![1, 0, NONE],
        starts: vec![2],
    };
    let e = check_path_cover(&cyc, &bad).unwrap_err();
    assert!(e.to_string().contains("visit"), "{e}");
    // wrong length
    let bad = PathCover {
        next: vec![1, 2, NONE],
        starts: vec![0],
    };
    assert!(check_path_cover(&g, &bad).is_err());
}

/// Regression: corrupting one link of the closed-form explicit cover is caught
/// by `check_path_cover`, which is the only check of that cover in the CLI.
#[test]
fn corrupted_closed_form_cover_is_rejected() {
    let mut corrupted = 0;
    for (seed, g) in mini_pgs() {
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let n = g.num_nodes();
        let w = Window::full(n);
        let csr = window_csr(&g, w, 1 + seed % 5);
        let (_, pc) = min_path_cover_blocks(&blocks, w, true);
        let pc = pc.unwrap();
        check_path_cover(&csr, &pc).unwrap();
        let linked: Vec<usize> = (0..n).filter(|&u| pc.next[u] != NONE).collect();
        for &u in linked.iter().take(25) {
            let v = pc.next[u];
            // (1) point the link at a node that is not a successor
            if let Some(x) = (0..n as u32).find(|&x| !csr.succ(u).contains(&x)) {
                let mut bad = pc.clone();
                bad.next[u] = x;
                assert!(check_path_cover(&csr, &bad).is_err(), "seed {seed} {u}→{x}");
                corrupted += 1;
            }
            // (2) cut the link without fixing starts
            let mut bad = pc.clone();
            bad.next[u] = NONE;
            assert!(check_path_cover(&csr, &bad).is_err(), "seed {seed} cut {u}");
            // (3) re-point the link at another successor that already has a predecessor
            if let Some(&x) = csr
                .succ(u)
                .iter()
                .find(|&&x| x != v && pc.starts.binary_search(&x).is_err())
            {
                let mut bad = pc.clone();
                bad.next[u] = x;
                assert!(check_path_cover(&csr, &bad).is_err(), "seed {seed} {u}→{x}");
            }
        }
    }
    assert!(corrupted > 50);
}

#[test]
fn path_cover_special_graphs() {
    let cases: Vec<(VecGraph, usize)> = vec![
        (testkit::chain(10), 1),
        (testkit::complete_dag(8), 1),
        (testkit::isolated(7), 7),
        (testkit::out_star(6), 5),
        (testkit::in_star(6), 5),
        (testkit::isolated(0), 0),
        (testkit::isolated(1), 1),
    ];
    for (g, want) in cases {
        let csr = csr_of(&g);
        let pc = min_path_cover_hk(&csr);
        assert_eq!(check_path_cover(&csr, &pc).unwrap(), want);
    }
}

#[test]
fn closed_form_mpc_equals_hk_on_mini_pg_windows() {
    let mut rng = SmallRng::seed_from_u64(1);
    for (seed, g) in mini_pgs() {
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let n = g.num_nodes();
        let mut ws = windows(n, &mut rng, 25);
        // windows starting exactly at, and just after, a block start (cutting through blocks)
        for b in [0, blocks.nb / 2, blocks.nb.saturating_sub(1)] {
            if b < blocks.nb {
                let f = blocks.first[b] as usize;
                ws.push(Window { lo: f, hi: n });
                ws.push(Window {
                    lo: (f + 1).min(n),
                    hi: n,
                });
                ws.push(Window { lo: 0, hi: f + 1 });
            }
        }
        for w in ws {
            let csr = window_csr(&g, w, 1 + seed % 5);
            let hk = hopcroft_karp(&csr);
            let (stats, pc) = min_path_cover_blocks(&blocks, w, true);
            assert_eq!(
                stats.matching, hk.size as u64,
                "seed {seed} window {w}: closed form vs HK"
            );
            assert_eq!(stats.n, w.len() as u64);
            assert_eq!(stats.mpc_d, stats.n - stats.matching);
            let pc = pc.unwrap();
            let lo = w.lo as u32;
            let rel = dag::width::PathCover {
                next: pc
                    .next
                    .iter()
                    .map(|&x| if x == NONE { NONE } else { x - lo })
                    .collect(),
                starts: pc.starts.iter().map(|&x| x - lo).collect(),
            };
            assert_eq!(
                check_path_cover(&csr, &rel).unwrap() as u64,
                stats.mpc_d,
                "seed {seed} window {w}: explicit cover"
            );
            let (stats2, none) = min_path_cover_blocks(&blocks, w, false);
            assert_eq!(stats2, stats);
            assert!(none.is_none());
        }
    }
}

// -------------------------------------------------------------------------------------------
// Frontier
// -------------------------------------------------------------------------------------------

#[test]
fn frontier_blocks_equals_generic_and_brute_force() {
    let mut rng = SmallRng::seed_from_u64(2);
    for (seed, g) in mini_pgs() {
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let n = g.num_nodes();
        for w in windows(n, &mut rng, 15) {
            let cg = testkit::chunked(&g, 1 + seed % 7);
            let fg = frontier_generic(&cg, w, true, &mut no_log()).unwrap();
            let fb = frontier_blocks(&blocks, w, true);
            let csr = window_csr(&g, w, 3);
            let (size, cut) = brute_frontier(&csr);
            assert_eq!(
                (fb.size, fb.cut),
                (size, w.lo + cut),
                "seed {seed} window {w}"
            );
            assert_eq!(
                (fg.size, fg.cut),
                (fb.size, fb.cut),
                "seed {seed} window {w}"
            );
            assert_eq!(fg.members, fb.members);
            let a = fb.members.unwrap();
            assert_eq!(a.len() as u64, size);
            check_antichain_blocks(&blocks, w, &a).unwrap();
            let rel: Vec<u32> = a.iter().map(|&x| x - w.lo as u32).collect();
            check_antichain(&csr, &rel).unwrap();
            let f2 = frontier_blocks(&blocks, w, false);
            assert_eq!((f2.size, f2.cut, f2.members), (fb.size, fb.cut, None));
        }
    }
}

#[test]
fn frontier_generic_on_random_dags() {
    let mut rng = SmallRng::seed_from_u64(3);
    for seed in 0..60u64 {
        let g = testkit::random_dag(5 + seed as usize % 40, 0.1, seed);
        let n = g.num_nodes();
        for w in windows(n, &mut rng, 6) {
            let cg = testkit::chunked(&g, 1 + seed % 4);
            let f = frontier_generic(&cg, w, true, &mut no_log()).unwrap();
            let csr = window_csr(&g, w, 5);
            let (size, cut) = brute_frontier(&csr);
            assert_eq!((f.size, f.cut), (size, w.lo + cut));
            let rel: Vec<u32> = f
                .members
                .unwrap()
                .iter()
                .map(|&x| x - w.lo as u32)
                .collect();
            check_antichain(&csr, &rel).unwrap();
        }
    }
}

#[test]
fn frontier_generic_rejects_back_arcs() {
    let g = testkit::with_back_arc(&testkit::random_dag(20, 0.2, 5));
    let cg = testkit::chunked(&g, 3);
    let err = frontier_generic(&cg, Window::full(20), false, &mut no_log()).unwrap_err();
    assert!(matches!(
        err.downcast_ref::<DagError>(),
        Some(DagError::NotTopological { .. })
    ));
}

// -------------------------------------------------------------------------------------------
// Dinic
// -------------------------------------------------------------------------------------------

/// Minimum s–t cut capacity by subset enumeration (`n ≤ 10`).
fn brute_min_cut(n: usize, edges: &[(usize, usize, i64)], s: usize, t: usize) -> i64 {
    let mut best = i64::MAX;
    for mask in 0u32..(1 << n) {
        if mask >> s & 1 == 0 || mask >> t & 1 == 1 {
            continue;
        }
        let c: i64 = edges
            .iter()
            .filter(|&&(u, v, _)| mask >> u & 1 == 1 && mask >> v & 1 == 0)
            .map(|e| e.2)
            .sum();
        best = best.min(c);
    }
    best
}

#[test]
fn dinic_and_push_relabel_match_brute_force_min_cut() {
    let mut rng = SmallRng::seed_from_u64(4);
    for round in 0..600 {
        let n = rng.random_range(2..=9usize);
        let mut edges = Vec::new();
        for _ in 0..rng.random_range(0..=3 * n) {
            let u = rng.random_range(0..n);
            let v = rng.random_range(0..n);
            if u != v {
                edges.push((u, v, rng.random_range(0..=7i64)));
            }
        }
        let undirected = round % 3 == 0;
        let mut net = Dinic::new(n, edges.len());
        let mut ids = Vec::new();
        for &(u, v, c) in &edges {
            ids.push(if undirected {
                net.add_edge_with_rev(u, v, c, c)
            } else {
                net.add_edge(u, v, c)
            });
        }
        net.finalize();
        let (s, t) = (rng.random_range(0..n), rng.random_range(0..n));
        if s == t {
            continue;
        }
        let mut all: Vec<(usize, usize, i64)> = edges.clone();
        if undirected {
            all.extend(edges.iter().map(|&(u, v, c)| (v, u, c)));
        }
        let want = brute_min_cut(n, &all, s, t);
        for algo in 0..2 {
            let mut d = net.clone();
            let f = if algo == 0 {
                d.max_flow(s, t)
            } else {
                d.max_flow_push_relabel(s, t)
            };
            assert_eq!(f, want, "round {round} algo {algo}");
            // the residual side of s is a minimum cut
            let side = d.min_cut_side(s);
            assert!(side[s] && !side[t]);
            let cut: i64 = all
                .iter()
                .filter(|&&(u, v, _)| side[u] && !side[v])
                .map(|e| e.2)
                .sum();
            assert_eq!(cut, f, "round {round} algo {algo}");
            // capacity bounds and conservation: the result is a flow, not a preflow
            let mut excess = vec![0i64; n];
            for (k, &(u, v, c)) in edges.iter().enumerate() {
                let x = d.flow(ids[k]);
                assert!(x <= c && (undirected || x >= 0) && x >= -c);
                excess[v] += x;
                excess[u] -= x;
            }
            for (v, &e) in excess.iter().enumerate() {
                if v != s && v != t {
                    assert_eq!(e, 0, "round {round} algo {algo}: excess at {v}");
                }
            }
            assert_eq!(excess[t], f);
            // a second call finds nothing more, with either algorithm
            assert_eq!(d.max_flow(s, t), 0);
            assert_eq!(d.max_flow_push_relabel(s, t), 0);
        }
    }
}

// -------------------------------------------------------------------------------------------
// Width
// -------------------------------------------------------------------------------------------

fn assert_certified(g: &Csr, r: &WidthResult) {
    check_width_certificate(g, r).unwrap_or_else(|e| panic!("{} certificate: {e:#}", r.method));
}

/// Every node of `g` appears exactly once in `c`.
fn assert_partition(g: &Csr, c: &Chains) {
    let mut seen = vec![0u32; g.n];
    for &x in &c.nodes {
        seen[x as usize] += 1;
    }
    assert!(
        seen.iter().all(|&k| k == 1),
        "the chains are not a partition"
    );
}

#[test]
fn width_methods_agree_with_brute_force() {
    for seed in 0..300u64 {
        let n = (seed % 17) as usize;
        let p = [0.05, 0.15, 0.3, 0.6][(seed / 17 % 4) as usize];
        let g = testkit::random_dag(n, p, seed);
        let csr = csr_of(&g);
        let brute = testkit::brute_width(&g) as u64;
        let mf = width_min_flow(&csr);
        let fk = width_fulkerson(&csr);
        let mf_d = width_min_flow_with(&csr, FlowAlgo::Dinic);
        assert_certified(&csr, &mf_d);
        assert_eq!(mf_d.width, brute, "seed {seed} min-flow (Dinic)");
        assert_eq!(mf.width, brute, "seed {seed} min-flow");
        assert_eq!(fk.width, brute, "seed {seed} Fulkerson");
        assert_certified(&csr, &mf);
        assert_certified(&csr, &fk);
        assert_partition(&csr, mf.chains.as_ref().unwrap());
        assert_partition(&csr, mf_d.chains.as_ref().unwrap());
        // frontier ≤ width ≤ MPC_d
        let (fr, _) = brute_frontier(&csr);
        let mpc = min_path_cover_hk(&csr).num_paths() as u64;
        assert!(fr <= brute && brute <= mpc);
    }
}

#[test]
fn width_methods_agree_on_larger_graphs() {
    for seed in 0..30u64 {
        let g = match seed % 3 {
            0 => testkit::random_dag(150, 0.03, seed),
            1 => testkit::layered(&[20, 35, 10, 40, 25], 0.1, seed),
            _ => testkit::random_dag(400, 0.008, seed),
        };
        let csr = csr_of(&g);
        let mf = width_min_flow(&csr);
        let fk = width_fulkerson(&csr);
        assert_eq!(mf.width, fk.width, "seed {seed}");
        assert_certified(&csr, &mf);
        assert_certified(&csr, &fk);
        // the min-flow chains partition the nodes (consecutive members are comparable, which
        // check_width_certificate verifies)
        assert_partition(&csr, mf.chains.as_ref().unwrap());
    }
}

#[test]
fn closure_width_equals_min_flow_on_mini_pg_windows() {
    let mut rng = SmallRng::seed_from_u64(5);
    for (seed, g) in mini_pgs() {
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let n = g.num_nodes();
        for w in windows(n, &mut rng, 12) {
            let csr = window_csr(&g, w, 2);
            let cl = width_closure_blocks(&blocks, w);
            check_antichain_blocks(&blocks, w, &cl.antichain).unwrap();
            assert_eq!(cl.antichain.len() as u64, cl.width);
            assert!(cl.chains.is_none());
            let mf = width_min_flow(&csr);
            assert_certified(&csr, &mf);
            assert_eq!(cl.width, mf.width, "seed {seed} window {w}");
            // the Dinic variants give the same widths, with valid certificates
            let cl_d = width_closure_blocks_with(&blocks, w, FlowAlgo::Dinic);
            check_antichain_blocks(&blocks, w, &cl_d.antichain).unwrap();
            assert_eq!(cl_d.width, cl.width);
            let mf_d = width_min_flow_with(&csr, FlowAlgo::Dinic);
            assert_certified(&csr, &mf_d);
            assert_eq!(mf_d.width, cl.width);
            if csr.n <= 1 << 12 {
                assert_eq!(width_fulkerson(&csr).width, cl.width);
            }
            if csr.n <= 16 {
                assert_eq!(testkit::brute_width(&to_vec_graph(&csr)) as u64, cl.width);
            }
            // the relative closure antichain is a generic antichain too
            let rel: Vec<u32> = cl.antichain.iter().map(|&x| x - w.lo as u32).collect();
            check_antichain(&csr, &rel).unwrap();
            // frontier ≤ width ≤ MPC_d
            let fr = frontier_blocks(&blocks, w, false).size;
            let (mpc, _) = min_path_cover_blocks(&blocks, w, false);
            assert!(
                fr <= cl.width && cl.width <= mpc.mpc_d,
                "seed {seed} window {w}"
            );
        }
    }
}

#[test]
fn tiny_mini_pg_windows_match_brute_width() {
    let mut checked = 0;
    for seed in 0..200u64 {
        let g = testkit::mini_pg(4 + seed as usize % 4, seed, &MiniPgOpts::default()).graph;
        let n = g.num_nodes();
        if n > 16 {
            continue;
        }
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let w = Window::full(n);
        assert_eq!(
            width_closure_blocks(&blocks, w).width,
            testkit::brute_width(&g) as u64,
            "seed {seed}"
        );
        checked += 1;
    }
    assert!(checked > 50);
}

#[test]
fn antichain_checks_reject_a_comparable_pair() {
    let mut rejected = 0;
    for (seed, g) in mini_pgs() {
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let n = g.num_nodes();
        let w = Window::full(n);
        let cl = width_closure_blocks(&blocks, w);
        let csr = csr_of(&g);
        // plant u → v with u in the antichain and v outside it
        let Some((u, v)) = cl.antichain.iter().find_map(|&u| {
            csr.succ(u as usize)
                .iter()
                .find(|v| cl.antichain.binary_search(v).is_err())
                .map(|&v| (u, v))
        }) else {
            continue;
        };
        let mut bad = cl.antichain.clone();
        bad.push(v);
        bad.sort_unstable();
        assert!(
            check_antichain_blocks(&blocks, w, &bad).is_err(),
            "seed {seed}: ({u}, {v})"
        );
        assert!(check_antichain(&csr, &bad).is_err());
        let r = WidthResult {
            width: bad.len() as u64,
            chains: None,
            antichain: bad,
            method: "planted",
        };
        assert!(check_width_certificate(&csr, &r).is_err());
        rejected += 1;
    }
    assert!(rejected > 6);
    // unsorted, duplicate and out-of-window members are rejected too
    let g = testkit::mini_pg(50, 1, &MiniPgOpts::default()).graph;
    let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
    let w = Window {
        lo: 5,
        hi: g.num_nodes(),
    };
    assert!(check_antichain_blocks(&blocks, w, &[7, 6]).is_err());
    assert!(check_antichain_blocks(&blocks, w, &[6, 6]).is_err());
    assert!(check_antichain_blocks(&blocks, w, &[1]).is_err());
}

#[test]
fn chain_checks_reject_bad_chains() {
    // 0 → 1, 0 → 2: width 2 with antichain {1, 2}
    let g = Csr::from_arcs(3, [(0, 1), (0, 2)]);
    let r = |nodes: Vec<u32>, offsets: Vec<u64>| WidthResult {
        width: 2,
        chains: Some(Chains { nodes, offsets }),
        antichain: vec![1, 2],
        method: "manual",
    };
    check_width_certificate(&g, &r(vec![0, 1, 2], vec![0, 2, 3])).unwrap();
    // 1 and 2 are incomparable
    assert!(check_width_certificate(&g, &r(vec![1, 2, 0], vec![0, 2, 3])).is_err());
    // node 2 is not covered
    assert!(check_width_certificate(&g, &r(vec![0, 1, 1], vec![0, 2, 3])).is_err());
    // an empty chain
    assert!(check_width_certificate(&g, &r(vec![0, 1, 2], vec![0, 0, 3])).is_err());
    // wrong number of chains
    assert!(check_width_certificate(&g, &r(vec![0, 1, 2], vec![0, 1, 2, 3])).is_err());
    // the antichain is not one
    let mut bad = r(vec![0, 1, 2], vec![0, 2, 3]);
    bad.antichain = vec![0, 1];
    assert!(check_width_certificate(&g, &bad).is_err());
    // comparable through a longer path (not an arc) is accepted
    let p = Csr::from_arcs(3, [(0, 1), (1, 2)]);
    let one = WidthResult {
        width: 1,
        chains: Some(Chains {
            nodes: vec![0, 2],
            offsets: vec![0, 2],
        }),
        antichain: vec![1],
        method: "manual",
    };
    // node 1 is uncovered, so this must fail; with 1 added it passes
    assert!(check_width_certificate(&p, &one).is_err());
    let ok = WidthResult {
        chains: Some(Chains {
            nodes: vec![0, 2, 1],
            offsets: vec![0, 3],
        }),
        ..one
    };
    assert!(check_width_certificate(&p, &ok).is_err()); // 2 does not reach 1
    let ok = WidthResult {
        chains: Some(Chains {
            nodes: vec![1, 0, 2],
            offsets: vec![0, 3],
        }),
        ..ok
    };
    assert!(check_width_certificate(&p, &ok).is_err()); // 1 does not reach 0
    let ok = WidthResult {
        chains: Some(Chains {
            nodes: vec![0, 1, 2],
            offsets: vec![0, 3],
        }),
        ..ok
    };
    check_width_certificate(&p, &ok).unwrap();
    let skip = WidthResult {
        chains: Some(Chains {
            nodes: vec![0, 2, 1],
            offsets: vec![0, 2, 3],
        }),
        width: 2,
        antichain: vec![1, 2],
        ..ok
    };
    // {1, 2} is not an antichain of the path
    assert!(check_width_certificate(&p, &skip).is_err());
}

#[test]
fn width_of_special_graphs_and_additivity() {
    for n in [1usize, 2, 7, 30] {
        let complete = csr_of(&testkit::complete_dag(n));
        assert_eq!(width_min_flow(&complete).width, 1);
        let iso = csr_of(&testkit::isolated(n));
        let r = width_min_flow(&iso);
        assert_eq!(r.width, n as u64);
        assert_certified(&iso, &r);
        assert_eq!(width_min_flow(&csr_of(&testkit::chain(n))).width, 1);
    }
    assert_eq!(width_min_flow(&csr_of(&testkit::out_star(9))).width, 8);
    assert_eq!(width_min_flow(&csr_of(&testkit::in_star(9))).width, 8);
    assert_eq!(width_min_flow(&csr_of(&testkit::diamonds(4))).width, 2);
    let empty = csr_of(&testkit::isolated(0));
    let r = width_min_flow(&empty);
    assert_eq!((r.width, r.chains.as_ref().unwrap().len()), (0, 0));
    assert_certified(&empty, &r);
    for seed in 0..20u64 {
        let parts: Vec<VecGraph> = (0..3)
            .map(|i| testkit::random_dag(10 + 7 * i, 0.15, seed * 3 + i as u64))
            .collect();
        let sum: u64 = parts.iter().map(|g| width_min_flow(&csr_of(g)).width).sum();
        let union = csr_of(&testkit::disjoint_union(&parts));
        let r = width_min_flow(&union);
        assert_eq!(r.width, sum, "seed {seed}");
        assert_certified(&union, &r);
        let mpc_sum: usize = parts
            .iter()
            .map(|g| min_path_cover_hk(&csr_of(g)).num_paths())
            .sum();
        assert_eq!(min_path_cover_hk(&union).num_paths(), mpc_sum);
    }
}

#[test]
fn empty_windows_and_tiny_graphs() {
    let g = testkit::mini_pg(30, 9, &MiniPgOpts::default()).graph;
    let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
    let n = g.num_nodes();
    for lo in [0, n / 2, n] {
        let w = Window { lo, hi: lo };
        let (st, pc) = min_path_cover_blocks(&blocks, w, true);
        assert_eq!((st.n, st.matching, st.mpc_d), (0, 0, 0));
        assert!(pc.unwrap().next.is_empty());
        let f = frontier_blocks(&blocks, w, true);
        assert_eq!((f.size, f.cut), (0, lo));
        let cg = testkit::chunked(&g, 4);
        let fg = frontier_generic(&cg, w, true, &mut no_log()).unwrap();
        assert_eq!((fg.size, fg.cut), (0, lo));
        let cl = width_closure_blocks(&blocks, w);
        assert_eq!((cl.width, cl.antichain.len()), (0, 0));
    }
    let one = csr_of(&testkit::isolated(1));
    assert_eq!(hopcroft_karp(&one).size, 0);
    assert_eq!(width_min_flow(&one).width, 1);
    assert_eq!(width_fulkerson(&one).width, 1);
}

#[test]
fn bv_round_trip_agrees_with_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    for seed in 0..4u64 {
        let g = testkit::mini_pg(300, seed, &MiniPgOpts::default()).graph;
        let base = testkit::write_bv(dir.path(), &format!("g{seed}"), &g);
        let conf = dag::graph::ChunkConf {
            chunk_arcs: 1 + seed * 13,
            max_chunk_nodes: 64,
        };
        let cg = dag::graph::open(&base, conf).unwrap();
        let (bv_blocks, _) = BlockIndex::build(&cg, &mut no_log()).unwrap();
        let (mem_blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let n = g.num_nodes();
        for w in [
            Window::full(n),
            Window { lo: n / 3, hi: n },
            Window {
                lo: n / 5,
                hi: 2 * n / 3,
            },
        ] {
            let f_bv = frontier_generic(&cg, w, true, &mut no_log()).unwrap();
            let f_mem = frontier_blocks(&mem_blocks, w, true);
            assert_eq!(f_bv, f_mem);
            assert_eq!(frontier_blocks(&bv_blocks, w, true), f_mem);
            let csr_bv = Csr::from_window(&cg, w, &mut no_log()).unwrap();
            assert_eq!(csr_bv, window_csr(&g, w, 7));
            assert_eq!(
                width_min_flow(&csr_bv).width,
                width_closure_blocks(&bv_blocks, w).width
            );
            assert_eq!(
                min_path_cover_blocks(&bv_blocks, w, false).0.matching,
                hopcroft_karp(&csr_bv).size as u64
            );
        }
    }
}

/// A "broom": `k` sources into the head of a path of `k` nodes, whose tail feeds `k` sinks.
/// Width `k`, but every one of the `k` flow paths walks the whole handle, so a decomposition
/// into paths would hold `Θ(k²)` ids.
fn broom(k: usize) -> Csr {
    let n = 3 * k;
    let mut arcs = Vec::new();
    for s in 0..k {
        arcs.push((s, k));
    }
    for x in k..2 * k - 1 {
        arcs.push((x, x + 1));
    }
    for t in 2 * k..3 * k {
        arcs.push((2 * k - 1, t));
    }
    Csr::from_arcs(n, arcs)
}

#[test]
fn min_flow_chains_are_a_partition_on_brooms() {
    for k in [1usize, 2, 7, 100, 1000, 4000] {
        let g = broom(k);
        for algo in [FlowAlgo::PushRelabel, FlowAlgo::Dinic] {
            let r = width_min_flow_with(&g, algo);
            assert_eq!(r.width, k as u64, "k = {k}");
            let c = r.chains.as_ref().unwrap();
            // n ids in all, not Θ(k²)
            assert_eq!(c.nodes.len(), g.n, "k = {k}");
            assert_partition(&g, c);
            assert_certified(&g, &r);
        }
    }
}

/// `(u, x, v)` with `u → x → v` and `u ↛ v` as an arc, all inside `[lo, hi)`.
fn two_step(g: &Csr) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for u in 0..g.n {
        for &x in g.succ(u) {
            for &v in g.succ(x as usize) {
                if g.succ(u).binary_search(&v).is_err() {
                    out.push((u as u32, x, v));
                }
            }
        }
    }
    out
}

#[test]
fn antichain_checks_follow_paths_of_length_two_or_more() {
    // hand-made: 0 → 1 → 2, 3 isolated; {0, 2} is comparable only through 1
    let g = Csr::from_arcs(4, [(0, 1), (1, 2)]);
    assert!(check_antichain(&g, &[0, 2]).is_err());
    assert!(check_antichain(&g, &[0, 3]).is_ok());
    let r = WidthResult {
        width: 2,
        chains: None,
        antichain: vec![0, 2],
        method: "planted",
    };
    assert!(check_width_certificate(&g, &r).is_err());
    // a longer path: 0 → 1 → 2 → 3 → 4
    let g = Csr::from_arcs(5, [(0, 1), (1, 2), (2, 3), (3, 4)]);
    assert!(check_antichain(&g, &[0, 4]).is_err());
    // mini_pg windows (line digraphs: u → x → v crosses two transactions)
    let mut rng = SmallRng::seed_from_u64(77);
    let mut rejected = 0;
    for (seed, g) in mini_pgs() {
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        for w in windows(g.num_nodes(), &mut rng, 6) {
            let csr = window_csr(&g, w, 7);
            let lo = w.lo as u32;
            let cands = two_step(&csr);
            for &(u, x, v) in cands.iter().step_by(cands.len().div_ceil(20).max(1)) {
                let a = [u, v];
                assert!(
                    check_antichain(&csr, &a).is_err(),
                    "seed {seed} {w}: {u} → {x} → {v}"
                );
                assert!(
                    check_antichain_blocks(&blocks, w, &[u + lo, v + lo]).is_err(),
                    "seed {seed} {w}: {u} → {x} → {v} (blocks)"
                );
                let r = WidthResult {
                    width: 2,
                    chains: None,
                    antichain: a.to_vec(),
                    method: "planted",
                };
                assert!(check_width_certificate(&csr, &r).is_err());
                rejected += 1;
            }
        }
    }
    assert!(rejected > 100, "only {rejected} planted pairs");
}

#[test]
fn antichain_checks_match_brute_force_pairwise() {
    let mut rng = SmallRng::seed_from_u64(5);
    for seed in 0..200u64 {
        let n = 2 + (seed % 30) as usize;
        let p = [0.05, 0.15, 0.4][(seed % 3) as usize];
        let g = testkit::random_dag(n, p, seed);
        let csr = csr_of(&g);
        let tc = testkit::closure(&g);
        for _ in 0..20 {
            let a: Vec<u32> = (0..n as u32).filter(|_| rng.random_bool(0.3)).collect();
            let brute = a.iter().all(|&u| {
                a.iter()
                    .all(|&v| !testkit::bit(&tc[u as usize], v as usize))
            });
            assert_eq!(
                check_antichain(&csr, &a).is_ok(),
                brute,
                "seed {seed}: {a:?}"
            );
        }
    }
    // the same on line digraphs, against the BlockIndex checker
    for (seed, g) in mini_pgs().into_iter().take(8) {
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let tc = testkit::closure(&g);
        let n = g.num_nodes();
        for w in windows(n, &mut rng, 5) {
            for _ in 0..10 {
                let a: Vec<u32> = (w.lo as u32..w.hi as u32)
                    .filter(|_| rng.random_bool(0.05))
                    .collect();
                let brute = a.iter().all(|&u| {
                    a.iter()
                        .all(|&v| !testkit::bit(&tc[u as usize], v as usize))
                });
                assert_eq!(
                    check_antichain_blocks(&blocks, w, &a).is_ok(),
                    brute,
                    "seed {seed} {w}: {a:?}"
                );
            }
        }
    }
}
