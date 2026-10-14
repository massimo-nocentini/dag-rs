//! Acceptance tests of the transitive reduction.
//!
//! Every algorithm is checked against brute-force oracles: a Floyd–Warshall boolean closure
//! (n ≤ 64) and the BFS closure of `testkit::closure` (larger n).

#[path = "support/cli.rs"]
mod cli;

use std::path::Path;

use dag::DagError;
use dag::blocks::BlockIndex;
use dag::csr::Csr;
use dag::error::BlockViolation;
use dag::graph::{self, ChunkConf, ChunkedGraph};
use dag::io::{MemBudget, read_array};
use dag::reduce::{
    self, ReductionReport, exact, h3_failures, h3_failures_in, line_digraph_window, min_max_succ,
    redundant_arcs_of, transitive_reduction_bitset, transitive_reduction_search, verify_reduced,
    verify_reduced_window,
};
use dag::sweep::SweepConf;
use dag::testkit::{self, MiniPgOpts, bit};
use dag::window::Window;
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;

type Arcs = Vec<(usize, usize)>;

fn npl() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

// ---------------------------------------------------------------------------------------------
// Oracles and helpers
// ---------------------------------------------------------------------------------------------

/// Floyd–Warshall boolean closure on n ≤ 64 (row u = nodes reachable by a path of length ≥ 1).
fn fw_closure(n: usize, arcs: &[(usize, usize)]) -> Vec<u64> {
    assert!(n <= 64);
    let mut c = vec![0u64; n];
    for &(u, v) in arcs {
        c[u] |= 1 << v;
    }
    for k in 0..n {
        for i in 0..n {
            if c[i] >> k & 1 == 1 {
                c[i] |= c[k];
            }
        }
    }
    c
}

/// The redundant arcs of `g` (sorted), by brute force: `(u, v)` is redundant iff some `x` has
/// `u ⇝ x` and `x ⇝ v`.
fn oracle_redundant(g: &VecGraph) -> Arcs {
    let n = g.num_nodes();
    let a = testkit::arcs(g);
    if n <= 64 {
        let c = fw_closure(n, &a);
        a.into_iter()
            .filter(|&(u, v)| (0..n).any(|x| c[u] >> x & 1 == 1 && c[x] >> v & 1 == 1))
            .collect()
    } else {
        let c = testkit::closure(g);
        let succ = testkit::succ_lists(g);
        a.into_iter()
            .filter(|&(u, v)| succ[u].iter().any(|&x| x != v && bit(&c[x], v)))
            .collect()
    }
}

/// The oracle transitive reduction of `g`.
fn oracle_tr(g: &VecGraph) -> Arcs {
    let red = oracle_redundant(g);
    testkit::arcs(g)
        .into_iter()
        .filter(|a| red.binary_search(a).is_err())
        .collect()
}

fn csr_arcs(c: &Csr) -> Arcs {
    (0..c.n)
        .flat_map(|u| c.succ(u).iter().map(move |&v| (u, v as usize)))
        .collect()
}

fn to_usize(red: &[(u32, u32)]) -> Arcs {
    red.iter().map(|&(u, v)| (u as usize, v as usize)).collect()
}

/// The subgraph induced by `w` (ids shifted by −lo).
fn induced(g: &VecGraph, w: Window) -> VecGraph {
    testkit::vec_graph(
        w.len(),
        testkit::arcs(g)
            .into_iter()
            .filter(|&(u, v)| w.contains(u) && w.contains(v))
            .map(|(u, v)| (u - w.lo, v - w.lo)),
    )
}

fn verify_generic(cg: &ChunkedGraph<VecGraph>, w: Window) -> (ReductionReport, Arcs) {
    let (r, red) = verify_reduced_window(
        cg,
        w,
        None,
        &SweepConf::default(),
        &mut MemBudget::unlimited(),
        &mut npl(),
    )
    .expect("verify");
    (r, to_usize(&red))
}

fn random_window(n: usize, r: &mut SmallRng) -> Window {
    let a = r.random_range(0..=n);
    let b = r.random_range(a..=n);
    Window::new(a, b, n).unwrap()
}

/// Checks the report fields against the redundant list.
fn check_report(r: &ReductionReport, red: &Arcs, arcs: usize) {
    assert_eq!(r.arcs, arcs as u64);
    assert_eq!(r.h1_violations, 0);
    assert_eq!(r.redundant, red.len() as u64);
    assert_eq!(r.is_transitively_reduced, red.is_empty());
    assert_eq!(r.certificate == "H3", r.h3_failing_nodes == 0);
    if r.certificate == "exact" {
        assert!(r.exact_checked_arcs >= r.redundant);
    }
}

// ---------------------------------------------------------------------------------------------
// Exact algorithms vs oracles
// ---------------------------------------------------------------------------------------------

#[test]
fn bitset_and_search_match_floyd_warshall() {
    let mut r = SmallRng::seed_from_u64(1);
    for seed in 0..400 {
        let n = r.random_range(0..=64);
        let p = [0.02, 0.05, 0.1, 0.3, 0.7][seed as usize % 5];
        let g = testkit::random_dag(n, p, seed);
        let want = oracle_tr(&g);
        let c = Csr::from_vec_graph(&g);
        let a = csr_arcs(&transitive_reduction_bitset(&c));
        let b = csr_arcs(&transitive_reduction_search(&c));
        assert_eq!(a, want, "bitset, seed {seed}, n {n}, p {p}");
        assert_eq!(b, want, "search, seed {seed}, n {n}, p {p}");
    }
}

#[test]
fn reduction_invariants_on_larger_dags() {
    for seed in 0..12 {
        let n = 100 + 20 * seed as usize;
        let p = [0.01, 0.03, 0.08][seed as usize % 3];
        let g = testkit::random_dag(n, p, 100 + seed);
        let c = Csr::from_vec_graph(&g);
        let tr = transitive_reduction_bitset(&c);
        assert_eq!(tr, transitive_reduction_search(&c), "seed {seed}");
        let trg = tr.to_vec_graph();
        // TR ⊆ G
        let ga = testkit::arcs(&g);
        let ta = testkit::arcs(&trg);
        assert!(ta.iter().all(|a| ga.binary_search(a).is_ok()));
        // same closure
        assert_eq!(testkit::closure(&trg), testkit::closure(&g), "seed {seed}");
        // minimality: no arc of TR is implied by a longer path of TR
        assert!(oracle_redundant(&trg).is_empty(), "seed {seed}");
        // idempotence
        assert_eq!(transitive_reduction_bitset(&tr), tr);
        assert_eq!(transitive_reduction_search(&tr), tr);
        // equals the oracle
        assert_eq!(ta, oracle_tr(&g), "seed {seed}");
    }
}

#[test]
fn planted_redundant_arcs_are_removed_and_reported() {
    let mut r = SmallRng::seed_from_u64(7);
    for seed in 0..60 {
        let n = r.random_range(5..90);
        let h = testkit::random_dag(n, 0.06, seed);
        let t = transitive_reduction_bitset(&Csr::from_vec_graph(&h)).to_vec_graph();
        let tarcs = testkit::arcs(&t);
        let c = testkit::closure(&t);
        // plant arcs (u, v) with v reachable from u but (u, v) not in T
        let mut planted = Vec::new();
        for _ in 0..n {
            let u = r.random_range(0..n);
            let reach: Vec<usize> = (u + 1..n)
                .filter(|&v| bit(&c[u], v) && tarcs.binary_search(&(u, v)).is_err())
                .collect();
            if !reach.is_empty() {
                planted.push((u, reach[r.random_range(0..reach.len())]));
            }
        }
        planted.sort_unstable();
        planted.dedup();
        let g = testkit::vec_graph(n, tarcs.iter().copied().chain(planted.iter().copied()));
        let gc = Csr::from_vec_graph(&g);
        assert_eq!(csr_arcs(&transitive_reduction_bitset(&gc)), tarcs);
        assert_eq!(csr_arcs(&transitive_reduction_search(&gc)), tarcs);
        let cg = testkit::chunked(&g, 1 + seed % 7);
        let (rep, red) = verify_generic(&cg, Window::full(n));
        assert_eq!(red, planted, "seed {seed}");
        check_report(&rep, &red, testkit::arcs(&g).len());
    }
}

#[test]
fn triangle_reports_the_shortcut() {
    // u = 0 → w = 1 → v = 2 and the shortcut 0 → 2
    let g = testkit::vec_graph(3, [(0, 1), (1, 2), (0, 2)]);
    let cg = testkit::chunked(&g, 1);
    let (rep, red) = verify_generic(&cg, Window::full(3));
    assert_eq!(red, vec![(0, 2)]);
    assert_eq!(rep.certificate, "exact");
    assert_eq!(rep.h3_failing_nodes, 1);
    assert!(!rep.is_transitively_reduced);
    check_report(&rep, &red, 3);
    // H3 threshold t_0 = F[1] = 2
    let mm = min_max_succ(&cg, Window::full(3), &mut npl()).unwrap();
    assert_eq!(mm.f, vec![1, 2, u32::MAX]);
    assert_eq!(mm.l, vec![2, 2, u32::MAX]);
    assert_eq!(
        h3_failures(&cg, &mm.f, &mm.l, &mut npl()).unwrap(),
        vec![(0, 2)]
    );
    let mut bits = dag::bitset::DirtyBitset::new();
    let mut stack = Vec::new();
    assert_eq!(redundant_arcs_of(&g, 0, 0, &mut bits, &mut stack), vec![2]);
    assert_eq!(
        redundant_arcs_of(&g, 1, 0, &mut bits, &mut stack),
        Vec::<u32>::new()
    );
    let c = Csr::from_vec_graph(&g);
    assert_eq!(
        csr_arcs(&transitive_reduction_bitset(&c)),
        vec![(0, 1), (1, 2)]
    );
    assert_eq!(
        csr_arcs(&transitive_reduction_search(&c)),
        vec![(0, 1), (1, 2)]
    );
    // a window [lo, hi) that contains both ends of the shortcut also contains the middle node
    // (ids are topological), so the shortcut is reported on [0, 4); windows holding only one
    // end of it, or the middle node and the head only, have no redundant arc
    let g2 = testkit::vec_graph(4, [(0, 1), (1, 3), (0, 3), (0, 2)]);
    let cg2 = testkit::chunked(&g2, 2);
    let (rep2, red2) = verify_generic(&cg2, Window::new(0, 4, 4).unwrap());
    assert_eq!(red2, vec![(0, 3)]);
    check_report(&rep2, &red2, 4);
    for (lo, hi) in [(0, 3), (1, 4), (0, 2)] {
        let (rep3, red3) = verify_generic(&cg2, Window::new(lo, hi, 4).unwrap());
        assert!(red3.is_empty(), "window [{lo}, {hi})");
        assert!(rep3.is_transitively_reduced, "window [{lo}, {hi})");
    }
}

#[test]
fn h3_implies_reduced() {
    let (mut holds, mut fails) = (0, 0);
    for seed in 0..300 {
        let n = 10 + (seed as usize % 50);
        let p = [0.01, 0.02, 0.04, 0.1][seed as usize % 4];
        let g = testkit::random_dag(n, p, 1000 + seed);
        let cg = testkit::chunked(&g, 1 + seed % 5);
        let mm = min_max_succ(&cg, Window::full(n), &mut npl()).unwrap();
        let f = h3_failures(&cg, &mm.f, &mm.l, &mut npl()).unwrap();
        let red = oracle_redundant(&g);
        if f.is_empty() {
            holds += 1;
            assert!(
                red.is_empty(),
                "H3 holds but seed {seed} has redundant arcs"
            );
        } else {
            fails += 1;
        }
        // every redundant arc (u, v) has a failing u with t_u ≤ v (completeness of the check)
        for &(u, v) in &red {
            let t = f.iter().find(|&&(x, _)| x as usize == u).map(|&(_, t)| t);
            assert!(
                matches!(t, Some(t) if t as usize <= v),
                "seed {seed}: ({u}, {v})"
            );
        }
    }
    assert!(holds > 20 && fails > 20, "holds {holds}, fails {fails}");
}

#[test]
fn verifier_matches_oracle_on_windows_and_chunk_sizes() {
    let mut r = SmallRng::seed_from_u64(11);
    for seed in 0..300 {
        let n = r.random_range(0..70);
        let p = [0.03, 0.08, 0.2][seed as usize % 3];
        let g = testkit::random_dag(n, p, 2000 + seed);
        let cg = testkit::chunked(&g, 1 + seed % 7);
        let w = if seed % 3 == 0 {
            Window::full(n)
        } else {
            random_window(n, &mut r)
        };
        let sub = induced(&g, w);
        let want: Arcs = oracle_redundant(&sub)
            .into_iter()
            .map(|(u, v)| (u + w.lo, v + w.lo))
            .collect();
        let (rep, red) = verify_generic(&cg, w);
        assert_eq!(red, want, "seed {seed}, window {w}");
        check_report(&rep, &red, testkit::arcs(&sub).len());
        assert!(!rep.line_digraph);
    }
}

#[test]
fn edge_cases() {
    let cases: Vec<(&str, VecGraph)> = vec![
        ("empty", testkit::isolated(0)),
        ("single", testkit::isolated(1)),
        ("isolated", testkit::isolated(5)),
        ("chain", testkit::chain(6)),
        ("complete", testkit::complete_dag(8)),
        ("out_star", testkit::out_star(7)),
        ("in_star", testkit::in_star(7)),
        ("diamonds", testkit::diamonds(4)),
        ("ladder", testkit::ladder(5)),
        (
            "union",
            testkit::disjoint_union(&[testkit::complete_dag(4), testkit::chain(3)]),
        ),
    ];
    for (name, g) in cases {
        let n = g.num_nodes();
        let want = oracle_tr(&g);
        let c = Csr::from_vec_graph(&g);
        assert_eq!(csr_arcs(&transitive_reduction_bitset(&c)), want, "{name}");
        assert_eq!(csr_arcs(&transitive_reduction_search(&c)), want, "{name}");
        let cg = testkit::chunked(&g, 3);
        let (rep, red) = verify_generic(&cg, Window::full(n));
        assert_eq!(red, oracle_redundant(&g), "{name}");
        check_report(&rep, &red, testkit::arcs(&g).len());
        // the empty window
        let (rep0, red0) = verify_generic(&cg, Window::new(n / 2, n / 2, n).unwrap());
        assert!(
            red0.is_empty() && rep0.arcs == 0 && rep0.certificate == "H3",
            "{name}"
        );
    }
    // the complete DAG reduces to the Hamiltonian path
    let k = transitive_reduction_bitset(&Csr::from_vec_graph(&testkit::complete_dag(9)));
    assert_eq!(csr_arcs(&k), (0..8).map(|i| (i, i + 1)).collect::<Arcs>());
}

#[test]
fn bitset_rows_cross_word_boundaries() {
    // long chains across several 64-bit words with shortcuts at word edges
    let n = 300;
    let mut a: Arcs = (0..n - 1).map(|i| (i, i + 1)).collect();
    for &(u, v) in &[
        (0, 64),
        (63, 64),
        (62, 129),
        (127, 128),
        (1, 299),
        (190, 256),
    ] {
        a.push((u, v));
    }
    let g = testkit::vec_graph(n, a);
    let c = Csr::from_vec_graph(&g);
    let want: Arcs = (0..n - 1).map(|i| (i, i + 1)).collect();
    assert_eq!(csr_arcs(&transitive_reduction_bitset(&c)), want);
    assert_eq!(csr_arcs(&transitive_reduction_search(&c)), want);
    let (rep, red) = verify_generic(&testkit::chunked(&g, 16), Window::full(n));
    assert_eq!(red, oracle_redundant(&g));
    // (63, 64) and (127, 128) duplicate chain arcs
    check_report(&rep, &red, n - 1 + 4);
    assert_eq!(red.len(), 4);
}

#[test]
fn back_arc_is_an_error() {
    for seed in 0..20 {
        let g = testkit::with_back_arc(&testkit::random_dag(30, 0.1, 3000 + seed));
        let cg = testkit::chunked(&g, 4);
        let err = verify_reduced(
            &cg,
            None,
            &SweepConf::default(),
            &mut MemBudget::unlimited(),
            &mut npl(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<DagError>(),
                Some(DagError::NotTopological { .. })
            ),
            "{err:#}"
        );
    }
    let g = testkit::with_self_loop(&testkit::chain(5));
    let err = verify_reduced(
        &testkit::chunked(&g, 2),
        None,
        &SweepConf::default(),
        &mut MemBudget::unlimited(),
        &mut npl(),
    )
    .unwrap_err();
    assert_eq!(
        err.downcast_ref::<DagError>().map(|e| e.to_string()),
        Some(DagError::NotTopological { u: 2, v: 2 }.to_string())
    );
}

// ---------------------------------------------------------------------------------------------
// Certificates on PG-like graphs
// ---------------------------------------------------------------------------------------------

#[test]
fn mini_pg_is_reduced_via_blocks_and_h3() {
    let mut r = SmallRng::seed_from_u64(5);
    for seed in 0..40 {
        let opts = MiniPgOpts {
            zero_output_prob: if seed % 2 == 0 { 0.0 } else { 0.1 },
            ..Default::default()
        };
        let pg = testkit::mini_pg(60 + seed as usize * 5, seed, &opts);
        let g = &pg.graph;
        let n = g.num_nodes();
        let m = testkit::arcs(g).len() as u64;
        let (idx, _) = BlockIndex::from_vec_graph(g).expect("mini_pg is a line digraph");
        let cg = testkit::chunked(g, 1 + seed % 7);
        let (rep, red) = verify_reduced(
            &cg,
            Some(&idx),
            &SweepConf::default(),
            &mut MemBudget::unlimited(),
            &mut npl(),
        )
        .unwrap();
        assert!(red.is_empty());
        assert_eq!(rep, ReductionReport::from_blocks(m));
        // generic H3 path: zero failing nodes
        let (rep2, red2) = verify_generic(&cg, Window::full(n));
        assert!(red2.is_empty() && rep2.h3_failing_nodes == 0 && rep2.certificate == "H3");
        assert_eq!(rep2.arcs, m);
        // window-local line-digraph check and windowed blocks certificate
        for _ in 0..4 {
            let w = random_window(n, &mut r);
            let sub_arcs = testkit::arcs(&induced(g, w)).len() as u64;
            let ld = line_digraph_window(&cg, w, &mut MemBudget::unlimited(), &mut npl()).unwrap();
            assert!(ld.ok(), "seed {seed}, window {w}: {ld:?}");
            assert_eq!(ld.arcs, sub_arcs);
            let (rw, _) = verify_reduced_window(
                &cg,
                w,
                Some(&idx),
                &SweepConf::default(),
                &mut MemBudget::unlimited(),
                &mut npl(),
            )
            .unwrap();
            assert_eq!(rw.arcs, sub_arcs);
            let (rg, redg) = verify_generic(&cg, w);
            assert!(redg.is_empty() && rg.certificate == "H3");
        }
        // the number of blocks is that of the ground truth
        let ld = line_digraph_window(
            &cg,
            Window::full(n),
            &mut MemBudget::unlimited(),
            &mut npl(),
        )
        .unwrap();
        assert_eq!(ld.blocks, pg.blocks().len() as u64);
        assert_eq!(oracle_redundant(g), Arcs::new());
    }
}

#[test]
fn mini_pg_double_spend_is_detected_and_checked_exactly() {
    let mut with_redundancy = 0;
    for seed in 0..60 {
        let opts = MiniPgOpts {
            double_spend: true,
            max_outputs: 3,
            coinbase_prob: 0.05,
            ..Default::default()
        };
        let pg = testkit::mini_pg(40 + seed as usize, seed, &opts);
        let g = &pg.graph;
        let n = g.num_nodes();
        let err = BlockIndex::from_vec_graph(g).unwrap_err();
        assert!(matches!(
            err.downcast_ref::<DagError>(),
            Some(DagError::NotBiclique { .. })
        ));
        let cg = testkit::chunked(g, 1 + seed % 5);
        let ld = line_digraph_window(
            &cg,
            Window::full(n),
            &mut MemBudget::unlimited(),
            &mut npl(),
        )
        .unwrap();
        assert!(!ld.ok(), "seed {seed}");
        let (rep, red) = verify_generic(&cg, Window::full(n));
        assert_eq!(red, oracle_redundant(g), "seed {seed}");
        check_report(&rep, &red, testkit::arcs(g).len());
        if !red.is_empty() {
            with_redundancy += 1;
        }
    }
    assert!(
        with_redundancy > 0,
        "no double spend created a redundant arc"
    );
}

#[test]
fn line_digraph_violations_are_pinned() {
    let check = |n: usize, arcs: &[(usize, usize)], want: Option<(usize, BlockViolation)>| {
        let g = testkit::vec_graph(n, arcs.iter().copied());
        for ca in 1..4 {
            let ld = line_digraph_window(
                &testkit::chunked(&g, ca),
                Window::full(n),
                &mut MemBudget::unlimited(),
                &mut npl(),
            )
            .unwrap();
            assert_eq!(ld.violation, want, "{arcs:?}");
        }
    };
    check(3, &[(0, 1), (1, 2)], None);
    check(3, &[(1, 0)], Some((1, BlockViolation::BackwardArc)));
    check(
        5,
        &[(0, 2), (0, 4)],
        Some((0, BlockViolation::NonContiguous)),
    );
    check(
        4,
        &[(0, 2), (0, 3), (1, 2)],
        Some((2, BlockViolation::SameStartDifferentLength)),
    );
    check(
        5,
        &[(0, 2), (0, 3), (1, 3), (1, 4)],
        Some((3, BlockViolation::Overlap)),
    );
    // identical lists are fine; restricting to a window can make a list contiguous again
    check(4, &[(0, 2), (0, 3), (1, 2), (1, 3)], None);
    let g = testkit::vec_graph(5, [(0, 2), (0, 4)]);
    let ld = line_digraph_window(
        &testkit::chunked(&g, 1),
        Window::new(0, 4, 5).unwrap(),
        &mut MemBudget::unlimited(),
        &mut npl(),
    )
    .unwrap();
    assert!(ld.ok() && ld.arcs == 1 && ld.blocks == 1);
}

#[test]
fn non_contiguous_mini_pg_falls_back_to_exact() {
    for seed in 0..20 {
        let opts = MiniPgOpts {
            non_contiguous: true,
            ..Default::default()
        };
        let pg = testkit::mini_pg(50, seed, &opts);
        let g = &pg.graph;
        let n = g.num_nodes();
        let cg = testkit::chunked(g, 3);
        let ld = line_digraph_window(
            &cg,
            Window::full(n),
            &mut MemBudget::unlimited(),
            &mut npl(),
        )
        .unwrap();
        assert!(!ld.ok());
        let (rep, red) = verify_generic(&cg, Window::full(n));
        assert_eq!(red, oracle_redundant(g));
        check_report(&rep, &red, testkit::arcs(g).len());
    }
}

// ---------------------------------------------------------------------------------------------
// Memory budget, thread counts
// ---------------------------------------------------------------------------------------------

#[test]
fn budget_is_enforced_and_threads_are_capped() {
    let g = testkit::random_dag(200, 0.05, 77);
    let cg = testkit::chunked(&g, 8);
    let n = g.num_nodes();
    // F and L do not fit
    let mut tiny = MemBudget::new(1e-9);
    let err = verify_reduced(&cg, None, &SweepConf::default(), &mut tiny, &mut npl()).unwrap_err();
    assert!(matches!(
        err.downcast_ref::<DagError>(),
        Some(DagError::OverBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    // exactly F, L, the H3-failure list (8 B per failing node, regression)
    // and one search thread: same answer as the unlimited run; one byte less fails
    let mm = min_max_succ(&cg, Window::full(n), &mut npl()).unwrap();
    let fails = h3_failures(&cg, &mm.f, &mm.l, &mut npl()).unwrap();
    assert!(!fails.is_empty());
    let span = fails
        .iter()
        .map(|&(u, _)| (mm.l[u as usize] - u) as usize)
        .max()
        .unwrap();
    let bytes = reduce::MinMax::bytes(n) + 8 * fails.len() as u64 + exact::search_bytes(span);
    let mut short = MemBudget::new((bytes - 1) as f64 / 1e9);
    let err = verify_reduced(&cg, None, &SweepConf::default(), &mut short, &mut npl()).unwrap_err();
    assert!(matches!(
        err.downcast_ref::<DagError>(),
        Some(DagError::OverBudget { .. })
    ));
    assert_eq!(short.used(), 0, "reservations are released on failure");
    let mut one = MemBudget::new(bytes as f64 / 1e9);
    let (r1, red1) =
        verify_reduced(&cg, None, &SweepConf::default(), &mut one, &mut npl()).unwrap();
    assert_eq!(one.used(), 0, "reservations are released");
    let (r2, red2) = verify_generic(&cg, Window::full(n));
    assert_eq!((r1, to_usize(&red1)), (r2, red2));
    // the line-digraph check needs 4 B per window node
    let mut small = MemBudget::new(3.0 * n as f64 / 1e9);
    assert!(line_digraph_window(&cg, Window::full(n), &mut small, &mut npl()).is_err());
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let g = testkit::random_dag(400, 0.02, 99);
    let c = Csr::from_vec_graph(&g);
    let mut prev = None;
    for t in [1, 4, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        let res = pool.install(|| {
            let cg = testkit::chunked(&g, 5);
            (
                verify_generic(&cg, Window::full(400)),
                transitive_reduction_search(&c),
            )
        });
        if let Some(p) = &prev {
            assert_eq!(p, &res, "threads {t}");
        }
        prev = Some(res);
    }
}

/// Regression: `tr reduce --algo search` reserved the whole pool at a
/// pessimistic `4·span` bytes of DFS stack per thread and refused to run (OverBudget on the
/// window `0..10⁸` of the Payment Graph under `--max-mem 20`). The per-thread reservation is now
/// the bitset plus a bounded stack allowance, and the thread count degrades to what fits.
#[test]
fn search_reduction_threads_are_capped_by_the_budget() {
    // a path 0 → 1 → … → n−1 plus the long arc 0 → n−1 (redundant): span n − 1
    let n = 1 << 16;
    let g = Csr::from_arcs(n, (0..n - 1).map(|u| (u, u + 1)).chain([(0, n - 1)]));
    assert_eq!(exact::search_span(&g), n - 1);
    let m = g.num_arcs() as u64;
    let per = exact::search_bytes(n - 1);
    let gb = |bytes: u64| bytes as f64 / 1e9;
    let lo = 5;
    let want = vec![(lo as u32, (n - 1 + lo) as u32)];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    pool.install(|| {
        for (bytes, threads) in [
            (m + 2 * per + per / 2, 2),
            (m + per + per / 2, 1),
            (u64::MAX / 2, 8),
        ] {
            let mut b = MemBudget::new(gb(bytes));
            let out = exact::redundant_arcs_search(&g, lo, &mut b, &mut npl()).unwrap();
            assert_eq!(out.threads, threads, "budget {bytes}");
            assert_eq!(out.redundant, want, "budget {bytes}");
            assert_eq!(b.used(), 0, "reservations are released");
        }
        // not even one thread fits
        let mut b = MemBudget::new(gb(m + per - 1));
        let err = exact::redundant_arcs_search(&g, lo, &mut b, &mut npl()).unwrap_err();
        assert!(matches!(
            err.downcast_ref::<DagError>(),
            Some(DagError::OverBudget { .. })
        ));
        assert_eq!(b.used(), 0, "reservations are released on failure");
    });
    // the stack allowance is bounded: on the full Payment Graph (span ≈ 2.2·10⁹) 112 threads
    // need under 50 GB, against about 1 TB with the old 4 B/id stack bound
    let pg_span = 2_181_021_971;
    assert!(112 * exact::search_bytes(pg_span) < 50_000_000_000);
    assert_eq!(
        exact::search_bytes(pg_span),
        pg_span as u64 / 8 + pg_span as u64 / 16 + 4 * exact::STACK_RESERVE_IDS as u64 + 64
    );
}

/// The budgeted in-memory variants return the same redundant arcs as the oracle, shifted by
/// `lo` and sorted, for both algorithms, and release every reservation (keep flags, bitsets,
/// the redundant-arc list).
#[test]
fn in_memory_redundant_arcs_match_oracle() {
    for seed in 0..8 {
        let n = 30 + 37 * seed as usize;
        let g = testkit::random_dag(n, 0.07, 9100 + seed);
        let c = Csr::from_vec_graph(&g);
        let lo = 3 * seed as usize;
        let want: Arcs = oracle_redundant(&g)
            .into_iter()
            .map(|(u, v)| (u + lo, v + lo))
            .collect();
        let mut b = MemBudget::unlimited();
        let a = exact::redundant_arcs_bitset(&c, lo, &mut b, &mut Option::<ProgressLogger>::None)
            .unwrap();
        assert_eq!((to_usize(&a.redundant), a.threads), (want.clone(), 1));
        assert_eq!(b.used(), 0);
        let s = exact::redundant_arcs_search(&c, lo, &mut b, &mut npl()).unwrap();
        assert_eq!(to_usize(&s.redundant), want, "seed {seed}");
        assert_eq!(b.used(), 0);
        // the bitset variant needs its whole working memory
        let need = exact::bitset_bytes(n) + c.num_arcs() as u64;
        let mut short = MemBudget::new((need - 1) as f64 / 1e9);
        assert!(
            exact::redundant_arcs_bitset(&c, lo, &mut short, &mut Option::<ProgressLogger>::None)
                .is_err()
        );
        assert_eq!(short.used(), 0);
    }
}

/// Many arc-balanced ranges (so the parallel prefix sums, the range-split keep flags and the
/// parallel dropped-arc fill all cross range boundaries): the two in-memory algorithms agree,
/// and the redundant arcs are exactly `G \ TR`, shifted and sorted.
#[test]
fn in_memory_paths_agree_across_many_ranges() {
    let n = 20_000;
    let mut rng = SmallRng::seed_from_u64(31337);
    let arcs: Vec<(usize, usize)> = (0..n)
        .flat_map(|u| {
            let k = rng.random_range(0..8);
            (0..k)
                .map(|_| (u, u + 1 + rng.random_range(0..300)))
                .filter(|&(_, v)| v < n)
                .collect::<Vec<_>>()
        })
        .collect();
    let c = Csr::from_arcs(n, arcs);
    assert!(c.num_arcs() > 50_000);
    let lo = 1_000_000;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(7)
        .build()
        .unwrap();
    let (a, s, tr) = pool.install(|| {
        let mut b = MemBudget::unlimited();
        (
            exact::redundant_arcs_bitset(&c, lo, &mut b, &mut Option::<ProgressLogger>::None)
                .unwrap(),
            exact::redundant_arcs_search(&c, lo, &mut b, &mut npl()).unwrap(),
            transitive_reduction_search(&c),
        )
    });
    assert_eq!(tr, transitive_reduction_bitset(&c));
    assert_eq!(a.redundant, s.redundant);
    let kept = csr_arcs(&tr);
    let diff: Arcs = csr_arcs(&c)
        .into_iter()
        .filter(|a| kept.binary_search(a).is_err())
        .map(|(u, v)| (u + lo, v + lo))
        .collect();
    assert!(!diff.is_empty());
    assert_eq!(to_usize(&s.redundant), diff);
}

/// Regression: the exact re-check of H3 failures sizes a thread by its bitset
/// and a bounded stack allowance, so a budget for `k` such threads runs `k` threads.
#[test]
fn exact_recheck_threads_follow_the_budget() {
    let g = testkit::random_dag(300, 0.05, 4242);
    let cg = testkit::chunked(&g, 8);
    let n = g.num_nodes();
    let w = Window::full(n);
    let mm = min_max_succ(&cg, w, &mut npl()).unwrap();
    let fails = h3_failures(&cg, &mm.f, &mm.l, &mut npl()).unwrap();
    assert!(!fails.is_empty());
    let span = fails
        .iter()
        .map(|&(u, _)| (mm.l[u as usize] - u) as usize)
        .max()
        .unwrap();
    let per = exact::search_bytes(span);
    let want = oracle_redundant(&g);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    pool.install(|| {
        for k in [1, 3, 8] {
            let mut b = MemBudget::new((k * per + per / 2) as f64 / 1e9);
            let out = reduce::exact_redundant(&g, w, &fails, &mm.l, &mut b, &mut npl()).unwrap();
            assert_eq!(out.threads, k as usize);
            assert_eq!(to_usize(&out.redundant), want, "{k} threads");
            assert_eq!(b.used(), 0);
        }
    });
}

/// Environment variable that turns [`search_reduction_heap_is_flat`] into its measuring child.
const RSS_CHILD: &str = "DAG_BIN_TEST_TR_RSS_CHILD";

/// A `/proc/self/status` field, in bytes.
fn proc_status_bytes(field: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = s.lines().find(|l| l.starts_with(field)).unwrap();
    line.split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 1024
}

/// Regression: the search reduction writes its output straight into a
/// flat CSR instead of one heap vector per node. Measured in a child process running only this
/// test (so no other test shares the heap): the peak RSS growth while reducing must stay within
/// the documented output bound [`exact::reduction_output_bytes`] plus a small slack. The old
/// `Vec<Vec<u32>>` collection needed about 3× that bound on this graph (24-byte headers plus
/// one small allocation per node, then a flat copy).
#[test]
fn search_reduction_heap_is_flat() {
    let n = 4 << 20;
    if std::env::var_os(RSS_CHILD).is_none() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["search_reduction_heap_is_flat", "--exact", "--nocapture"])
            .env(RSS_CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "child failed: {stdout}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let grown: u64 = stdout
            .lines()
            .find_map(|l| l.strip_prefix("TR_RSS_GROWTH="))
            .expect("child did not report")
            .trim()
            .parse()
            .unwrap();
        let m = 2 * n - 3;
        let bound = exact::reduction_output_bytes(n, m) + (24 << 20);
        assert!(
            grown <= bound,
            "search reduction grew the RSS by {grown} B, bound {bound} B"
        );
        return;
    }
    if !Path::new("/proc/self/clear_refs").exists() {
        println!("TR_RSS_GROWTH=0");
        return;
    }
    // u → u+1 (kept) and u → u+2 (redundant)
    let g = Csr::from_arcs(
        n,
        (0..n)
            .flat_map(|u| [(u, u + 1), (u, u + 2)])
            .filter(|&(_, v)| v < n),
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| rayon::iter::ParallelIterator::for_each(rayon::iter::repeatn(0, 64), drop));
    // reset VmHWM to the current RSS (Linux ≥ 4.0)
    std::fs::write("/proc/self/clear_refs", "5").unwrap();
    let before = proc_status_bytes("VmRSS:");
    let tr = pool.install(|| transitive_reduction_search(&g));
    let peak = proc_status_bytes("VmHWM:");
    assert_eq!(tr.num_arcs(), n - 1);
    println!("TR_RSS_GROWTH={}", peak.saturating_sub(before));
}

// ---------------------------------------------------------------------------------------------
// BV round trips
// ---------------------------------------------------------------------------------------------

fn bv_arcs(base: &Path) -> Arcs {
    let g = graph::load_ra(base).unwrap();
    testkit::arcs(&g)
}

#[test]
fn bv_verify_and_reduce_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    for seed in 0..6 {
        let n = 40 + 15 * seed as usize;
        let g = testkit::random_dag(n, 0.08, 4000 + seed);
        let name = format!("g{seed}");
        let base = testkit::write_bv(dir.path(), &name, &g);
        let conf = ChunkConf {
            chunk_arcs: 1 + seed * 3,
            max_chunk_nodes: 7,
        };
        let cg = graph::open(&base, conf).unwrap();
        let (rep, red) = verify_reduced(
            &cg,
            None,
            &SweepConf { decoders: 3 },
            &mut MemBudget::unlimited(),
            &mut npl(),
        )
        .unwrap();
        assert_eq!(to_usize(&red), oracle_redundant(&g));
        let (rep_vec, _) = verify_generic(&testkit::chunked(&g, 4), Window::full(n));
        assert_eq!(rep, rep_vec);
        // full reduction
        let out = dir.path().join(format!("r{seed}"));
        reduce::reduce_bv(&cg, &red, &out).unwrap();
        assert_eq!(bv_arcs(&out), oracle_tr(&g), "seed {seed}");
        let t = graph::transpose_basename(&out).expect("transpose written");
        let tr_t: Arcs = {
            let mut a: Arcs = oracle_tr(&g).into_iter().map(|(u, v)| (v, u)).collect();
            a.sort_unstable();
            a
        };
        assert_eq!(bv_arcs(&t), tr_t);
        // window reduction
        let w = Window::new(n / 4, n - n / 5, n).unwrap();
        let (rw, redw) = verify_reduced_window(
            &cg,
            w,
            None,
            &SweepConf::default(),
            &mut MemBudget::unlimited(),
            &mut npl(),
        )
        .unwrap();
        assert_eq!(rw.redundant, redw.len() as u64);
        let outw = dir.path().join(format!("w{seed}"));
        let (nn, mm) = reduce::reduce_bv_window(
            &cg,
            w,
            &redw,
            &outw,
            &dir.path().join("tmp"),
            MemoryUsage::BatchSize(7),
            &mut Option::<ProgressLogger>::None,
        )
        .unwrap();
        let want = oracle_tr(&induced(&g, w));
        assert_eq!(nn, w.len());
        assert_eq!(mm, want.len() as u64);
        assert_eq!(bv_arcs(&outw), want, "seed {seed}");
        // the H3 thresholds over a window agree with those of the induced graph
        let mm_w = min_max_succ(&cg, w, &mut npl()).unwrap();
        let fw = h3_failures_in(&cg, w, &mm_w.f, &mm_w.l, &mut npl()).unwrap();
        let sub = induced(&g, w);
        let cs = testkit::chunked(&sub, 2);
        let mm_s = min_max_succ(&cs, Window::full(w.len()), &mut npl()).unwrap();
        let fs = h3_failures(&cs, &mm_s.f, &mm_s.l, &mut npl()).unwrap();
        let shifted: Vec<(u32, u32)> = fs
            .iter()
            .map(|&(u, t)| (u + w.lo as u32, t + w.lo as u32))
            .collect();
        assert_eq!(fw, shifted);
    }
}

// ---------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------

use cli::{p, read_json, run_expect, run_ok};

#[test]
fn cli_verify_and_reduce() {
    let dir = tempfile::tempdir().unwrap();
    // a PG-like graph: reduced, certificate blocks (auto), H3 (none), blocks (PREFIX)
    let pg = testkit::mini_pg(300, 3, &MiniPgOpts::default()).graph;
    let b = testkit::write_bv(dir.path(), "pg", &pg);
    let o = dir.path().join("v_auto");
    run_ok(&["tr", "verify", "--graph", &p(&b), "--out", &p(&o)]);
    let j = read_json(&o.with_extension("json"));
    assert_eq!(j["results"]["report"]["certificate"], "blocks");
    assert_eq!(j["results"]["report"]["is_transitively_reduced"], true);
    let o = dir.path().join("v_none");
    run_ok(&[
        "tr",
        "verify",
        "--graph",
        &p(&b),
        "--blocks",
        "none",
        "--out",
        &p(&o),
    ]);
    let j = read_json(&o.with_extension("json"));
    assert_eq!(j["results"]["report"]["certificate"], "H3");
    assert_eq!(j["results"]["report"]["h3_failing_nodes"], 0);
    let prefix = dir.path().join("idx");
    run_ok(&["blocks", "--graph", &p(&b), "--out", &p(&prefix)]);
    let o = dir.path().join("v_prefix");
    run_ok(&[
        "tr",
        "verify",
        "--graph",
        &p(&b),
        "--blocks",
        &p(&prefix),
        "--window",
        "10..200",
        "--out",
        &p(&o),
    ]);
    let j = read_json(&o.with_extension("json"));
    assert_eq!(j["results"]["report"]["certificate"], "blocks");
    assert_eq!(
        j["results"]["report"]["arcs"],
        testkit::arcs(&induced(&pg, Window::new(10, 200, pg.num_nodes()).unwrap())).len()
    );

    // a graph with redundant arcs: exit 1, redundant list = oracle
    let g = testkit::random_dag(80, 0.1, 5);
    let want_red = oracle_redundant(&g);
    assert!(!want_red.is_empty());
    let b = testkit::write_bv(dir.path(), "rnd", &g);
    let o = dir.path().join("v_rnd");
    run_expect(&["tr", "verify", "--graph", &p(&b), "--out", &p(&o)], 1);
    let j = read_json(&o.with_extension("json"));
    assert_eq!(j["results"]["report"]["certificate"], "exact");
    assert!(!j["results"]["line_digraph_check"]["violation"].is_null());
    let red: Vec<(u32, u32)> = read_array(&o.with_extension("redundant.u32x2le")).unwrap();
    assert_eq!(to_usize(&red), want_red);

    // tr reduce with every algorithm, full and windowed
    let want = oracle_tr(&g);
    for algo in ["auto", "bitset", "search"] {
        let out = dir.path().join(format!("red_{algo}"));
        run_ok(&[
            "tr",
            "reduce",
            "--graph",
            &p(&b),
            "--algo",
            algo,
            "--out",
            &p(&out),
        ]);
        assert_eq!(bv_arcs(&out), want, "{algo}");
        let j = read_json(&out.with_extension("json"));
        assert_eq!(j["results"]["redundant"], want_red.len());
        // auto takes the bitset path on 80 nodes; search records the budget-capped pool size
        let t = j["results"]["threads"].as_u64().expect("threads recorded");
        if algo == "search" {
            assert!(t >= 1, "{algo}");
        } else {
            assert_eq!(t, 1, "{algo}");
        }
        let out = dir.path().join(format!("redw_{algo}"));
        run_ok(&[
            "tr",
            "reduce",
            "--graph",
            &p(&b),
            "--algo",
            algo,
            "--window",
            "20..70",
            "--out",
            &p(&out),
        ]);
        let w = Window::new(20, 70, 80).unwrap();
        assert_eq!(bv_arcs(&out), oracle_tr(&induced(&g, w)), "{algo} window");
    }
    // tr reduce auto above the bitset limit takes the verification path: emulate with a
    // large isolated graph prefix (n > 2^16) plus the random DAG
    let big = testkit::disjoint_union(&[testkit::isolated(reduce::BITSET_MAX_NODES), g.clone()]);
    let b = testkit::write_bv(dir.path(), "big", &big);
    let out = dir.path().join("red_big");
    run_ok(&[
        "tr",
        "reduce",
        "--graph",
        &p(&b),
        "--blocks",
        "none",
        "--out",
        &p(&out),
    ]);
    let j = read_json(&out.with_extension("json"));
    assert_eq!(j["params"]["algo"], "verify");
    assert_eq!(j["params"]["blocks"], "none");
    let shift = reduce::BITSET_MAX_NODES;
    assert_eq!(
        bv_arcs(&out),
        want.iter()
            .map(|&(u, v)| (u + shift, v + shift))
            .collect::<Arcs>()
    );
}

/// Regression: `tr reduce` releases every reservation (window CSR, algorithm
/// memory, `--blocks PREFIX` index) before sizing the external sort, so a `--max-mem` of
/// `SORT_MEM_MIN` plus the redundant-arc list and a few KiB suffices; before the fix the
/// leaked reservations (tens of KiB for the CSR paths, 8 B/node for the index) made the sort
/// fail with OverBudget. Also checks that `--blocks` is recorded as null when it is ignored.
#[test]
fn cli_reduce_releases_reservations_before_the_sort() {
    let dir = tempfile::tempdir().unwrap();
    let min = dag::topo::SORT_MEM_MIN;
    let gb = |bytes: u64| format!("{}", bytes as f64 / 1e9);
    // in-memory algorithms on a sparse random DAG
    let g = testkit::random_dag(400, 0.01, 11);
    let want = oracle_tr(&g);
    let nred = oracle_redundant(&g).len() as u64;
    let b = testkit::write_bv(dir.path(), "rnd", &g);
    let mem = gb(min + 8 * nred + 4096);
    for algo in ["bitset", "search"] {
        let out = dir.path().join(format!("tight_{algo}"));
        run_ok(&[
            "--max-mem",
            &mem,
            "tr",
            "reduce",
            "--graph",
            &p(&b),
            "--algo",
            algo,
            "--out",
            &p(&out),
        ]);
        assert_eq!(bv_arcs(&out), want, "{algo}");
        let j = read_json(&out.with_extension("json"));
        assert!(
            j["params"]["blocks"].is_null(),
            "{algo}: --blocks is ignored"
        );
    }
    // the verification path with a block index (n > 2^16 so that auto does not use bitset)
    let pg = testkit::mini_pg(300, 3, &MiniPgOpts::default()).graph;
    let big = testkit::disjoint_union(&[testkit::isolated(reduce::BITSET_MAX_NODES), pg.clone()]);
    let b = testkit::write_bv(dir.path(), "bigpg", &big);
    let prefix = dir.path().join("bigidx");
    run_ok(&["blocks", "--graph", &p(&b), "--out", &p(&prefix)]);
    let out = dir.path().join("tight_prefix");
    run_ok(&[
        "--max-mem",
        &gb(min + 4096),
        "tr",
        "reduce",
        "--graph",
        &p(&b),
        "--blocks",
        &p(&prefix),
        "--out",
        &p(&out),
    ]);
    let j = read_json(&out.with_extension("json"));
    assert_eq!(j["params"]["algo"], "verify");
    assert_eq!(j["results"]["report"]["certificate"], "blocks");
    assert_eq!(bv_arcs(&out), testkit::arcs(&big));
}
