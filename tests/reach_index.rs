//! Acceptance tests of the exact reachability index ChronoChain.
//!
//! Oracles: [`closure`] (BFS bitset rows) and [`bfs_reach`]; label oracles are recomputed from
//! the closure and explicit successor lists.

use dag::blocks::BlockIndex;
use dag::graph::{self, ChunkConf, ChunkedGraph};
use dag::io::{LoadMode, NONE};
use dag::reach::bench::{BenchConf, cones64, run_bench, run_workload, workloads};
use dag::reach::dagview::{BlockQuotient, GraphDag, Mode, SuccDag, succ_lists};
use dag::reach::index::{Decider, QueryConf, ReachIndex, Scratch, ScratchPool, reach};
use dag::sweep::SweepConf;
use dag::testkit::*;
use dag::window::Window;
use dsi_progress_logger::prelude::*;
use webgraph::prelude::*;

fn quiet() -> Option<ProgressLogger> {
    None
}

fn sweep2() -> SweepConf {
    SweepConf { decoders: 2 }
}

/// The DAG of a [`SuccDag`] as a `VecGraph` (for the oracles).
fn as_vec_graph(dag: &dyn SuccDag) -> VecGraph {
    let lists = succ_lists(dag);
    vec_graph(
        dag.num_nodes(),
        lists
            .iter()
            .enumerate()
            .flat_map(|(x, s)| s.iter().map(move |&y| (x, y as usize))),
    )
}

/// Reflexive closure test.
fn reaches(cl: &[Vec<u64>], x: usize, y: usize) -> bool {
    x == y || bit(&cl[x], y)
}

const BUDGETS: [u32; 4] = [0, 1, 3, u32::MAX];

/// Checks every label against the oracle, every filter in isolation on every pair, the
/// combined test and the exact query under every budget. Returns the number of undecided pairs.
fn check_index(idx: &ReachIndex, dag: &dyn SuccDag, what: &str) -> usize {
    let n = dag.num_nodes();
    assert_eq!(idx.n_dag, n, "{what}");
    let g = as_vec_graph(dag);
    let cl = closure(&g);
    let lists = succ_lists(dag);
    // ---- labels against the oracle ----
    let mut lvl = vec![0u32; n];
    for x in 0..n {
        for &y in &lists[x] {
            lvl[y as usize] = lvl[y as usize].max(lvl[x] + 1);
        }
    }
    let mut ht = vec![0u32; n];
    for x in (0..n).rev() {
        for &y in &lists[x] {
            ht[x] = ht[x].max(ht[y as usize] + 1);
        }
    }
    for x in 0..n {
        assert_eq!(idx.lvl[x], lvl[x], "{what}: lvl({x})");
        assert_eq!(idx.ht[x], ht[x], "{what}: ht({x})");
        let hi = (0..n).filter(|&z| reaches(&cl, x, z)).max().unwrap();
        let lo = (0..n).filter(|&z| reaches(&cl, z, x)).min().unwrap();
        assert_eq!(idx.hi[x] as usize, hi, "{what}: hi({x})");
        assert_eq!(idx.lo[x] as usize, lo, "{what}: lo({x})");
    }
    // ---- chains: maximal paths, chain 0 is a longest path ----
    let maxlvl = lvl.iter().copied().max().unwrap_or(0);
    for (c, ch) in idx.chains.iter().enumerate() {
        let nodes: &[u32] = &ch.nodes;
        assert!(!nodes.is_empty(), "{what}: chain {c} empty");
        for p in nodes.windows(2) {
            assert!(
                lists[p[0] as usize].contains(&p[1]),
                "{what}: chain {c} is not a path at {p:?}"
            );
        }
        assert_eq!(
            lvl[nodes[0] as usize], 0,
            "{what}: chain {c} starts at a non-source"
        );
        assert!(
            lists[*nodes.last().unwrap() as usize].is_empty(),
            "{what}: chain {c} ends at a non-sink"
        );
        if c == 0 {
            assert_eq!(
                nodes.len() as u32,
                maxlvl + 1,
                "{what}: chain 0 is not a longest path"
            );
        }
        for x in 0..n {
            let out = nodes
                .iter()
                .position(|&z| reaches(&cl, x, z as usize))
                .map_or(NONE, |p| p as u32);
            let inn = nodes
                .iter()
                .rposition(|&z| reaches(&cl, z as usize, x))
                .map_or(0, |p| p as u32 + 1);
            assert_eq!(ch.out[x], out, "{what}: out_{c}({x})");
            assert_eq!(ch.inn[x], inn, "{what}: in_{c}({x})");
            assert_eq!(
                idx.on_chain(c, x),
                nodes.contains(&(x as u32)),
                "{what}: on_{c}({x})"
            );
        }
    }
    // ---- every filter in isolation, the test, and the exact query ----
    let mut s = Scratch::new();
    let mut undecided = 0;
    for x in 0..n {
        for y in 0..n {
            let truth = reaches(&cl, x, y);
            if x < y {
                let mut verdicts = vec![
                    ("hilo_bounds", idx.filter_hilo_bounds(x, y)),
                    ("level", idx.filter_level(x, y)),
                    ("hilo_dominance", idx.filter_hilo_dominance(x, y)),
                ];
                for c in 0..idx.k() {
                    verdicts.push(("chain_positive", idx.filter_chain_positive(c, x, y)));
                    verdicts.push(("chain_exact", idx.filter_chain_exact(c, x, y)));
                    verdicts.push(("chain_negative", idx.filter_chain_negative(c, x, y)));
                }
                for (name, v) in verdicts {
                    if let Some(r) = v {
                        assert_eq!(r, truth, "{what}: filter {name} unsound on ({x}, {y})");
                    }
                }
            }
            match idx.test(x, y) {
                Some((r, d)) => {
                    assert_eq!(r, truth, "{what}: test({x}, {y}) by {d:?}");
                    assert!(d.is_label());
                }
                None => undecided += 1,
            }
            for budget in BUDGETS {
                let q = QueryConf { budget };
                let (r, d, _) = idx.reach_dag(dag, x, y, &q, &mut s);
                assert_eq!(
                    r, truth,
                    "{what}: reach_dag({x}, {y}) budget {budget} by {d:?}"
                );
                if budget == 0 {
                    assert_ne!(d, Decider::Search, "{what}: budget 0 must skip the search");
                }
                if budget == u32::MAX {
                    assert_ne!(d, Decider::WindowScan, "{what}: unlimited budget scanned");
                }
                if idx.test(x, y).is_none() {
                    let want = if budget == 0 {
                        Decider::WindowScan
                    } else if budget == u32::MAX {
                        Decider::Search
                    } else {
                        d
                    };
                    assert_eq!(d, want, "{what}: fallback of ({x}, {y}) budget {budget}");
                }
            }
        }
    }
    undecided
}

fn graph_index(g: &VecGraph, k: usize, chunk_arcs: u64, what: &str) -> usize {
    let cg = chunked(g, chunk_arcs);
    let dag = GraphDag::new(&cg, sweep2());
    let idx = ReachIndex::build(&dag, k, &mut quiet()).unwrap();
    assert_eq!(idx.mode, Mode::Graph);
    assert!(idx.k() <= k);
    check_index(&idx, &dag, what)
}

/// The adversarial shapes of the test plan, plus random DAGs.
fn shapes() -> Vec<(String, VecGraph)> {
    let mut v: Vec<(String, VecGraph)> = vec![
        ("empty".into(), isolated(0)),
        ("single".into(), isolated(1)),
        ("isolated".into(), isolated(20)),
        ("chain".into(), chain(50)),
        ("complete".into(), complete_dag(30)),
        ("out_star".into(), out_star(40)),
        ("in_star".into(), in_star(40)),
        ("diamonds".into(), diamonds(10)),
        ("ladder".into(), ladder(20)),
        ("layered".into(), layered(&[5, 8, 3, 10, 4, 6], 0.3, 5)),
        (
            "union".into(),
            disjoint_union(&[
                chain(10),
                isolated(5),
                diamonds(4),
                random_dag(30, 0.1, 9),
                out_star(6),
                in_star(6),
            ]),
        ),
    ];
    for (i, (n, p)) in [
        (300, 0.005),
        (300, 0.02),
        (200, 0.05),
        (120, 0.2),
        (60, 0.5),
    ]
    .into_iter()
    .enumerate()
    {
        v.push((
            format!("random_dag({n}, {p})"),
            random_dag(n, p, 100 + i as u64),
        ));
    }
    v
}

#[test]
fn graph_mode_exact_on_all_pairs_for_k_0_1_4() {
    let mut undecided = [0usize; 3];
    for (name, g) in shapes() {
        for (i, k) in [0, 1, 4].into_iter().enumerate() {
            for chunk_arcs in [1, 7, 1 << 20] {
                undecided[i] += graph_index(
                    &g,
                    k,
                    chunk_arcs,
                    &format!("{name} k={k} chunk={chunk_arcs}"),
                );
            }
        }
    }
    // the fallbacks must actually be exercised, and more chains must decide more pairs
    assert!(undecided.iter().all(|&u| u > 0), "{undecided:?}");
    assert!(
        undecided[0] >= undecided[1] && undecided[1] >= undecided[2],
        "{undecided:?}"
    );
}

#[test]
fn more_random_dags() {
    for seed in 0..40 {
        let n = 20 + (seed as usize * 7) % 120;
        let p = [0.01, 0.03, 0.08, 0.15][seed as usize % 4];
        graph_index(
            &random_dag(n, p, seed),
            (seed % 5) as usize,
            3,
            &format!("seed {seed}"),
        );
    }
}

#[test]
fn nodes_off_every_chain_are_handled_soundly() {
    // with k = 1 the spine lives in the chain component; the other components neither reach
    // nor are reached by any chain node (out = NONE, in = 0), yet every filter stays sound
    let g = disjoint_union(&[chain(30), random_dag(40, 0.1, 3), isolated(4)]);
    let cg = chunked(&g, 5);
    let dag = GraphDag::new(&cg, sweep2());
    let idx = ReachIndex::build(&dag, 1, &mut quiet()).unwrap();
    let off: Vec<usize> = (0..dag.num_nodes())
        .filter(|&x| idx.chains[0].out[x] == NONE && idx.chains[0].inn[x] == 0)
        .collect();
    assert!(
        off.len() >= 44,
        "expected nodes off chain 0, got {}",
        off.len()
    );
    check_index(&idx, &dag, "off-chain");
}

#[test]
fn full_chain_cover_decides_every_pair_by_labels() {
    // Jagadish: with a chain cover, the labels alone are exact
    for seed in 0..10 {
        let g = random_dag(50, 0.06, seed);
        let cg = chunked(&g, 4);
        let dag = GraphDag::new(&cg, sweep2());
        let idx = ReachIndex::build(&dag, 255, &mut quiet()).unwrap();
        let covered: usize = (0..50)
            .filter(|&x| (0..idx.k()).any(|c| idx.on_chain(c, x)))
            .count();
        assert_eq!(covered, 50, "greedy chains must cover every node");
        assert_eq!(check_index(&idx, &dag, "full cover"), 0);
    }
}

#[test]
fn chain_selection_stops_when_everything_is_covered() {
    let cg = chunked(&chain(10), 3);
    let dag = GraphDag::new(&cg, sweep2());
    let (idx, stats) = ReachIndex::build_with_stats(&dag, 4, &mut quiet()).unwrap();
    assert_eq!(idx.k(), 1);
    assert_eq!(stats.chain_lens, vec![10]);
    assert_eq!(stats.max_lvl, 9);
    assert_eq!(stats.covered, 10);
}

/// The filter classes of [`ReachIndex::test`], in check order, with the decider each reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Filter {
    HiLoBounds,
    Level,
    ChainPositive(u8),
    ChainExact(u8),
    ChainNegative(u8),
    HiLoDominance,
}

impl Filter {
    fn decider(self) -> Decider {
        match self {
            Self::HiLoBounds | Self::HiLoDominance => Decider::HiLo,
            Self::Level => Decider::Level,
            Self::ChainPositive(c) => Decider::ChainPositive(c),
            Self::ChainExact(c) => Decider::ChainExact(c),
            Self::ChainNegative(c) => Decider::ChainNegative(c),
        }
    }

    /// The class without the chain index.
    fn class(self) -> &'static str {
        match self {
            Self::HiLoBounds => "hilo_bounds",
            Self::Level => "level",
            Self::ChainPositive(_) => "chain_positive",
            Self::ChainExact(_) => "chain_exact",
            Self::ChainNegative(_) => "chain_negative",
            Self::HiLoDominance => "hilo_dominance",
        }
    }
}

/// The filters that fire on `(x, y)`, `x < y`.
fn firing(idx: &ReachIndex, x: usize, y: usize) -> Vec<(Filter, bool)> {
    let mut v = Vec::new();
    let mut push = |f: Filter, r: Option<bool>| {
        if let Some(r) = r {
            v.push((f, r));
        }
    };
    push(Filter::HiLoBounds, idx.filter_hilo_bounds(x, y));
    push(Filter::Level, idx.filter_level(x, y));
    for c in 0..idx.k() {
        let cc = c as u8;
        push(
            Filter::ChainPositive(cc),
            idx.filter_chain_positive(c, x, y),
        );
        push(Filter::ChainExact(cc), idx.filter_chain_exact(c, x, y));
        push(
            Filter::ChainNegative(cc),
            idx.filter_chain_negative(c, x, y),
        );
    }
    push(Filter::HiLoDominance, idx.filter_hilo_dominance(x, y));
    v
}

/// Fixed DAGs on which every filter decides some pair on its own.
fn firing_shapes() -> Vec<(String, VecGraph)> {
    let mut v: Vec<(String, VecGraph)> = vec![
        ("diamonds".into(), diamonds(6)),
        ("ladder".into(), ladder(12)),
        ("layered".into(), layered(&[5, 8, 3, 10, 4, 6], 0.3, 5)),
    ];
    for seed in 0..6 {
        v.push((
            format!("random_dag(80, 0.04, {seed})"),
            random_dag(80, 0.04, seed),
        ));
    }
    v
}

/// Regression test: every label filter must not only
/// be sound but actually fire, and [`ReachIndex::test`] must report the first firing filter in
/// check order. Two filters are subsumed by others: `ChainExact(c)` by `ChainPositive(c)` /
/// `ChainNegative(c)` (index module documentation) and the hi/lo bounds by the hi/lo dominance
/// (`Y > hi(X)` gives `hi(Y) ≥ Y > hi(X)`; `lo(Y) > X` gives `lo(Y) > X ≥ lo(X)`), so they never
/// decide a pair alone; every other filter must decide some pairs alone (all other filters
/// silent, up to the subsumed ones), with the right answer and decider. Disabling any filter,
/// or dropping it from `test`, makes this fail.
#[test]
fn every_filter_fires_and_some_pairs_are_decided_by_it_alone() {
    use std::collections::BTreeMap;
    let mut fired = BTreeMap::<&str, u64>::new();
    let mut alone = BTreeMap::<&str, u64>::new();
    for (name, g) in firing_shapes() {
        let cl = closure(&g);
        let cg = chunked(&g, 5);
        let dag = GraphDag::new(&cg, sweep2());
        for k in [1, 2, 4] {
            let idx = ReachIndex::build(&dag, k, &mut quiet()).unwrap();
            let n = dag.num_nodes();
            for x in 0..n {
                for y in x + 1..n {
                    let f = firing(&idx, x, y);
                    let mut classes: Vec<&str> = f.iter().map(|(f, _)| f.class()).collect();
                    classes.dedup();
                    for c in &classes {
                        *fired.entry(c).or_default() += 1;
                    }
                    let main: Vec<(Filter, bool)> = f
                        .iter()
                        .copied()
                        .filter(|(f, _)| !matches!(f, Filter::ChainExact(_) | Filter::HiLoBounds))
                        .collect();
                    if let [(only, r)] = main[..] {
                        *alone.entry(only.class()).or_default() += 1;
                        let truth = reaches(&cl, x, y);
                        assert_eq!(r, truth, "{name} k={k}: {only:?} on ({x}, {y})");
                        let (tr, td) = idx.test(x, y).expect("decided");
                        assert_eq!(tr, truth, "{name} k={k}: ({x}, {y})");
                        // the subsumed filter of the same family may report first
                        let ok = match only {
                            Filter::ChainNegative(c) => {
                                td == Decider::ChainNegative(c) || td == Decider::ChainExact(c)
                            }
                            _ => td == only.decider(),
                        };
                        assert!(
                            ok,
                            "{name} k={k}: ({x}, {y}) decided alone by {only:?}, test says {td:?}"
                        );
                    }
                    // the test reports the first firing filter in check order
                    let first = f
                        .iter()
                        .min_by_key(|(f, _)| *f)
                        .map(|&(f, r)| (r, f.decider()));
                    assert_eq!(idx.test(x, y), first, "{name} k={k}: ({x}, {y})");
                }
            }
        }
    }
    for class in [
        "hilo_bounds",
        "level",
        "chain_positive",
        "chain_exact",
        "chain_negative",
        "hilo_dominance",
    ] {
        assert!(
            fired.get(class).copied().unwrap_or(0) > 0,
            "{class} never fires: {fired:?}"
        );
    }
    for class in [
        "level",
        "chain_positive",
        "chain_negative",
        "hilo_dominance",
    ] {
        assert!(
            alone.get(class).copied().unwrap_or(0) > 0,
            "{class} never decides a pair alone: {alone:?}"
        );
    }
    // the subsumed filters never fire without their subsuming filter
    assert!(!alone.contains_key("chain_exact") && !alone.contains_key("hilo_bounds"));
}

/// Hand-checked single-filter pairs (regression test).
#[test]
fn hand_built_single_filter_pairs() {
    // diamonds(3): 0 → {1, 2} → 3 → {4, 5} → 6 → {7, 8} → 9; chain 0 = 0 1 3 4 6 7 9
    let cg = chunked(&diamonds(3), 4);
    let dag = GraphDag::new(&cg, sweep2());
    let idx = ReachIndex::build(&dag, 1, &mut quiet()).unwrap();
    assert_eq!(&idx.chains[0].nodes[..], &[0, 1, 3, 4, 6, 7, 9]);
    // 1 and 2 are siblings: same level, nothing else says no
    assert_eq!(idx.filter_level(1, 2), Some(false));
    assert_eq!(idx.filter_hilo_bounds(1, 2), None);
    assert_eq!(idx.test(1, 2), Some((false, Decider::Level)));
    // 2 ⇝ 5 off the chain: chain 0 proves it (out_0(2) = pos(3) = 2 < inn_0(5) = 3)
    assert_eq!(idx.filter_chain_positive(0, 2, 5), Some(true));
    assert_eq!(idx.test(2, 5), Some((true, Decider::ChainPositive(0))));
    // 0 ⇝ 9 with both on the chain
    assert_eq!(idx.test(0, 9), Some((true, Decider::ChainPositive(0))));
    // a sink before a later node: hi(Y) bound
    let g = vec_graph(4, [(0, 1), (2, 3)]);
    let cg = chunked(&g, 4);
    let dag = GraphDag::new(&cg, sweep2());
    let idx = ReachIndex::build(&dag, 1, &mut quiet()).unwrap();
    // 1 is a sink: hi(1) = 1 < 2
    assert_eq!(idx.filter_hilo_bounds(1, 2), Some(false));
    assert_eq!(idx.test(1, 2), Some((false, Decider::HiLo)));
}

/// Regression test: neither the search (unbounded
/// budget) nor the window scan (budget 0) may expand nodes the label test rules out. The
/// visited count is at most the unpruned exploration's (every node of `[x, y)` reachable from
/// `x`, which is exactly what either visits on a no-answer without pruning), and strictly
/// smaller on some undecided no-answers.
#[test]
fn search_pruning_visits_fewer_nodes_than_an_unpruned_dfs() {
    for (budget, decider) in [(u32::MAX, Decider::Search), (0, Decider::WindowScan)] {
        check_pruning(QueryConf { budget }, decider);
    }
}

fn check_pruning(q: QueryConf, decider: Decider) {
    let mut s = Scratch::new();
    let (mut searched, mut strictly_fewer) = (0usize, 0usize);
    for (name, g) in [
        ("ladder", ladder(20)),
        ("random_dag(120, 0.04, 1)", random_dag(120, 0.04, 1)),
        ("random_dag(200, 0.02, 2)", random_dag(200, 0.02, 2)),
    ] {
        let cl = closure(&g);
        let cg = chunked(&g, 7);
        let dag = GraphDag::new(&cg, sweep2());
        for k in [0, 1] {
            let idx = ReachIndex::build(&dag, k, &mut quiet()).unwrap();
            let n = dag.num_nodes();
            for x in 0..n {
                for y in x + 1..n {
                    if idx.test(x, y).is_some() {
                        continue;
                    }
                    let (r, d, visited) = idx.reach_dag(&dag, x, y, &q, &mut s);
                    assert_eq!(d, decider);
                    assert_eq!(r, reaches(&cl, x, y), "{name} k={k}: ({x}, {y})");
                    let unpruned = (x..y).filter(|&w| reaches(&cl, x, w)).count();
                    assert!(
                        visited as usize <= unpruned,
                        "{name}: ({x}, {y}) visited {visited} > {unpruned}"
                    );
                    if !r {
                        searched += 1;
                        if (visited as usize) < unpruned {
                            strictly_fewer += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(searched > 0, "no undecided no-answer");
    assert!(
        strictly_fewer > 0,
        "{decider:?}: pruning never saved a visit on {searched} searches"
    );
}

fn mini(txs: usize, seed: u64) -> MiniPg {
    let opts = MiniPgOpts {
        zero_output_prob: 0.05,
        ..Default::default()
    };
    mini_pg(txs, seed, &opts)
}

#[test]
fn blocks_mode_g_level_equals_closure() {
    for seed in 0..12 {
        let pg = mini(60 + 10 * seed as usize, seed);
        let g = &pg.graph;
        let (bi, _) = BlockIndex::from_vec_graph(g).unwrap();
        let cl = closure(g);
        let gt = transpose(g);
        for k in [0, 1, 4] {
            let q = BlockQuotient::new(&bi).with_batch(1 + seed as usize % 5);
            let idx = ReachIndex::build(&q, k, &mut quiet()).unwrap();
            assert_eq!(idx.mode, Mode::Blocks);
            // Q-level: all filters and queries
            check_index(&idx, &q, &format!("Q seed {seed} k {k}"));
            // G-level (Lemma 1 + map tests)
            let mut s = Scratch::new();
            for u in 0..g.num_nodes() {
                for v in 0..g.num_nodes() {
                    for budget in [0, u32::MAX] {
                        let (r, d, _) =
                            reach(&idx, &q, Some(&bi), u, v, &QueryConf { budget }, &mut s);
                        assert_eq!(
                            r,
                            reaches(&cl, u, v),
                            "G seed {seed} k {k} ({u}, {v}) by {d:?}"
                        );
                        match d {
                            Decider::DirectArc => {
                                assert!(g.successors(u).into_iter().any(|x| x == v))
                            }
                            Decider::SinkSource => {
                                assert!(g.outdegree(u) == 0 || gt.outdegree(v) == 0)
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn windows_agree_with_the_full_closure() {
    // graph mode: window convexity makes window answers equal full-graph answers
    let g = random_dag(150, 0.04, 77);
    let cl = closure(&g);
    let cg = chunked(&g, 3);
    for (lo, hi) in [(0, 150), (10, 90), (37, 38), (60, 150), (5, 5)] {
        let w = Window { lo, hi };
        let dag = GraphDag::with_window(&cg, sweep2(), w);
        let idx = ReachIndex::build(&dag, 2, &mut quiet()).unwrap();
        check_index(&idx, &dag, &format!("graph window {lo}..{hi}"));
        let mut s = Scratch::new();
        for u in lo..hi {
            for v in lo..hi {
                let (r, _, _) = reach(&idx, &dag, None, u, v, &QueryConf::default(), &mut s);
                assert_eq!(r, reaches(&cl, u, v), "graph window {lo}..{hi} ({u}, {v})");
            }
        }
    }
    // blocks mode: windows cutting through blocks and containing sources
    for seed in 0..6 {
        let pg = mini(150, 1000 + seed);
        let g = &pg.graph;
        let n = g.num_nodes();
        let (bi, _) = BlockIndex::from_vec_graph(g).unwrap();
        let cl = closure(g);
        for (lo, hi) in [(0, n), (n / 5, 4 * n / 5), (n / 3 + 1, n), (0, n / 2 + 1)] {
            let w = Window { lo, hi };
            let q = BlockQuotient::with_window(&bi, w).with_batch(2);
            let idx = ReachIndex::build(&q, 3, &mut quiet()).unwrap();
            assert_eq!(idx.info.base, q.block_range().start);
            check_index(&idx, &q, &format!("blocks window {lo}..{hi}"));
            let mut s = Scratch::new();
            for u in lo..hi {
                for v in lo..hi {
                    let (r, d, _) =
                        reach(&idx, &q, Some(&bi), u, v, &QueryConf { budget: 2 }, &mut s);
                    assert_eq!(
                        r,
                        reaches(&cl, u, v),
                        "blocks window {lo}..{hi} ({u}, {v}) by {d:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn bv_round_trip_gives_identical_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let g = random_dag(200, 0.03, 5);
    let base = write_bv(dir.path(), "g", &g);
    let bv = graph::open(
        &base,
        ChunkConf {
            chunk_arcs: 11,
            max_chunk_nodes: 13,
        },
    )
    .unwrap();
    let mem = ChunkedGraph::new(g.clone(), ChunkConf::default()).unwrap();
    let a = ReachIndex::build(&GraphDag::new(&bv, sweep2()), 3, &mut quiet()).unwrap();
    let b = ReachIndex::build(&GraphDag::new(&mem, sweep2()), 3, &mut quiet()).unwrap();
    assert_same(&a, &b);
    check_index(&a, &GraphDag::new(&bv, sweep2()), "bv");
    // mini_pg through BV and BlockIndex::build
    let pg = mini(200, 4);
    let base = write_bv(dir.path(), "pg", &pg.graph);
    let bv = graph::open(
        &base,
        ChunkConf {
            chunk_arcs: 5,
            max_chunk_nodes: 7,
        },
    )
    .unwrap();
    let mut cpl = Option::<ConcurrentWrapper<ProgressLogger>>::None;
    let (bi, _) = BlockIndex::build(&bv, &mut cpl).unwrap();
    let (bi2, _) = BlockIndex::from_vec_graph(&pg.graph).unwrap();
    let a = ReachIndex::build(&BlockQuotient::new(&bi), 4, &mut quiet()).unwrap();
    let b = ReachIndex::build(&BlockQuotient::new(&bi2).with_batch(3), 4, &mut quiet()).unwrap();
    assert_same(&a, &b);
}

fn assert_same(a: &ReachIndex, b: &ReachIndex) {
    assert_eq!(a.mode, b.mode);
    assert_eq!(a.info, b.info);
    assert_eq!(&*a.lvl, &*b.lvl);
    assert_eq!(&*a.ht, &*b.ht);
    assert_eq!(&*a.hi, &*b.hi);
    assert_eq!(&*a.lo, &*b.lo);
    assert_eq!(a.k(), b.k());
    for (x, y) in a.chains.iter().zip(&b.chains) {
        assert_eq!(&*x.nodes, &*y.nodes);
        assert_eq!(&*x.out, &*y.out);
        assert_eq!(&*x.inn, &*y.inn);
    }
}

#[test]
fn build_is_identical_across_thread_counts_and_batches() {
    let pg = mini(400, 21);
    let (bi, _) = BlockIndex::from_vec_graph(&pg.graph).unwrap();
    let reference = ReachIndex::build(&BlockQuotient::new(&bi), 4, &mut quiet()).unwrap();
    for threads in [1, 4, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for batch in [1, 7, 1 << 18] {
            let idx = pool.install(|| {
                ReachIndex::build(&BlockQuotient::new(&bi).with_batch(batch), 4, &mut quiet())
                    .unwrap()
            });
            assert_same(&reference, &idx);
        }
    }
}

#[test]
fn save_load_gives_identical_answers() {
    let dir = tempfile::tempdir().unwrap();
    let pg = mini(150, 8);
    let (bi, _) = BlockIndex::from_vec_graph(&pg.graph).unwrap();
    let w = Window {
        lo: 3,
        hi: pg.graph.num_nodes() - 2,
    };
    let q = BlockQuotient::with_window(&bi, w);
    let idx = ReachIndex::build(&q, 3, &mut quiet()).unwrap();
    let prefix = dir.path().join("idx");
    idx.save(&prefix).unwrap();
    for mode in [LoadMode::Ram, LoadMode::Mmap] {
        let back = ReachIndex::load(&prefix, mode).unwrap();
        assert_same(&idx, &back);
        let (mut s1, mut s2) = (Scratch::new(), Scratch::new());
        for u in w.lo..w.hi {
            for v in w.lo..w.hi {
                let qc = QueryConf { budget: 1 };
                assert_eq!(
                    reach(&idx, &q, Some(&bi), u, v, &qc, &mut s1),
                    reach(&back, &q, Some(&bi), u, v, &qc, &mut s2)
                );
            }
        }
    }
    // the empty DAG and k = 0 round-trip too
    let cg = chunked(&isolated(0), 1);
    let e = ReachIndex::build(&GraphDag::new(&cg, sweep2()), 0, &mut quiet()).unwrap();
    let p2 = dir.path().join("empty");
    e.save(&p2).unwrap();
    assert_same(&e, &ReachIndex::load(&p2, LoadMode::Mmap).unwrap());
}

#[test]
fn cones64_matches_bfs() {
    let g = random_dag(250, 0.02, 31);
    let cg = chunked(&g, 6);
    let dag = GraphDag::new(&cg, sweep2());
    for sources in [
        vec![],
        vec![0u32],
        (0..64).map(|i| i * 3).collect::<Vec<u32>>(),
    ] {
        let cones = cones64(&dag, &sources, &mut quiet()).unwrap();
        for (i, &s) in sources.iter().enumerate() {
            for (y, &m) in cones.iter().enumerate() {
                assert_eq!(m >> i & 1 != 0, bfs_reach(&g, s as usize, y), "({s}, {y})");
            }
        }
        if sources.is_empty() {
            assert!(cones.iter().all(|&m| m == 0));
        }
    }
    assert!(cones64(&dag, &[0; 65], &mut quiet()).is_err());
    // on the block quotient
    let pg = mini(300, 2);
    let (bi, _) = BlockIndex::from_vec_graph(&pg.graph).unwrap();
    let q = BlockQuotient::new(&bi).with_batch(5);
    let qg = as_vec_graph(&q);
    let src: Vec<u32> = (0..q.num_nodes().min(64) as u32).collect();
    let cones = cones64(&q, &src, &mut quiet()).unwrap();
    for (i, &s) in src.iter().enumerate() {
        for (y, &m) in cones.iter().enumerate() {
            assert_eq!(m >> i & 1 != 0, bfs_reach(&qg, s as usize, y));
        }
    }
}

#[test]
fn bench_reports_full_agreement() {
    let g = random_dag(400, 0.01, 12);
    let cg = chunked(&g, 64);
    let dag = GraphDag::new(&cg, sweep2());
    let idx = ReachIndex::build(&dag, 2, &mut quiet()).unwrap();
    for budget in [0, 4, 1 << 20] {
        let conf = BenchConf {
            sources: 70,
            queries: 2000,
            seed: 3,
            max_query_secs: 60.0,
            query: QueryConf { budget },
            threads: 4,
        };
        let rep = run_bench(&idx, &dag, &conf, &mut quiet()).unwrap();
        assert_eq!(rep.sources.len(), 70);
        assert_eq!(rep.disagreements(), 0);
        let uni = &rep.workloads[0];
        let bal = &rep.workloads[1];
        assert_eq!(uni.queries, 2000);
        assert!(
            bal.queries > 1000,
            "balanced workload too small: {}",
            bal.queries
        );
        let pos = bal.positives as f64 / bal.queries as f64;
        assert!((0.3..=0.7).contains(&pos), "balanced positive rate {pos}");
        let total: u64 = uni.deciders.values().sum();
        assert_eq!(total as usize, uni.queries);
        assert!((uni.label_share + uni.search_share + uni.window_scan_share - 1.0).abs() < 1e-9);
        if budget == 0 {
            assert_eq!(uni.search_share, 0.0);
        }
    }
    // a zero time limit aborts
    let conf = BenchConf {
        sources: 4,
        queries: 50,
        seed: 1,
        max_query_secs: 0.0,
        query: QueryConf::default(),
        threads: 1,
    };
    assert!(run_bench(&idx, &dag, &conf, &mut quiet()).is_err());
}

/// Regression test: the agreement check can fail. Flipping the
/// ground truth of some queries must be counted exactly, in both the sequential and the
/// throughput run (the latter is cross-checked against the former).
#[test]
fn bench_counts_disagreements() {
    let g = random_dag(300, 0.015, 5);
    let cg = chunked(&g, 32);
    let dag = GraphDag::new(&cg, sweep2());
    let idx = ReachIndex::build(&dag, 2, &mut quiet()).unwrap();
    for budget in [0, 1 << 20] {
        let conf = BenchConf {
            sources: 20,
            queries: 600,
            seed: 9,
            max_query_secs: 60.0,
            query: QueryConf { budget },
            threads: 3,
        };
        let w = workloads(&dag, &conf, &mut quiet()).unwrap();
        let ok = run_workload(&idx, &dag, "uniform", &w.uniform, &conf).unwrap();
        assert_eq!(ok.disagreements, 0);
        assert_eq!(ok.first_disagreement, None);
        let mut bad = w.balanced.clone();
        let flipped: Vec<usize> = (0..bad.len()).step_by(7).collect();
        for &i in &flipped {
            bad[i].truth = !bad[i].truth;
        }
        let rep = run_workload(&idx, &dag, "balanced", &bad, &conf).unwrap();
        assert_eq!(rep.disagreements, flipped.len(), "budget {budget}");
        let q0 = bad[flipped[0]];
        assert_eq!(rep.first_disagreement, Some((q0.x, q0.y, q0.truth)));
    }
}

/// Regression test: the parallel queries use exactly one scratch per
/// thread, so scratch memory stays within `threads · Scratch::max_bytes` however rayon splits
/// the batch, and the answers equal the sequential ones in order.
#[test]
fn scratch_pool_keeps_one_scratch_per_thread() {
    let g = random_dag(500, 0.01, 8);
    let cl = closure(&g);
    let cg = chunked(&g, 64);
    let dag = GraphDag::new(&cg, sweep2());
    let idx = ReachIndex::build(&dag, 1, &mut quiet()).unwrap();
    let n = dag.num_nodes();
    let pairs: Vec<(usize, usize)> = (0..20_000)
        .map(|i| ((i * 7919) % n, (i * 104_729 + 13) % n))
        .collect();
    let max_succ = succ_lists(&dag).iter().map(Vec::len).max().unwrap_or(0);
    for threads in [1, 4] {
        let pool = ScratchPool::new(threads).unwrap();
        assert_eq!(pool.threads(), threads);
        let q = QueryConf { budget: 0 };
        let ans = pool.map(&pairs, |&(x, y), s| {
            if x <= y {
                idx.reach_dag(&dag, x, y, &q, s).0
            } else {
                false
            }
        });
        for (&(x, y), &r) in pairs.iter().zip(&ans) {
            assert_eq!(r, x <= y && reaches(&cl, x, y), "({x}, {y})");
        }
        let bound = threads as u64 * Scratch::max_bytes(n, 0, max_succ);
        assert!(pool.bytes() > 0);
        assert!(pool.bytes() as u64 <= bound, "{} > {bound}", pool.bytes());
    }
    // the bench reports the same bound
    let conf = BenchConf {
        sources: 10,
        queries: 3000,
        seed: 2,
        max_query_secs: 60.0,
        query: QueryConf { budget: 0 },
        threads: 4,
    };
    let rep = run_bench(&idx, &dag, &conf, &mut quiet()).unwrap();
    for w in &rep.workloads {
        assert_eq!(w.par_threads, 4);
        assert!(w.par_scratch_bytes as u64 <= 4 * Scratch::max_bytes(n, 0, max_succ));
    }
}
