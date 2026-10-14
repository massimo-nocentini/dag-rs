//! Acceptance tests of the dominator family.
//!
//! Oracles: vertex removal + BFS (`testkit::brute_dominators`), and a test-local reference
//! implementation of the iterative algorithm of Cooper, Harvey and Kennedy on DFS postorder
//! numbers (independent of the topological order used by the crate).

use std::collections::VecDeque;

use dag::blocks::BlockIndex;
use dag::dom::stats::{
    DomStats, dmin_blocks, dmin_sweep, dom_stats, sizes_from_idom, tree_stats, write_top_tsv,
};
use dag::dom::{
    DomConf, DomTree, ROOT, UNSET, check_depth_bound, dominators_blocks, dominators_push,
    post_dominators_blocks, post_dominators_pull, verify_dominators, verify_post_dominators,
};
use dag::graph::{self, ChunkConf};
use dag::io::{self, read_array};
use dag::sweep::SweepConf;
use dag::testkit::*;
use dag::{DagError, NONE, Window};
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;

fn quiet() -> Option<ProgressLogger> {
    None
}

fn cquiet() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

const BUDGETS: [u32; 5] = [0, 1, 3, 64, u32::MAX];

fn conf(w: Window, budget: u32) -> DomConf {
    DomConf {
        window: w,
        budget,
        sweep: SweepConf { decoders: 3 },
    }
}

/// Generic dominator tree of `g` on `w` with chunks of `chunk` arcs.
fn push(g: &VecGraph, w: Window, budget: u32, chunk: u64) -> DomTree {
    dominators_push(&chunked(g, chunk), &conf(w, budget), &mut quiet()).unwrap()
}

/// Generic post-dominator tree of `g` on `w`.
fn pull(g: &VecGraph, w: Window, budget: u32, chunk: u64) -> DomTree {
    post_dominators_pull(&chunked(g, chunk), &conf(w, budget), &mut quiet()).unwrap()
}

fn full(g: &VecGraph) -> Window {
    Window::full(g.num_nodes())
}

fn opt_to_public(v: Option<usize>) -> u32 {
    v.map_or(NONE, |d| d as u32)
}

/// Brute-force immediate dominators (vertex removal + BFS).
fn brute_dom(g: &VecGraph) -> Vec<u32> {
    brute_dominators(g).into_iter().map(opt_to_public).collect()
}

/// The reverse graph with mirrored ids `v ↦ n − 1 − v` (identity order stays topological).
fn mirror(g: &VecGraph) -> VecGraph {
    let n = g.num_nodes();
    vec_graph(n, arcs(g).into_iter().map(|(u, v)| (n - 1 - v, n - 1 - u)))
}

/// Brute-force immediate post-dominators: dominators of the mirrored reverse graph.
fn brute_post(g: &VecGraph) -> Vec<u32> {
    let n = g.num_nodes();
    let r = brute_dominators(&mirror(g));
    (0..n)
        .map(|u| opt_to_public(r[n - 1 - u].map(|d| n - 1 - d)))
        .collect()
}

/// Reference: Cooper–Harvey–Kennedy iterative dominators on DFS postorder numbers, with a
/// virtual root (index n) linked to every source. Works on any flow graph, ignores the ids'
/// order.
fn chk_dom(g: &VecGraph) -> Vec<u32> {
    let n = g.num_nodes();
    let root = n;
    let succ: Vec<Vec<usize>> = (0..=n)
        .map(|u| {
            if u == root {
                (0..n).filter(|&v| indeg(g, v) == 0).collect()
            } else {
                g.successors(u).collect()
            }
        })
        .collect();
    let mut pred = vec![Vec::new(); n + 1];
    for (u, s) in succ.iter().enumerate() {
        for &v in s {
            pred[v].push(u);
        }
    }
    // iterative DFS postorder from the root
    let mut post = vec![usize::MAX; n + 1];
    let mut order = Vec::with_capacity(n + 1);
    let mut seen = vec![false; n + 1];
    let mut stack = vec![(root, 0usize)];
    seen[root] = true;
    while let Some(&mut (u, ref mut i)) = stack.last_mut() {
        if *i < succ[u].len() {
            let v = succ[u][*i];
            *i += 1;
            if !seen[v] {
                seen[v] = true;
                stack.push((v, 0));
            }
        } else {
            post[u] = order.len();
            order.push(u);
            stack.pop();
        }
    }
    let mut idom = vec![usize::MAX; n + 1];
    idom[root] = root;
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while post[a] < post[b] {
                a = idom[a];
            }
            while post[b] < post[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &v in order.iter().rev() {
            if v == root {
                continue;
            }
            let mut new = usize::MAX;
            for &p in &pred[v] {
                if idom[p] != usize::MAX {
                    new = if new == usize::MAX {
                        p
                    } else {
                        intersect(&idom, p, new)
                    };
                }
            }
            if idom[v] != new {
                idom[v] = new;
                changed = true;
            }
        }
    }
    (0..n)
        .map(|v| {
            if idom[v] == root {
                NONE
            } else {
                idom[v] as u32
            }
        })
        .collect()
}

fn indeg(g: &VecGraph, v: usize) -> usize {
    // small graphs only
    (0..g.num_nodes())
        .filter(|&u| g.successors(u).any(|x| x == v))
        .count()
}

fn chk_post(g: &VecGraph) -> Vec<u32> {
    let n = g.num_nodes();
    let r = chk_dom(&mirror(g));
    (0..n)
        .map(|u| match r[n - 1 - u] {
            NONE => NONE,
            d => (n - 1 - d as usize) as u32,
        })
        .collect()
}

/// The subgraph induced by `w`, ids shifted by `−w.lo`.
fn induced(g: &VecGraph, w: Window) -> VecGraph {
    vec_graph(
        w.len(),
        arcs(g)
            .into_iter()
            .filter(|&(u, v)| w.contains(u) && w.contains(v))
            .map(|(u, v)| (u - w.lo, v - w.lo)),
    )
}

fn shift(a: &[u32], lo: usize) -> Vec<u32> {
    a.iter()
        .map(|&x| if x == NONE { NONE } else { x + lo as u32 })
        .collect()
}

fn depths(t: &DomTree) -> Vec<u32> {
    let w = t.window();
    (w.lo..w.hi).map(|v| t.depth(v)).collect()
}

fn random_window(n: usize, r: &mut SmallRng) -> Window {
    let a = r.random_range(0..=n);
    let b = r.random_range(0..=n);
    Window {
        lo: a.min(b),
        hi: a.max(b),
    }
}

fn mini(txs: usize, seed: u64, zero_out: f64) -> VecGraph {
    let opts = MiniPgOpts {
        zero_output_prob: zero_out,
        ..Default::default()
    };
    mini_pg(txs, seed, &opts).graph
}

// ---------------------------------------------------------------------------------------------
// Oracles
// ---------------------------------------------------------------------------------------------

#[test]
fn random_dags_match_brute_force() {
    for seed in 0..500u64 {
        let n = 1 + (seed as usize % 30);
        let p = [0.05, 0.1, 0.2, 0.4][seed as usize % 4];
        let g = random_dag(n, p, seed);
        let budget = BUDGETS[seed as usize % BUDGETS.len()];
        let chunk = seed % 7 + 1;
        let d = push(&g, full(&g), budget, chunk);
        assert_eq!(d.idoms(), brute_dom(&g), "dominators, seed {seed}");
        let pd = pull(&g, full(&g), budget, chunk);
        assert_eq!(pd.idoms(), brute_post(&g), "post-dominators, seed {seed}");
    }
}

#[test]
fn chk_reference_matches_brute_force() {
    // the test-local reference is itself checked against the brute-force oracle
    for seed in 0..100u64 {
        let g = random_dag(1 + seed as usize % 25, 0.15, seed);
        assert_eq!(chk_dom(&g), brute_dom(&g), "seed {seed}");
        assert_eq!(chk_post(&g), brute_post(&g), "seed {seed}");
    }
}

#[test]
fn larger_random_dags_match_chk_reference() {
    for seed in 0..40u64 {
        let n = 200 + 37 * seed as usize;
        let g = random_dag(n, 2.5 / n as f64, 1000 + seed);
        let budget = BUDGETS[seed as usize % BUDGETS.len()];
        let d = push(&g, full(&g), budget, 64);
        assert_eq!(d.idoms(), chk_dom(&g), "dominators, seed {seed}");
        let pd = pull(&g, full(&g), budget, 64);
        assert_eq!(pd.idoms(), chk_post(&g), "post-dominators, seed {seed}");
    }
    for seed in 0..10u64 {
        let g = layered(&[5, 40, 3, 60, 2, 30, 30], 0.08, seed);
        assert_eq!(push(&g, full(&g), 3, 16).idoms(), chk_dom(&g));
        assert_eq!(pull(&g, full(&g), 3, 16).idoms(), chk_post(&g));
    }
}

#[test]
fn special_families() {
    let families: Vec<(&str, VecGraph)> = vec![
        ("empty", isolated(0)),
        ("single", isolated(1)),
        ("isolated", isolated(7)),
        ("chain", chain(12)),
        ("out_star", out_star(9)),
        ("in_star", in_star(9)),
        ("diamonds", diamonds(5)),
        ("ladder", ladder(20)),
        ("complete", complete_dag(9)),
        ("layered", layered(&[3, 4, 3, 4], 0.5, 7)),
        (
            "union",
            disjoint_union(&[chain(4), diamonds(2), out_star(5), isolated(2)]),
        ),
    ];
    for (name, g) in &families {
        for chunk in [1, 3, 100] {
            assert_eq!(
                push(g, full(g), 64, chunk).idoms(),
                brute_dom(g),
                "{name} dom"
            );
            assert_eq!(
                pull(g, full(g), 64, chunk).idoms(),
                brute_post(g),
                "{name} post"
            );
        }
    }
    // explicit expectations
    let c = chain(5);
    assert_eq!(push(&c, full(&c), 64, 2).idoms(), [NONE, 0, 1, 2, 3]);
    assert_eq!(pull(&c, full(&c), 64, 2).idoms(), [1, 2, 3, 4, NONE]);
    assert_eq!(depths(&push(&c, full(&c), 64, 2)), [1, 2, 3, 4, 5]);
    let s = out_star(4);
    assert_eq!(push(&s, full(&s), 64, 2).idoms(), [NONE, 0, 0, 0]);
    assert_eq!(pull(&s, full(&s), 64, 2).idoms(), [NONE; 4]);
    let s = in_star(4);
    assert_eq!(push(&s, full(&s), 64, 2).idoms(), [NONE; 4]);
    assert_eq!(pull(&s, full(&s), 64, 2).idoms(), [3, 3, 3, NONE]);
    let l = ladder(6); // x_i = 5 + i − 1 has predecessors {c_0, c_i}
    let t = push(&l, full(&l), 0, 4).idoms();
    assert_eq!(&t[..6], &[NONE, 0, 1, 2, 3, 4]);
    assert!(t[6..].iter().all(|&d| d == 0));
    let e = isolated(0);
    assert!(push(&e, full(&e), 64, 1).idoms().is_empty());
}

// ---------------------------------------------------------------------------------------------
// PG fast path
// ---------------------------------------------------------------------------------------------

#[test]
fn mini_pg_blocks_equal_generic() {
    let mut r = SmallRng::seed_from_u64(42);
    for seed in 0..60u64 {
        let txs = 20 + 7 * seed as usize;
        let g = mini(txs, seed, if seed % 3 == 0 { 0.2 } else { 0.0 });
        let (idx, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let n = g.num_nodes();
        let mut windows = vec![Window::full(n)];
        for _ in 0..4 {
            windows.push(random_window(n, &mut r));
        }
        for w in windows {
            for &budget in &[0, 2, 64] {
                let c = conf(w, budget);
                let db = dominators_blocks(&idx, &c, &mut quiet()).unwrap();
                let dg = push(&g, w, budget, 5);
                assert_eq!(db.idoms(), dg.idoms(), "dom seed {seed} {w:?}");
                assert_eq!(depths(&db), depths(&dg), "dom depth seed {seed} {w:?}");
                let pb = post_dominators_blocks(&idx, &c, &mut quiet()).unwrap();
                let pg = pull(&g, w, budget, 5);
                assert_eq!(pb.idoms(), pg.idoms(), "post seed {seed} {w:?}");
                assert_eq!(depths(&pb), depths(&pg), "post depth seed {seed} {w:?}");
            }
        }
        if n <= 120 {
            let w = Window::full(n);
            let db = dominators_blocks(&idx, &conf(w, 64), &mut quiet()).unwrap();
            assert_eq!(db.idoms(), brute_dom(&g), "brute dom seed {seed}");
            let pb = post_dominators_blocks(&idx, &conf(w, 64), &mut quiet()).unwrap();
            assert_eq!(pb.idoms(), brute_post(&g), "brute post seed {seed}");
        }
    }
}

#[test]
fn blocks_builders_reject_bad_windows() {
    let g = mini(30, 1, 0.0);
    let (idx, _) = BlockIndex::from_vec_graph(&g).unwrap();
    let w = Window {
        lo: 0,
        hi: g.num_nodes() + 1,
    };
    let e = dominators_blocks(&idx, &conf(w, 64), &mut quiet()).unwrap_err();
    assert!(matches!(
        e.downcast_ref::<DagError>(),
        Some(DagError::BadWindow { .. })
    ));
}

// ---------------------------------------------------------------------------------------------
// Kernel properties
// ---------------------------------------------------------------------------------------------

#[test]
fn budgets_give_identical_trees() {
    let mut graphs = vec![ladder(80), diamonds(30), complete_dag(20)];
    for seed in 0..20 {
        graphs.push(random_dag(150, 0.03, seed));
        graphs.push(mini(100, seed, 0.1));
    }
    for g in &graphs {
        let w = full(g);
        let base_d = push(g, w, 64, 8);
        let base_p = pull(g, w, 64, 8);
        for b in BUDGETS {
            let d = push(g, w, b, 8);
            assert_eq!(d.idoms(), base_d.idoms());
            assert_eq!(depths(&d), depths(&base_d));
            let p = pull(g, w, b, 8);
            assert_eq!(p.idoms(), base_p.idoms());
            assert_eq!(depths(&p), depths(&base_p));
            if b == u32::MAX {
                assert_eq!(d.stats.jump_fallbacks, 0);
            }
        }
    }
    // ladder: budget 0 always falls back; the walk costs grow with the rung
    let l = ladder(80);
    let d0 = push(&l, full(&l), 0, 8);
    assert!(d0.stats.jump_fallbacks > 0);
    let dinf = push(&l, full(&l), u32::MAX, 8);
    assert_eq!(dinf.stats.jump_fallbacks, 0);
    assert_eq!(dinf.stats.calls, d0.stats.calls);
    let hist = &dinf.stats.walk_steps_log2_hist;
    assert!(
        hist[6..].iter().sum::<u64>() > 0,
        "long walks expected: {hist:?}"
    );
    assert_eq!(
        dinf.stats.top_exits + dinf.stats.walk_steps_log2_hist.iter().sum::<u64>(),
        dinf.stats.calls
    );
}

#[test]
fn nca_and_level_ancestor_match_naive() {
    let mut r = SmallRng::seed_from_u64(5);
    for _ in 0..50 {
        let n = r.random_range(1..400usize);
        // a random tree: parent of x is uniform in [0, x), biased to long paths half the time
        let deep = r.random_bool(0.5);
        let mut par = vec![0u32; n + 1];
        let mut t = DomTree::with_capacity(n, false, 2);
        for x in 1..=n as u32 {
            let p = if deep && x > 1 && r.random_bool(0.9) {
                x - 1
            } else {
                r.random_range(0..x)
            };
            par[x as usize] = p;
            t.add_leaf(x, p);
        }
        let anc = |mut x: u32| {
            let mut v = vec![x];
            while x != ROOT {
                x = par[x as usize];
                v.push(x);
            }
            v
        };
        for _ in 0..200 {
            let a = r.random_range(0..=n as u32);
            let b = r.random_range(0..=n as u32);
            let (aa, ab) = (anc(a), anc(b));
            let want = *aa.iter().find(|x| ab.contains(x)).unwrap();
            for budget in BUDGETS {
                t.set_budget(budget);
                assert_eq!(t.nca(a, b), want);
                assert_eq!(t.nca_tracked(b, a), want);
            }
            let d = t.depth_index(a);
            let k = r.random_range(0..=d);
            assert_eq!(t.level_ancestor(a, k), aa[(d - k) as usize]);
        }
    }
}

/// The root child above the internal index `x` (naive climb), `ROOT` for the root.
fn naive_top(t: &DomTree, mut x: u32) -> u32 {
    while x != ROOT && t.parent(x) != ROOT {
        x = t.parent(x);
    }
    x
}

/// Asserts that `nca_tracked(a, b)` exits at the `top` test exactly when `a` and `b` lie below
/// different root children (the fast path must fire, not only give the right answer).
fn assert_top_exit(t: &mut DomTree, a: u32, b: u32, ctx: &str) {
    let before = t.stats.top_exits;
    let c = t.nca_tracked(a, b);
    let (ta, tb) = (naive_top(t, a), naive_top(t, b));
    let exited = t.stats.top_exits == before + 1;
    assert_eq!(exited, ta != tb, "{ctx}: a {a} b {b} tops {ta} {tb}");
    if exited {
        assert_eq!(c, ROOT, "{ctx}");
    }
}

#[test]
fn top_test_fires_across_root_children() {
    // kernel: root children 1, 2 (add_leaf) and 5 (add_leaf_like of the root child 1)
    let mut t = DomTree::with_capacity(7, false, 64);
    t.add_leaf(1, ROOT);
    t.add_leaf(2, ROOT);
    t.add_leaf(3, 1);
    t.add_leaf(4, 2);
    t.add_leaf_like(5, 1);
    t.add_leaf(6, 5);
    t.add_leaf_like(7, 3);
    for a in 0..=7 {
        for b in 0..=7 {
            assert_top_exit(&mut t, a, b, "kernel");
        }
    }
    // builders: every pair of nodes of mini-PG trees (blocks builders use add_leaf_like)
    for seed in 0..12u64 {
        let g = mini(40, seed, 0.1);
        let (idx, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let w = full(&g);
        let n = g.num_nodes() as u32;
        for post in [false, true] {
            let c = conf(w, 64);
            let mut trees = vec![if post {
                pull(&g, w, 64, 4)
            } else {
                push(&g, w, 64, 4)
            }];
            trees.push(if post {
                post_dominators_blocks(&idx, &c, &mut quiet()).unwrap()
            } else {
                dominators_blocks(&idx, &c, &mut quiet()).unwrap()
            });
            for mut t in trees {
                // the builders' own queries: merges of different lineages exit at the top test
                assert!(
                    t.stats.top_exits > 0,
                    "seed {seed} post {post} {:?}",
                    t.stats
                );
                for a in 0..=n {
                    for b in 0..=n {
                        assert_top_exit(&mut t, a, b, &format!("seed {seed} post {post}"));
                    }
                }
            }
        }
    }
    // chains of 4 nodes merged into one sink: their only common dominator is the root, so
    // every query of the push builder exits at the top test
    let k = 5;
    let mut ar = Vec::new();
    for c in 0..k {
        ar.extend((0..3).map(|i| (4 * c + i, 4 * c + i + 1)));
        ar.push((4 * c + 3, 4 * k));
    }
    let g = vec_graph(4 * k + 1, ar);
    let t = push(&g, full(&g), 64, 3);
    assert_eq!(t.idom_public(4 * k), NONE);
    assert!(t.stats.calls >= 1);
    assert_eq!(t.stats.top_exits, t.stats.calls, "{:?}", t.stats);
}

#[test]
fn with_capacity_slots_are_unset() {
    let t = DomTree::with_capacity(10, false, 64);
    assert_eq!(t.pending(ROOT), ROOT);
    assert_eq!(t.depth_index(ROOT), 0);
    for x in 1..=10 {
        assert_eq!(t.pending(x), UNSET, "slot {x}");
    }
    // regression: with ROOT-initialised slots the push builder would make every node a root
    // child (ROOT absorbs the NCA); a chain needs the UNSET distinction
    let c = chain(6);
    assert_eq!(push(&c, full(&c), 64, 1).idoms(), [NONE, 0, 1, 2, 3, 4]);
    let w = Window { lo: 3, hi: 9 };
    let t = DomTree::for_window(w, true, 64);
    assert_eq!(t.len(), 6);
    assert_eq!(t.index_of(8), 1);
    assert_eq!(t.node_of(6), 3);
}

#[test]
fn idempotence_on_tree_arcs() {
    for seed in 0..60u64 {
        let g = if seed % 2 == 0 {
            random_dag(80, 0.05, seed)
        } else {
            mini(50, seed, 0.1)
        };
        let n = g.num_nodes();
        let d = push(&g, full(&g), 64, 4).idoms();
        let tg = vec_graph(
            n,
            (0..n).filter(|&v| d[v] != NONE).map(|v| (d[v] as usize, v)),
        );
        assert_eq!(push(&tg, full(&tg), 64, 4).idoms(), d, "dom seed {seed}");
        let p = pull(&g, full(&g), 64, 4).idoms();
        let tp = vec_graph(
            n,
            (0..n).filter(|&v| p[v] != NONE).map(|v| (v, p[v] as usize)),
        );
        assert_eq!(pull(&tp, full(&tp), 64, 4).idoms(), p, "post seed {seed}");
    }
}

#[test]
fn sizes_and_preorder_are_consistent() {
    for seed in 0..40u64 {
        let g = random_dag(60, 0.06, seed);
        let n = g.num_nodes();
        for post in [false, true] {
            let t = if post {
                pull(&g, full(&g), 3, 5)
            } else {
                push(&g, full(&g), 3, 5)
            };
            let size = t.sizes();
            let pre = t.preorder(&size);
            assert_eq!(size[0] as usize, n + 1);
            // size = 1 + Σ children; root children cover the window
            let mut want = vec![1u32; n + 1];
            for x in (1..=n as u32).rev() {
                want[t.parent(x) as usize] += want[x as usize];
            }
            assert_eq!(size, want);
            let root_cover: u32 = (1..=n as u32)
                .filter(|&x| t.parent(x) == ROOT)
                .map(|x| size[x as usize])
                .sum();
            assert_eq!(root_cover as usize, n);
            let psize: Vec<u32> = (0..n).map(|v| size[t.index_of(v) as usize]).collect();
            let st = tree_stats(&t, &size, 3);
            assert_naive_stats(&st, &t.idoms(), &depths(&t), &psize, full(&g), post);
            // pre is a permutation of 0..=n
            let mut sorted = pre.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..=n as u32).collect::<Vec<_>>());
            // interval query == ancestor relation
            for a in 0..=n as u32 {
                for x in 0..=n as u32 {
                    let mut y = x;
                    let mut is_anc = y == a;
                    while y != ROOT && !is_anc {
                        y = t.parent(y);
                        is_anc = y == a;
                    }
                    let q = pre[a as usize] <= pre[x as usize]
                        && pre[x as usize] < pre[a as usize] + size[a as usize];
                    assert_eq!(q, is_anc, "seed {seed} post {post} a {a} x {x}");
                }
            }
        }
    }
}

#[test]
fn windowed_runs_equal_induced_subgraphs() {
    let mut r = SmallRng::seed_from_u64(9);
    for seed in 0..80u64 {
        let g = if seed % 2 == 0 {
            random_dag(70, 0.06, seed)
        } else {
            mini(40, seed, 0.1)
        };
        let w = random_window(g.num_nodes(), &mut r);
        let h = induced(&g, w);
        let hw = full(&h);
        assert_eq!(
            push(&g, w, 3, 4).idoms(),
            shift(&push(&h, hw, 3, 4).idoms(), w.lo),
            "dom seed {seed} {w:?}"
        );
        assert_eq!(
            pull(&g, w, 3, 4).idoms(),
            shift(&pull(&h, hw, 3, 4).idoms(), w.lo),
            "post seed {seed} {w:?}"
        );
        assert_eq!(depths(&push(&g, w, 3, 4)), depths(&push(&h, hw, 3, 4)));
    }
}

#[test]
fn back_arcs_are_rejected() {
    for g in [
        with_back_arc(&random_dag(30, 0.1, 3)),
        with_self_loop(&chain(6)),
    ] {
        let w = full(&g);
        let cg = chunked(&g, 4);
        let e = dominators_push(&cg, &conf(w, 64), &mut quiet()).unwrap_err();
        assert!(matches!(
            e.downcast_ref::<DagError>(),
            Some(DagError::NotTopological { .. })
        ));
        let e = post_dominators_pull(&cg, &conf(w, 64), &mut quiet()).unwrap_err();
        assert!(matches!(
            e.downcast_ref::<DagError>(),
            Some(DagError::NotTopological { .. })
        ));
        let t = vec![NONE; g.num_nodes()];
        let e = verify_dominators(&cg, &t, w, &mut cquiet()).unwrap_err();
        assert!(matches!(
            e.downcast_ref::<DagError>(),
            Some(DagError::NotTopological { .. })
        ));
    }
}

// ---------------------------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------------------------

fn check(g: &VecGraph, t: &[u32], w: Window, post: bool) -> dag::dom::VerifyReport {
    let cg = chunked(g, 3);
    if post {
        verify_post_dominators(&cg, t, w, &mut cquiet()).unwrap()
    } else {
        verify_dominators(&cg, t, w, &mut cquiet()).unwrap()
    }
}

#[test]
fn verifier_accepts_correct_trees_and_rejects_mutations() {
    let mut r = SmallRng::seed_from_u64(77);
    let mut rejected = [0usize; 4];
    for seed in 0..120u64 {
        let g = match seed % 3 {
            0 => random_dag(50, 0.08, seed),
            1 => mini(40, seed, 0.15),
            _ => layered(&[4, 6, 5, 6, 4], 0.3, seed),
        };
        let n = g.num_nodes();
        let w = if seed % 4 == 3 {
            random_window(n, &mut r)
        } else {
            Window::full(n)
        };
        for post in [false, true] {
            let tree = if post {
                pull(&g, w, 3, 4)
            } else {
                push(&g, w, 3, 4)
            };
            let t = tree.idoms();
            let rep = check(&g, &t, w, post);
            assert!(
                rep.ok,
                "correct tree rejected: seed {seed} post {post} {rep:?}"
            );
            assert_eq!(rep.first_failure, None);
            if w.is_empty() {
                continue;
            }
            // parent of v (absolute) in public terms
            let par = |v: usize| t[v - w.lo];
            // "precedes" in the tree order
            let before = |a: usize, b: usize| if post { a > b } else { a < b };
            for _ in 0..6 {
                let v = r.random_range(w.lo..w.hi);
                let p = par(v);
                // 1. an ancestor: the grandparent (or the root)
                if p != NONE {
                    let mut m = t.clone();
                    m[v - w.lo] = par(p as usize);
                    let rep = check(&g, &m, w, post);
                    assert!(
                        !rep.ok,
                        "grandparent accepted: seed {seed} post {post} v {v}"
                    );
                    rejected[0] += 1;
                }
                // 2. a sibling that precedes v
                if let Some(s) = (w.lo..w.hi).find(|&s| s != v && par(s) == p && before(s, v)) {
                    let mut m = t.clone();
                    m[v - w.lo] = s as u32;
                    let rep = check(&g, &m, w, post);
                    assert!(!rep.ok, "sibling accepted: seed {seed} post {post} v {v}");
                    rejected[1] += 1;
                }
                // 3. a later node (violates the order: V0)
                let later = if post {
                    (v > w.lo).then(|| r.random_range(w.lo..v))
                } else {
                    (v + 1 < w.hi).then(|| r.random_range(v + 1..w.hi))
                };
                if let Some(l) = later {
                    let mut m = t.clone();
                    m[v - w.lo] = l as u32;
                    let rep = check(&g, &m, w, post);
                    assert!(!rep.ok && rep.v0_fail > 0, "later node accepted: {rep:?}");
                    assert!(rep.first_failure.as_ref().is_some_and(|f| f.0 <= v));
                    rejected[2] += 1;
                }
                // 4. any other value that precedes v
                let cands: Vec<u32> = std::iter::once(NONE)
                    .chain((w.lo..w.hi).filter(|&s| before(s, v)).map(|s| s as u32))
                    .filter(|&c| c != p)
                    .collect();
                if !cands.is_empty() {
                    let c = cands[r.random_range(0..cands.len())];
                    let mut m = t.clone();
                    m[v - w.lo] = c;
                    let rep = check(&g, &m, w, post);
                    assert!(!rep.ok, "mutation {p} -> {c} accepted: seed {seed} v {v}");
                    rejected[3] += 1;
                }
            }
        }
    }
    assert!(
        rejected.iter().all(|&c| c > 50),
        "too few mutations tried: {rejected:?}"
    );
}

#[test]
fn verifier_reports_conditions() {
    // diamond 0 → {1, 2} → 3: idom(3) = 0
    let g = diamonds(1);
    let w = full(&g);
    let ok = [NONE, 0, 0, 0];
    assert!(check(&g, &ok, w, false).ok);
    // idom(3) = 1: V1 fails on the arc (2, 3)
    let rep = check(&g, &[NONE, 0, 0, 1], w, false);
    assert_eq!((rep.v0_fail, rep.v1_fail), (0, 1));
    assert_eq!(rep.first_failure.as_ref().unwrap().0, 3);
    // regression: a node whose only predecessor fails V1 counts in v1_fail, not in v2_fail
    // (0 → 2, 1 isolated, idom(2) = 1)
    let h = vec_graph(3, [(0, 2)]);
    let rep = check(&h, &[NONE, NONE, 1], full(&h), false);
    assert_eq!((rep.v0_fail, rep.v1_fail, rep.v2_fail), (0, 1, 0));
    assert!(!rep.ok);
    // chain 0 → 1 → 2 with idom(2) = 0: V1 holds, V2 fails (one predecessor, no parent arc)
    let c = chain(3);
    let rep = check(&c, &[NONE, 0, 0], full(&c), false);
    assert_eq!((rep.v0_fail, rep.v1_fail, rep.v2_fail), (0, 0, 1));
    // a source with a parent: V0
    let rep = check(&c, &[1, 0, 1], full(&c), false);
    assert!(!rep.ok && rep.v0_fail >= 1);
    // wrong length
    let cg = chunked(&c, 2);
    assert!(verify_dominators(&cg, &[NONE], full(&c), &mut cquiet()).is_err());
}

// ---------------------------------------------------------------------------------------------
// BV round trip, threads, outputs, statistics
// ---------------------------------------------------------------------------------------------

#[test]
fn bv_round_trip_agrees_with_vec_graph() {
    let dir = tempfile::tempdir().unwrap();
    for seed in 0..6u64 {
        let g = if seed % 2 == 0 {
            random_dag(300, 0.01, seed)
        } else {
            mini(200, seed, 0.1)
        };
        let base = write_bv(dir.path(), &format!("g{seed}"), &g);
        let cg = graph::open(
            &base,
            ChunkConf {
                chunk_arcs: 17,
                max_chunk_nodes: 13,
            },
        )
        .unwrap();
        let w = full(&g);
        let d = dominators_push(&cg, &conf(w, 3), &mut quiet()).unwrap();
        assert_eq!(d.idoms(), push(&g, w, 3, 5).idoms());
        let p = post_dominators_pull(&cg, &conf(w, 3), &mut quiet()).unwrap();
        assert_eq!(p.idoms(), pull(&g, w, 3, 5).idoms());
        assert!(
            verify_dominators(&cg, &d.idoms(), w, &mut cquiet())
                .unwrap()
                .ok
        );
        assert!(
            verify_post_dominators(&cg, &p.idoms(), w, &mut cquiet())
                .unwrap()
                .ok
        );
    }
}

#[test]
fn results_do_not_depend_on_threads() {
    let g = mini(400, 3, 0.1);
    let (idx, _) = BlockIndex::from_vec_graph(&g).unwrap();
    let w = full(&g);
    let run = || {
        let d = push(&g, w, 3, 9).idoms();
        let p = pull(&g, w, 3, 9).idoms();
        let db = dominators_blocks(&idx, &conf(w, 3), &mut quiet())
            .unwrap()
            .idoms();
        let rep = verify_dominators(&chunked(&g, 9), &d, w, &mut cquiet()).unwrap();
        (d, p, db, rep)
    };
    let mut res = Vec::new();
    for t in [1, 4, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        res.push(pool.install(run));
    }
    assert!(res.windows(2).all(|r| r[0] == r[1]));
    assert!(res[0].3.ok);
}

#[test]
fn written_arrays_match_the_tree() {
    let dir = tempfile::tempdir().unwrap();
    let g = mini(150, 11, 0.1);
    let n = g.num_nodes();
    let w = Window {
        lo: n / 5,
        hi: n - n / 7,
    };
    for post in [false, true] {
        let t = if post {
            pull(&g, w, 3, 4)
        } else {
            push(&g, w, 3, 4)
        };
        let size = t.sizes();
        let pre = t.preorder(&size);
        let prefix = dir.path().join(if post { "p" } else { "d" });
        t.write(&prefix, Some(&size), Some(&pre)).unwrap();
        let (idom_name, tree_name) = t.field_names();
        let rd = |f: &str| read_array::<u32>(&io::array_path::<u32>(&prefix, f)).unwrap();
        assert_eq!(rd(idom_name), t.idoms());
        assert_eq!(rd(&format!("{tree_name}.depth")), depths(&t));
        let fsize = rd(&format!("{tree_name}.size"));
        let fpre = rd(&format!("{tree_name}.pre"));
        for v in w.lo..w.hi {
            let x = t.index_of(v) as usize;
            assert_eq!(fsize[v - w.lo], size[x]);
            assert_eq!(fpre[v - w.lo], pre[x]);
        }
        // statistics from the files == statistics from the tree
        assert_eq!(sizes_from_idom(&t.idoms(), w, post), fsize);
        let a = dom_stats(&t.idoms(), &depths(&t), &fsize, w, 5);
        let b = tree_stats(&t, &size, 5);
        assert_eq!(a, b);
        assert_eq!(a.root_children_size_sum as usize, w.len());
        assert_naive_stats(&a, &t.idoms(), &depths(&t), &fsize, w, post);
        assert!(
            a.top
                .windows(2)
                .all(|p| (p[0].size, p[1].node) >= (p[1].size, p[0].node))
        );
        let tsv = dir.path().join("top.tsv");
        write_top_tsv(&tsv, &a).unwrap();
        let text = std::fs::read_to_string(&tsv).unwrap();
        assert_eq!(text.lines().count(), a.top.len() + 1);
    }
}

/// Recomputes every field of `st` but `top` with plain loops over the public arrays and
/// compares (an oracle independent of the parallel accumulator of `dom_stats`/`tree_stats`).
fn assert_naive_stats(
    st: &DomStats,
    idom: &[u32],
    depth: &[u32],
    size: &[u32],
    w: Window,
    post: bool,
) {
    let n = w.len();
    let hist = |xs: &[u32]| -> Vec<u64> {
        let mut h: Vec<u64> = Vec::new();
        for &x in xs {
            let b = if x == 0 { 0 } else { x.ilog2() as usize + 1 };
            if h.len() <= b {
                h.resize(b + 1, 0);
            }
            h[b] += 1;
        }
        h
    };
    let roots: Vec<usize> = (0..n).filter(|&i| idom[i] == NONE).collect();
    assert_eq!(st.n, n);
    assert_eq!(st.root_children, roots.len() as u64);
    assert_eq!(
        st.root_children_cut,
        roots.iter().filter(|&&i| size[i] > 1).count() as u64
    );
    assert_eq!(
        st.root_children_dominated,
        roots.iter().map(|&i| size[i] as u64 - 1).sum::<u64>()
    );
    assert_eq!(
        st.root_children_size_sum,
        roots.iter().map(|&i| size[i] as u64).sum::<u64>()
    );
    assert_eq!(st.cut_nodes, size.iter().filter(|&&s| s > 1).count() as u64);
    let maxd = depth.iter().copied().max().unwrap_or(0);
    assert_eq!(st.max_depth, maxd);
    assert_eq!(
        st.argmax_depth,
        depth.iter().position(|&d| d == maxd).map(|i| w.lo + i)
    );
    let mean = depth.iter().map(|&d| d as f64).sum::<f64>() / n.max(1) as f64;
    assert!(
        (st.mean_depth - mean).abs() < 1e-9,
        "{} {mean}",
        st.mean_depth
    );
    assert_eq!(st.depth_log2_hist, hist(depth));
    assert_eq!(st.size_log2_hist, hist(size));
    // size: 1 + Σ sizes of the children, from idom alone
    let mut want = vec![1u32; n];
    let order: Vec<usize> = if post {
        (0..n).collect() // post-dominators: parents follow their children
    } else {
        (0..n).rev().collect()
    };
    for i in order {
        if idom[i] != NONE {
            let s = want[i];
            want[idom[i] as usize - w.lo] += s;
        }
    }
    assert_eq!(size, &want[..]);
}

/// Unit-length BFS distance from the window sources (or to the window sinks).
fn bfs_dmin(g: &VecGraph, w: Window, post: bool) -> Vec<u32> {
    let h = if post {
        mirror(&induced(g, w))
    } else {
        induced(g, w)
    };
    let n = h.num_nodes();
    let mut indeg = vec![0usize; n];
    for (_, v) in arcs(&h) {
        indeg[v] += 1;
    }
    let mut d = vec![u32::MAX; n];
    let mut q: VecDeque<usize> = (0..n).filter(|&v| indeg[v] == 0).collect();
    for &s in &q {
        d[s] = 0;
    }
    while let Some(u) = q.pop_front() {
        for v in h.successors(u) {
            if d[v] == u32::MAX {
                d[v] = d[u] + 1;
                q.push_back(v);
            }
        }
    }
    if post {
        d.reverse();
    }
    d
}

#[test]
fn depth_bound_holds_and_dmin_is_exact() {
    let mut r = SmallRng::seed_from_u64(1);
    for seed in 0..40u64 {
        let g = mini(80, seed, 0.1);
        let (idx, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let w = if seed % 2 == 0 {
            full(&g)
        } else {
            random_window(g.num_nodes(), &mut r)
        };
        for post in [false, true] {
            let want = bfs_dmin(&g, w, post);
            assert_eq!(
                dmin_blocks(&idx, w, post, &mut quiet()),
                want,
                "blocks seed {seed} post {post}"
            );
            let ds = dmin_sweep(
                &chunked(&g, 3),
                w,
                &SweepConf { decoders: 2 },
                post,
                &mut quiet(),
            )
            .unwrap();
            assert_eq!(ds, want, "sweep seed {seed} post {post}");
            let t = if post {
                pull(&g, w, 3, 4)
            } else {
                push(&g, w, 3, 4)
            };
            let b = check_depth_bound(&t, &want);
            assert_eq!(b.violations, 0, "{b:?}");
            assert_eq!(b.checked as usize, w.len());
        }
    }
    // the bound is tight on a chain
    let c = chain(6);
    let t = push(&c, full(&c), 64, 2);
    let b = check_depth_bound(&t, &bfs_dmin(&c, full(&c), false));
    assert_eq!((b.violations, b.max_slack, b.max_dmin), (0, 0, 5));
    // diamond 0 → {1, 2} → 3: depth(3) = 2 (idom 0), dmin(3) = 2, slack 1
    let g = diamonds(1);
    let t = push(&g, full(&g), 64, 2);
    let d = bfs_dmin(&g, full(&g), false);
    assert_eq!(d, [0, 1, 1, 2]);
    let b = check_depth_bound(&t, &d);
    assert_eq!((b.violations, b.max_slack, b.max_dmin), (0, 1, 2));
    assert!((b.mean_dmin - 1.0).abs() < 1e-12);
    // a planted violation is found
    let b = check_depth_bound(&t, &[0, 1, 1, 0]);
    assert_eq!((b.violations, b.first_violation), (1, Some(3)));
}
