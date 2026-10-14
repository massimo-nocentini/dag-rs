//! Acceptance tests of the critical path method.

#[path = "support/dp.rs"]
mod support;

use std::collections::{BTreeMap, BTreeSet};

use dag::blocks::BlockIndex;
use dag::cpm::{CpmResult, cpm};
use dag::dp::{ArcWeights, Const, DpInput, Kernel, Node};
use dag::semiring::MaxPlus;
use dag::testkit::*;
use dag::{NONE, Window};
use rand::RngExt;
use support::*;
use webgraph::prelude::*;

const SEEDS: u64 = 300;

/// The weight of a path under `w` (MaxPlus<u32>: a sum).
fn path_weight<W: ArcWeights<MaxPlus<u32>>>(w: &W, p: &[usize]) -> u32 {
    let s = MaxPlus::<u32>::new();
    p.windows(2).map(|a| w.arc(&s, a[0], a[1])).sum()
}

/// Runs the CPM with every kernel available for `g` and checks that they agree.
fn cpm_all<W: ArcWeights<MaxPlus<u32>>>(
    g: &VecGraph,
    b: Option<&BlockIndex>,
    win: Window,
    w: &W,
    chunk: u64,
) -> CpmResult<u32> {
    let gt = transpose(g);
    let (cg, cgt) = (chunked(g, chunk), chunked(&gt, chunk));
    // (forward, backward) kernels: push/pull from the graph alone, pull/pull with the
    // transpose, blocks/blocks with the index (also what Auto picks then)
    let graph_only = DpInput::graph(&cg);
    let both = DpInput::with_transpose(&cg, &cgt);
    // Kernel::Push runs E with push and H with pull (there is no backward push kernel)
    let mut runs = vec![
        ("push+pull", graph_only, Kernel::Auto),
        ("push", DpInput::graph(&cg), Kernel::Push),
        ("pull", both, Kernel::Pull),
    ];
    if let Some(b) = b {
        runs.push((
            "blocks",
            DpInput::with_transpose(&cg, &cgt).and_blocks(b),
            Kernel::Blocks,
        ));
        runs.push(("auto", DpInput::graph(&cg).and_blocks(b), Kernel::Auto));
    }
    let mut out: Option<CpmResult<u32>> = None;
    for (what, input, k) in runs {
        let r = cpm::<u32, _, _>(&input, w, &conf(win, k, true, 2), &mut quiet())
            .unwrap_or_else(|e| panic!("{what}: {e:#}"));
        if let Some(o) = &out {
            assert_eq!(o, &r, "{what} differs");
        }
        out = Some(r);
    }
    out.unwrap()
}

fn check_invariants(r: &CpmResult<u32>, g: &VecGraph) {
    let win = r.window;
    let t = r.t;
    if !win.is_empty() {
        assert_eq!(Some(t), r.e.iter().copied().max());
        assert_eq!(Some(t), r.h.iter().copied().max());
    }
    for i in 0..win.len() {
        assert!(r.e[i] + r.h[i] <= t, "E + H > T at {}", i + win.lo);
        assert_eq!(r.slack(i + win.lo), t - r.e[i] - r.h[i]);
    }
    let sl = r.slacks();
    assert_eq!(sl.len(), win.len());
    // E is 0 exactly on window sources, H exactly on window sinks
    let succ = window_succ(g, win);
    for v in win.lo..win.hi {
        let has_pred = succ.iter().any(|s| s.contains(&v));
        if !has_pred {
            assert_eq!(r.e[v - win.lo], 0);
        }
        if succ[v].is_empty() {
            assert_eq!(r.h[v - win.lo], 0);
        }
    }
}

#[test]
fn cpm_matches_brute_force_longest_paths() {
    for seed in 0..SEEDS {
        let mut r = rng(seed);
        let (g, b) = if seed % 2 == 0 {
            (
                random_dag(r.random_range(0..=11), r.random_range(0.1..0.6), seed),
                None,
            )
        } else {
            let opts = MiniPgOpts {
                max_outputs: 2,
                ..Default::default()
            };
            let pg = mini_pg(r.random_range(1..7), seed, &opts).graph;
            let b = BlockIndex::from_vec_graph(&pg).unwrap().0;
            (pg, Some(b))
        };
        let n = g.num_nodes();
        if n > 12 {
            continue;
        }
        let win = random_window(&mut r, n);
        let dur: Vec<u32> = (0..n).map(|u| (mix(seed ^ u as u64) % 4) as u32).collect();
        let node = Node {
            alpha: |u: usize| dur[u],
            beta: |_: usize| 0u32,
        };
        for unit in [true, false] {
            let res = if unit {
                cpm_all(&g, b.as_ref(), win, &Const(1u32), 1 + seed % 3)
            } else {
                cpm_all(&g, b.as_ref(), win, &node, 1 + seed % 3)
            };
            check_invariants(&res, &g);
            let paths = window_paths(&g, win);
            let weight = |p: &[usize]| {
                if unit {
                    path_weight(&Const(1u32), p)
                } else {
                    path_weight(&node, p)
                }
            };
            let tmax = paths.iter().map(|p| weight(p)).max().unwrap_or(0);
            assert_eq!(res.t, tmax, "seed {seed} unit {unit}");
            let longest: Vec<&Vec<usize>> = paths.iter().filter(|p| weight(p) == tmax).collect();
            let crit_nodes: BTreeSet<usize> =
                longest.iter().flat_map(|p| p.iter().copied()).collect();
            let crit_arcs: BTreeSet<(usize, usize)> = longest
                .iter()
                .flat_map(|p| p.windows(2).map(|a| (a[0], a[1])))
                .collect();
            for v in win.lo..win.hi {
                assert_eq!(
                    res.is_critical(v),
                    crit_nodes.contains(&v),
                    "seed {seed} node {v}"
                );
            }
            let cg = chunked(&g, 2);
            let arcs = if unit {
                res.critical_arcs(&cg, &Const(1u32), &mut cquiet()).unwrap()
            } else {
                res.critical_arcs(&cg, &node, &mut cquiet()).unwrap()
            };
            assert_eq!(arcs as usize, crit_arcs.len(), "seed {seed} unit {unit}");
            // summary
            let sum = res.summary(unit);
            assert_eq!(sum.t, tmax as u64);
            assert_eq!(sum.critical_nodes as usize, crit_nodes.len());
            let mut layers: BTreeMap<u32, usize> = BTreeMap::new();
            for &v in &crit_nodes {
                *layers.entry(res.e[v - win.lo]).or_default() += 1;
            }
            assert_eq!(
                sum.max_critical_width as usize,
                layers.values().copied().max().unwrap_or(0)
            );
            if unit {
                let w1 = layers.values().filter(|&&c| c == 1).count();
                assert_eq!(sum.width1_layers, Some(w1 as u64));
                // with unit weights a width-1 layer is a node on every longest path
                for (&d, &c) in &layers {
                    if c == 1 {
                        let v = crit_nodes
                            .iter()
                            .find(|&&v| res.e[v - win.lo] == d)
                            .unwrap();
                        assert!(longest.iter().all(|p| p.contains(v)));
                    }
                }
            } else {
                assert_eq!(sum.width1_layers, None);
            }
            if !win.is_empty() {
                let end = (win.lo..win.hi)
                    .find(|&v| res.e[v - win.lo] == res.t)
                    .unwrap();
                assert_eq!(sum.longest_path_end, end);
                let p: Vec<usize> = sum
                    .longest_path
                    .unwrap()
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                assert_eq!(*p.last().unwrap(), end);
                assert_eq!(weight(&p), tmax);
                assert!(paths.contains(&p), "seed {seed}: {p:?} is not a path");
            }
        }
    }
}

#[test]
fn critical_arcs_form_t_length_paths() {
    for seed in 0..100 {
        let pg = mini_pg(60, seed, &MiniPgOpts::default()).graph;
        let n = pg.num_nodes();
        let b = BlockIndex::from_vec_graph(&pg).unwrap().0;
        let res = cpm_all(&pg, Some(&b), Window::full(n), &Const(1u32), 4);
        check_invariants(&res, &pg);
        // every critical arc (u, v) extends to a path of length T: E(u) + 1 + H(v) = T
        let mut count = 0u64;
        for u in 0..n {
            for v in pg.successors(u) {
                if res.is_critical(u) && res.is_critical(v) && res.e[u] + 1 == res.e[v] {
                    assert_eq!(res.e[u] + 1 + res.h[v], res.t);
                    count += 1;
                }
            }
        }
        let arcs = res
            .critical_arcs(&chunked(&pg, 3), &Const(1u32), &mut cquiet())
            .unwrap();
        assert_eq!(arcs, count);
        // and the critical subgraph has a source-to-sink path of every node
        assert!(res.summary(true).critical_nodes as u32 > res.t);
    }
}

#[test]
fn unit_e_equals_the_analysis_path_dmax_recurrence() {
    for seed in 0..SEEDS {
        let mut r = rng(seed);
        let g = if seed % 2 == 0 {
            random_dag(r.random_range(0..200), r.random_range(0.0..0.05), seed)
        } else {
            mini_pg(r.random_range(1..150), seed, &MiniPgOpts::default()).graph
        };
        let n = g.num_nodes();
        // analysis_path: one sequential pass, d_max(v) = max over predecessors of d_max(u) + 1
        // (first-seen predecessor on ties), 0 at sources
        let mut dmax = vec![u32::MAX; n];
        let mut src = vec![NONE; n];
        for u in 0..n {
            if dmax[u] == u32::MAX {
                dmax[u] = 0;
                src[u] = u as u32;
            }
            for v in g.successors(u) {
                if dmax[v] == u32::MAX || dmax[u] + 1 > dmax[v] {
                    dmax[v] = dmax[u] + 1;
                    src[v] = src[u];
                }
            }
        }
        let b = BlockIndex::from_vec_graph(&g).ok().map(|x| x.0);
        let res = cpm_all(&g, b.as_ref(), Window::full(n), &Const(1u32), 1 + seed % 5);
        assert_eq!(res.e, dmax, "seed {seed}");
        // the pred chain ends at the first-seen source of the recurrence
        let pred = res.pred.as_ref().unwrap();
        for (v, &sv) in src.iter().enumerate() {
            let path = dag::dp::path::path_to(pred, Window::full(n), v);
            assert_eq!(*path.last().unwrap(), sv, "seed {seed} node {v}");
        }
    }
}

#[test]
fn windowed_cpm_equals_cpm_on_the_induced_subgraph() {
    for seed in 0..100 {
        let mut r = rng(seed);
        let pg = mini_pg(80, seed, &MiniPgOpts::default()).graph;
        let n = pg.num_nodes();
        let b = BlockIndex::from_vec_graph(&pg).unwrap().0;
        let win = random_window(&mut r, n);
        let h = induced(&pg, win);
        let hb = BlockIndex::from_vec_graph(&h).unwrap().0;
        let a = cpm_all(&pg, Some(&b), win, &Const(1u32), 3);
        let z = cpm_all(&h, Some(&hb), Window::full(win.len()), &Const(1u32), 2);
        assert_eq!(a.e, z.e);
        assert_eq!(a.h, z.h);
        assert_eq!(a.t, z.t);
        assert_eq!(
            a.pred.as_ref().unwrap(),
            &shift_pred(z.pred.as_ref().unwrap(), win.lo)
        );
        let (sa, sz) = (a.summary(true), z.summary(true));
        assert_eq!(sa.critical_nodes, sz.critical_nodes);
        assert_eq!(sa.width1_layers, sz.width1_layers);
        assert_eq!(sa.longest_path_end, sz.longest_path_end + win.lo);
    }
}

#[test]
fn cpm_u64_agrees_with_u32() {
    let pg = mini_pg(300, 5, &MiniPgOpts::default()).graph;
    let n = pg.num_nodes();
    let cg = chunked(&pg, 7);
    let c = conf(Window::full(n), Kernel::Auto, false, 3);
    let a = cpm::<u32, _, _>(&DpInput::graph(&cg), &Const(1u32), &c, &mut quiet()).unwrap();
    let b = cpm::<u64, _, _>(&DpInput::graph(&cg), &Const(1u64), &c, &mut quiet()).unwrap();
    assert_eq!(a.t as u64, b.t);
    assert!(a.e.iter().zip(&b.e).all(|(&x, &y)| x as u64 == y));
    assert_eq!(a.summary(true), b.summary(true));
}

#[test]
fn empty_window() {
    let g = chain(4);
    let cg = chunked(&g, 1);
    let r = cpm::<u32, _, _>(
        &DpInput::graph(&cg),
        &Const(1u32),
        &conf(Window { lo: 2, hi: 2 }, Kernel::Auto, true, 1),
        &mut quiet(),
    )
    .unwrap();
    assert!(r.e.is_empty() && r.h.is_empty());
    assert_eq!(r.t, 0);
    let s = r.summary(true);
    assert_eq!(s.critical_nodes, 0);
    assert_eq!(s.longest_path, None);
}

/// The layer profile of `r` computed directly: (critical nodes, max width, width-1 layers).
fn brute_layers(r: &CpmResult<u32>) -> (u64, u64, u64) {
    let mut layers: BTreeMap<u32, u64> = BTreeMap::new();
    for i in 0..r.e.len() {
        if r.e[i] + r.h[i] == r.t {
            *layers.entry(r.e[i]).or_default() += 1;
        }
    }
    (
        layers.values().sum(),
        layers.values().copied().max().unwrap_or(0),
        layers.values().filter(|&&c| c == 1).count() as u64,
    )
}

/// Brute force of the widest critical layer (smallest `E` on ties), its node range (window
/// starting at 0) and the log2 histogram of the widths of the layers `0..=T` (trailing zeros
/// dropped).
fn brute_widest(r: &dag::cpm::CpmResult<u32>) -> (u64, Option<(usize, usize)>, Vec<u64>) {
    // critical nodes per non-empty layer (T can be ~10⁸ here: no dense array)
    let mut width: BTreeMap<u32, u64> = BTreeMap::new();
    for i in 0..r.e.len() {
        if r.e[i] + r.h[i] == r.t {
            *width.entry(r.e[i]).or_default() += 1;
        }
    }
    let max = width.values().copied().max().unwrap_or(0);
    let layer = width
        .iter()
        .find(|&(_, &c)| c == max)
        .map_or(0, |(&e, _)| e);
    let nodes: Vec<usize> = (0..r.e.len())
        .filter(|&i| r.e[i] + r.h[i] == r.t && r.e[i] == layer)
        .collect();
    let mut hist = vec![0u64; 66];
    // bin 0: layers without a critical node; bin k: [2^(k-1), 2^k)
    hist[0] = r.t as u64 + 1 - width.len() as u64;
    for &c in width.values() {
        hist[(u64::BITS - c.leading_zeros()) as usize] += 1;
    }
    while hist.last() == Some(&0) {
        hist.pop();
    }
    let range = (max > 0).then(|| (nodes[0], *nodes.last().unwrap()));
    (u64::from(layer), range, hist)
}

#[test]
fn layer_profile_histogram_and_sort_fallback_agree_with_brute_force() {
    for seed in 0..40 {
        let g = random_dag(200, 0.02 + (seed % 5) as f64 * 0.01, seed);
        let n = g.num_nodes();
        let cg = chunked(&g, 5);
        let c = conf(Window::full(n), Kernel::Auto, false, 2);
        // small durations (T < 2²⁰: histogram) and huge ones (T ≥ 2²⁰ > 2·n: sort fallback)
        for scale in [1u32, 1 << 21] {
            let dur: Vec<u32> = (0..n)
                .map(|u| (mix(seed ^ u as u64) % 4) as u32 * scale)
                .collect();
            let w = Node {
                alpha: |u: usize| dur[u],
                beta: |_: usize| 0u32,
            };
            let r = cpm::<u32, _, _>(&DpInput::graph(&cg), &w, &c, &mut quiet()).unwrap();
            let s = r.summary(true);
            let (crit, maxw, w1) = brute_layers(&r);
            assert_eq!(
                (s.critical_nodes, s.max_critical_width, s.width1_layers),
                (crit, maxw, Some(w1))
            );
            // regression: the layer profile was reduced to two scalars
            let (layer, range, widths) = brute_widest(&r);
            assert_eq!(
                (
                    s.max_critical_layer,
                    s.max_critical_layer_nodes,
                    &s.critical_width_log2_hist
                ),
                (layer, range, &widths),
                "seed {seed} scale {scale}"
            );
            let hist = (r.t as u64) < 1 << 20;
            assert_eq!(hist, scale == 1 || r.t == 0, "seed {seed} T {}", r.t);
            let bytes = if hist {
                4 * (r.t as u64 + 1)
            } else {
                8 * n as u64
            };
            assert_eq!(r.summary_bytes(), bytes);
        }
    }
    // T = 0: every node is critical and sits in the single layer E = 0 (one 4-byte counter)
    let g = isolated(1000);
    let cg = chunked(&g, 3);
    let r = cpm::<u32, _, _>(
        &DpInput::graph(&cg),
        &Const(1u32),
        &conf(Window::full(1000), Kernel::Auto, false, 2),
        &mut quiet(),
    )
    .unwrap();
    assert_eq!(r.t, 0);
    assert_eq!(r.summary_bytes(), 4);
    let s = r.summary(true);
    assert_eq!(
        (s.critical_nodes, s.max_critical_width, s.width1_layers),
        (1000, 1000, Some(0))
    );
}
