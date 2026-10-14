//! Acceptance tests of the shared core.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use dag::bitset::DirtyBitset;
use dag::blocks::BlockIndex;
use dag::csr::Csr;
use dag::graph::{self, ChunkConf, ChunkedGraph};
use dag::io::{self, Dump, LoadMode, MemBudget, U32Array};
use dag::sweep::{SweepConf, par_chunks, sweep_backward, sweep_forward, sweep_forward_preds};
use dag::testkit::*;
use dag::topo;
use dag::{BlockViolation, DagError, NONE, Window};
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;

type Lists = Vec<(usize, Vec<u32>)>;

fn quiet() -> Option<ProgressLogger> {
    None
}

fn cquiet() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

/// Expected `(u, succ(u) ∩ [0, hi))` for `u ∈ w`.
fn expected<G: RandomAccessGraph>(g: &G, w: Window) -> Lists {
    (w.lo..w.hi)
        .map(|u| {
            (
                u,
                g.successors(u)
                    .into_iter()
                    .filter(|&v| v < w.hi)
                    .map(|v| v as u32)
                    .collect(),
            )
        })
        .collect()
}

fn sweep_lists<G: RandomAccessGraph + Sync>(
    cg: &ChunkedGraph<G>,
    w: Window,
    decoders: usize,
    backward: bool,
) -> Result<Lists> {
    let conf = SweepConf { decoders };
    let mut out: Lists = Vec::new();
    let mut pl = quiet();
    if backward {
        sweep_backward(cg, w, &conf, &mut pl, |c| {
            for (u, s) in c.iter_rev() {
                out.push((u, s.to_vec()));
            }
            Ok(())
        })?;
        out.reverse();
    } else {
        sweep_forward(cg, w, &conf, &mut pl, |c| {
            assert_eq!(c.num_arcs(), c.iter().map(|(_, s)| s.len()).sum::<usize>());
            for (u, s) in c.iter() {
                out.push((u, s.to_vec()));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

fn windows(n: usize, rng: &mut SmallRng) -> Vec<Window> {
    let mut ws = vec![
        Window::full(n),
        Window { lo: 0, hi: 0 },
        Window { lo: n, hi: n },
    ];
    for _ in 0..4 {
        let a = rng.random_range(0..=n);
        let b = rng.random_range(a..=n);
        ws.push(Window { lo: a, hi: b });
    }
    ws
}

fn check_sweeps<G: RandomAccessGraph + Sync>(cg: &ChunkedGraph<G>, rng: &mut SmallRng) {
    let n = cg.num_nodes();
    for w in windows(n, rng) {
        let want = expected(&cg.graph, w);
        for d in [1, 2, 3, 8] {
            assert_eq!(
                sweep_lists(cg, w, d, false).unwrap(),
                want,
                "forward, window {w}, decoders {d}"
            );
            assert_eq!(
                sweep_lists(cg, w, d, true).unwrap(),
                want,
                "backward, window {w}, decoders {d}"
            );
        }
    }
}

#[test]
fn sweeps_visit_every_list_in_order() {
    let mut rng = SmallRng::seed_from_u64(1);
    for seed in 0..12u64 {
        let n = 1 + (seed as usize * 7) % 60;
        let g = random_dag(n, 0.15, seed);
        for chunk in [1u64, 2, 3, 5, 7, 1 << 20] {
            check_sweeps(&chunked(&g, chunk), &mut rng);
        }
    }
    // chunk sequences alternating dense and empty lists (buffer recycling)
    let g = disjoint_union(&[
        complete_dag(9),
        isolated(20),
        complete_dag(6),
        isolated(3),
        out_star(12),
    ]);
    for chunk in 1..=7u64 {
        check_sweeps(&chunked(&g, chunk), &mut rng);
    }
}

#[test]
fn sweeps_on_bv_graphs() {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = SmallRng::seed_from_u64(2);
    for seed in 0..4u64 {
        let g = random_dag(80, 0.08, seed);
        let base = write_bv(dir.path(), &format!("g{seed}"), &g);
        for chunk in [1u64, 4, 7, 1 << 24] {
            let conf = ChunkConf {
                chunk_arcs: chunk,
                max_chunk_nodes: 5,
            };
            let cg = graph::open(&base, conf).unwrap();
            assert_eq!(cg.num_arcs, g.num_arcs());
            check_sweeps(&cg, &mut rng);
        }
        // with a DCF on disk
        let bv = graph::load_ra(&base).unwrap();
        graph::store_dcf(&bv, &base.with_extension("dcf")).unwrap();
        let cg = graph::open(
            &base,
            ChunkConf {
                chunk_arcs: 3,
                max_chunk_nodes: 4,
            },
        )
        .unwrap();
        check_cuts(&cg);
        check_sweeps(&cg, &mut rng);
    }
}

fn check_cuts<G: RandomAccessGraph + Sync>(cg: &ChunkedGraph<G>) {
    let n = cg.num_nodes();
    assert_eq!(cg.cut[0], 0);
    assert_eq!(*cg.cut.last().unwrap(), n);
    assert!(cg.cut.windows(2).all(|p| p[0] < p[1]) || n == 0);
    let maxdeg = (0..n)
        .map(|u| cg.graph.outdegree(u) as u64)
        .max()
        .unwrap_or(0);
    for p in cg.cut.windows(2) {
        assert!(p[1] - p[0] <= cg.conf.max_chunk_nodes);
        let arcs: u64 = (p[0]..p[1]).map(|u| cg.graph.outdegree(u) as u64).sum();
        assert!(
            arcs <= cg.conf.chunk_arcs + maxdeg,
            "chunk {p:?} has {arcs} arcs"
        );
    }
}

#[test]
fn chunk_cut_points() {
    for seed in 0..10 {
        let g = random_dag(100, 0.05, seed);
        for chunk in [1, 3, 10, 1000] {
            let cg = ChunkedGraph::new(
                &g,
                ChunkConf {
                    chunk_arcs: chunk,
                    max_chunk_nodes: 7,
                },
            )
            .unwrap();
            check_cuts(&cg);
            assert_eq!(cg.num_arcs, g.num_arcs());
            let w = Window { lo: 13, hi: 77 };
            let c = cg.cut_in(w);
            assert_eq!((c[0], *c.last().unwrap()), (13, 77));
            assert!(c.windows(2).all(|p| p[0] < p[1]));
        }
    }
    let e = ChunkedGraph::new(VecGraph::empty(0), ChunkConf::default()).unwrap();
    assert_eq!(&*e.cut, &[0]);
    assert!(e.chunks_in(Window::full(0)).is_empty());
}

#[test]
fn back_arc_is_not_topological_at_smallest_node() {
    for seed in 0..20u64 {
        let n = 40;
        let g = random_dag(n, 0.1, seed);
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut a = arcs(&g);
        let u1 = rng.random_range(5..n);
        let u2 = rng.random_range(5..n);
        let v1 = rng.random_range(0..=u1);
        let v2 = rng.random_range(0..=u2);
        a.push((u1, v1));
        a.push((u2, v2));
        let bad = vec_graph(n, a);
        let umin = u1.min(u2);
        let vmin = bad
            .successors(umin)
            .into_iter()
            .find(|&v| v <= umin)
            .unwrap();
        for chunk in [1, 3, 1 << 20] {
            let cg = chunked(&bad, chunk);
            for d in [1, 2, 5] {
                let err = sweep_lists(&cg, Window::full(n), d, false).unwrap_err();
                match err.downcast_ref::<DagError>() {
                    Some(&DagError::NotTopological { u, v }) => assert_eq!((u, v), (umin, vmin)),
                    other => panic!("unexpected {other:?}"),
                }
            }
            // a single back arc is also caught by the backward sweep
            let one = vec_graph(n, arcs(&g).into_iter().chain([(u1, v1)]));
            let err = sweep_lists(&chunked(&one, chunk), Window::full(n), 2, true).unwrap_err();
            let v = one.successors(u1).into_iter().find(|&v| v <= u1).unwrap();
            assert!(
                matches!(err.downcast_ref::<DagError>(), Some(&DagError::NotTopological { u, v: vv }) if u == u1 && vv == v)
            );
        }
    }
    let err = sweep_lists(
        &chunked(&with_self_loop(&chain(5)), 2),
        Window::full(5),
        1,
        false,
    )
    .unwrap_err();
    assert!(matches!(
        err.downcast_ref::<DagError>(),
        Some(DagError::NotTopological { u: 2, v: 2 })
    ));
}

/// `sweep_forward_preds` over the transpose yields, for every `v ∈ w` in ascending order, the
/// predecessors of `v` in `[w.lo, v)`, ascending (brute force from the forward lists).
#[test]
fn pred_sweep_matches_brute_force_predecessors() {
    let mut rng = SmallRng::seed_from_u64(11);
    for seed in 0..12u64 {
        let n = 1 + (seed as usize * 11) % 70;
        let g = random_dag(n, 0.12, seed);
        let gt = transpose(&g);
        let mut preds = vec![Vec::new(); n];
        for u in 0..n {
            for v in g.successors(u) {
                preds[v].push(u as u32);
            }
        }
        for chunk in [1u64, 2, 3, 7, 1 << 20] {
            let cgt = chunked(&gt, chunk);
            for w in windows(n, &mut rng) {
                let want: Lists = (w.lo..w.hi)
                    .map(|v| {
                        let p = preds[v].iter().copied().filter(|&u| u as usize >= w.lo);
                        (v, p.collect())
                    })
                    .collect();
                for d in [1, 2, 5] {
                    let mut got: Lists = Vec::new();
                    sweep_forward_preds(&cgt, w, &SweepConf { decoders: d }, &mut quiet(), |c| {
                        for (v, p) in c.iter() {
                            got.push((v, p.to_vec()));
                        }
                        Ok(())
                    })
                    .unwrap();
                    assert_eq!(
                        got, want,
                        "seed {seed}, chunk {chunk}, window {w}, decoders {d}"
                    );
                }
            }
        }
    }
    // a forward graph given as "transpose" has predecessors larger than the node: rejected at
    // the smallest offending node, as the forward arc (u, v) with u ≥ v
    let g = chain(4);
    let err = sweep_forward_preds(
        &chunked(&g, 1),
        Window::full(4),
        &SweepConf { decoders: 2 },
        &mut quiet(),
        |_| Ok(()),
    )
    .unwrap_err();
    assert!(matches!(
        err.downcast_ref::<DagError>(),
        Some(DagError::NotTopological { u: 1, v: 0 })
    ));
}

/// `read_array` (8 concurrent streams of 8 MiB positioned reads) and `write_array` (128 MiB
/// conversion batches) round-trip lengths around every split point: fewer elements than
/// streams, uneven parts, several pieces per stream, and elements wider than 8 bytes.
#[test]
fn array_io_round_trips_across_split_points() {
    let dir = tempfile::tempdir().unwrap();
    let piece = (1usize << 23) / 4; // u32 elements per 8 MiB piece
    for n in [
        0usize,
        1,
        7,
        8,
        9,
        63,
        1000,
        8 * piece - 1,
        8 * piece + 13,
        17 * piece + 5,
    ] {
        let v: Vec<u32> = (0..n as u32)
            .map(|x| x.wrapping_mul(2_654_435_761))
            .collect();
        let f = dir.path().join(format!("a{n}.u32le"));
        io::write_array(&f, &v).unwrap();
        assert_eq!(std::fs::metadata(&f).unwrap().len(), 4 * n as u64);
        assert_eq!(io::read_array::<u32>(&f).unwrap(), v, "n = {n}");
    }
    let w: Vec<(u64, u128)> = (0..100_003u64)
        .map(|x| (x, (x as u128) << 70 | 5))
        .collect();
    let f = dir.path().join("pairs.bin");
    io::write_array(&f, &w).unwrap();
    assert_eq!(io::read_array::<(u64, u128)>(&f).unwrap(), w);
    // rewriting an existing (longer) array replaces it
    let f = dir.path().join("again.u32le");
    io::write_array(&f, &[1u32, 2, 3, 4, 5]).unwrap();
    io::write_array(&f, &[9u32, 8]).unwrap();
    assert_eq!(io::read_array::<u32>(&f).unwrap(), [9, 8]);
    // a truncated file is rejected
    std::fs::write(dir.path().join("bad.u32le"), [1u8, 2, 3]).unwrap();
    assert!(io::read_array::<u32>(&dir.path().join("bad.u32le")).is_err());
}

#[test]
fn consumer_errors_propagate() {
    let g = random_dag(50, 0.2, 3);
    let cg = chunked(&g, 4);
    let mut seen = 0;
    let err = sweep_forward(
        &cg,
        Window::full(50),
        &SweepConf { decoders: 3 },
        &mut quiet(),
        |c| {
            seen += c.num_nodes();
            anyhow::ensure!(c.end <= 20, "stop at {}", c.start);
            Ok(())
        },
    )
    .unwrap_err();
    assert!(err.to_string().starts_with("stop at"));
    assert!(seen < 50);
}

#[test]
fn par_chunks_folds_and_reports_lowest_chunk_error() {
    for seed in 0..10 {
        let g = random_dag(120, 0.05, seed);
        let cg = chunked(&g, 3);
        let w = Window { lo: 7, hi: 101 };
        let (nodes, arcs_in) = par_chunks(
            &cg,
            w,
            &mut cquiet(),
            |it| {
                let mut a = (0usize, 0usize);
                while let Some((u, s)) = it.next_node() {
                    assert!(w.contains(u));
                    assert!(s.iter().all(|&v| (v as usize) < w.hi));
                    a.0 += 1;
                    a.1 += s.len();
                }
                Ok(a)
            },
            |a, b| (a.0 + b.0, a.1 + b.1),
        )
        .unwrap();
        assert_eq!(nodes, w.len());
        assert_eq!(
            arcs_in,
            expected(&g, w).iter().map(|(_, s)| s.len()).sum::<usize>()
        );
        // errors in several chunks: the lowest chunk wins, deterministically
        for _ in 0..5 {
            let err = par_chunks(
                &cg,
                Window::full(120),
                &mut cquiet(),
                |it| {
                    while let Some((u, _)) = it.next_node() {
                        if u % 17 == 16 {
                            anyhow::bail!("node {u}");
                        }
                    }
                    Ok(())
                },
                |_, _| (),
            )
            .unwrap_err();
            assert_eq!(err.to_string(), "node 16");
        }
    }
}

#[test]
fn check_identity_counts() {
    let dir = tempfile::tempdir().unwrap();
    for seed in 0..6u64 {
        let n = 70;
        let g = random_dag(n, 0.06, seed);
        let sinks = (0..n).filter(|&u| g.outdegree(u) == 0).count();
        let sources = (0..n).filter(|&v| transpose(&g).outdegree(v) == 0).count();
        let cg = chunked(&g, 5);
        let gt = transpose(&g);
        let ct = chunked(&gt, 3);
        let r = topo::check_identity(&cg, Some(&ct), &mut cquiet()).unwrap();
        assert!(r.identity_topological);
        assert_eq!(
            (r.n, r.m, r.sinks, r.sources),
            (n, g.num_arcs(), sinks, Some(sources))
        );
        let bad = with_back_arc(&g);
        let (u, v) = *arcs(&g).last().unwrap();
        let r = topo::check_identity(
            &chunked(&bad, 2),
            Some(&chunked(&transpose(&bad), 2)),
            &mut cquiet(),
        )
        .unwrap();
        assert!(!r.identity_topological);
        assert_eq!(r.first_violation, Some((v, u)));
        // BV round trip
        let base = write_bv(dir.path(), &format!("c{seed}"), &g);
        let cg = graph::open(
            &base,
            ChunkConf {
                chunk_arcs: 7,
                max_chunk_nodes: 9,
            },
        )
        .unwrap();
        let ct = graph::open(
            &graph::transpose_basename(&base).unwrap(),
            ChunkConf::default(),
        )
        .unwrap();
        let r2 = topo::check_identity(&cg, Some(&ct), &mut cquiet()).unwrap();
        assert_eq!(
            (r2.sinks, r2.sources, r2.m),
            (sinks, Some(sources), g.num_arcs())
        );
    }
}

/// Regression: a self-loop `(u, u)` violates `u < v` in both scans; the
/// comparisons must be strict-violation (`v <= u`, `u >= v`).
#[test]
fn check_identity_self_loop() {
    for seed in 0..4u64 {
        let n = 41;
        let g = random_dag(n, 0.08, seed);
        let sl = with_self_loop(&g);
        let want = Some((n / 2, n / 2));
        // forward scan only
        let r = topo::check_identity(&chunked(&sl, 3), None, &mut cquiet()).unwrap();
        assert!(!r.identity_topological);
        assert_eq!(r.first_violation, want);
        // forward and transpose
        let r = topo::check_identity(
            &chunked(&sl, 3),
            Some(&chunked(&transpose(&sl), 4)),
            &mut cquiet(),
        )
        .unwrap();
        assert!(!r.identity_topological);
        assert_eq!(r.first_violation, want);
        // the transpose scan alone must catch it: the forward graph is a DAG with the same
        // number of arcs (one extra forward arc instead of the loop)
        let extra = (0..n - 1)
            .find(|&u| !g.successors(u).into_iter().any(|v| v == n - 1))
            .expect("some node does not reach n - 1 directly");
        let mut a = arcs(&g);
        a.push((extra, n - 1));
        let fwd = vec_graph(n, a);
        assert_eq!(fwd.num_arcs(), sl.num_arcs());
        let r = topo::check_identity(
            &chunked(&fwd, 3),
            Some(&chunked(&transpose(&sl), 4)),
            &mut cquiet(),
        )
        .unwrap();
        assert!(!r.identity_topological);
        assert_eq!(r.first_violation, want);
    }
}

/// Regression: `map_back` rejects a non-permutation and handles empty arrays.
#[test]
fn map_back_checks_the_permutation() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    io::write_array(&d.join("v.u32le"), &[10u32, 20, 30]).unwrap();
    io::write_array(&d.join("o.u32le"), &[2u32, 0, 1]).unwrap();
    topo::map_back::<u32>(&d.join("v.u32le"), &d.join("o.u32le"), &d.join("r.u32le")).unwrap();
    assert_eq!(
        io::read_array::<u32>(&d.join("r.u32le")).unwrap(),
        vec![20, 30, 10]
    );
    for bad in [[2u32, 0, 0], [0, 1, 3]] {
        io::write_array(&d.join("b.u32le"), &bad).unwrap();
        let e = topo::map_back::<u32>(&d.join("v.u32le"), &d.join("b.u32le"), &d.join("x.u32le"))
            .unwrap_err();
        assert!(e.to_string().contains("not a permutation"), "{e}");
    }
    io::write_array::<u32>(&d.join("e.u32le"), &[]).unwrap();
    topo::map_back::<u32>(&d.join("e.u32le"), &d.join("e.u32le"), &d.join("re.u32le")).unwrap();
    assert!(
        io::read_array::<u32>(&d.join("re.u32le"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn topo_order_and_relabel() {
    for seed in 0..30u64 {
        let (g, rank0) = shuffled_dag(40, 0.1, seed);
        let ord = topo::topo_order(&g, &mut quiet()).unwrap();
        for u in 0..40 {
            assert_eq!(ord.order[ord.rank[u]], u);
        }
        let h = topo::relabel_vec(&g, &ord.rank);
        assert!(arcs(&h).iter().all(|&(u, v)| u < v));
        assert_eq!(h.num_arcs(), g.num_arcs());
        // the generator's rank is topological too
        let h0 = topo::relabel_vec(&g, &rank0);
        assert_eq!(arcs(&h0), arcs(&random_dag(40, 0.1, seed)));
        // cycles
        let c = with_back_arc(&random_dag(30, 0.2, seed));
        let err = topo::topo_order(&c, &mut quiet()).unwrap_err();
        match err.downcast_ref::<DagError>() {
            Some(&DagError::Cycle { u, v }) => {
                assert!(c.successors(u).into_iter().any(|x| x == v));
                assert!(bfs_reach(&c, v, u), "arc ({u}, {v}) is not on a cycle");
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    let err = topo::topo_order(&with_self_loop(&chain(4)), &mut quiet()).unwrap_err();
    assert!(matches!(
        err.downcast_ref::<DagError>(),
        Some(DagError::Cycle { u: 2, v: 2 })
    ));
}

#[test]
fn relabel_bv_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    for seed in 0..3u64 {
        let (g, _) = shuffled_dag(60, 0.08, seed);
        let base = write_bv(dir.path(), &format!("s{seed}"), &g);
        let ord = topo::topo_order(&g, &mut quiet()).unwrap();
        let out = dir.path().join(format!("r{seed}"));
        topo::relabel_bv(
            &base,
            &out,
            &ord.rank,
            &dir.path().join("tmp"),
            &mut quiet(),
        )
        .unwrap();
        let bv = graph::load_ra(&out).unwrap();
        let want = topo::relabel_vec(&g, &ord.rank);
        webgraph::traits::graph::eq(&want, &bv).unwrap();
        let bvt = graph::load_ra(&graph::transpose_basename(&out).unwrap()).unwrap();
        webgraph::traits::graph::eq(&transpose(&want), &bvt).unwrap();
        assert!(out.with_extension("dcf").exists());
        let order = io::read_array::<u32>(&io::suffixed(&out, "order.u32le")).unwrap();
        let rank = io::read_array::<u32>(&io::suffixed(&out, "rank.u32le")).unwrap();
        assert!(
            order
                .iter()
                .enumerate()
                .all(|(i, &o)| ord.order[i] == o as usize)
        );
        assert!(
            rank.iter()
                .enumerate()
                .all(|(i, &r)| ord.rank[i] == r as usize)
        );
        // map_back: per-node values in new ids back to old ids
        let vals: Vec<u64> = (0..60u64).map(|i| i * 10).collect();
        io::write_array(&dir.path().join("vals.u64le"), &vals).unwrap();
        topo::map_back::<u64>(
            &dir.path().join("vals.u64le"),
            &io::suffixed(&out, "order.u32le"),
            &dir.path().join("old.u64le"),
        )
        .unwrap();
        let old = io::read_array::<u64>(&dir.path().join("old.u64le")).unwrap();
        for u in 0..60 {
            assert_eq!(old[u], vals[ord.rank[u]]);
        }
        // the post-check fires when `order` is passed instead of `rank`
        if ord.order != ord.rank {
            let err = topo::relabel_bv(
                &base,
                &dir.path().join(format!("w{seed}")),
                &ord.order,
                &dir.path().join("tmp"),
                &mut quiet(),
            );
            let err = err.expect_err("inverted permutation must be rejected");
            assert!(
                matches!(
                    err.downcast_ref::<DagError>(),
                    Some(DagError::NotTopological { .. })
                ),
                "{err:#}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// BlockIndex
// ---------------------------------------------------------------------------------------------

fn check_against_ground_truth(mp: &MiniPg, idx: &BlockIndex, rep: &dag::blocks::BlockReport) {
    let g = &mp.graph;
    let n = g.num_nodes();
    let truth = mp.blocks();
    assert_eq!(idx.nb, truth.len());
    for (b, &(f, l, i)) in truth.iter().enumerate() {
        assert_eq!(
            (idx.first[b], idx.len[b], idx.nin[b]),
            (f, l, i),
            "block {b}"
        );
    }
    // per node
    let indeg: Vec<usize> = (0..n).map(|v| transpose(g).outdegree(v)).collect();
    for (v, &d) in indeg.iter().enumerate() {
        assert_eq!(idx.indegree(v), d);
        assert_eq!(idx.outdegree(v), g.outdegree(v));
        assert_eq!(
            idx.block_of(v).is_none(),
            d == 0,
            "sources have create = NONE"
        );
        let s: Vec<usize> = g.successors(v).collect();
        match idx.spender(v) {
            None => assert!(s.is_empty()),
            Some(b) => {
                assert_eq!(idx.outputs(b).collect::<Vec<_>>(), s);
                assert_eq!(idx.first_succ(v), Some(s[0]));
            }
        }
        if let Some(b) = idx.block_of(v) {
            assert!(idx.outputs(b).contains(&v));
        }
    }
    let sinks = (0..n).filter(|&u| g.outdegree(u) == 0).count();
    let sources = indeg.iter().filter(|&&d| d == 0).count();
    assert_eq!(
        (rep.n, rep.m, rep.nb, rep.sinks, rep.sources),
        (n, g.num_arcs(), truth.len(), sinks, sources)
    );
    assert_eq!(rep.sum_len, (n - sources) as u64);
    assert_eq!(rep.sum_nin, (n - sinks) as u64);
    assert_eq!(rep.sum_len_times_nin, g.num_arcs());
    for hi in [0, n / 3, n] {
        let bhi = idx.block_bound(hi);
        assert!((0..idx.nb).all(|b| ((idx.first[b] as usize) < hi) == (b < bhi)));
    }
}

#[test]
fn block_index_matches_mini_pg() {
    let dir = tempfile::tempdir().unwrap();
    for seed in 0..40u64 {
        let opts = MiniPgOpts {
            zero_output_prob: if seed % 2 == 0 { 0.0 } else { 0.1 },
            max_outputs: 1 + seed as usize % 6,
            ..Default::default()
        };
        let mp = mini_pg(60 + seed as usize * 5, seed, &opts);
        let (idx, rep) = BlockIndex::from_vec_graph(&mp.graph).unwrap();
        check_against_ground_truth(&mp, &idx, &rep);
        for chunk in [1, 2, 7] {
            let (idx2, rep2) =
                BlockIndex::build(&chunked(&mp.graph, chunk), &mut cquiet()).unwrap();
            assert_eq!(rep2, rep);
            assert_eq!(&*idx2.create, &*idx.create);
            assert_eq!(&*idx2.spend, &*idx.spend);
        }
        if seed < 4 {
            let base = write_bv(dir.path(), &format!("m{seed}"), &mp.graph);
            let cg = graph::open(
                &base,
                ChunkConf {
                    chunk_arcs: 5,
                    max_chunk_nodes: 8,
                },
            )
            .unwrap();
            let (idx3, rep3) = BlockIndex::build(&cg, &mut cquiet()).unwrap();
            assert_eq!(rep3, rep);
            assert_eq!(&*idx3.first, &*idx.first);
            assert_eq!(&*idx3.create, &*idx.create);
        }
    }
}

#[test]
fn block_index_identical_across_thread_counts() {
    let mp = mini_pg(3000, 7, &MiniPgOpts::default());
    let build = |t: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        pool.install(|| {
            let (idx, rep) = BlockIndex::build(&chunked(&mp.graph, 16), &mut cquiet()).unwrap();
            (
                rep,
                idx.create.to_vec(),
                idx.spend.to_vec(),
                idx.first.to_vec(),
                idx.len.to_vec(),
                idx.nin.to_vec(),
            )
        })
    };
    let r1 = build(1);
    assert_eq!(build(4), r1);
    assert_eq!(build(16), r1);
    // violations are deterministic too
    let bad = mini_pg(
        3000,
        7,
        &MiniPgOpts {
            double_spend: true,
            non_contiguous: true,
            ..Default::default()
        },
    );
    let err = |t: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        pool.install(|| {
            BlockIndex::build(&chunked(&bad.graph, 16), &mut cquiet())
                .unwrap_err()
                .to_string()
        })
    };
    let e1 = err(1);
    assert_eq!(err(4), e1);
    assert_eq!(err(16), e1);
}

/// Regression: `build_budgeted` reserves `8n` before pass 1 and `12·nb` after
/// it, with `nb` up to `n − 1` (a chain), and releases everything on failure.
#[test]
fn block_index_budget() {
    let n = 1000;
    let g = chain(n);
    let need = BlockIndex::bytes(n, n - 1);
    // exactly enough
    let mut b = MemBudget::new(need as f64 / 1e9);
    let (idx, _) = BlockIndex::build_budgeted(&chunked(&g, 64), &mut cquiet(), &mut b).unwrap();
    assert_eq!(idx.nb, n - 1);
    assert_eq!(b.used(), need);
    // the old n/2 estimate is not enough: fails after pass 1 and releases the 8n
    let mut b = MemBudget::new((8 * n as u64 + 12 * (n as u64 / 2 + 1)) as f64 / 1e9);
    let e = BlockIndex::build_budgeted(&chunked(&g, 64), &mut cquiet(), &mut b).unwrap_err();
    assert!(matches!(
        e.downcast_ref::<DagError>(),
        Some(DagError::OverBudget { .. })
    ));
    assert_eq!(b.used(), 0);
    // a NotBiclique failure releases too
    let bad = with_back_arc(&g);
    let mut b = MemBudget::unlimited();
    b.need(5, "other").unwrap();
    BlockIndex::build_budgeted(&chunked(&bad, 64), &mut cquiet(), &mut b).unwrap_err();
    assert_eq!(b.used(), 5);
}

/// The reported violation is the smallest pass-1 violation even when a pass-2 `Overlap` lies
/// at a smaller node (documented contract of `BlockIndex::build`).
#[test]
fn block_index_pass1_violation_wins() {
    let g = vec_graph(8, [(0, 2), (0, 3), (1, 3), (4, 5), (4, 7)]);
    assert_eq!(violation(&g), (BlockViolation::NonContiguous, 4));
    let g = vec_graph(8, [(0, 2), (0, 3), (1, 3)]);
    assert_eq!(violation(&g), (BlockViolation::Overlap, 3));
}

fn violation(g: &VecGraph) -> (BlockViolation, usize) {
    let err = BlockIndex::from_vec_graph(g)
        .map(|_| ())
        .expect_err("must not be a biclique digraph");
    match err.downcast_ref::<DagError>() {
        Some(&DagError::NotBiclique { kind, node }) => (kind, node),
        other => panic!("unexpected {other:?}: {err}"),
    }
}

#[test]
fn block_index_violations() {
    use BlockViolation::*;
    assert_eq!(violation(&vec_graph(3, [(0, 1), (2, 1)])), (BackwardArc, 2));
    assert_eq!(
        violation(&vec_graph(5, [(0, 2), (0, 4)])),
        (NonContiguous, 0)
    );
    assert_eq!(
        violation(&vec_graph(4, [(0, 2), (0, 3), (1, 2)])),
        (SameStartDifferentLength, 2)
    );
    assert_eq!(
        violation(&vec_graph(4, [(0, 2), (0, 3), (1, 3)])),
        (Overlap, 3)
    );
    // the smallest node wins
    assert_eq!(
        violation(&vec_graph(6, [(0, 3), (0, 5), (4, 1)])),
        (NonContiguous, 0)
    );
    assert_eq!(
        violation(&vec_graph(6, [(1, 3), (1, 5), (0, 0)])),
        (BackwardArc, 0)
    );
    // injected anomalies in simulated graphs
    for seed in 0..30u64 {
        for (ds, nc) in [(true, false), (false, true), (true, true)] {
            let mp = mini_pg(
                80,
                seed,
                &MiniPgOpts {
                    double_spend: ds,
                    non_contiguous: nc,
                    ..Default::default()
                },
            );
            let _ = violation(&mp.graph);
        }
    }
    // cycles and self loops are not biclique digraphs either
    let _ = violation(&with_back_arc(
        &mini_pg(50, 1, &MiniPgOpts::default()).graph,
    ));
}

#[test]
fn block_index_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let mp = mini_pg(500, 11, &MiniPgOpts::default());
    let (idx, rep) = BlockIndex::from_vec_graph(&mp.graph).unwrap();
    let prefix = dir.path().join("p");
    idx.save(&prefix).unwrap();
    for mode in [LoadMode::Ram, LoadMode::Mmap] {
        let l = BlockIndex::load(&prefix, mode).unwrap();
        assert_eq!((l.n, l.m, l.nb), (idx.n, idx.m, idx.nb));
        assert_eq!(&*l.create, &*idx.create);
        assert_eq!(&*l.spend, &*idx.spend);
        assert_eq!(&*l.first, &*idx.first);
        assert_eq!(&*l.len, &*idx.len);
        assert_eq!(&*l.nin, &*idx.nin);
        assert_eq!(l.block_report().unwrap(), rep);
    }
    // empty graph
    let (e, r) = BlockIndex::from_vec_graph(&VecGraph::empty(0)).unwrap();
    assert_eq!((e.nb, r.sources, r.sinks), (0, 0, 0));
    e.save(&dir.path().join("e")).unwrap();
    let l = BlockIndex::load(&dir.path().join("e"), LoadMode::Mmap).unwrap();
    assert_eq!(l.n, 0);
}

// ---------------------------------------------------------------------------------------------
// Csr, Window, io, DirtyBitset
// ---------------------------------------------------------------------------------------------

#[test]
fn csr_from_window_is_the_induced_subgraph() {
    let mut rng = SmallRng::seed_from_u64(5);
    for seed in 0..10u64 {
        let g = random_dag(90, 0.07, seed);
        for chunk in [1, 4, 1 << 20] {
            let cg = chunked(&g, chunk);
            for w in windows(90, &mut rng) {
                let c = Csr::from_window(&cg, w, &mut cquiet()).unwrap();
                let want = Csr::from_arcs(
                    w.len(),
                    arcs(&g)
                        .into_iter()
                        .filter(|&(u, v)| w.contains(u) && w.contains(v))
                        .map(|(u, v)| (u - w.lo, v - w.lo)),
                );
                assert_eq!(c, want, "window {w}");
                c.check_topological().unwrap();
            }
        }
    }
    let c = Csr::from_arcs(4, [(2, 3), (0, 1), (0, 1), (0, 3)]);
    assert_eq!(
        (c.num_arcs(), c.succ(0), c.outdegree(1)),
        (3, &[1u32, 3][..], 0)
    );
    let t = c.transpose();
    assert_eq!(t.succ(3), &[0, 2]);
    assert_eq!(Csr::from_vec_graph(&c.to_vec_graph()), c);
    let bad = Csr::from_arcs(3, [(0, 1), (2, 1)]);
    let e = bad.check_topological().unwrap_err();
    assert!(matches!(
        e.downcast_ref::<DagError>(),
        Some(DagError::NotTopological { u: 2, v: 1 })
    ));
}

#[test]
fn window_parse() {
    let n = 100;
    assert_eq!(Window::parse("", n).unwrap(), Window::full(n));
    assert_eq!(Window::parse("..", n).unwrap(), Window::full(n));
    assert_eq!(
        Window::parse("10..20", n).unwrap(),
        Window { lo: 10, hi: 20 }
    );
    assert_eq!(
        Window::parse("10..", n).unwrap(),
        Window { lo: 10, hi: 100 }
    );
    assert_eq!(Window::parse("..20", n).unwrap(), Window { lo: 0, hi: 20 });
    assert_eq!(
        Window::parse(" 1_0..2_0 ", n).unwrap(),
        Window { lo: 10, hi: 20 }
    );
    assert_eq!(Window::parse("5..5", n).unwrap().len(), 0);
    assert!(Window::parse("100..100", n).unwrap().is_empty());
    for bad in ["20..10", "0..101", "101..", "x..3", "3", "1..2..3"] {
        let e = Window::parse(bad, n);
        assert!(e.is_err(), "{bad:?} accepted");
    }
    let e = Window::parse("20..10", n).unwrap_err();
    assert!(matches!(
        e.downcast_ref::<DagError>(),
        Some(DagError::BadWindow {
            lo: 20,
            hi: 10,
            n: 100
        })
    ));
    let w = Window { lo: 3, hi: 7 };
    assert!(w.contains(3) && !w.contains(7) && !w.is_full(7) && Window::full(7).is_full(7));
    assert_eq!(w.to_string(), "3..7");
}

fn round_trip<T: Dump + PartialEq + std::fmt::Debug>(dir: &Path, data: &[T]) {
    let path = io::array_path::<T>(&dir.join("rt"), "x");
    io::write_array(&path, data).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len() as usize,
        data.len() * T::BYTES
    );
    assert_eq!(io::read_array::<T>(&path).unwrap(), data);
    let mut w = io::array_writer::<T>(&path).unwrap();
    for &x in data {
        w.push(x).unwrap();
    }
    assert_eq!(w.finish().unwrap(), data.len());
    assert_eq!(io::read_array::<T>(&path).unwrap(), data);
    let mut buf = vec![0u8; T::BYTES];
    for &x in data {
        x.put(&mut buf);
        assert_eq!(T::get(&buf), x);
    }
}

#[test]
fn dump_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    round_trip(d, &[0u8, 1, 255]);
    round_trip(d, &[0u32, 1, NONE, 123_456_789]);
    round_trip(d, &[0u64, u64::MAX, 1 << 40]);
    round_trip(d, &[i64::MIN, -1, 0, i64::MAX]);
    round_trip(d, &[u128::MAX, 0, 1 << 100]);
    round_trip(d, &[0.5f32, -1.25, f32::INFINITY]);
    round_trip(d, &[0.1f64, -0.0, f64::NEG_INFINITY, 1e300]);
    round_trip(d, &[true, false, true]);
    round_trip(d, &[(1u32, 2u64), (NONE, u64::MAX)]);
    round_trip::<u32>(d, &[]);
    assert_eq!(<(u32, f64)>::BYTES, 12);
    assert_eq!(<u32 as Dump>::DTYPE, "u32le");
    assert_eq!(
        io::suffixed(Path::new("a/run.v1"), "E.u32le"),
        Path::new("a/run.v1.E.u32le")
    );
    // larger than one conversion batch
    let big: Vec<u32> = (0..5_000_000u32)
        .map(|i| i.wrapping_mul(2654435761))
        .collect();
    round_trip(d, &big);
    let path = d.join("big.u32le");
    io::write_array(&path, &big).unwrap();
    for mode in [LoadMode::Ram, LoadMode::Mmap] {
        let a = U32Array::load(&path, mode).unwrap();
        assert_eq!(&*a, &big[..]);
    }
}

#[test]
fn dirty_bitset_matches_hashset() {
    let mut rng = SmallRng::seed_from_u64(9);
    let mut bs = DirtyBitset::new();
    for _ in 0..300 {
        let base = rng.random_range(0..1000);
        let len = rng.random_range(0..3000);
        bs.reset(base, len);
        let mut model = HashSet::new();
        for _ in 0..rng.random_range(0..200) {
            if len == 0 {
                break;
            }
            let x = base + rng.random_range(0..len);
            assert_eq!(bs.insert(x), model.insert(x));
        }
        for _ in 0..50 {
            let x = rng.random_range(0..base + len + 70);
            assert_eq!(bs.contains(x), model.contains(&x));
        }
        let from = rng.random_range(0..base + len + 70);
        let mut want: Vec<usize> = model.iter().copied().filter(|&x| x >= from).collect();
        want.sort_unstable();
        assert_eq!(bs.iter_from(from).collect::<Vec<_>>(), want);
        assert!(bs.bytes() >= len / 8);
    }
}

#[test]
fn huge_vec_atomics_budget() {
    let v = io::huge_vec(3_000_000, 7u32);
    assert_eq!(v.len(), 3_000_000);
    assert!(v.iter().all(|&x| x == 7));
    let mut w = io::huge_vec(10, 0u32);
    {
        let a = io::as_atomic_u32(&mut w);
        a[3].fetch_add(5, std::sync::atomic::Ordering::Relaxed);
    }
    assert_eq!(w[3], 5);
    let mut x = vec![1u64; 4];
    io::as_atomic_u64(&mut x)[0].fetch_max(9, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(x[0], 9);
    let mut b = MemBudget::new(1.0);
    b.need(600_000_000, "a").unwrap();
    let e = b.need(600_000_000, "b").unwrap_err();
    assert!(matches!(
        e.downcast_ref::<DagError>(),
        Some(DagError::OverBudget { .. })
    ));
    assert_eq!(b.used(), 600_000_000);
    b.release(600_000_000);
    b.need(1_000_000_000, "c").unwrap();
    assert_eq!(b.available(), 0);
}

#[test]
fn run_info_json() {
    let dir = tempfile::tempdir().unwrap();
    let mut info = io::RunInfo::new(None, 10, 20, Window { lo: 1, hi: 5 });
    info.files.push(io::FileInfo::new::<u32>(
        "x.u32le",
        4,
        Some("u32::MAX = NONE"),
        "test",
    ));
    info.results = serde_json::json!({ "ok": true });
    let prefix = dir.path().join("run");
    info.write(&prefix).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("run.json")).unwrap())
            .unwrap();
    assert_eq!(v["tool"], "dag");
    assert_eq!(v["window"]["lo"], 1);
    assert_eq!(v["files"][0]["dtype"], "u32le");
    assert!(v["peak_rss_bytes"].as_u64().unwrap() > 0);
}

// ---------------------------------------------------------------------------------------------
// testkit self-checks
// ---------------------------------------------------------------------------------------------

#[test]
fn testkit_oracles() {
    // dominators: diamonds have idom = the previous join node
    let d = brute_dominators(&diamonds(3));
    assert_eq!(d[0], None);
    assert_eq!((d[1], d[2], d[3]), (Some(0), Some(0), Some(0)));
    assert_eq!((d[4], d[5], d[6]), (Some(3), Some(3), Some(3)));
    let l = brute_dominators(&ladder(5));
    assert_eq!(l[5], Some(0)); // x_1: preds {c0, c1}
    assert_eq!(l[8], Some(0)); // x_4: preds {c0, c4}
    assert_eq!(l[4], Some(3));
    assert_eq!(brute_dominators(&in_star(4)), vec![None, None, None, None]);
    // width and matching
    assert_eq!(brute_width(&complete_dag(6)), 1);
    assert_eq!(brute_width(&isolated(7)), 7);
    assert_eq!(brute_width(&out_star(6)), 5);
    assert_eq!(brute_width(&diamonds(2)), 2);
    assert_eq!(brute_matching(&chain(9)), 8);
    assert_eq!(brute_matching(&out_star(9)), 1);
    assert_eq!(brute_matching(&complete_dag(6)), 5);
    // paths
    let mut count = 0;
    all_paths(&complete_dag(5), |p| {
        assert!(p.windows(2).all(|w| w[0] < w[1]));
        count += 1;
    });
    assert_eq!(count, 31); // non-empty subsets of 5 nodes
    let c = closure(&chain(70));
    assert!(bit(&c[0], 69) && !bit(&c[69], 0) && !bit(&c[3], 3));
    let bf = bellman_ford(
        &diamonds(1),
        |u, v| if (u, v) == (0, 2) { -5 } else { 1 },
        &[0],
    );
    assert_eq!(bf, vec![Some(0), Some(1), Some(-5), Some(-4)]);
    assert!(!bfs_reach(&chain(3), 2, 0) && bfs_reach(&chain(3), 0, 2));
    // generators
    assert_eq!(ladder(4).num_nodes(), 7);
    assert_eq!(layered(&[2, 3, 1], 1.0, 0).num_arcs(), 9);
    let u = disjoint_union(&[chain(3), chain(2)]);
    assert_eq!(arcs(&u), vec![(0, 1), (1, 2), (3, 4)]);
    let mp = mini_pg(
        200,
        3,
        &MiniPgOpts {
            zero_output_prob: 0.2,
            ..Default::default()
        },
    );
    assert_eq!(mp.tx_inputs.len(), 200);
    let n = mp.graph.num_nodes();
    assert_eq!(n as u32, mp.tx_outputs.last().unwrap().end);
    // every output is spent at most once
    let mut spent = vec![false; n];
    for i in mp.tx_inputs.iter().flatten() {
        assert!(!std::mem::replace(&mut spent[*i as usize], true));
    }
}

// ---------------------------------------------------------------------------------------------
// Core semiring instances (§3.10)
// ---------------------------------------------------------------------------------------------

fn axioms<S: dag::semiring::Semiring>(
    s: &S,
    sample: &mut dyn FnMut() -> S::Elem,
    bits: impl Fn(S::Elem) -> u128,
) {
    let (z, o) = (s.zero(), s.one());
    assert_eq!(bits(s.plus(z, z)), bits(z), "{}: 0 + 0", s.name());
    for _ in 0..10_000 {
        let (a, b, c) = (sample(), sample(), sample());
        for x in [a, z, o] {
            assert_eq!(bits(s.plus(z, x)), bits(x), "{}: 0 + {x:?}", s.name());
            assert_eq!(bits(s.plus(x, z)), bits(x), "{}: {x:?} + 0", s.name());
            assert_eq!(bits(s.times(o, x)), bits(x), "{}: 1 * {x:?}", s.name());
            assert_eq!(bits(s.times(x, o)), bits(x), "{}: {x:?} * 1", s.name());
            assert_eq!(bits(s.times(z, x)), bits(z), "{}: 0 * {x:?}", s.name());
            assert_eq!(bits(s.times(x, z)), bits(z), "{}: {x:?} * 0", s.name());
        }
        assert!(s.plus(a, b) == s.plus(b, a), "{}: commutativity", s.name());
        assert!(
            s.plus(s.plus(a, b), c) == s.plus(a, s.plus(b, c)),
            "{}: + associativity",
            s.name()
        );
        assert!(
            s.times(s.times(a, b), c) == s.times(a, s.times(b, c)),
            "{}: * associativity {a:?} {b:?} {c:?}",
            s.name()
        );
        assert!(
            s.times(a, s.plus(b, c)) == s.plus(s.times(a, b), s.times(a, c)),
            "{}: left distributivity",
            s.name()
        );
        assert!(
            s.times(s.plus(a, b), c) == s.plus(s.times(a, c), s.times(b, c)),
            "{}: right distributivity",
            s.name()
        );
        if S::SELECTIVE {
            let p = s.plus(a, b);
            assert!(p == a || p == b);
            assert!(!(s.better(a, b) && s.better(b, a)));
        }
    }
}

#[test]
fn core_semiring_axioms() {
    use dag::semiring::{Boolean, MaxPlus, MinPlus, Semiring};
    let mut r = SmallRng::seed_from_u64(42);
    let u32s = |r: &mut SmallRng| match r.random_range(0..10) {
        0 => u32::MAX,
        1 => u32::MAX - 1,
        2 => u32::MAX - r.random_range(2..1000),
        _ => r.random_range(0..1_000_000),
    };
    axioms(&MaxPlus::<u32>::new(), &mut || u32s(&mut r), |x| x as u128);
    let mut r7 = SmallRng::seed_from_u64(7);
    axioms(&MinPlus::<u32>::new(), &mut || u32s(&mut r7), |x| x as u128);
    let mut r = SmallRng::seed_from_u64(43);
    axioms(
        &MaxPlus::<u64>::new(),
        &mut || {
            if r.random_bool(0.1) {
                u64::MAX
            } else {
                r.random_range(0..1u64 << 62)
            }
        },
        |x| x as u128,
    );
    axioms(
        &MinPlus::<u64>::new(),
        &mut || {
            if r.random_bool(0.1) {
                u64::MAX
            } else {
                r.random_range(0..1u64 << 62)
            }
        },
        |x| x as u128,
    );
    let lim = 1i64 << 60;
    axioms(
        &MaxPlus::<i64>::new(),
        &mut || {
            if r.random_bool(0.1) {
                i64::MIN
            } else {
                r.random_range(-lim..=lim)
            }
        },
        |x| x as u128,
    );
    axioms(
        &MinPlus::<i64>::new(),
        &mut || {
            if r.random_bool(0.1) {
                i64::MAX
            } else {
                r.random_range(-lim..=lim)
            }
        },
        |x| x as u128,
    );
    // floats: integers in a range where + is exact, plus signed zeros
    let f = |r: &mut SmallRng, z: f64| match r.random_range(0..10) {
        0 => z,
        1 => -0.0,
        _ => r.random_range(-1_000_000i64..1_000_000) as f64,
    };
    let mut r1 = SmallRng::seed_from_u64(44);
    axioms(
        &MaxPlus::<f64>::new(),
        &mut || f(&mut r1, f64::NEG_INFINITY),
        |x| x.to_bits() as u128,
    );
    let mut r2 = SmallRng::seed_from_u64(45);
    axioms(
        &MinPlus::<f64>::new(),
        &mut || f(&mut r2, f64::INFINITY),
        |x| x.to_bits() as u128,
    );
    let mut r3 = SmallRng::seed_from_u64(46);
    axioms(&Boolean::new(), &mut || r3.random_bool(0.5), |x| x as u128);
    // unsigned MaxPlus: MAX is −∞ and sums saturate at MAX − 1
    let s = MaxPlus::<u32>::new();
    assert_eq!(s.times(u32::MAX - 1, 5), u32::MAX - 1);
    assert!(s.better(3, u32::MAX) && !s.better(u32::MAX, 3));
}

/// Throughput of the sweep engine on a window of the real PG (ignored; needs `PG_BASENAME`,
/// optional `PG_WINDOW`, default `0..100000000`).
#[test]
#[ignore]
fn sweep_pg_window() {
    let Ok(base) = std::env::var("PG_BASENAME") else {
        return;
    };
    let cg = graph::open(Path::new(&base), ChunkConf::default()).unwrap();
    let w = Window::parse(
        &std::env::var("PG_WINDOW").unwrap_or("0..100000000".into()),
        cg.num_nodes(),
    )
    .unwrap();
    let conf = SweepConf::default();
    for backward in [false, true] {
        let t = std::time::Instant::now();
        let (mut nodes, mut arcs) = (0usize, 0usize);
        let mut f = |c: &dag::sweep::Chunk| {
            nodes += c.num_nodes();
            arcs += c.num_arcs();
            Ok(())
        };
        if backward {
            sweep_backward(&cg, w, &conf, &mut quiet(), &mut f).unwrap();
        } else {
            sweep_forward(&cg, w, &conf, &mut quiet(), &mut f).unwrap();
        }
        eprintln!(
            "{} sweep of {w}: {nodes} nodes, {arcs} arcs in {:?} ({} decoders)",
            if backward { "backward" } else { "forward" },
            t.elapsed(),
            conf.decoders
        );
        assert_eq!(nodes, w.len());
    }
}
