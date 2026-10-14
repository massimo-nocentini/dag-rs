//! Tests of the aggregate measures (`dag::summary`) and of the per-family distributions built
//! on them, against brute-force oracles: distributions and time buckets, the frontier curve,
//! the dominator-tree and block distributions, `aggregate` on raw bytes, and the run metadata
//! (memory-budget log, RSS counters, phases).
//!
//! These are the regression tests of the aggregate measures: per-half-year breakdowns, histograms
//! and top-k lists for every family, the frontier time series and the recorded memory
//! estimates.

use dag::blocks::BlockIndex;
use dag::cli::aggregate::{Absent, Dtype, aggregate_bytes};
use dag::dom::{DomConf, dominators_push, post_dominators_pull, tree_distributions, tree_stats};
use dag::io::{Dump, MemBudget, RunInfo, budget_log};
use dag::summary::{
    Bucket, Buckets, CHUNK, Dist, DistConf, DistReport, LOG2_BINS, distribution, distribution_with,
};
use dag::testkit::{self, MiniPgOpts};
use dag::width::{
    CutRange, frontier_at_cuts_blocks, frontier_at_cuts_generic, frontier_blocks,
    frontier_curve_blocks, frontier_curve_generic,
};
use dag::{NONE, Window};
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;

fn no_log() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

fn quiet() -> Option<ProgressLogger> {
    None
}

// ---------------------------------------------------------------------------------------------
// Brute-force distribution
// ---------------------------------------------------------------------------------------------

/// The bin of a value by the definition (0 → 0; `[2^(k−1), 2^k)` → k; clamped), computed with
/// repeated halving rather than the exponent bits.
fn bin_by_definition(x: f64) -> usize {
    if x == 0.0 {
        return 0;
    }
    let mut k = 1;
    let mut hi = 2.0;
    while x >= hi && k < LOG2_BINS - 1 {
        hi *= 2.0;
        k += 1;
    }
    k
}

/// The distribution of `vals` restricted to `keep`, entry `i` having id `id(i)` and node
/// `node(i)`, by sorting.
fn brute(
    vals: &[Option<f64>],
    keep: impl Fn(usize) -> bool,
    id: impl Fn(usize) -> usize,
    node: Option<&dyn Fn(usize) -> usize>,
    k: usize,
) -> Dist {
    let idx: Vec<usize> = (0..vals.len()).filter(|&i| keep(i)).collect();
    let present: Vec<(f64, usize)> = idx
        .iter()
        .filter_map(|&i| vals[i].filter(|x| !x.is_nan()).map(|x| (x, i)))
        .collect();
    let count = present.len() as u64;
    let absent = idx.len() as u64 - count;
    let sum: f64 = present.iter().map(|p| p.0).sum();
    let mut bins = vec![0u64; LOG2_BINS];
    let (mut negative, mut below_one) = (0, 0);
    for &(x, _) in &present {
        if x < 0.0 {
            negative += 1;
        } else if x > 0.0 && x < 1.0 {
            below_one += 1;
        } else {
            bins[bin_by_definition(x)] += 1;
        }
    }
    while bins.last() == Some(&0) {
        bins.pop();
    }
    let mut sorted = present.clone();
    sorted.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let max = sorted.first().copied();
    let min = present
        .iter()
        .copied()
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    Dist {
        count,
        absent,
        sum,
        mean: (count > 0).then(|| sum / count as f64),
        min: min.map(|m| m.0),
        argmin: min.map(|m| id(m.1)),
        max: max.map(|m| m.0),
        argmax: max.map(|m| id(m.1)),
        argmax_node: node.and_then(|f| max.map(|m| f(m.1))),
        log2_hist: dag::summary::Log2Hist {
            negative,
            below_one,
            bins,
        },
        top: sorted
            .iter()
            .take(k)
            .map(|&(value, i)| dag::summary::Top {
                id: id(i),
                value,
                node: node.map(|f| f(i)),
            })
            .collect(),
    }
}

/// Random values: integers (exact sums), with absent entries, zeros, NaNs, fractions and
/// negative values mixed in, and many ties.
fn random_values(len: usize, rng: &mut SmallRng) -> Vec<Option<f64>> {
    (0..len)
        .map(|_| match rng.random_range(0..20u32) {
            0 => None,
            1 => Some(0.0),
            2 => Some(f64::NAN),
            3 => Some(0.25),
            4 => Some(-(rng.random_range(1..5u32) as f64)),
            5 => Some(1e30),
            _ => Some(rng.random_range(1..3000u32) as f64),
        })
        .collect()
}

/// Random sorted buckets over `[0, span)` with gaps, empty buckets and a bucket past the end.
fn random_buckets(span: usize, rng: &mut SmallRng) -> Buckets {
    let mut cuts: Vec<usize> = (0..rng.random_range(1..8usize))
        .map(|_| rng.random_range(0..=span))
        .collect();
    cuts.push(0);
    cuts.push(span);
    cuts.sort_unstable();
    let mut b = Vec::new();
    for (j, w) in cuts.windows(2).enumerate() {
        // every third range is a gap
        if j % 3 != 2 {
            b.push(Bucket {
                label: format!("b{j}"),
                lo: w[0],
                hi: w[1],
            });
        }
    }
    b.push(Bucket {
        label: "beyond".into(),
        lo: span + 10,
        hi: span + 20,
    });
    Buckets::new(b).unwrap()
}

/// `got == want`, with the floating-point sums compared to a relative 10⁻⁹ (the brute force
/// adds in another order) and everything else exactly.
fn assert_dist_eq(got: &Dist, want: &Dist, what: &str) {
    let close = |a: f64, b: f64| a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs());
    assert!(
        close(got.sum, want.sum),
        "{what}: sum {} vs {}",
        got.sum,
        want.sum
    );
    match (got.mean, want.mean) {
        (Some(a), Some(b)) => assert!(close(a, b), "{what}: mean {a} vs {b}"),
        (a, b) => assert_eq!(a, b, "{what}: mean"),
    }
    let mut w = want.clone();
    (w.sum, w.mean) = (got.sum, got.mean);
    assert_eq!(got, &w, "{what}");
}

fn check_report(
    got: &DistReport,
    vals: &[Option<f64>],
    id: &dyn Fn(usize) -> usize,
    node: Option<&dyn Fn(usize) -> usize>,
    k: usize,
    buckets: Option<&Buckets>,
    what: &str,
) {
    let all = brute(vals, |_| true, id, node, k);
    assert_dist_eq(&got.all, &all, &format!("{what}: overall"));
    let node_of = |i: usize| node.map_or_else(|| id(i), |f| f(i));
    match buckets {
        None => assert!(got.by_bucket.is_none() && got.unbucketed.is_none()),
        Some(bk) => {
            let per = got.by_bucket.as_ref().unwrap();
            assert_eq!(per.len(), bk.len());
            let mut covered = 0;
            for (d, b) in per.iter().zip(bk.as_slice()) {
                assert_eq!(
                    (d.label.as_str(), d.lo, d.hi),
                    (b.label.as_str(), b.lo, b.hi)
                );
                let want = brute(vals, |i| (b.lo..b.hi).contains(&node_of(i)), id, node, k);
                assert_dist_eq(&d.dist, &want, &format!("{what}: bucket {}", b.label));
                covered += d.dist.count + d.dist.absent;
            }
            assert_eq!(
                got.unbucketed,
                Some(vals.len() as u64 - covered),
                "{what}: unbucketed"
            );
        }
    }
}

#[test]
fn distribution_matches_brute_force_with_and_without_buckets() {
    let mut rng = SmallRng::seed_from_u64(7);
    for round in 0..60 {
        let len = [0, 1, 2, 17, 300, 1000][round % 6];
        let vals = random_values(len, &mut rng);
        let lo = rng.random_range(0..50usize);
        let k = [0, 1, 3, 25][round % 4];
        let buckets = random_buckets(lo + len, &mut rng);
        for bk in [None, Some(&buckets)] {
            let conf = DistConf {
                top: k,
                buckets: bk,
            };
            let got = distribution(len, |i| vals[i], lo, &conf);
            check_report(
                &got,
                &vals,
                &|i| lo + i,
                None,
                k,
                bk,
                &format!("round {round}"),
            );
        }
        // per-block ids: block id lo + i, node 3·i + 5 (bucketed by node)
        let node = |i: usize| 3 * i + 5;
        let conf = DistConf {
            top: k,
            buckets: Some(&buckets),
        };
        let got = distribution_with(len, |i| vals[i], &|i| lo + i, Some(&node), &conf);
        check_report(
            &got,
            &vals,
            &|i| lo + i,
            Some(&node),
            k,
            Some(&buckets),
            &format!("blocks round {round}"),
        );
    }
}

#[test]
fn distribution_across_chunks_is_exact_and_thread_independent() {
    // more than two chunks, with bucket boundaries inside chunks and at chunk boundaries
    let len = 2 * CHUNK + CHUNK / 2 + 3;
    let mut rng = SmallRng::seed_from_u64(11);
    let vals = random_values(len, &mut rng);
    let b = |label: &str, lo: usize, hi: usize| Bucket {
        label: label.into(),
        lo,
        hi,
    };
    let buckets = Buckets::new(vec![
        b("a", 0, 1000),
        b("b", 1000, CHUNK),
        b("c", CHUNK + 7, 2 * CHUNK + 100),
        b("d", 2 * CHUNK + 100, len),
    ])
    .unwrap();
    let conf = DistConf {
        top: 20,
        buckets: Some(&buckets),
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| distribution(len, |i| vals[i], 0, &conf))
    };
    let one = run(1);
    check_report(&one, &vals, &|i| i, None, 20, Some(&buckets), "chunks");
    for t in [2, 5, 16] {
        let r = run(t);
        assert_eq!(r, one, "{t} threads");
        // bit-identical sums, not only equal ones
        assert_eq!(r.all.sum.to_bits(), one.all.sum.to_bits());
    }
    // the overall sum does not depend on the buckets
    let plain = distribution(
        len,
        |i| vals[i],
        0,
        &DistConf {
            top: 20,
            buckets: None,
        },
    );
    assert_eq!(plain.all, one.all);
}

#[test]
fn buckets_parse_chunk_boundaries_and_generic_files() {
    let tsv = "chunk_id\tstart_date\tend_date\tstart_ts\tend_ts_excl\tfirst_node\tlast_node_exclusive\tnode_count\n\
               1\t2009-01-01 00:00:00 Europe/Rome\t2009-07-01 00:00:00 Europe/Rome\t1\t2\t0\t18620\t18620\n\
               2\t2009-07-01 00:00:00 Europe/Rome\t2010-01-01 00:00:00 Europe/Rome\t1\t2\t18620\t32777\t14157\n";
    let b = Buckets::parse(tsv).unwrap();
    let labels: Vec<_> = b.as_slice().iter().map(|x| x.label.as_str()).collect();
    assert_eq!(labels, ["2009H1", "2009H2"]);
    assert_eq!((b.as_slice()[1].lo, b.as_slice()[1].hi), (18620, 32777));
    assert_eq!(b.locate(18619), Some(0));
    assert_eq!(b.locate(18620), Some(1));
    assert_eq!(b.locate(32777), None);

    let csv = "# comment\nlabel,lo,hi\nearly,0,1_000\n\nlate,2000,3000\n";
    let b = Buckets::parse(csv).unwrap();
    assert_eq!(b.len(), 2);
    assert_eq!(b.locate(1500), None);
    assert_eq!(b.as_slice()[0].hi, 1000);

    // a file with only ids gets line-number labels
    let b = Buckets::parse("lo\thi\n0\t5\n5\t9\n").unwrap();
    assert_eq!(b.as_slice()[1].label, "3");

    for bad in [
        "",
        "label,lo\nx,0\n",
        "label,lo,hi\n",
        "label,lo,hi\nx,0,10\ny,5,20\n",
        "label,lo,hi\nx,10,5\n",
        "label,lo,hi\nx,a,5\n",
        "label,lo,hi\nx,1\n",
    ] {
        assert!(Buckets::parse(bad).is_err(), "{bad:?} must be rejected");
    }
}

#[test]
fn the_real_chunk_boundaries_cover_the_payment_graph() {
    // the 28 half-year buckets of the 2022 PG, shipped as data/chunk_boundaries.tsv
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/chunk_boundaries.tsv");
    let b = Buckets::load(&path).unwrap();
    assert_eq!(b.len(), 28);
    assert_eq!(b.as_slice()[0].label, "2009H1");
    assert_eq!(b.as_slice()[27].label, "2022H2");
    assert_eq!(b.as_slice()[27].hi, 2_181_021_971);
    assert!(b.as_slice().windows(2).all(|w| w[0].hi == w[1].lo));
}

// ---------------------------------------------------------------------------------------------
// Frontier curve
// ---------------------------------------------------------------------------------------------

/// `|A_c|` for every cut `c ∈ [lo, hi]` by the definition.
fn brute_alive(g: &VecGraph, w: Window) -> Vec<u64> {
    (w.lo..=w.hi)
        .map(|c| {
            (w.lo..c)
                .filter(|&u| {
                    g.successors(u)
                        .filter(|&v| v < w.hi)
                        .min()
                        .is_none_or(|m| m >= c)
                })
                .count() as u64
        })
        .collect()
}

fn brute_curve(alive: &[u64], w: Window, ranges: &[(String, usize, usize)]) -> Vec<CutRange> {
    dag::width::cut_ranges(w, ranges)
        .into_iter()
        .map(|(label, a, b)| {
            let at = |c: usize| alive[c - w.lo];
            let (mut max, mut argmax) = (at(a), a);
            for c in a..=b {
                if at(c) > max {
                    (max, argmax) = (at(c), c);
                }
            }
            CutRange {
                label,
                lo: a,
                hi: b,
                at_lo: at(a),
                at_hi: at(b),
                max,
                argmax,
            }
        })
        .collect()
}

fn random_ranges(n: usize, rng: &mut SmallRng) -> Vec<(String, usize, usize)> {
    let mut v = vec![
        ("all".to_owned(), 0, n),
        ("start".to_owned(), 0, 0),
        ("end".to_owned(), n, n),
    ];
    for j in 0..6 {
        let a = rng.random_range(0..=n);
        let b = rng.random_range(0..=n);
        v.push((format!("r{j}"), a.min(b), a.max(b)));
    }
    v
}

#[test]
fn frontier_curve_matches_brute_force() {
    let mut rng = SmallRng::seed_from_u64(3);
    for seed in 0..16u64 {
        let opts = MiniPgOpts {
            zero_output_prob: if seed % 3 == 0 { 0.15 } else { 0.0 },
            ..Default::default()
        };
        let g = testkit::mini_pg([20, 60, 200][seed as usize % 3], seed, &opts).graph;
        let n = g.num_nodes();
        let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
        let cg = testkit::chunked(&g, 1 + seed % 7);
        let a = rng.random_range(0..=n);
        let b = rng.random_range(0..=n);
        for w in [
            Window::full(n),
            Window {
                lo: a.min(b),
                hi: a.max(b),
            },
        ] {
            let alive = brute_alive(&g, w);
            let ranges = random_ranges(n, &mut rng);
            let want = brute_curve(&alive, w, &ranges);
            assert_eq!(
                frontier_curve_blocks(&blocks, w, &ranges),
                want,
                "seed {seed} {w} blocks"
            );
            assert_eq!(
                frontier_curve_generic(&cg, w, &ranges, &mut no_log()).unwrap(),
                want,
                "seed {seed} {w} generic"
            );
            // |A_c| at sorted cuts (with repeats and both ends)
            let mut cuts: Vec<usize> = (0..12).map(|_| rng.random_range(w.lo..=w.hi)).collect();
            cuts.extend([w.lo, w.hi]);
            cuts.sort_unstable();
            let want_cuts: Vec<u64> = cuts.iter().map(|&c| alive[c - w.lo]).collect();
            assert_eq!(
                frontier_at_cuts_blocks(&blocks, w, &cuts).unwrap(),
                want_cuts
            );
            assert_eq!(
                frontier_at_cuts_generic(&cg, w, &cuts, &mut no_log()).unwrap(),
                want_cuts
            );
            if w.len() > 1 {
                // unsorted or out-of-window cuts are rejected
                assert!(frontier_at_cuts_blocks(&blocks, w, &[w.hi, w.lo]).is_err());
                assert!(frontier_at_cuts_blocks(&blocks, w, &[w.hi + 1]).is_err());
            }
            // the maximum over the whole window is the frontier
            let f = frontier_blocks(&blocks, w, false);
            let whole = frontier_curve_blocks(&blocks, w, &[("w".into(), w.lo, w.hi)]);
            assert_eq!((whole[0].max, whole[0].argmax), (f.size, f.cut));
        }
    }
    // a random DAG (generic only)
    for seed in 0..10 {
        let g = testkit::random_dag(120, 0.05, seed);
        let w = Window::full(120);
        let ranges = random_ranges(120, &mut rng);
        let want = brute_curve(&brute_alive(&g, w), w, &ranges);
        let cg = testkit::chunked(&g, 4);
        assert_eq!(
            frontier_curve_generic(&cg, w, &ranges, &mut no_log()).unwrap(),
            want
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Dominator-tree and block distributions
// ---------------------------------------------------------------------------------------------

#[test]
fn dom_distributions_agree_with_dom_stats_and_brute_force() {
    for seed in 0..12u64 {
        let g = testkit::mini_pg(150, seed, &MiniPgOpts::default()).graph;
        let n = g.num_nodes();
        let cg = testkit::chunked(&g, 3);
        let w = Window { lo: n / 5, hi: n };
        let buckets = random_buckets(n, &mut SmallRng::seed_from_u64(seed));
        let conf = DistConf {
            top: 5,
            buckets: Some(&buckets),
        };
        for post in [false, true] {
            let c = DomConf::new(w);
            let tree = if post {
                post_dominators_pull(&cg, &c, &mut quiet()).unwrap()
            } else {
                dominators_push(&cg, &c, &mut quiet()).unwrap()
            };
            let sizes = tree.sizes();
            let st = tree_stats(&tree, &sizes, 5);
            let d = tree_distributions(&tree, &sizes, &conf);
            // the same conventions as DomStats
            assert_eq!(d.depth.all.log2_hist.bins, st.depth_log2_hist);
            assert_eq!(d.size.all.log2_hist.bins, st.size_log2_hist);
            assert_eq!(d.depth.all.max, Some(f64::from(st.max_depth)));
            assert_eq!(d.depth.all.argmax, st.argmax_depth);
            assert_eq!(d.root_children.all.count, st.root_children);
            assert_eq!(d.root_children.all.sum as u64, st.root_children_size_sum);
            assert_eq!(d.root_children.all.sum as usize, w.len());
            let top: Vec<_> = d.size.all.top.iter().map(|t| t.id).collect();
            let st_top: Vec<_> = st.top.iter().map(|t| t.node).collect();
            assert_eq!(top, st_top);
            // brute force from the public arrays
            let mut vals_depth = Vec::new();
            let mut vals_roots = Vec::new();
            for v in w.lo..w.hi {
                let x = tree.index_of(v);
                vals_depth.push(Some(f64::from(tree.depth_index(x))));
                let root = tree.idom_public(v) == NONE;
                vals_roots.push(root.then(|| f64::from(sizes[x as usize])));
            }
            let id = |i: usize| w.lo + i;
            check_report(&d.depth, &vals_depth, &id, None, 5, Some(&buckets), "depth");
            check_report(
                &d.root_children,
                &vals_roots,
                &id,
                None,
                5,
                Some(&buckets),
                "roots",
            );
        }
    }
}

#[test]
fn block_distributions_match_the_ground_truth_blocks() {
    let mp = testkit::mini_pg(300, 5, &MiniPgOpts::default());
    let g = &mp.graph;
    let (idx, report) = BlockIndex::from_vec_graph(g).unwrap();
    let buckets = random_buckets(g.num_nodes(), &mut SmallRng::seed_from_u64(1));
    let d = dag::cli::blocks::distributions(
        &idx,
        &DistConf {
            top: 4,
            buckets: Some(&buckets),
        },
    );
    let truth = mp.blocks(); // (first, len, nin)
    assert_eq!(d.len.all.count as usize, truth.len());
    assert_eq!(d.len.all.sum as u64, report.sum_len);
    assert_eq!(d.nin.all.sum as u64, report.sum_nin);
    assert_eq!(d.arcs.all.sum as u64, report.m);
    assert_eq!(d.len.all.max, Some(f64::from(report.max_len)));
    let first = |i: usize| truth[i].0 as usize;
    let lens: Vec<Option<f64>> = truth.iter().map(|t| Some(f64::from(t.1))).collect();
    let nins: Vec<Option<f64>> = truth.iter().map(|t| Some(f64::from(t.2))).collect();
    check_report(
        &d.len,
        &lens,
        &|i| i,
        Some(&first),
        4,
        Some(&buckets),
        "len",
    );
    check_report(
        &d.nin,
        &nins,
        &|i| i,
        Some(&first),
        4,
        Some(&buckets),
        "nin",
    );
}

// ---------------------------------------------------------------------------------------------
// aggregate, reach-counts summaries
// ---------------------------------------------------------------------------------------------

fn bytes_of<T: Dump>(v: &[T]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * T::BYTES];
    for (o, x) in out.chunks_exact_mut(T::BYTES).zip(v) {
        x.put(o);
    }
    out
}

#[test]
fn aggregate_reads_every_dtype_and_sentinel() {
    let mut rng = SmallRng::seed_from_u64(9);
    let buckets = random_buckets(400, &mut rng);
    let conf = DistConf {
        top: 3,
        buckets: Some(&buckets),
    };
    let u: Vec<u32> = (0..300)
        .map(|i| {
            if i % 7 == 0 {
                u32::MAX
            } else {
                rng.random_range(0..100)
            }
        })
        .collect();
    let first: Vec<u32> = (0..400).map(|b| b as u32).collect();
    // the expected values of an array whose sentinel ("max") reads as `max` in f64
    let expect = |vals: &[f64], max: f64, absent: Absent| -> Vec<Option<f64>> {
        vals.iter()
            .map(|&x| match absent {
                Absent::Max if x == max => None,
                Absent::Zero if x == 0.0 => None,
                _ => Some(x),
            })
            .collect()
    };
    let as_f64 = |sentinel: f64| -> Vec<f64> {
        u.iter()
            .map(|&x| {
                if x == u32::MAX {
                    sentinel
                } else {
                    f64::from(x)
                }
            })
            .collect()
    };
    for absent in [Absent::None, Absent::Max, Absent::Zero] {
        let id = |i: usize| 50 + i;
        let bk = Some(&buckets);
        let got = aggregate_bytes(&bytes_of(&u), Dtype::U32le, absent, 50, None, &conf);
        let m = f64::from(u32::MAX);
        check_report(
            &got,
            &expect(&as_f64(m), m, absent),
            &id,
            None,
            3,
            bk,
            "u32",
        );
        let u64s: Vec<u64> = u
            .iter()
            .map(|&x| {
                if x == u32::MAX {
                    u64::MAX
                } else {
                    u64::from(x)
                }
            })
            .collect();
        let got = aggregate_bytes(&bytes_of(&u64s), Dtype::U64le, absent, 50, None, &conf);
        let m = u64::MAX as f64;
        check_report(
            &got,
            &expect(&as_f64(m), m, absent),
            &id,
            None,
            3,
            bk,
            "u64",
        );
        let i64s: Vec<i64> = u
            .iter()
            .map(|&x| {
                if x == u32::MAX {
                    i64::MAX
                } else {
                    i64::from(x)
                }
            })
            .collect();
        let got = aggregate_bytes(&bytes_of(&i64s), Dtype::I64le, absent, 50, None, &conf);
        let m = i64::MAX as f64;
        check_report(
            &got,
            &expect(&as_f64(m), m, absent),
            &id,
            None,
            3,
            bk,
            "i64",
        );
        let f64s = as_f64(f64::INFINITY);
        let got = aggregate_bytes(&bytes_of(&f64s), Dtype::F64le, absent, 50, None, &conf);
        let want_f = expect(&f64s, f64::INFINITY, absent);
        check_report(&got, &want_f, &id, None, 3, bk, "f64");
        let f32s: Vec<f32> = f64s.iter().map(|&x| x as f32).collect();
        let got = aggregate_bytes(&bytes_of(&f32s), Dtype::F32le, absent, 50, None, &conf);
        check_report(&got, &want_f, &id, None, 3, bk, "f32");
    }
    // per-block entries: blocks 10.., bucketed by first[b] = b
    let got = aggregate_bytes(
        &bytes_of(&u),
        Dtype::U32le,
        Absent::Max,
        10,
        Some(&first),
        &conf,
    );
    let want: Vec<Option<f64>> = u
        .iter()
        .map(|&x| (x != u32::MAX).then(|| f64::from(x)))
        .collect();
    let node = |i: usize| 10 + i;
    check_report(
        &got,
        &want,
        &|i| 10 + i,
        Some(&node),
        3,
        Some(&buckets),
        "blocks",
    );
    assert_eq!(
        Dtype::of_path("x.dom.depth.u32le".as_ref()),
        Some(Dtype::U32le)
    );
    assert_eq!(Dtype::of_path("x.anc.f32le".as_ref()), Some(Dtype::F32le));
    assert_eq!(Dtype::of_path("x.extf".as_ref()), None);
}

#[test]
fn reach_counts_summary_reports_absolute_ids() {
    // regression: argmax was relative to the window, or a block-window index
    let a = [0.0f32, 3.0, 7.5, 7.5, 1.0];
    let conf = DistConf {
        top: 2,
        buckets: None,
    };
    let s = dag::cli::reach_counts::summarize(&a, 10, &|i| 100 + i, None, &conf);
    assert_eq!((s.argmax, s.max, s.zeros), (Some(102), 7.5, 1));
    assert_eq!(s.argmax_node, None);
    assert_eq!(s.sum, 19.0);
    assert_eq!(s.mean, 19.0 / 5.0);
    assert_eq!(s.mean_over_window, 19.0 / 5.0 / 10.0);
    assert_eq!(s.top.iter().map(|t| t.id).collect::<Vec<_>>(), [102, 103]);
    let first = |i: usize| 1000 + 10 * i;
    let s = dag::cli::reach_counts::summarize(&a, 10, &|i| 40 + i, Some(&first), &conf);
    assert_eq!((s.argmax, s.argmax_node), (Some(42), Some(1020)));
    assert_eq!(s.top[1].node, Some(1030));
}

// ---------------------------------------------------------------------------------------------
// Run metadata
// ---------------------------------------------------------------------------------------------

#[test]
fn run_info_records_budget_rss_and_phases() {
    let mut b = MemBudget::new(1.0);
    b.need(123_456_789, "summary test reservation").unwrap();
    b.release(123_456_789);
    b.need(1000, "summary test small").unwrap();
    assert_eq!(b.peak(), 123_456_789);
    assert!(b.need(2_000_000_000, "too much").is_err());
    let log = budget_log();
    assert!(log.estimate_peak_bytes >= 123_456_789);
    assert!(
        log.reservations
            .iter()
            .any(|r| r.what == "summary test reservation" && r.bytes == 123_456_789)
            || log.dropped > 0
    );
    let dir = tempfile::tempdir().unwrap();
    let mut info = RunInfo::new(None, 5, 0, Window::full(5));
    info.phases_s.add("kernel", 1.5);
    let v: u32 = info.phases_s.time("summary", || 7);
    assert_eq!(v, 7);
    assert_eq!(info.phases_s.get("kernel"), Some(1.5));
    let prefix = dir.path().join("run");
    info.write(&prefix).unwrap();
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("run.json")).unwrap())
            .unwrap();
    assert_eq!(j["phases_s"]["kernel"], 1.5);
    assert!(j["phases_s"]["summary"].as_f64().unwrap() >= 0.0);
    assert!(j["mem"]["estimate_peak_bytes"].as_u64().unwrap() >= 123_456_789);
    assert!(j["mem"]["max_bytes"].is_u64());
    if cfg!(target_os = "linux") {
        assert!(j["rss"]["rss_anon"].as_u64().unwrap() > 0);
        assert!(j["rss"]["vm_hwm"].as_u64().unwrap() >= j["rss"]["vm_rss"].as_u64().unwrap());
    }
}
