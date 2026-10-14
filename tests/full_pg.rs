//! Full-scale validation on the 2022 Bitcoin Payment Graph.
//!
//! Every test is `#[ignore]`d: each one makes one or more passes over a graph of
//! 2,181,021,971 nodes and 8,639,773,499 arcs and needs tens of GB of memory. They check the
//! library (not the CLI, which `scripts/validate_pg.sh` covers) against the exact values of
//! §10.4 and against the per-node arrays of earlier, independent studies in
//! `PG_ANALYSIS_DIR` (`chains/longest.u32le`, `layers/height.u32le`, `bicliques/*`).
//!
//! Run them one at a time (they share one [`BlockIndex`] and must not run concurrently):
//!
//! ```text
//! PG_BASENAME=/data/bitcoin/2022/pg/pg PG_BLOCKS=<prefix of a saved index> \
//!     cargo test --release --test full_pg -- --ignored --test-threads 1
//! ```
//!
//! Environment: `PG_BASENAME` (default `/data/bitcoin/2022/pg/pg`; the transpose `pg-t` must be
//! next to it), `PG_ANALYSIS_DIR` (default `/data/bitcoin/2022/pg_analysis/results`) and the
//! optional `PG_BLOCKS`, the prefix of a [`BlockIndex`] saved by `dag blocks` (loaded in RAM;
//! without it the index is built, about one minute and 27 GB).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use dag::blocks::BlockIndex;
use dag::cpm::cpm;
use dag::csr::Csr;
use dag::dom::{
    DomConf, dominators_blocks, post_dominators_blocks, tree_stats, verify_dominators,
    verify_post_dominators,
};
use dag::dp::certify::{Tolerance, certify};
use dag::dp::solve::solve;
use dag::dp::{Const, DpConf, DpInput, Init, Kernel};
use dag::graph::{self, BvRa, ChunkConf, ChunkedGraph};
use dag::io::{LoadMode, MemBudget, U32Array};
use dag::paths::{CountKind, count_paths};
use dag::reach::bench::{BenchConf, run_bench};
use dag::reach::dagview::BlockQuotient;
use dag::reach::hash::SeededBuildHasher;
use dag::reach::index::{QueryConf, ReachIndex};
use dag::reach::sketch::{SketchConf, counts_blocks, rsd};
use dag::reduce::verify_reduced;
use dag::semiring::{MaxPlus, MinPlus, Semiring};
use dag::sweep::{DynLog, SweepConf};
use dag::topo::check_identity;
use dag::width::{
    frontier_blocks, min_path_cover_blocks, min_path_cover_hk, width_closure_blocks, width_min_flow,
};
use dag::{Dir, NONE, Which, Window};
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;

const N: usize = 2_181_021_971;
const M: u64 = 8_639_773_499;
const SOURCES: usize = 2_883_482;
const SINKS: usize = 136_124_643;
const LONGEST: u32 = 3_559_224;
const LONGEST_END: usize = 2_181_015_947;
const LONGEST_SOURCE: usize = 84_564;

fn basename() -> PathBuf {
    std::env::var_os("PG_BASENAME").map_or_else(|| "/data/bitcoin/2022/pg/pg".into(), Into::into)
}

fn analysis(rel: &str) -> PathBuf {
    std::env::var_os("PG_ANALYSIS_DIR")
        .map_or_else(
            || PathBuf::from("/data/bitcoin/2022/pg_analysis/results"),
            PathBuf::from,
        )
        .join(rel)
}

fn pl() -> ProgressLogger {
    let mut p = ProgressLogger::default();
    p.display_memory(true);
    p
}

fn cpl() -> ConcurrentWrapper<ProgressLogger> {
    let mut p = concurrent_progress_logger![];
    p.display_memory(true);
    p
}

fn open(base: &Path) -> ChunkedGraph<BvRa> {
    graph::open(base, ChunkConf::default()).expect("the PG opens")
}

fn g() -> &'static ChunkedGraph<BvRa> {
    static G: OnceLock<ChunkedGraph<BvRa>> = OnceLock::new();
    G.get_or_init(|| open(&basename()))
}

fn gt() -> &'static ChunkedGraph<BvRa> {
    static GT: OnceLock<ChunkedGraph<BvRa>> = OnceLock::new();
    GT.get_or_init(|| {
        let t = graph::transpose_basename(&basename()).expect("the transpose pg-t exists");
        open(&t)
    })
}

/// The shared [`BlockIndex`]: loaded from `PG_BLOCKS` or built from the graph.
fn blocks() -> &'static BlockIndex {
    static B: OnceLock<BlockIndex> = OnceLock::new();
    B.get_or_init(|| match std::env::var_os("PG_BLOCKS") {
        Some(p) => BlockIndex::load(&PathBuf::from(p), LoadMode::Ram).expect("PG_BLOCKS loads"),
        None => {
            BlockIndex::build(g(), &mut cpl())
                .expect("the PG is a biclique digraph")
                .0
        }
    })
}

fn full() -> Window {
    Window::full(N)
}

fn mismatches(a: &[u32], b: &[u32]) -> (usize, Option<usize>) {
    assert_eq!(a.len(), b.len());
    let bad = a
        .par_chunks(1 << 20)
        .zip(b.par_chunks(1 << 20))
        .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
        .sum();
    let first = (bad > 0).then(|| a.iter().zip(b).position(|(p, q)| p != q).unwrap());
    (bad, first)
}

fn mean(v: &[u32]) -> f64 {
    v.par_chunks(1 << 20)
        .map(|c| c.iter().map(|&x| x as u64).sum::<u64>())
        .sum::<u64>() as f64
        / v.len() as f64
}

/// The first argmax and the max.
fn argmax(v: &[u32]) -> (usize, u32) {
    let m = *v.par_iter().max().unwrap();
    (v.iter().position(|&x| x == m).unwrap(), m)
}

#[test]
#[ignore = "full PG: one parallel scan of pg and pg-t"]
fn topology() {
    let r = check_identity(g(), Some(gt()), &mut cpl()).unwrap();
    assert_eq!((r.n, r.m), (N, M));
    assert!(r.identity_topological, "{:?}", r.first_violation);
    assert_eq!(r.sinks, SINKS);
    assert_eq!(r.sources, Some(SOURCES));
}

#[test]
#[ignore = "full PG: BlockIndex (27 GB) and the bicliques arrays (18 GB, mapped)"]
fn block_index_matches_the_bicliques_lens() {
    let b = blocks();
    let r = b.block_report().unwrap();
    assert_eq!((r.n, r.m, r.nb), (N, M, 777_851_179));
    assert_eq!(
        (r.sum_len, r.sum_nin, r.sum_len_times_nin),
        (2_178_138_489, 2_044_897_328, M)
    );
    assert_eq!((r.max_len, r.max_nin), (13_107, 20_000));
    assert_eq!((r.sources, r.sinks), (SOURCES, SINKS));
    // blocks.u32x3 = (first, |I|, |O|) per block, ascending
    let rec = U32Array::load(&analysis("bicliques/blocks.u32x3"), LoadMode::Mmap).unwrap();
    assert_eq!(rec.len(), 3 * r.nb);
    let bad: usize = (0..r.nb)
        .into_par_iter()
        .with_min_len(1 << 16)
        .filter(|&k| {
            (rec[3 * k], rec[3 * k + 1], rec[3 * k + 2]) != (b.first[k], b.nin[k], b.len[k])
        })
        .count();
    assert_eq!(bad, 0, "blocks differing from bicliques/blocks.u32x3");
    // blk_start[v] = first output of the block containing v, NONE for sources
    let start = U32Array::load(&analysis("bicliques/blk_start.u32"), LoadMode::Mmap).unwrap();
    assert_eq!(start.len(), N);
    let bad: usize = (0..N)
        .into_par_iter()
        .with_min_len(1 << 20)
        .filter(|&v| {
            let want = match b.create[v] {
                NONE => NONE,
                k => b.first[k as usize],
            };
            start[v] != want
        })
        .count();
    assert_eq!(
        bad, 0,
        "nodes whose create differs from bicliques/blk_start.u32"
    );
}

#[test]
#[ignore = "full PG: three DP passes on the blocks, three certification scans"]
fn longest_and_shortest_paths_match_prior_results() {
    let input = DpInput::with_transpose(g(), gt()).and_blocks(blocks());
    let mut conf = DpConf::new(full());
    conf.kernel = Kernel::Blocks;

    // d_max: byte-identical to chains/longest.u32le, with a pred chain back to source 84,564
    let s = MaxPlus::<u32>::new();
    let init = Init::Sources(s.one());
    conf.pred = true;
    let fwd = solve(
        &s,
        &input,
        Dir::Forward,
        &Const(1u32),
        &init,
        &conf,
        &mut pl(),
    )
    .unwrap();
    let longest = U32Array::load(&analysis("chains/longest.u32le"), LoadMode::Mmap).unwrap();
    assert_eq!(
        mismatches(&fwd.val, &longest),
        (0, None),
        "d_max vs longest.u32le"
    );
    assert_eq!(argmax(&fwd.val), (LONGEST_END, LONGEST));
    assert!((mean(&fwd.val) - 2_397_482.094).abs() < 5e-4);
    let pred = fwd.pred.as_ref().unwrap();
    let (mut v, mut steps) = (LONGEST_END, 0u32);
    while pred[v] != NONE {
        v = pred[v] as usize;
        steps += 1;
    }
    assert_eq!((v, steps), (LONGEST_SOURCE, LONGEST));
    let c = certify(
        &s,
        gt(),
        Dir::Forward,
        &Const(1u32),
        &init,
        &fwd,
        Tolerance::Exact,
        &mut cpl(),
    )
    .unwrap();
    assert_eq!(
        (c.violations, c.pred_violations, c.checked),
        (0, 0, N as u64),
        "{c:?}"
    );
    drop(fwd);

    // height: byte-identical to layers/height.u32le
    conf.pred = false;
    let bwd = solve(
        &s,
        &input,
        Dir::Backward,
        &Const(1u32),
        &init,
        &conf,
        &mut pl(),
    )
    .unwrap();
    let height = U32Array::load(&analysis("layers/height.u32le"), LoadMode::Mmap).unwrap();
    assert_eq!(
        mismatches(&bwd.val, &height),
        (0, None),
        "height vs height.u32le"
    );
    assert_eq!(argmax(&bwd.val), (LONGEST_SOURCE, LONGEST));
    let c = certify(
        &s,
        g(),
        Dir::Backward,
        &Const(1u32),
        &init,
        &bwd,
        Tolerance::Exact,
        &mut cpl(),
    )
    .unwrap();
    assert_eq!(c.violations, 0, "{c:?}");
    drop(bwd);

    // d_min: the multi-source BFS depth
    let s = MinPlus::<u32>::new();
    let init = Init::Sources(s.one());
    let dmin = solve(
        &s,
        &input,
        Dir::Forward,
        &Const(1u32),
        &init,
        &conf,
        &mut pl(),
    )
    .unwrap();
    assert_eq!(argmax(&dmin.val), (479_412_379, 25_348));
    assert!((mean(&dmin.val) - 76.34).abs() < 5e-3);
    let c = certify(
        &s,
        gt(),
        Dir::Forward,
        &Const(1u32),
        &init,
        &dmin,
        Tolerance::Exact,
        &mut cpl(),
    )
    .unwrap();
    assert_eq!(c.violations, 0, "{c:?}");
}

#[test]
#[ignore = "full PG: E and H block passes"]
fn critical_path_method() {
    let input = DpInput::blocks(blocks());
    let r = cpm::<u32, _, _>(&input, &Const(1u32), &DpConf::new(full()), &mut pl()).unwrap();
    assert_eq!(r.t, LONGEST);
    let critical =
        r.e.par_iter()
            .zip(r.h.par_iter())
            .filter(|&(&e, &h)| e + h == r.t)
            .count();
    assert_eq!(critical, 5_459_477);
    assert!(
        r.e.par_iter()
            .zip(r.h.par_iter())
            .all(|(&e, &h)| e + h <= r.t)
    );
}

#[test]
#[ignore = "full PG: two concurrent Product<ExtFloat, M61> block passes (~165 GB)"]
fn path_count_duality() {
    let input = DpInput::blocks(blocks());
    let r = count_paths(
        &input,
        Which::Both,
        &[CountKind::ExtFloat, CountKind::M61],
        &DpConf::new(full()),
        &mut pl(),
    )
    .unwrap();
    let d = r.summary.duality.as_ref().expect("both directions ran");
    assert!(d.ok, "{d:?}");
    assert_eq!(d.mod61_equal, Some(true));
    assert!(d.extfloat_rel_diff.unwrap() <= 1e-6, "{d:?}");
    eprintln!(
        "ExtFloat duality: relative difference {:e}",
        d.extfloat_rel_diff.unwrap()
    );
    eprintln!(
        "total paths: 2^{} (mod 2^61-1: {})",
        r.total_log2, r.total_mod61
    );
}

#[test]
#[ignore = "full PG: one generic H3 scan plus the blocks certificate"]
fn transitively_reduced() {
    let (rep, red) = verify_reduced(
        g(),
        Some(blocks()),
        &SweepConf::default(),
        &mut MemBudget::unlimited(),
        &mut cpl(),
    )
    .unwrap();
    assert!(rep.is_transitively_reduced && red.is_empty());
    assert_eq!((rep.certificate, rep.arcs), ("blocks", M));
    let (rep, red) = verify_reduced(
        g(),
        None,
        &SweepConf::default(),
        &mut MemBudget::unlimited(),
        &mut cpl(),
    )
    .unwrap();
    assert_eq!(rep.h3_failing_nodes, 0, "{rep:?}");
    assert!(rep.is_transitively_reduced && red.is_empty());
}

#[test]
#[ignore = "full PG: closed forms on the blocks, Hopcroft–Karp and flows on windows"]
fn path_cover_frontier_and_window_widths() {
    let b = blocks();
    let (st, _) = min_path_cover_blocks(b, full(), false);
    assert_eq!((st.matching, st.mpc_d), (1_016_828_057, 1_164_193_914));
    let fr = frontier_blocks(b, full(), false);
    assert_eq!((fr.size, fr.cut), (136_188_373, 2_180_649_454));
    for w in [
        Window {
            lo: 0,
            hi: 1_000_000,
        },
        Window {
            lo: 1_000_000_000,
            hi: 1_001_000_000,
        },
        Window {
            lo: 2_180_000_000,
            hi: 2_181_000_000,
        },
    ] {
        let csr = Csr::from_window(g(), w, &mut cpl()).unwrap();
        let (st, _) = min_path_cover_blocks(b, w, false);
        assert_eq!(
            st.mpc_d as usize,
            min_path_cover_hk(&csr).num_paths(),
            "window {w}"
        );
    }
    let w = Window {
        lo: 0,
        hi: 1_000_000,
    };
    let csr = Csr::from_window(g(), w, &mut cpl()).unwrap();
    let closure = width_closure_blocks(b, w);
    let minflow = width_min_flow(&csr);
    assert_eq!(closure.width, minflow.width, "width of {w}");
    assert!(frontier_blocks(b, w, false).size <= closure.width);
}

#[test]
#[ignore = "full PG: both trees from the blocks (~110 GB) and two verifier scans"]
fn dominator_trees() {
    let b = blocks();
    let conf = DomConf::new(full());
    let dom = dominators_blocks(b, &conf, &mut pl()).unwrap();
    let st = tree_stats(&dom, &dom.sizes(), 0);
    assert!(st.root_children >= SOURCES as u64, "{}", st.root_children);
    assert!(st.max_depth <= 25_349, "{}", st.max_depth);
    assert_eq!(st.root_children_size_sum, N as u64);
    let idom = dom.idoms();
    drop(dom);
    let r = verify_dominators(g(), &idom, full(), &mut cpl()).unwrap();
    assert!(
        r.ok && (r.v0_fail, r.v1_fail, r.v2_fail) == (0, 0, 0),
        "{r:?}"
    );
    drop(idom);

    let pdom = post_dominators_blocks(b, &conf, &mut pl()).unwrap();
    let st = tree_stats(&pdom, &pdom.sizes(), 0);
    assert!(st.root_children >= SINKS as u64, "{}", st.root_children);
    let ipdom = pdom.idoms();
    drop(pdom);
    let r = verify_post_dominators(g(), &ipdom, full(), &mut cpl()).unwrap();
    assert!(
        r.ok && (r.v0_fail, r.v1_fail, r.v2_fail) == (0, 0, 0),
        "{r:?}"
    );
}

#[test]
#[ignore = "full PG: two concurrent HyperLogLog block passes (~40 GB, ~15 min)"]
fn hyperloglog_counts() {
    let b = blocks();
    let conf = SketchConf {
        log2m: 10,
        hasher: SeededBuildHasher { seed: 0 },
        window: full(),
        sweep: SweepConf::default(),
    };
    let c = counts_blocks(b, Which::Both, &conf, &mut pl()).unwrap();
    assert!(c.peak_live <= 5_139_210, "peak live {}", c.peak_live);
    for p in &c.passes {
        assert_eq!(p.stats.live_at_end, 0, "{}: leaked sketches", p.name);
    }
    let anc = c.anc.as_ref().unwrap();
    let nonzero = (0..N)
        .into_par_iter()
        .filter(|&v| b.create[v] == NONE && anc[v] != 0.0)
        .count();
    assert_eq!(nonzero, 0, "sources with a non-zero ancestor estimate");
    let desc = c.desc.as_ref().unwrap();
    let mean = desc
        .par_chunks(1 << 20)
        .map(|x| x.iter().map(|&d| d as f64).sum::<f64>())
        .sum::<f64>()
        / N as f64;
    let frac = mean / N as f64;
    // the reference 0.4195 is the exact mean over 64 sampled cones
    assert!(
        (frac - 0.4195).abs() <= 3.0 * rsd(10) * 0.4195,
        "mean desc / N = {frac}"
    );
}

#[test]
#[ignore = "full PG: ChronoChain on the block quotient (~40 GB) and 2·10⁴ checked queries"]
fn reachability_index_on_the_block_quotient() {
    let b = blocks();
    let dag = BlockQuotient::new(b);
    let mut log = pl();
    let (idx, stats) = ReachIndex::build_with_stats(&dag, 4, &mut DynLog(&mut log)).unwrap();
    // a G path of L arcs from a coinbase output visits L blocks: L − 1 arcs of Q
    assert_eq!(stats.max_lvl, LONGEST - 1);
    let conf = BenchConf {
        sources: 64,
        queries: 10_000,
        seed: 1,
        max_query_secs: 60.0,
        query: QueryConf::default(),
        threads: rayon::current_num_threads(),
    };
    let rep = run_bench(&idx, &dag, &conf, &mut DynLog(&mut log)).unwrap();
    assert_eq!(rep.disagreements(), 0);
    // the uniform workload has exactly `queries` queries; in the balanced one a source with
    // too few reachable (or unreachable) targets contributes fewer (reach::bench docs)
    let total: usize = rep.workloads.iter().map(|w| w.queries).sum();
    assert!(total >= 10_000, "{total} queries");
    for w in &rep.workloads {
        assert!(
            w.queries <= 10_000 && w.queries > 0,
            "{}: {}",
            w.name,
            w.queries
        );
        if w.name == "uniform" {
            assert_eq!(w.queries, 10_000);
        }
        assert!(
            w.par_max_query_s < 60.0,
            "{}: {} s",
            w.name,
            w.par_max_query_s
        );
    }
}
