//! Acceptance tests of the HyperLogLog ancestor/descendant counts.
//!
//! Oracles: the brute-force closure of `testkit`, sketches built by inserting the exact sets
//! (F3 Lemma 4), and HyperBall run to stabilization.

#[path = "support/cli.rs"]
mod cli;

use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hasher};
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use anyhow::Result;
use card_est_array::traits::{EstimationLogic, MergeEstimationLogic, SliceEstimationLogic};
use dag::blocks::BlockIndex;
use dag::graph::{self, ChunkConf, ChunkedGraph};
use dag::io::{MemBudget, read_array};
use dag::reach::hash::{GAMMA, SeededBuildHasher, mix64};
use dag::reach::sketch::{
    self, Counts, Logic, SketchConf, anc_pass_blocks, anc_pass_blocks_paged, anc_pass_generic,
    block_window, counts_blocks, counts_blocks_batched, counts_blocks_paged, counts_generic,
    counts_generic_batched, desc_pass_blocks, desc_pass_blocks_paged, desc_pass_generic,
    reflexive_sketch_estimates,
};
use dag::sweep::SweepConf;
use dag::testkit::*;
use dag::{DagError, NONE, Which, Window};
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;
use webgraph_algo::distances::hyperball::HyperBallBuilder;

type Sketches = Vec<Vec<usize>>;

fn quiet() -> Option<ProgressLogger> {
    None
}

fn sweep_conf(decoders: usize) -> SweepConf {
    SweepConf { decoders }
}

fn conf(log2m: u32, seed: u64, window: Window) -> SketchConf<SeededBuildHasher> {
    SketchConf {
        log2m,
        hasher: SeededBuildHasher::new(seed),
        window,
        sweep: sweep_conf(3),
    }
}

/// A random window of `[0, n)` (possibly empty or full).
fn random_window(r: &mut SmallRng, n: usize) -> Window {
    let a = r.random_range(0..=n);
    let b = r.random_range(0..=n);
    Window {
        lo: a.min(b),
        hi: a.max(b),
    }
}

/// The sketch obtained by inserting `set` (F3 Lemma 4 oracle).
fn sketch_of<H: BuildHasher + Clone>(
    logic: &Logic<H>,
    set: impl IntoIterator<Item = usize>,
) -> Vec<usize> {
    let mut s = vec![0usize; logic.backend_len()];
    for x in set {
        logic.add(&mut s, x);
    }
    s
}

/// `a ≤ b` register-wise, i.e. `max(a, b) == b`.
fn dominated<H: BuildHasher + Clone>(logic: &Logic<H>, a: &[usize], b: &[usize]) -> bool {
    let mut m = b.to_vec();
    logic.merge(&mut m, a);
    m == b
}

/// Exact proper ancestors and descendants inside `w`, indexed by `v − lo`.
fn exact_sets(cl: &[Vec<u64>], w: Window) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let anc = (w.lo..w.hi)
        .map(|v| (w.lo..v).filter(|&u| bit(&cl[u], v)).collect())
        .collect();
    let desc = (w.lo..w.hi)
        .map(|u| (u + 1..w.hi).filter(|&v| bit(&cl[u], v)).collect())
        .collect();
    (anc, desc)
}

/// Visits of a pass as `(position, sketch)`, checking the visiting order.
fn collect(visits: Vec<(usize, Vec<usize>)>, len: usize, ascending: bool) -> Sketches {
    assert_eq!(visits.len(), len, "every position visited once");
    for (k, (pos, _)) in visits.iter().enumerate() {
        let want = if ascending { k } else { len - 1 - k };
        assert_eq!(*pos, want, "visiting order");
    }
    let mut out: Sketches = visits.into_iter().map(|(_, s)| s).collect();
    if !ascending {
        out.reverse();
    }
    out
}

fn generic_sketches<G: RandomAccessGraph + Sync, H: BuildHasher + Clone + Send + Sync>(
    g: &ChunkedGraph<G>,
    w: Window,
    logic: &Logic<H>,
    decoders: usize,
) -> Result<(Sketches, Sketches)> {
    let mut va = Vec::new();
    anc_pass_generic(g, w, logic, &sweep_conf(decoders), &mut quiet(), |p, s| {
        va.push((p, s.to_vec()))
    })?;
    let mut vd = Vec::new();
    desc_pass_generic(g, w, logic, &sweep_conf(decoders), &mut quiet(), |p, s| {
        vd.push((p, s.to_vec()))
    })?;
    Ok((collect(va, w.len(), true), collect(vd, w.len(), false)))
}

/// Block sketches `A(X)`, `D(X)` for `X ∈ block_window(w)` and the two pass reports.
fn block_sketches<H: BuildHasher + Clone>(
    idx: &BlockIndex,
    w: Window,
    logic: &Logic<H>,
) -> (Sketches, Sketches, sketch::PassStats, sketch::PassStats) {
    let nbw = block_window(idx, w).len();
    let c = AtomicUsize::new(0);
    let mut va = Vec::new();
    let sa = anc_pass_blocks(idx, w, logic, &c, |p, s| va.push((p, s.to_vec())));
    let mut vd = Vec::new();
    let sd = desc_pass_blocks(idx, w, logic, &c, |p, s| vd.push((p, s.to_vec())));
    assert_eq!(
        c.into_inner(),
        2 * w.len(),
        "progress counts every node once per pass"
    );
    (collect(va, nbw, true), collect(vd, nbw, false), sa, sd)
}

/// Exact `A(X) ∩ w` and `D(X) ∩ w` for the blocks of the window.
fn exact_block_sets(
    idx: &BlockIndex,
    cl: &[Vec<u64>],
    w: Window,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let bw = block_window(idx, w);
    let mut a = Vec::new();
    let mut d = Vec::new();
    for x in bw {
        let o = idx.outputs(x);
        let f = o.start;
        a.push((w.lo..f).filter(|&u| bit(&cl[u], f)).collect());
        let mut ds: Vec<usize> = o.clone().filter(|&v| v < w.hi).collect();
        for v in o.clone().filter(|&v| v < w.hi) {
            ds.extend((v + 1..w.hi).filter(|&y| bit(&cl[v], y)));
        }
        ds.sort_unstable();
        ds.dedup();
        d.push(ds);
    }
    (a, d)
}

/// `max_t #{X ∈ bw : min input of X in w ≤ t < first(X)}`: the live-set bound `LB`.
fn live_bound(idx: &BlockIndex, w: Window) -> usize {
    let bw = block_window(idx, w);
    let mut minin = vec![usize::MAX; bw.len()];
    for u in w.lo..w.hi {
        let x = idx.spend[u];
        if x != NONE && bw.contains(&(x as usize)) {
            let m = &mut minin[x as usize - bw.start];
            *m = (*m).min(u);
        }
    }
    (w.lo..w.hi)
        .map(|t| {
            bw.clone()
                .filter(|&x| minin[x - bw.start] <= t && t < idx.first[x] as usize)
                .count()
        })
        .max()
        .unwrap_or(0)
}

fn mini(seed: u64, r: &mut SmallRng) -> MiniPg {
    let opts = MiniPgOpts {
        coinbase_prob: [0.05, 0.1, 0.3][r.random_range(0..3)],
        max_outputs: r.random_range(1..=5),
        max_inputs: r.random_range(1..=6),
        zero_output_prob: [0.0, 0.0, 0.2][r.random_range(0..3)],
        ..Default::default()
    };
    mini_pg(r.random_range(1..=160), seed, &opts)
}

fn assert_f32_eq(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: position {i}: {x} vs {y}");
    }
}

fn assert_counts_eq(a: &Counts, b: &Counts, blocks: bool, what: &str) {
    for (x, y, f) in [(&a.anc, &b.anc, "anc"), (&a.desc, &b.desc, "desc")] {
        assert_f32_eq(
            x.as_deref().unwrap(),
            y.as_deref().unwrap(),
            &format!("{what} {f}"),
        );
    }
    if blocks {
        for (x, y, f) in [
            (&a.blk_anc, &b.blk_anc, "blk_anc"),
            (&a.blk_desc, &b.blk_desc, "blk_desc"),
        ] {
            assert_f32_eq(
                x.as_deref().unwrap(),
                y.as_deref().unwrap(),
                &format!("{what} {f}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Hasher
// ---------------------------------------------------------------------------------------------

/// The hash of `x` is the `(x + 1)`-th output of SplitMix64 from state `mix64(seed)` (reference
/// outputs of splitmix64.c seeded with 0, as `mix64(0) = 0`), integer widths agree, and ids
/// never collide.
#[test]
fn hasher_matches_splitmix64() {
    let h = SeededBuildHasher::new(0);
    let want = [
        0xE220_A839_7B1D_CDAFu64,
        0x6E78_9E6A_A1B9_65F4,
        0x06C4_5D18_8009_454F,
    ];
    for (x, &w) in want.iter().enumerate() {
        assert_eq!(h.hash_one(x), w);
        assert_eq!(h.hash_one(x as u64), w);
        assert_eq!(h.hash_one(x as u32), w);
        assert_eq!(h.hash_u64(x as u64), w);
    }
    // a sequential SplitMix64 generator gives the same stream
    assert_eq!(mix64(0), 0);
    let mut state = mix64(12345);
    let h = SeededBuildHasher::new(12345);
    assert_eq!(h.state(), state);
    for x in 0..1000u64 {
        state = state.wrapping_add(GAMMA);
        assert_eq!(h.hash_one(x as usize), mix64(state));
    }
    let mut seen = std::collections::HashSet::new();
    for x in 0..200_000usize {
        assert!(seen.insert(h.hash_one(x)), "collision at {x}");
    }
    assert_ne!(
        SeededBuildHasher::new(1).hash_one(5usize),
        SeededBuildHasher::new(2).hash_one(5usize)
    );
    // byte tails are length-tagged
    let bytes = |b: &[u8]| {
        let mut s = SeededBuildHasher::new(7).build_hasher();
        s.write(b);
        s.finish()
    };
    assert_ne!(bytes(&[0]), bytes(&[0, 0]));
    assert_ne!(bytes(&[]), bytes(&[0]));
    assert_eq!(
        bytes(&5u64.to_le_bytes()),
        SeededBuildHasher::new(7).hash_one(5u64)
    );
}

/// Regression: seeds differing by a multiple of γ are not shifted copies of one
/// hash function, `h_{s+kγ}(x) ≠ h_s(x + k)`, and the sketches of the same id-contiguous cones
/// under two such seeds are not identical.
#[test]
fn gamma_shifted_seeds_are_not_shifts() -> Result<()> {
    for s in [0u64, 1, 7, 12345, u64::MAX] {
        for k in [1u64, 2, 1000] {
            let h = SeededBuildHasher::new(s);
            let hk = SeededBuildHasher::new(s.wrapping_add(k.wrapping_mul(GAMMA)));
            let shifted = (0..1000u64)
                .filter(|&x| hk.hash_u64(x) == h.hash_u64(x + k))
                .count();
            assert_eq!(shifted, 0, "seed {s}, k {k}");
        }
    }
    // sketches of the cones {0..c} and {k..k+c} under seeds s + kγ and s
    let (s, k) = (3u64, 5u64);
    let la = sketch::logic(
        1 << 20,
        8,
        SeededBuildHasher::new(s.wrapping_add(k.wrapping_mul(GAMMA))),
    )?;
    let lb = sketch::logic(1 << 20, 8, SeededBuildHasher::new(s))?;
    for c in [100usize, 10_000] {
        assert_ne!(
            sketch_of(&la, 0..c),
            sketch_of(&lb, k as usize..k as usize + c)
        );
    }
    Ok(())
}

/// The 5-bit group merge equals card-est-array's merge (register-wise maximum) on arbitrary
/// words (every register value 0..=31, straddling registers included) and on real sketches.
#[test]
fn fast_merge_equals_card_est_array_merge() -> Result<()> {
    let mut r = SmallRng::seed_from_u64(99);
    for log2m in [6, 7, 10, 12] {
        let logic = sketch::logic(1 << 20, log2m, SeededBuildHasher::new(log2m.into()))?;
        let words = logic.backend_len();
        for round in 0..2000 {
            let gen_words = |r: &mut SmallRng| -> Vec<usize> {
                (0..words)
                    .map(|_| match round % 4 {
                        0 => r.random::<u64>() as usize,
                        1 => usize::MAX,
                        2 => 0,
                        _ => (r.random::<u64>() & r.random::<u64>()) as usize,
                    })
                    .collect()
            };
            let x = gen_words(&mut r);
            let y = if round % 3 == 0 {
                gen_words(&mut r)
            } else {
                (0..words).map(|_| r.random::<u64>() as usize).collect()
            };
            let mut want = x.clone();
            logic.merge(&mut want, &y);
            let mut got = x.clone();
            sketch::merge_max5(&mut got, &y);
            assert_eq!(got, want, "log2m {log2m} round {round}");
        }
        for k in [1, 10, 1000, 100_000] {
            let a = sketch_of(&logic, (0..k).map(|_| r.random_range(0..1usize << 40)));
            let b = sketch_of(&logic, (0..k).map(|_| r.random_range(0..1usize << 40)));
            let mut want = a.clone();
            logic.merge(&mut want, &b);
            let mut got = a.clone();
            sketch::merge_max5(&mut got, &b);
            assert_eq!(got, want);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Bit identity (F3 Lemma 4)
// ---------------------------------------------------------------------------------------------

/// Generic passes: every node's sketch equals the sketch of its exact proper ancestor
/// (descendant) set in the window; in-memory graphs with tiny chunks and BV round trips.
#[test]
fn generic_sketches_equal_exact_sets() -> Result<()> {
    let dir = tempfile::tempdir()?;
    for seed in 0..60u64 {
        let mut r = SmallRng::seed_from_u64(seed);
        let n = r.random_range(1..=120);
        let p = [0.01, 0.05, 0.2][r.random_range(0..3)];
        let g = random_dag(n, p, seed);
        let cl = closure(&g);
        let log2m = [6, 8, 10][r.random_range(0..3)];
        let logic = sketch::logic(n, log2m, SeededBuildHasher::new(seed))?;
        let cg = chunked(&g, r.random_range(1..=7));
        let bv = if seed % 4 == 0 {
            let base = write_bv(dir.path(), &format!("g{seed}"), &g);
            Some(graph::open(
                &base,
                ChunkConf {
                    chunk_arcs: 3,
                    max_chunk_nodes: 3,
                },
            )?)
        } else {
            None
        };
        for w in [Window::full(n), random_window(&mut r, n)] {
            let (ea, ed) = exact_sets(&cl, w);
            let (sa, sd) = generic_sketches(&cg, w, &logic, r.random_range(1..=4))?;
            for i in 0..w.len() {
                assert_eq!(
                    sa[i],
                    sketch_of(&logic, ea[i].iter().copied()),
                    "seed {seed} {w} anc {i}"
                );
                assert_eq!(
                    sd[i],
                    sketch_of(&logic, ed[i].iter().copied()),
                    "seed {seed} {w} desc {i}"
                );
            }
            if let Some(bv) = &bv {
                let (ba, bd) = generic_sketches(bv, w, &logic, 2)?;
                assert_eq!(ba, sa, "BV anc seed {seed}");
                assert_eq!(bd, sd, "BV desc seed {seed}");
            }
        }
    }
    Ok(())
}

/// Block passes: `A(X)` and `D(X)` equal the sketches of the exact sets, on mini_pg graphs and
/// windows cutting through blocks; no slot leaks, and the live set stays within `LB + 1`.
#[test]
fn block_sketches_equal_exact_sets() -> Result<()> {
    for seed in 0..80u64 {
        let mut r = SmallRng::seed_from_u64(1000 + seed);
        let m = mini(seed, &mut r);
        let n = m.graph.num_nodes();
        if n == 0 {
            continue;
        }
        let (idx, _) = BlockIndex::from_vec_graph(&m.graph)?;
        let cl = closure(&m.graph);
        let logic = sketch::logic(
            n,
            [6, 7, 10][r.random_range(0..3)],
            SeededBuildHasher::new(seed),
        )?;
        for w in [
            Window::full(n),
            random_window(&mut r, n),
            random_window(&mut r, n),
        ] {
            let (ea, ed) = exact_block_sets(&idx, &cl, w);
            let (sa, sd, st_a, st_d) = block_sketches(&idx, w, &logic);
            for x in 0..ea.len() {
                assert_eq!(
                    sa[x],
                    sketch_of(&logic, ea[x].iter().copied()),
                    "seed {seed} {w} A({x})"
                );
                assert_eq!(
                    sd[x],
                    sketch_of(&logic, ed[x].iter().copied()),
                    "seed {seed} {w} D({x})"
                );
            }
            let lb = live_bound(&idx, w);
            for (st, name) in [(st_a, "anc"), (st_d, "desc")] {
                assert_eq!(st.live_at_end, 0, "seed {seed} {w} {name}: leaked slots");
                assert_eq!(st.closed, ea.len());
                assert!(
                    st.peak_live <= lb + 1,
                    "seed {seed} {w} {name}: peak {} > LB {lb} + 1",
                    st.peak_live
                );
            }
        }
    }
    Ok(())
}

/// Regression: with slab pages of 1–3 slots, merges between slots of different
/// pages (both orders) and page growth run on every graph, and the block sketches still equal
/// the sketches of the exact sets; the full block mode equals the default-page result.
#[test]
fn tiny_slab_pages_equal_exact_sets() -> Result<()> {
    for seed in 0..40u64 {
        let mut r = SmallRng::seed_from_u64(6000 + seed);
        let m = mini(seed, &mut r);
        let n = m.graph.num_nodes();
        if n == 0 {
            continue;
        }
        let (idx, _) = BlockIndex::from_vec_graph(&m.graph)?;
        let cl = closure(&m.graph);
        let logic = sketch::logic(n, 6, SeededBuildHasher::new(seed))?;
        for w in [Window::full(n), random_window(&mut r, n)] {
            let (ea, ed) = exact_block_sets(&idx, &cl, w);
            let c = conf(6, seed, w);
            let want = counts_blocks(&idx, Which::Both, &c, &mut quiet())?;
            for slots in 1..=3 {
                let p = AtomicUsize::new(0);
                let mut va = Vec::new();
                let sa = anc_pass_blocks_paged(&idx, w, &logic, &p, slots, |i, s| {
                    va.push((i, s.to_vec()))
                });
                let mut vd = Vec::new();
                let sd = desc_pass_blocks_paged(&idx, w, &logic, &p, slots, |i, s| {
                    vd.push((i, s.to_vec()))
                });
                let (sa_, sd_) = (collect(va, ea.len(), true), collect(vd, ea.len(), false));
                for x in 0..ea.len() {
                    let what = format!("seed {seed} {w} pages of {slots}");
                    assert_eq!(
                        sa_[x],
                        sketch_of(&logic, ea[x].iter().copied()),
                        "{what} A({x})"
                    );
                    assert_eq!(
                        sd_[x],
                        sketch_of(&logic, ed[x].iter().copied()),
                        "{what} D({x})"
                    );
                }
                assert_eq!((sa.live_at_end, sd.live_at_end), (0, 0));
                let b = counts_blocks_paged(&idx, Which::Both, &c, 3, Some(slots), &mut quiet())?;
                assert_counts_eq(&b, &want, true, &format!("seed {seed} {w} paged {slots}"));
            }
        }
    }
    Ok(())
}

/// Block mode equals generic mode on mini_pg (full graph and windows), and the per-node arrays
/// are the per-block arrays expanded through `create` and `spend`.
#[test]
fn blocks_equal_generic() -> Result<()> {
    for seed in 0..60u64 {
        let mut r = SmallRng::seed_from_u64(2000 + seed);
        let m = mini(seed, &mut r);
        let n = m.graph.num_nodes();
        let (idx, _) = BlockIndex::from_vec_graph(&m.graph)?;
        let cg = chunked(&m.graph, r.random_range(1..=9));
        for w in [Window::full(n), random_window(&mut r, n)] {
            let c = conf(8, seed, w);
            let b = counts_blocks(&idx, Which::Both, &c, &mut quiet())?;
            let generic = counts_generic(
                &cg,
                Which::Both,
                &c,
                &mut MemBudget::unlimited(),
                &mut quiet(),
            )?;
            assert_counts_eq(&b, &generic, false, &format!("seed {seed} {w}"));
            let bw = b.blocks.clone().unwrap();
            assert_eq!(bw, block_window(&idx, w));
            let (ba, bd) = (b.blk_anc.as_ref().unwrap(), b.blk_desc.as_ref().unwrap());
            let (anc, desc) = (b.anc.as_ref().unwrap(), b.desc.as_ref().unwrap());
            for v in w.lo..w.hi {
                let cr = idx.create[v];
                let want = if cr != NONE && bw.contains(&(cr as usize)) {
                    ba[cr as usize - bw.start]
                } else {
                    0.0
                };
                assert_eq!(anc[v - w.lo].to_bits(), want.to_bits());
                let sp = idx.spend[v];
                let want = if sp != NONE && bw.contains(&(sp as usize)) {
                    bd[sp as usize - bw.start]
                } else {
                    0.0
                };
                assert_eq!(desc[v - w.lo].to_bits(), want.to_bits());
            }
            // one direction at a time gives the same arrays
            let fa = counts_blocks(&idx, Which::Forward, &c, &mut quiet())?;
            assert!(fa.desc.is_none() && fa.blk_desc.is_none());
            assert_f32_eq(fa.anc.as_deref().unwrap(), anc, "anc only");
            let fd = counts_generic(
                &cg,
                Which::Backward,
                &c,
                &mut MemBudget::unlimited(),
                &mut quiet(),
            )?;
            assert!(fd.anc.is_none());
            assert_f32_eq(fd.desc.as_deref().unwrap(), desc, "desc only");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// HyperBall oracle
// ---------------------------------------------------------------------------------------------

/// With HyperBall's hasher, `num_elements = n` and the same `log2m`, the reflexive estimates
/// equal `HyperBall::reachable_nodes()` exactly.
#[test]
fn reflexive_estimates_equal_hyperball() -> Result<()> {
    for seed in 0..25u64 {
        let mut r = SmallRng::seed_from_u64(3000 + seed);
        let n = r.random_range(1..=200);
        let g = random_dag(n, [0.005, 0.02, 0.1][r.random_range(0..3)], seed);
        let log2m = [6, 7, 9][r.random_range(0..3)];
        let ours = reflexive_sketch_estimates(
            &chunked(&g, r.random_range(1..=6)),
            log2m,
            BuildHasherDefault::<DefaultHasher>::default(),
            &sweep_conf(2),
        )?;
        let dcf = g.build_dcf();
        let mut hb =
            HyperBallBuilder::with_hyper_log_log(&g, None::<&VecGraph>, &dcf, log2m, None)?
                .build(&mut quiet());
        hb.run_until_done(SmallRng::seed_from_u64(seed), &mut quiet())?;
        let theirs = hb.reachable_nodes()?;
        assert_f32_eq(&ours, &theirs, &format!("seed {seed} n {n} log2m {log2m}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------------------------

/// Along every arc `u → v`: `S_anc(u) ∪ {u} ≤ S_anc(v)` and `S_desc(v) ∪ {v} ≤ S_desc(u)`
/// register-wise.
#[test]
fn registers_are_monotone_along_arcs() -> Result<()> {
    for seed in 0..15u64 {
        let g = random_dag(150, 0.03, seed);
        let logic = sketch::logic(150, 6, SeededBuildHasher::new(seed))?;
        let (sa, sd) = generic_sketches(&chunked(&g, 4), Window::full(150), &logic, 2)?;
        for (u, v) in arcs(&g) {
            let mut au = sa[u].clone();
            logic.add(&mut au, u);
            assert!(dominated(&logic, &au, &sa[v]), "anc {u} -> {v}");
            let mut dv = sd[v].clone();
            logic.add(&mut dv, v);
            assert!(dominated(&logic, &dv, &sd[u]), "desc {u} -> {v}");
        }
    }
    Ok(())
}

/// `Σ|anc| = Σ|desc|` on the exact sets (both count the comparable pairs), and the estimated
/// totals are within `4·rsd` of it.
#[test]
fn exact_totals_agree() -> Result<()> {
    for seed in 0..5u64 {
        let n = 2500;
        let g = random_dag(n, 0.002, seed);
        let cl = closure(&g);
        let (ea, ed) = exact_sets(&cl, Window::full(n));
        let ta: usize = ea.iter().map(Vec::len).sum();
        let td: usize = ed.iter().map(Vec::len).sum();
        assert_eq!(ta, td);
        let c = counts_generic(
            &chunked(&g, 64),
            Which::Both,
            &conf(10, seed, Window::full(n)),
            &mut MemBudget::unlimited(),
            &mut quiet(),
        )?;
        let tol = 4.0 * sketch::rsd(10);
        for a in [c.anc.unwrap(), c.desc.unwrap()] {
            let s: f64 = a.iter().map(|&x| f64::from(x)).sum();
            assert!(
                (s / ta as f64 - 1.0).abs() <= tol,
                "seed {seed}: total {s} vs {ta}"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Accuracy
// ---------------------------------------------------------------------------------------------

/// Over 30 independent seeds (a different random DAG and an independent hash function each),
/// `|est − exact| ≤ 4·rsd·exact` for at least 95 % of the (seed, node, direction) samples with
/// `exact > 0`, and the estimate is exactly 0 when `exact = 0`. Also a long chain, where
/// `|anc(v)| = v`.
#[test]
fn accuracy_over_independent_seeds() -> Result<()> {
    let log2m = 8;
    let tol = 4.0 * sketch::rsd(log2m);
    let (mut good, mut total) = (0usize, 0usize);
    for seed in 0..30u64 {
        let n = 1200;
        let g = random_dag(n, 0.004, 4000 + seed);
        let cl = closure(&g);
        let (ea, ed) = exact_sets(&cl, Window::full(n));
        let c = counts_generic(
            &chunked(&g, 256),
            Which::Both,
            &conf(log2m, 0xC0FFEE ^ (seed * 7919), Window::full(n)),
            &mut MemBudget::unlimited(),
            &mut quiet(),
        )?;
        for (est, ex) in [(c.anc.unwrap(), ea), (c.desc.unwrap(), ed)] {
            for (e, s) in est.iter().zip(&ex) {
                if s.is_empty() {
                    assert_eq!(*e, 0.0, "seed {seed}: empty set must estimate 0");
                    continue;
                }
                total += 1;
                good += usize::from((f64::from(*e) / s.len() as f64 - 1.0).abs() <= tol);
            }
        }
    }
    assert!(total > 30 * 1000, "too few samples: {total}");
    assert!(
        good as f64 >= 0.95 * total as f64,
        "{good} of {total} within 4·rsd"
    );

    let n = 20_000;
    let c = counts_generic(
        &chunked(&chain(n), 1 << 10),
        Which::Both,
        &conf(log2m, 9, Window::full(n)),
        &mut MemBudget::unlimited(),
        &mut quiet(),
    )?;
    let anc = c.anc.unwrap();
    let ok = (1..n)
        .filter(|&v| (f64::from(anc[v]) / v as f64 - 1.0).abs() <= tol)
        .count();
    assert!(
        ok as f64 >= 0.95 * (n - 1) as f64,
        "chain: {ok} of {} within 4·rsd",
        n - 1
    );
    let desc = c.desc.unwrap();
    assert_eq!(anc[0], 0.0);
    assert_eq!(desc[n - 1], 0.0);
    Ok(())
}

/// Small cones are nearly exact: on a complete DAG with `p = 10`, the estimate of a set of
/// `k ≤ 40` elements is within 1 of `k` (LogLog-β has a small bias at tiny cardinalities).
#[test]
fn small_cones_are_nearly_exact() -> Result<()> {
    let n = 41;
    let c = counts_generic(
        &chunked(&complete_dag(n), 7),
        Which::Both,
        &conf(10, 1, Window::full(n)),
        &mut MemBudget::unlimited(),
        &mut quiet(),
    )?;
    let (anc, desc) = (c.anc.unwrap(), c.desc.unwrap());
    for v in 0..n {
        assert!(
            (f64::from(anc[v]) - v as f64).abs() <= 1.0,
            "anc {v}: {}",
            anc[v]
        );
        assert!(
            (f64::from(desc[v]) - (n - 1 - v) as f64).abs() <= 1.0,
            "desc {v}: {}",
            desc[v]
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Edge cases, errors, determinism
// ---------------------------------------------------------------------------------------------

/// Sources have `anc = 0` and sinks `desc = 0` exactly, in both modes and in windows.
#[test]
fn sources_and_sinks_are_zero() -> Result<()> {
    for seed in 0..30u64 {
        let mut r = SmallRng::seed_from_u64(5000 + seed);
        let m = mini(seed, &mut r);
        let n = m.graph.num_nodes();
        let (idx, _) = BlockIndex::from_vec_graph(&m.graph)?;
        let t = transpose(&m.graph);
        for w in [Window::full(n), random_window(&mut r, n)] {
            let c = conf(6, seed, w);
            let b = counts_blocks(&idx, Which::Both, &c, &mut quiet())?;
            let generic = counts_generic(
                &chunked(&m.graph, 3),
                Which::Both,
                &c,
                &mut MemBudget::unlimited(),
                &mut quiet(),
            )?;
            for counts in [&b, &generic] {
                let (anc, desc) = (counts.anc.as_ref().unwrap(), counts.desc.as_ref().unwrap());
                for v in w.lo..w.hi {
                    let has_pred = t.successors(v).into_iter().any(|u| u >= w.lo);
                    let has_succ = m.graph.successors(v).into_iter().any(|x| x < w.hi);
                    assert_eq!(anc[v - w.lo] == 0.0, !has_pred, "seed {seed} {w} anc {v}");
                    assert_eq!(desc[v - w.lo] == 0.0, !has_succ, "seed {seed} {w} desc {v}");
                }
            }
        }
    }
    Ok(())
}

/// Empty graphs and windows, single nodes and isolated nodes.
#[test]
fn edge_cases() -> Result<()> {
    let c = conf(10, 0, Window::full(0));
    let g0 = chunked(&isolated(0), 1);
    let r = counts_generic(
        &g0,
        Which::Both,
        &c,
        &mut MemBudget::unlimited(),
        &mut quiet(),
    )?;
    assert_eq!((r.anc.unwrap().len(), r.desc.unwrap().len()), (0, 0));
    let (i0, _) = BlockIndex::from_vec_graph(&isolated(0))?;
    let r = counts_blocks(&i0, Which::Both, &c, &mut quiet())?;
    assert_eq!(r.blk_anc.unwrap().len(), 0);
    assert!(
        reflexive_sketch_estimates(&g0, 10, SeededBuildHasher::new(0), &sweep_conf(1))?.is_empty()
    );

    for n in [1, 5, 300] {
        let g = isolated(n);
        let (idx, _) = BlockIndex::from_vec_graph(&g)?;
        for w in [
            Window::full(n),
            Window { lo: 1, hi: 1 },
            Window { lo: 0, hi: n / 2 },
        ] {
            let c = conf(6, 3, w);
            let a = counts_generic(
                &chunked(&g, 2),
                Which::Both,
                &c,
                &mut MemBudget::unlimited(),
                &mut quiet(),
            )?;
            let b = counts_blocks(&idx, Which::Both, &c, &mut quiet())?;
            for x in [a.anc, a.desc, b.anc, b.desc] {
                let x = x.unwrap();
                assert_eq!(x.len(), w.len());
                assert!(x.iter().all(|&e| e == 0.0));
            }
            assert_eq!(b.blk_anc.unwrap().len(), 0);
        }
        let refl = reflexive_sketch_estimates(
            &chunked(&g, 2),
            10,
            SeededBuildHasher::new(1),
            &sweep_conf(1),
        )?;
        assert!(
            refl.iter().all(|&e| (e - 1.0).abs() < 0.01),
            "a singleton set estimates ≈ 1"
        );
    }
    Ok(())
}

/// Back arcs, self-loops, bad parameters and bad windows are errors.
#[test]
fn errors() -> Result<()> {
    let g = random_dag(40, 0.1, 1);
    let mut b = MemBudget::unlimited();
    for bad in [with_back_arc(&g), with_self_loop(&g)] {
        let c = conf(8, 0, Window::full(40));
        for which in [Which::Forward, Which::Backward] {
            let e = counts_generic(&chunked(&bad, 3), which, &c, &mut b, &mut quiet()).unwrap_err();
            assert!(
                matches!(
                    e.downcast_ref::<DagError>(),
                    Some(DagError::NotTopological { .. })
                ),
                "{e:#}"
            );
        }
        assert!(
            reflexive_sketch_estimates(
                &chunked(&bad, 3),
                8,
                SeededBuildHasher::new(0),
                &sweep_conf(1)
            )
            .is_err()
        );
    }
    for log2m in [4, 5, 19] {
        assert!(
            counts_generic(
                &chunked(&g, 3),
                Which::Both,
                &conf(log2m, 0, Window::full(40)),
                &mut b,
                &mut quiet()
            )
            .is_err()
        );
    }
    let e = counts_generic(
        &chunked(&g, 3),
        Which::Both,
        &conf(8, 0, Window { lo: 0, hi: 41 }),
        &mut b,
        &mut quiet(),
    )
    .unwrap_err();
    assert!(matches!(
        e.downcast_ref::<DagError>(),
        Some(DagError::BadWindow { .. })
    ));
    let (idx, _) = BlockIndex::from_vec_graph(&mini_pg(20, 1, &MiniPgOpts::default()).graph)?;
    let e = counts_blocks(
        &idx,
        Which::Both,
        &conf(
            8,
            0,
            Window {
                lo: 0,
                hi: idx.n + 1,
            },
        ),
        &mut quiet(),
    )
    .unwrap_err();
    assert!(matches!(
        e.downcast_ref::<DagError>(),
        Some(DagError::BadWindow { .. })
    ));
    // the generic sketch array is refused by the memory budget
    let mut tiny = MemBudget::new(1e-6);
    let e = counts_generic(
        &chunked(&g, 3),
        Which::Both,
        &conf(10, 0, Window::full(40)),
        &mut tiny,
        &mut quiet(),
    )
    .unwrap_err();
    assert!(
        matches!(
            e.downcast_ref::<DagError>(),
            Some(DagError::OverBudget { .. })
        ),
        "{e:#}"
    );
    assert_eq!(tiny.used(), 0, "nothing stays reserved after a refusal");
    Ok(())
}

/// Results do not depend on the estimation batch size (double buffering, ascending and
/// descending carving), the number of decoders or the thread count (1 thread included: the
/// collector must not deadlock; the block passes run in the installed pool, see
/// `blocks_respect_the_installed_pool`).
#[test]
fn deterministic_across_batches_and_threads() -> Result<()> {
    let m = mini_pg(400, 77, &MiniPgOpts::default());
    let n = m.graph.num_nodes();
    let (idx, _) = BlockIndex::from_vec_graph(&m.graph)?;
    let w = Window {
        lo: n / 7,
        hi: n - n / 9,
    };
    let c = conf(7, 5, w);
    let reference = counts_blocks(&idx, Which::Both, &c, &mut quiet())?;
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        pool.install(|| -> Result<()> {
            for batch in [1, 2, 7, 64, 1 << 16] {
                let b = counts_blocks_batched(&idx, Which::Both, &c, batch, &mut quiet())?;
                assert_counts_eq(
                    &b,
                    &reference,
                    true,
                    &format!("blocks threads {threads} batch {batch}"),
                );
                let mut cc = c.clone();
                cc.sweep = sweep_conf(threads);
                let g = counts_generic_batched(
                    &chunked(&m.graph, 5),
                    Which::Both,
                    &cc,
                    batch,
                    &mut MemBudget::unlimited(),
                    &mut quiet(),
                )?;
                assert_counts_eq(
                    &g,
                    &reference,
                    false,
                    &format!("generic threads {threads} batch {batch}"),
                );
            }
            Ok(())
        })?;
    }
    // with a real logger (progress forwarded from the helper thread)
    let mut pl = ProgressLogger::default();
    pl.log_interval(std::time::Duration::from_millis(1));
    let b = counts_blocks(&idx, Which::Both, &c, &mut pl)?;
    assert_counts_eq(&b, &reference, true, "with logger");
    Ok(())
}

/// A hasher that counts the hashes computed on threads whose name lacks `prefix` (regression:
/// the block passes used to run on the global pool whatever pool the caller installed).
#[derive(Clone)]
struct PoolCheckHasher {
    inner: SeededBuildHasher,
    prefix: &'static str,
    calls: std::sync::Arc<AtomicUsize>,
    foreign: std::sync::Arc<AtomicUsize>,
}

impl BuildHasher for PoolCheckHasher {
    type Hasher = <SeededBuildHasher as BuildHasher>::Hasher;
    fn build_hasher(&self) -> Self::Hasher {
        use std::sync::atomic::Ordering::Relaxed;
        self.calls.fetch_add(1, Relaxed);
        let ours = std::thread::current()
            .name()
            .is_some_and(|n| n.starts_with(self.prefix));
        if !ours {
            self.foreign.fetch_add(1, Relaxed);
        }
        self.inner.build_hasher()
    }
}

/// Block mode runs inside the rayon pool installed by the caller (1 and 2 threads), with the
/// same results as on the global pool, and a progress logger on the calling worker thread.
#[test]
fn blocks_respect_the_installed_pool() -> Result<()> {
    use std::sync::atomic::Ordering::Relaxed;
    let m = mini_pg(300, 5, &MiniPgOpts::default());
    let n = m.graph.num_nodes();
    let (idx, _) = BlockIndex::from_vec_graph(&m.graph)?;
    let c = conf(8, 9, Window::full(n));
    let reference = counts_blocks(&idx, Which::Both, &c, &mut quiet())?;
    for threads in [1, 2] {
        let prefix = if threads == 1 { "u3a-one-" } else { "u3a-two-" };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |i| format!("{prefix}{i}"))
            .build()?;
        let hasher = PoolCheckHasher {
            inner: SeededBuildHasher::new(9),
            prefix,
            calls: Default::default(),
            foreign: Default::default(),
        };
        let cc = SketchConf {
            log2m: c.log2m,
            hasher: hasher.clone(),
            window: c.window,
            sweep: c.sweep.clone(),
        };
        for batch in [1, 3, 1 << 16] {
            let mut pl = ProgressLogger::default();
            pl.log_interval(std::time::Duration::from_millis(1));
            let b =
                pool.install(|| counts_blocks_batched(&idx, Which::Both, &cc, batch, &mut pl))?;
            assert_counts_eq(
                &b,
                &reference,
                true,
                &format!("pool {threads} batch {batch}"),
            );
        }
        assert!(hasher.calls.load(Relaxed) > 0, "the passes hash nodes");
        assert_eq!(
            hasher.foreign.load(Relaxed),
            0,
            "{threads}-thread pool: block passes ran outside the installed pool"
        );
    }
    Ok(())
}

/// Different seeds give different (independent) estimates; the same seed is reproducible.
#[test]
fn seeds_change_the_hash() -> Result<()> {
    let g = random_dag(500, 0.01, 3);
    let cg = chunked(&g, 16);
    let run = |seed| {
        counts_generic(
            &cg,
            Which::Forward,
            &conf(8, seed, Window::full(500)),
            &mut MemBudget::unlimited(),
            &mut quiet(),
        )
        .map(|c| c.anc.unwrap())
    };
    let (a, b, a2) = (run(1)?, run(2)?, run(1)?);
    assert_f32_eq(&a, &a2, "same seed");
    assert!(a.iter().zip(&b).any(|(x, y)| x != y));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------

fn read_f32(path: &Path) -> Vec<f32> {
    read_array::<f32>(path).unwrap()
}

/// `dag reach-counts` in block and generic mode writes the library's arrays and a JSON
/// record; bad parameters give exit code 2.
#[test]
fn cli_reach_counts() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let m = mini_pg(300, 11, &MiniPgOpts::default());
    let n = m.graph.num_nodes();
    let base = write_bv(dir.path(), "mini", &m.graph);
    let (idx, _) = BlockIndex::from_vec_graph(&m.graph)?;
    let w = Window {
        lo: n / 5,
        hi: n - n / 6,
    };
    let c = conf(8, 42, w);
    let want = counts_blocks(&idx, Which::Both, &c, &mut quiet())?;
    let win = format!("{}..{}", w.lo, w.hi);
    for blocks in ["auto", "none"] {
        let out = dir.path().join(format!("rc-{blocks}"));
        cli::run_ok(&[
            "--threads",
            "2",
            "reach-counts",
            "--graph",
            &cli::p(&base),
            "--blocks",
            blocks,
            "--window",
            &win,
            "--log2m",
            "8",
            "--seed",
            "42",
            "--out",
            &cli::p(&out),
        ]);
        let f = |field: &str| read_f32(&dag::io::array_path::<f32>(&out, field));
        assert_f32_eq(&f("anc"), want.anc.as_deref().unwrap(), "cli anc");
        assert_f32_eq(&f("desc"), want.desc.as_deref().unwrap(), "cli desc");
        let j = cli::read_json(&dag::io::suffixed(&out, "json"));
        assert_eq!(j["params"]["seed"], 42);
        assert_eq!(j["params"]["log2m"], 8);
        assert_eq!(j["window"]["lo"], w.lo);
        assert_eq!(j["results"]["anc"]["len"], w.len());
        if blocks == "auto" {
            assert_eq!(j["results"]["mode"], "blocks");
            assert_f32_eq(
                &f("blk.anc"),
                want.blk_anc.as_deref().unwrap(),
                "cli blk anc",
            );
            assert_f32_eq(
                &f("blk.desc"),
                want.blk_desc.as_deref().unwrap(),
                "cli blk desc",
            );
            assert_eq!(j["results"]["sources_check"]["ok"], true);
            // regression: every source of the window is counted, not only the
            // nodes without a creating transaction
            let mut has_pred = vec![false; n];
            for u in w.lo..w.hi {
                for v in m.graph.successors(u) {
                    if v < w.hi {
                        has_pred[v] = true;
                    }
                }
            }
            let sources = (w.lo..w.hi).filter(|&v| !has_pred[v]).count();
            let global = (w.lo..w.hi).filter(|&v| idx.create[v] == NONE).count();
            assert!(
                sources > global,
                "the test window must have non-global sources"
            );
            assert_eq!(j["results"]["sources_check"]["sources"], sources);
            assert_eq!(j["results"]["passes"][0]["stats"]["live_at_end"], 0);
        } else {
            assert_eq!(j["results"]["mode"], "generic");
            assert!(!dag::io::array_path::<f32>(&out, "blk.anc").exists());
        }
    }
    // saved block index, one direction
    let prefix = dir.path().join("mini");
    idx.save(&prefix)?;
    let out = dir.path().join("rc-prefix");
    cli::run_ok(&[
        "reach-counts",
        "--graph",
        &cli::p(&base),
        "--blocks",
        &cli::p(&prefix),
        "--which",
        "desc",
        "--window",
        &win,
        "--log2m",
        "8",
        "--seed",
        "42",
        "--out",
        &cli::p(&out),
    ]);
    assert_f32_eq(
        &read_f32(&dag::io::array_path::<f32>(&out, "desc")),
        want.desc.as_deref().unwrap(),
        "prefix desc",
    );
    assert!(!dag::io::array_path::<f32>(&out, "anc").exists());
    // errors
    let bad = dir.path().join("bad");
    cli::run_expect(
        &[
            "reach-counts",
            "--graph",
            &cli::p(&base),
            "--log2m",
            "5",
            "--out",
            &cli::p(&bad),
        ],
        2,
    );
    cli::run_expect(
        &[
            "reach-counts",
            "--graph",
            &cli::p(&base),
            "--window",
            "5..2",
            "--out",
            &cli::p(&bad),
        ],
        2,
    );
    Ok(())
}
