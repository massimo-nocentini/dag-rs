//! Integration tests on a real web graph, cnr-2000.
//!
//! cnr-2000 (325,557 nodes, 3,216,152 arcs, from the WebGraph data sets) is not a DAG, so it
//! is turned into one in two ways:
//!
//! - **(a) forward**: keep the arcs `u → v` with `u < v`, so the identity is a topological order;
//! - **(b) condensation**: the quotient by the strongly connected components (Tarjan), with
//!   duplicate and intra-component arcs removed, relabelled by a verified topological order
//!   ([`topo::topo_order`] + [`topo::relabel_vec`]).
//!
//! On both DAGs every family runs on the library API and is checked against an independent
//! oracle that never shares code with the kernel under test: the topological check, DP kernel
//! agreement plus certification, dominator and post-dominator trees against the verifier and
//! against brute-force vertex removal on sampled pairs, the reachability index against BFS,
//! HyperLogLog counts against exact BFS cone sizes, transitive reduction against closure
//! preservation and the H3 verifier, and frontier ≤ MPC_d. cnr-2000 is not a line digraph, so
//! [`BlockIndex::build`] must fail with [`DagError::NotBiclique`] (the generic fallback path).
//!
//! The tests are skipped (with a message) when the graph is absent. The basename is taken from
//! `CNR2000_BASENAME`, by default `../webgraph-rs/data/cnr-2000`, i.e. a webgraph-rs checkout next to this repository.

#[path = "support/cli.rs"]
mod cli;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use cli::{p, read_json, run_expect, run_ok};
use dag::blocks::BlockIndex;
use dag::csr::Csr;
use dag::dom::{
    DomConf, DomTree, dominators_push, post_dominators_pull, verify_dominators,
    verify_post_dominators,
};
use dag::dp::certify::{Tolerance, certify};
use dag::dp::solve::solve;
use dag::dp::{Const, DpConf, DpInput, DpResult, Init, Kernel, One};
use dag::graph::{self, ChunkConf, ChunkedGraph};
use dag::io::MemBudget;
use dag::reach::dagview::GraphDag;
use dag::reach::hash::SeededBuildHasher;
use dag::reach::index::{QueryConf, ReachIndex, Scratch, reach};
use dag::reach::sketch::{SketchConf, counts_generic, rsd};
use dag::reduce::{transitive_reduction_search, verify_reduced_window};
use dag::semiring::modular::M61;
use dag::semiring::{MaxPlus, Semiring};
use dag::sweep::SweepConf;
use dag::testkit::{transpose, vec_graph};
use dag::topo::{self, check_identity};
use dag::width::{check_path_cover, frontier_generic, min_path_cover_hk};
use dag::{DagError, Dir, NONE, Which, Window};
use dsi_progress_logger::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use webgraph::prelude::*;
use webgraph_algo::sccs;

/// Chunks small enough that the sweeps of a 3·10⁶-arc graph use many chunks and batches.
const CHUNKS: ChunkConf = ChunkConf {
    chunk_arcs: 1 << 14,
    max_chunk_nodes: 1 << 12,
};

fn quiet() -> Option<ProgressLogger> {
    None
}

fn cquiet() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

fn sweep() -> SweepConf {
    SweepConf { decoders: 8 }
}

/// The cnr-2000 basename, or `None` (test skipped) if the graph is absent.
fn cnr_basename() -> Option<PathBuf> {
    let base = std::env::var_os("CNR2000_BASENAME").map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../webgraph-rs/data/cnr-2000")
                .to_path_buf()
        },
        PathBuf::from,
    );
    if base.with_extension("graph").exists() && base.with_extension("ef").exists() {
        Some(base)
    } else {
        eprintln!(
            "cnr2000: {} not found, test skipped (set CNR2000_BASENAME)",
            base.display()
        );
        None
    }
}

/// The two DAG-ifications of cnr-2000, `(a)` and `(b)`, built once per test binary.
fn dags() -> Option<&'static [(&'static str, VecGraph); 2]> {
    static DAGS: OnceLock<Option<[(&'static str, VecGraph); 2]>> = OnceLock::new();
    DAGS.get_or_init(|| {
        let base = cnr_basename()?;
        let g = graph::load_ra(&base).expect("cnr-2000 loads");
        let n = g.num_nodes();
        let all: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| g.successors(u).map(move |v| (u, v)))
            .collect();
        assert_eq!(all.len() as u64, g.num_arcs(), "arc count of cnr-2000");
        let forward = vec_graph(n, all.iter().copied().filter(|&(u, v)| u < v));
        let full = vec_graph(n, all.iter().copied());
        Some([("forward", forward), ("condensation", condensation(&full))])
    })
    .as_ref()
}

/// The condensation of `g` (quotient by the SCCs, no loops, no duplicate arcs), relabelled so
/// that the identity is a topological order.
fn condensation(g: &VecGraph) -> VecGraph {
    let sccs = sccs::tarjan(g, &mut quiet());
    let comp = sccs.components();
    let k = sccs.num_components();
    assert!(k < g.num_nodes(), "cnr-2000 has non-trivial SCCs");
    let q = vec_graph(
        k,
        (0..g.num_nodes()).flat_map(|u| {
            g.successors(u)
                .filter(move |&v| comp[u] != comp[v])
                .map(move |v| (comp[u], comp[v]))
        }),
    );
    let order = topo::topo_order(&q, &mut quiet()).expect("a condensation is acyclic");
    topo::relabel_vec(&q, &order.rank)
}

/// One DAG-ification with its transpose, both chunked.
struct Fixture<'a> {
    name: &'a str,
    g: &'a VecGraph,
    n: usize,
    cg: ChunkedGraph<VecGraph>,
    cgt: ChunkedGraph<VecGraph>,
}

fn fixtures() -> Vec<Fixture<'static>> {
    let Some(d) = dags() else {
        return Vec::new();
    };
    d.iter()
        .map(|(name, g)| Fixture {
            name,
            g,
            n: g.num_nodes(),
            cg: ChunkedGraph::new(g.clone(), CHUNKS).unwrap(),
            cgt: ChunkedGraph::new(transpose(g), CHUNKS).unwrap(),
        })
        .collect()
}

/// Nodes reachable from `src` (reflexive), by BFS over `g`, skipping the node `skip`.
fn bfs(g: &VecGraph, src: &[usize], skip: Option<usize>) -> Vec<bool> {
    let mut seen = vec![false; g.num_nodes()];
    let mut stack = Vec::new();
    for &s in src {
        if Some(s) != skip && !seen[s] {
            seen[s] = true;
            stack.push(s);
        }
    }
    while let Some(u) = stack.pop() {
        for v in g.successors(u) {
            if Some(v) != skip && !seen[v] {
                seen[v] = true;
                stack.push(v);
            }
        }
    }
    seen
}

/// `k` uniform draws (with repetition) from `0..n`.
fn sample(rng: &mut SmallRng, n: usize, k: usize) -> Vec<usize> {
    (0..k).map(|_| rng.random_range(0..n)).collect()
}

// ---------------------------------------------------------------------------------------------

#[test]
fn identity_is_topological_on_both_dags() {
    for f in fixtures() {
        let r = check_identity(&f.cg, Some(&f.cgt), &mut cquiet()).unwrap();
        assert!(
            r.identity_topological,
            "{}: {:?}",
            f.name, r.first_violation
        );
        assert_eq!(r.n, f.n);
        assert_eq!(r.m, f.g.num_arcs(), "{}", f.name);
        let sinks = (0..f.n).filter(|&u| f.g.outdegree(u) == 0).count();
        let sources = (0..f.n).filter(|&v| f.cgt.graph.outdegree(v) == 0).count();
        assert_eq!(r.sinks, sinks, "{}", f.name);
        assert_eq!(r.sources, Some(sources), "{}", f.name);
    }
}

/// The CLI agrees with the library on a BV round trip of DAG (a): `topo-check` and the
/// generic (`--blocks none`) `tr verify` of a window.
#[test]
fn cli_round_trip_of_the_forward_dag() {
    let Some(d) = dags() else { return };
    let g = &d[0].1;
    let dir = tempfile::tempdir().unwrap();
    let b = dag::testkit::write_bv(dir.path(), "cnr-fwd", g);
    let o = dir.path().join("topo");
    run_ok(&["topo-check", "--graph", &p(&b), "--out", &p(&o)]);
    let j = read_json(&o.with_extension("json"));
    let r = &j["results"];
    assert_eq!(r["identity_topological"], true, "{j}");
    assert_eq!(r["m"], g.num_arcs());

    let w = Window { lo: 0, hi: 20_000 };
    let o = dir.path().join("tr");
    // exit code 1: the window is not transitively reduced
    run_expect(
        &[
            "tr",
            "verify",
            "--graph",
            &p(&b),
            "--blocks",
            "none",
            "--window",
            &w.to_string(),
            "--out",
            &p(&o),
        ],
        1,
    );
    let j = read_json(&o.with_extension("json"));
    let cg = ChunkedGraph::new(g.clone(), CHUNKS).unwrap();
    let (rep, red) = verify_reduced_window(
        &cg,
        w,
        None,
        &sweep(),
        &mut MemBudget::unlimited(),
        &mut cquiet(),
    )
    .unwrap();
    let jr = &j["results"]["report"];
    assert_eq!(jr["redundant"], rep.redundant, "{j}");
    assert_eq!(jr["redundant"], red.len() as u64);
    assert_eq!(jr["h3_failing_nodes"], rep.h3_failing_nodes);
    assert_eq!(jr["is_transitively_reduced"], rep.is_transitively_reduced);
    assert!(!rep.is_transitively_reduced);
}

/// Longest paths (max-plus, unit arc length) and path counts mod 2⁶¹ − 1: push == pull,
/// backward pull, certification with exact tolerance, and an independent sequential DP.
#[test]
fn dp_kernels_agree_and_certify() {
    for f in fixtures() {
        let w = Window::full(f.n);
        let input = DpInput::with_transpose(&f.cg, &f.cgt);
        let run = |dir: Dir, kernel: Kernel| -> DpResult<u32> {
            let s = MaxPlus::<u32>::new();
            let mut conf = DpConf::new(w);
            conf.kernel = kernel;
            conf.sweep = sweep();
            conf.pred = true;
            solve(
                &s,
                &input,
                dir,
                &Const(1u32),
                &Init::Sources(s.one()),
                &conf,
                &mut quiet(),
            )
            .unwrap()
        };
        let push = run(Dir::Forward, Kernel::Push);
        let pull = run(Dir::Forward, Kernel::Pull);
        assert_eq!(push, pull, "{}: forward push != pull", f.name);
        let back = run(Dir::Backward, Kernel::Pull);

        // independent oracle: dmax by a plain forward recurrence, height by a backward one
        let mut dmax = vec![0u32; f.n];
        for u in 0..f.n {
            for v in f.g.successors(u) {
                dmax[v] = dmax[v].max(dmax[u] + 1);
            }
        }
        let mut height = vec![0u32; f.n];
        for u in (0..f.n).rev() {
            for v in f.g.successors(u) {
                height[u] = height[u].max(height[v] + 1);
            }
        }
        assert_eq!(push.val, dmax, "{}: forward values", f.name);
        assert_eq!(back.val, height, "{}: backward values", f.name);
        let longest = *dmax.iter().max().unwrap();
        assert_eq!(longest, *height.iter().max().unwrap());

        // the predecessor chain of the argmax is a real path of the right length
        let end = (0..f.n).find(|&v| dmax[v] == longest).unwrap();
        let (mut v, mut len) = (end, 0u32);
        let pred = push.pred.as_ref().unwrap();
        while pred[v] != NONE {
            let u = pred[v] as usize;
            assert!(f.g.successors(u).into_iter().any(|x| x == v));
            v = u;
            len += 1;
        }
        assert_eq!(len, longest, "{}: pred chain of {end}", f.name);
        assert_eq!(f.cgt.graph.outdegree(v), 0, "pred chain ends at a source");

        let s = MaxPlus::<u32>::new();
        let init = Init::Sources(s.one());
        for (dir, g, res) in [(Dir::Forward, &f.cgt, &push), (Dir::Backward, &f.cg, &back)] {
            let c = certify(
                &s,
                g,
                dir,
                &Const(1u32),
                &init,
                res,
                Tolerance::Exact,
                &mut cquiet(),
            )
            .unwrap();
            assert_eq!(c.violations, 0, "{} {dir:?}: {c:?}", f.name);
            assert_eq!(c.pred_violations, 0, "{} {dir:?}: {c:?}", f.name);
            assert_eq!(c.checked, f.n as u64);
        }

        // path counts mod p: push == pull == certified; totals agree in both directions
        let m = M61::default();
        let count = |dir: Dir, kernel: Kernel| {
            let mut conf = DpConf::new(w);
            conf.kernel = kernel;
            conf.sweep = sweep();
            solve(
                &m,
                &input,
                dir,
                &One,
                &Init::Sources(m.one()),
                &conf,
                &mut quiet(),
            )
            .unwrap()
        };
        let fp = count(Dir::Forward, Kernel::Push);
        assert_eq!(fp, count(Dir::Forward, Kernel::Pull), "{}: M61", f.name);
        let bp = count(Dir::Backward, Kernel::Pull);
        let c = certify(
            &m,
            &f.cgt,
            Dir::Forward,
            &One,
            &Init::Sources(m.one()),
            &fp,
            Tolerance::Exact,
            &mut cquiet(),
        )
        .unwrap();
        assert_eq!(c.violations, 0, "{}: {c:?}", f.name);
        let sum = |it: &mut dyn Iterator<Item = u64>| it.fold(m.zero(), |a, x| m.plus(a, x));
        let into_sinks = sum(&mut (0..f.n)
            .filter(|&u| f.g.outdegree(u) == 0)
            .map(|u| fp.val[u]));
        let from_sources = sum(&mut (0..f.n)
            .filter(|&v| f.cgt.graph.outdegree(v) == 0)
            .map(|v| bp.val[v]));
        assert_eq!(into_sinks, from_sources, "{}: path-count duality", f.name);
    }
}

/// Whether `d` is a proper ancestor of `v` in `tree` (public ids).
fn tree_ancestor(tree: &DomTree, d: usize, v: usize) -> bool {
    let mut x = tree.idom_public(v);
    while x != NONE {
        if x as usize == d {
            return true;
        }
        x = tree.idom_public(x as usize);
    }
    false
}

/// Dominators (push) and post-dominators (pull): the independent verifier accepts both, and
/// on 200 sampled pairs `(d, v)` per tree, `d` is a proper tree ancestor of `v` iff removing
/// `d` disconnects `v` from every source (resp. every sink), by BFS.
#[test]
fn dominator_trees_verify_and_match_vertex_removal() {
    let mut rng = SmallRng::seed_from_u64(2000);
    for f in fixtures() {
        let w = Window::full(f.n);
        let mut conf = DomConf::new(w);
        conf.sweep = sweep();
        let dom = dominators_push(&f.cg, &conf, &mut quiet()).unwrap();
        let pdom = post_dominators_pull(&f.cg, &conf, &mut quiet()).unwrap();
        let r = verify_dominators(&f.cg, &dom.idoms(), w, &mut cquiet()).unwrap();
        assert!(r.ok, "{}: {r:?}", f.name);
        let r = verify_post_dominators(&f.cg, &pdom.idoms(), w, &mut cquiet()).unwrap();
        assert!(r.ok, "{}: {r:?}", f.name);

        let gt = &f.cgt.graph;
        let sources: Vec<usize> = (0..f.n).filter(|&v| gt.outdegree(v) == 0).collect();
        let sinks: Vec<usize> = (0..f.n).filter(|&u| f.g.outdegree(u) == 0).collect();
        for (tree, fwd, roots) in [(&dom, f.g, &sources), (&pdom, gt, &sinks)] {
            let mut hits = 0;
            for i in 0..200 {
                let v = loop {
                    let v = rng.random_range(0..f.n);
                    if tree.idom_public(v) != NONE || i % 4 == 3 {
                        break v;
                    }
                };
                // half the pairs use a tree ancestor (expected to dominate), half a random node
                let d = if i % 2 == 0 && tree.idom_public(v) != NONE {
                    let mut chain = Vec::new();
                    let mut x = tree.idom_public(v);
                    while x != NONE {
                        chain.push(x as usize);
                        x = tree.idom_public(x as usize);
                    }
                    chain[rng.random_range(0..chain.len())]
                } else {
                    rng.random_range(0..f.n)
                };
                if d == v {
                    continue;
                }
                let expect = tree_ancestor(tree, d, v);
                let brute = !bfs(fwd, roots, Some(d))[v];
                assert_eq!(expect, brute, "{}: d = {d}, v = {v}", f.name);
                hits += usize::from(expect);
            }
            assert!(hits >= 50, "{}: too few dominating pairs sampled", f.name);
        }
    }
}

/// The ChronoChain index answers 10⁴ random pairs exactly (100 BFS sources × 100 targets,
/// half of them drawn from the source's cone), with the default budget and with budget 0
/// (every undecided query goes to the window scan).
#[test]
fn reach_index_matches_bfs() {
    let mut rng = SmallRng::seed_from_u64(77);
    for f in fixtures() {
        let dag = GraphDag::new(&f.cg, sweep());
        let idx = ReachIndex::build(&dag, 4, &mut quiet()).unwrap();
        let mut s = Scratch::new();
        let (mut yes, mut total) = (0, 0);
        for x in sample(&mut rng, f.n, 100) {
            let cone = bfs(f.g, &[x], None);
            let members: Vec<usize> = (0..f.n).filter(|&y| cone[y]).collect();
            for j in 0..100 {
                let y = if j % 2 == 0 {
                    members[rng.random_range(0..members.len())]
                } else {
                    rng.random_range(0..f.n)
                };
                let budget = if j % 10 == 0 { 0 } else { 1 << 20 };
                let (r, how, _) = reach(&idx, &dag, None, x, y, &QueryConf { budget }, &mut s);
                assert_eq!(r, cone[y], "{}: {x} -> {y} decided by {how:?}", f.name);
                yes += usize::from(r);
                total += 1;
            }
        }
        assert_eq!(total, 10_000);
        assert!(yes >= 5_000, "{}: positive queries {yes}", f.name);
    }
}

/// HyperLogLog ancestor/descendant counts (p = 10) are within 5σ (σ = rsd · exact count, plus
/// 2 nodes of absolute slack for tiny cones) of exact BFS cone sizes on 100 sampled nodes.
#[test]
fn hll_counts_are_within_five_sigma_of_exact_cones() {
    let mut rng = SmallRng::seed_from_u64(5);
    for f in fixtures() {
        let conf = SketchConf {
            log2m: 10,
            hasher: SeededBuildHasher { seed: 2000 },
            window: Window::full(f.n),
            sweep: sweep(),
        };
        let c = counts_generic(
            &f.cg,
            Which::Both,
            &conf,
            &mut MemBudget::unlimited(),
            &mut quiet(),
        )
        .unwrap();
        let (anc, desc) = (c.anc.unwrap(), c.desc.unwrap());
        let sigma = rsd(10);
        for v in sample(&mut rng, f.n, 100) {
            let exact_desc = bfs(f.g, &[v], None).iter().filter(|&&b| b).count() - 1;
            let exact_anc = bfs(&f.cgt.graph, &[v], None).iter().filter(|&&b| b).count() - 1;
            for (what, est, exact) in [("desc", desc[v], exact_desc), ("anc", anc[v], exact_anc)] {
                let tol = 5.0 * sigma * exact as f64 + 2.0;
                assert!(
                    (est as f64 - exact as f64).abs() <= tol,
                    "{}: {what}({v}) = {est}, exact {exact}",
                    f.name
                );
            }
        }
    }
}

/// Transitive reduction of a window by search preserves reachability from 50 sampled nodes,
/// keeps a subgraph, and the removed arcs are exactly the redundant arcs reported by the
/// H3-then-exact verifier.
#[test]
fn transitive_reduction_preserves_closure_and_matches_the_verifier() {
    let mut rng = SmallRng::seed_from_u64(9);
    for f in fixtures() {
        let w = Window {
            lo: 1_000,
            hi: f.n.min(101_000),
        };
        let csr = Csr::from_window(&f.cg, w, &mut cquiet()).unwrap();
        let tr = transitive_reduction_search(&csr);
        let arcs = |c: &Csr| -> Vec<(usize, usize)> {
            (0..c.n)
                .flat_map(|u| c.succ(u).iter().map(move |&v| (u, v as usize)))
                .collect()
        };
        let (ga, ta) = (arcs(&csr), arcs(&tr));
        let removed: Vec<(u32, u32)> = ga
            .iter()
            .filter(|a| ta.binary_search(a).is_err())
            .map(|&(u, v)| ((u + w.lo) as u32, (v + w.lo) as u32))
            .collect();
        assert_eq!(
            ga.len() - ta.len(),
            removed.len(),
            "{}: TR is a subgraph",
            f.name
        );
        let (gv, tv) = (csr.to_vec_graph(), tr.to_vec_graph());
        for x in sample(&mut rng, csr.n, 50) {
            assert_eq!(
                bfs(&gv, &[x], None),
                bfs(&tv, &[x], None),
                "{}: closure of {x}",
                f.name
            );
        }
        let (rep, mut red) = verify_reduced_window(
            &f.cg,
            w,
            None,
            &sweep(),
            &mut MemBudget::unlimited(),
            &mut cquiet(),
        )
        .unwrap();
        red.sort_unstable();
        assert_eq!(red, removed, "{}: verifier vs search", f.name);
        assert_eq!(rep.redundant, removed.len() as u64);
        assert_eq!(rep.arcs, ga.len() as u64);
        assert_eq!(rep.is_transitively_reduced, removed.is_empty());
        assert!(
            !rep.line_digraph,
            "{}: cnr-2000 is not a line digraph",
            f.name
        );
        assert!(
            !removed.is_empty(),
            "{}: a web graph has redundant arcs",
            f.name
        );
    }
}

/// `frontier ≤ width ≤ MPC_d`, `frontier ≥ #sinks` (the cut at `n`), and the explicit
/// Hopcroft–Karp path cover is valid.
#[test]
fn frontier_is_at_most_the_path_cover() {
    for f in fixtures() {
        let fr = frontier_generic(&f.cg, Window::full(f.n), true, &mut cquiet()).unwrap();
        let csr = Csr::from_vec_graph(f.g);
        let pc = min_path_cover_hk(&csr);
        let paths = check_path_cover(&csr, &pc).unwrap();
        assert_eq!(paths, pc.num_paths());
        let sinks = (0..f.n).filter(|&u| f.g.outdegree(u) == 0).count() as u64;
        assert!(sinks <= fr.size, "{}", f.name);
        assert!(
            fr.size <= pc.num_paths() as u64,
            "{}: {} > {}",
            f.name,
            fr.size,
            pc.num_paths()
        );
        let members = fr.members.unwrap();
        assert_eq!(members.len() as u64, fr.size);
        // A_c is an antichain: no member has a successor among the members (all successors of a
        // member lie at or after the cut, all members before it)
        for &u in &members {
            assert!((u as usize) < fr.cut);
            assert!(f.g.successors(u as usize).into_iter().all(|v| v >= fr.cut));
        }
    }
}

/// cnr-2000 is not a union of bicliques: the PG fast path refuses it.
#[test]
fn block_index_is_refused() {
    for f in fixtures() {
        let err = BlockIndex::build(&f.cg, &mut cquiet()).unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<DagError>(),
                Some(DagError::NotBiclique { .. })
            ),
            "{}: {err}",
            f.name
        );
    }
}
