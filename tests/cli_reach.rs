//! Command-line tests of `dag reach-index build|query|bench`.

#[path = "support/cli.rs"]
mod cli;

use std::path::Path;

use cli::{p, read_json, run_expect, run_ok};
use dag::io::{self, LoadMode};
use dag::reach::index::ReachIndex;
use dag::testkit::*;
use webgraph::prelude::*;

fn reaches(cl: &[Vec<u64>], u: usize, v: usize) -> bool {
    u == v || bit(&cl[u], v)
}

/// All pairs of `[lo, hi)` as a `.u32x2le`-style pair file.
fn write_pairs(path: &Path, lo: usize, hi: usize) -> Vec<(u32, u32)> {
    let pairs: Vec<(u32, u32)> = (lo..hi)
        .flat_map(|u| (lo..hi).map(move |v| (u as u32, v as u32)))
        .collect();
    io::write_array(path, &pairs).unwrap();
    pairs
}

fn check_answers(prefix: &Path, pairs: &[(u32, u32)], cl: &[Vec<u64>]) {
    let ans: Vec<u8> = io::read_array(&io::suffixed(prefix, "answers.u8")).unwrap();
    assert_eq!(ans.len(), pairs.len());
    for (&(u, v), &a) in pairs.iter().zip(&ans) {
        assert_eq!(a == 1, reaches(cl, u as usize, v as usize), "({u}, {v})");
    }
    let json = read_json(&io::suffixed(prefix, "json"));
    let total: u64 = json["results"]["deciders"]
        .as_object()
        .unwrap()
        .values()
        .map(|x| x.as_u64().unwrap())
        .sum();
    assert_eq!(total as usize, pairs.len());
}

#[test]
fn graph_mode_build_query_bench() {
    let dir = tempfile::tempdir().unwrap();
    let g = random_dag(120, 0.03, 4);
    let cl = closure(&g);
    let base = p(&write_bv(dir.path(), "g", &g));
    let idx = dir.path().join("idx");
    // `auto` falls back to the graph: a random DAG is not a biclique digraph
    run_ok(&[
        "--chunk-arcs",
        "13",
        "reach-index",
        "build",
        "--graph",
        &base,
        "--out",
        &p(&idx),
        "--chains",
        "3",
    ]);
    let json = read_json(&io::suffixed(&idx, "json"));
    assert_eq!(json["params"]["mode"], "Graph");
    assert_eq!(json["results"]["stats"]["n_dag"], 120);
    for f in json["files"].as_array().unwrap() {
        assert!(Path::new(f["path"].as_str().unwrap()).exists());
    }
    let loaded = ReachIndex::load(&idx, LoadMode::Mmap).unwrap();
    assert_eq!(loaded.n_dag, 120);
    // --pair prints one JSON line
    for (u, v) in [(0usize, 119usize), (5, 5), (100, 3), (17, 64)] {
        let o = run_ok(&[
            "reach-index",
            "query",
            "--graph",
            &base,
            "--blocks",
            "none",
            "--index",
            &p(&idx),
            "--pair",
            &u.to_string(),
            &v.to_string(),
        ]);
        let line: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
        assert_eq!(line["reachable"], reaches(&cl, u, v), "({u}, {v})");
    }
    // --pairs with --out, two budgets
    let pf = dir.path().join("pairs.u32x2le");
    let pairs = write_pairs(&pf, 0, 120);
    for budget in ["0", "1048576"] {
        let q = dir.path().join(format!("q{budget}"));
        run_ok(&[
            "reach-index",
            "query",
            "--graph",
            &base,
            "--index",
            &p(&idx),
            "--pairs",
            &p(&pf),
            "--budget",
            budget,
            "--out",
            &p(&q),
        ]);
        check_answers(&q, &pairs, &cl);
    }
    // --pairs without --out prints `u v answer decider visited`
    let small = dir.path().join("small.u32x2le");
    let sp = write_pairs(&small, 30, 40);
    let o = run_ok(&[
        "reach-index",
        "query",
        "--graph",
        &base,
        "--index",
        &p(&idx),
        "--pairs",
        &p(&small),
    ]);
    let text = String::from_utf8(o.stdout).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), sp.len());
    for (line, &(u, v)) in lines.iter().zip(&sp) {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(
            (f[0], f[1]),
            (u.to_string().as_str(), v.to_string().as_str())
        );
        assert_eq!(f[2] == "1", reaches(&cl, u as usize, v as usize));
    }
    // bench: 100 % agreement, exit 0
    let b = dir.path().join("bench");
    run_ok(&[
        "reach-index",
        "bench",
        "--graph",
        &base,
        "--index",
        &p(&idx),
        "--sources",
        "10",
        "--queries",
        "500",
        "--seed",
        "7",
        "--out",
        &p(&b),
    ]);
    let json = read_json(&io::suffixed(&b, "json"));
    for w in json["results"]["workloads"].as_array().unwrap() {
        assert_eq!(w["disagreements"], 0);
    }
    // bench --budget 0 (regression): every undecided query takes the window scan
    let b0 = dir.path().join("bench0");
    run_ok(&[
        "reach-index",
        "bench",
        "--graph",
        &base,
        "--index",
        &p(&idx),
        "--sources",
        "10",
        "--queries",
        "500",
        "--budget",
        "0",
        "--out",
        &p(&b0),
    ]);
    let json = read_json(&io::suffixed(&b0, "json"));
    assert_eq!(json["params"]["conf"]["query"]["budget"], 0);
    let mut scans = 0;
    for w in json["results"]["workloads"].as_array().unwrap() {
        assert_eq!(w["disagreements"], 0);
        let d = w["deciders"].as_object().unwrap();
        assert!(
            !d.contains_key("Search"),
            "budget 0 must skip the search: {d:?}"
        );
        scans += d.get("WindowScan").and_then(|v| v.as_u64()).unwrap_or(0);
    }
    assert!(scans > 0, "no query reached the window scan");
    // the labels are charged to --max-mem before they are loaded (regression):
    // with a truncated label file and a tiny budget the run fails on the budget, not on the file
    let trunc = dir.path().join("trunc");
    for f in std::fs::read_dir(dir.path()).unwrap() {
        let f = f.unwrap().path();
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        if let Some(rest) = name.strip_prefix("idx.reach.") {
            std::fs::copy(&f, dir.path().join(format!("trunc.reach.{rest}"))).unwrap();
        }
    }
    std::fs::write(io::suffixed(&trunc, "reach.lvl.u32le"), [0u8; 4]).unwrap();
    let o = run_expect(
        &[
            "--max-mem",
            "0.000000001",
            "reach-index",
            "query",
            "--graph",
            &base,
            "--index",
            &p(&trunc),
            "--pair",
            "0",
            "1",
        ],
        2,
    );
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("exceeds --max-mem"), "{err}");
    assert!(!err.contains("entries, expected"), "{err}");
    // errors: both or neither of --pair/--pairs, a pair outside the graph
    run_expect(
        &[
            "reach-index",
            "query",
            "--graph",
            &base,
            "--index",
            &p(&idx),
        ],
        2,
    );
    run_expect(
        &[
            "reach-index",
            "query",
            "--graph",
            &base,
            "--index",
            &p(&idx),
            "--pair",
            "0",
            "500",
        ],
        2,
    );
}

#[test]
fn blocks_mode_with_window() {
    let dir = tempfile::tempdir().unwrap();
    let pg = mini_pg(200, 3, &MiniPgOpts::default());
    let g = &pg.graph;
    let n = g.num_nodes();
    let cl = closure(g);
    let base = p(&write_bv(dir.path(), "pg", g));
    // a saved BlockIndex
    let blk = dir.path().join("blk");
    run_ok(&["blocks", "--graph", &base, "--out", &p(&blk)]);
    let (lo, hi) = (n / 4, 3 * n / 4);
    let window = format!("{lo}..{hi}");
    let idx = dir.path().join("idx");
    run_ok(&[
        "reach-index",
        "build",
        "--graph",
        &base,
        "--blocks",
        &p(&blk),
        "--window",
        &window,
        "--out",
        &p(&idx),
    ]);
    let json = read_json(&io::suffixed(&idx, "json"));
    assert_eq!(json["params"]["mode"], "Blocks");
    assert_eq!(json["window"]["lo"], lo);
    // the same index from `--blocks auto`
    let idx2 = dir.path().join("idx2");
    run_ok(&[
        "reach-index",
        "build",
        "--graph",
        &base,
        "--window",
        &window,
        "--out",
        &p(&idx2),
    ]);
    for f in ReachIndex::files(&idx, 4) {
        let name = f
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .replacen("idx", "idx2", 1);
        assert_eq!(
            std::fs::read(&f).unwrap(),
            std::fs::read(dir.path().join(name)).unwrap()
        );
    }
    let pf = dir.path().join("pairs.u32x2le");
    let pairs = write_pairs(&pf, lo, hi);
    for blocks in [p(&blk), "auto".to_owned()] {
        let q = dir.path().join("q");
        run_ok(&[
            "--load",
            "mmap",
            "reach-index",
            "query",
            "--graph",
            &base,
            "--blocks",
            &blocks,
            "--index",
            &p(&idx),
            "--pairs",
            &p(&pf),
            "--budget",
            "2",
            "--out",
            &p(&q),
        ]);
        check_answers(&q, &pairs, &cl);
    }
    // a blocks index cannot be queried without a BlockIndex, nor outside its window
    run_expect(
        &[
            "reach-index",
            "query",
            "--graph",
            &base,
            "--blocks",
            "none",
            "--index",
            &p(&idx),
            "--pairs",
            &p(&pf),
        ],
        2,
    );
    run_expect(
        &[
            "reach-index",
            "query",
            "--graph",
            &base,
            "--index",
            &p(&idx),
            "--pair",
            "0",
            &(n - 1).to_string(),
        ],
        2,
    );
    let b = dir.path().join("bench");
    run_ok(&[
        "reach-index",
        "bench",
        "--graph",
        &base,
        "--blocks",
        &p(&blk),
        "--index",
        &p(&idx),
        "--sources",
        "70",
        "--queries",
        "1000",
        "--out",
        &p(&b),
    ]);
    let json = read_json(&io::suffixed(&b, "json"));
    assert_eq!(json["results"]["sources"].as_array().unwrap().len(), 70);
    for w in json["results"]["workloads"].as_array().unwrap() {
        assert_eq!(w["disagreements"], 0);
    }
}

/// Regression test: `bench` exits 1 when answers disagree with
/// the cones. The index is built on the path `0 → 1 → … → 59` and benchmarked against a DAG
/// with the same nodes and arc count (`0 → 1` plus the even and odd paths `i → i + 2`), where
/// `x ≥ 1` never reaches a `y` of the other parity, but the path index says it does.
#[test]
fn bench_exits_1_on_disagreement() {
    let dir = tempfile::tempdir().unwrap();
    let n = 60;
    let path = vec_graph(n, (0..n - 1).map(|i| (i, i + 1)));
    let two = vec_graph(
        n,
        std::iter::once((0, 1)).chain((0..n - 2).map(|i| (i, i + 2))),
    );
    assert_eq!(path.num_arcs(), two.num_arcs());
    let a = p(&write_bv(dir.path(), "a", &path));
    let b = p(&write_bv(dir.path(), "b", &two));
    let idx = dir.path().join("idx");
    run_ok(&[
        "reach-index",
        "build",
        "--graph",
        &a,
        "--blocks",
        "none",
        "--out",
        &p(&idx),
        "--chains",
        "1",
    ]);
    let out = dir.path().join("bench");
    run_expect(
        &[
            "reach-index",
            "bench",
            "--graph",
            &b,
            "--blocks",
            "none",
            "--index",
            &p(&idx),
            "--sources",
            "8",
            "--queries",
            "200",
            "--out",
            &p(&out),
        ],
        1,
    );
    let json = read_json(&io::suffixed(&out, "json"));
    let bal = &json["results"]["workloads"][1];
    assert!(bal["disagreements"].as_u64().unwrap() > 0);
    assert!(!bal["first_disagreement"].is_null());
}
