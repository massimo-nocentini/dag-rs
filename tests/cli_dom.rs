//! `dag dom build|verify|stats` on BV graphs written by the test kit.

#[path = "support/cli.rs"]
mod cli;

use std::path::Path;

use cli::{p, read_json, run_expect, run_ok};
use dag::io::{array_path, read_array, write_array};
use dag::testkit::*;
use dag::{NONE, Window};
use webgraph::prelude::*;

fn brute(g: &VecGraph) -> Vec<u32> {
    brute_dominators(g)
        .into_iter()
        .map(|d| d.map_or(NONE, |d| d as u32))
        .collect()
}

fn brute_post(g: &VecGraph) -> Vec<u32> {
    let n = g.num_nodes();
    let r = vec_graph(n, arcs(g).into_iter().map(|(u, v)| (n - 1 - v, n - 1 - u)));
    let d = brute_dominators(&r);
    (0..n)
        .map(|u| d[n - 1 - u].map_or(NONE, |x| (n - 1 - x) as u32))
        .collect()
}

fn arr(prefix: &Path, field: &str) -> Vec<u32> {
    read_array::<u32>(&array_path::<u32>(prefix, field)).unwrap()
}

#[test]
fn build_verify_stats_on_mini_pg() {
    let dir = tempfile::tempdir().unwrap();
    let g = mini_pg(60, 4, &MiniPgOpts::default()).graph;
    let n = g.num_nodes();
    let base = write_bv(dir.path(), "g", &g);
    let blk = dir.path().join("b");
    run_ok(&["blocks", "--graph", &p(&base), "--out", &p(&blk)]);

    // blocks (loaded), blocks (auto) and generic builders agree with the oracle
    let pb = dir.path().join("pb");
    let pa = dir.path().join("pa");
    let pn = dir.path().join("pn");
    for (out, blocks) in [(&pb, p(&blk)), (&pa, "auto".into()), (&pn, "none".into())] {
        let o = p(out);
        run_ok(&[
            "--chunk-arcs",
            "7",
            "dom",
            "build",
            "--graph",
            &p(&base),
            "--blocks",
            &blocks,
            "--out",
            &o,
            "--sizes",
            "--pre",
            "--depth-bound",
            "--budget",
            "2",
        ]);
        assert_eq!(arr(out, "idom"), brute(&g), "{blocks}");
        assert_eq!(arr(out, "ipdom"), brute_post(&g), "{blocks}");
        let j = read_json(&out.with_extension("json"));
        assert_eq!(j["params"]["blocks"], blocks != "none");
        assert_eq!(j["results"]["dom"]["depth_bound"]["violations"], 0);
        assert_eq!(j["results"]["pdom"]["stats"]["root_children_size_sum"], n);
        assert!(j["results"]["dom"]["nca"]["calls"].is_u64());
    }
    for f in [
        "idom",
        "ipdom",
        "dom.depth",
        "dom.size",
        "dom.pre",
        "pdom.depth",
        "pdom.size",
    ] {
        assert_eq!(arr(&pb, f), arr(&pn, f), "{f}");
        assert_eq!(arr(&pa, f), arr(&pn, f), "{f}");
    }

    // verify: exit 0 on the built trees, 1 on a corrupted one
    let rep = dir.path().join("v");
    run_ok(&[
        "dom",
        "verify",
        "--graph",
        &p(&base),
        "--idom",
        &p(&array_path::<u32>(&pb, "idom")),
        "--out",
        &p(&rep),
    ]);
    assert_eq!(
        read_json(&rep.with_extension("json"))["results"]["report"]["ok"],
        true
    );
    run_ok(&[
        "dom",
        "verify",
        "--graph",
        &p(&base),
        "--post",
        "--idom",
        &p(&array_path::<u32>(&pb, "ipdom")),
        "--out",
        &p(&rep),
    ]);
    let mut bad = arr(&pb, "idom");
    let v = (0..n).rev().find(|&v| bad[v] != NONE).unwrap();
    bad[v] = NONE;
    let bad_path = dir.path().join("bad.u32le");
    write_array(&bad_path, &bad).unwrap();
    run_expect(
        &[
            "dom",
            "verify",
            "--graph",
            &p(&base),
            "--idom",
            &p(&bad_path),
            "--out",
            &p(&rep),
        ],
        1,
    );
    let j = read_json(&rep.with_extension("json"));
    assert_eq!(j["results"]["report"]["ok"], false);
    assert_eq!(j["results"]["report"]["first_failure"][0], v);
    // a wrong-length array is an error (exit 2)
    write_array(&bad_path, &bad[..n - 1]).unwrap();
    run_expect(
        &[
            "dom",
            "verify",
            "--graph",
            &p(&base),
            "--idom",
            &p(&bad_path),
            "--out",
            &p(&rep),
        ],
        2,
    );

    // stats from the files
    run_ok(&["dom", "stats", "--prefix", &p(&pb), "--top", "5"]);
    let s = read_json(&dir.path().join("pb.stats.json"));
    assert_eq!(s["results"]["dom"]["root_children_size_sum"], n);
    assert_eq!(s["results"]["pdom"]["root_children_size_sum"], n);
    let j = read_json(&pb.with_extension("json"));
    assert_eq!(
        s["results"]["dom"]["cut_nodes"],
        j["results"]["dom"]["stats"]["cut_nodes"]
    );
    let tsv = std::fs::read_to_string(dir.path().join("pb.dom.top.tsv")).unwrap();
    assert!(tsv.starts_with("node\tsize\tdepth\tidom\n"));
    assert_eq!(tsv.lines().count(), 6.min(n + 1));
}

#[test]
fn windowed_build_and_only() {
    let dir = tempfile::tempdir().unwrap();
    let g = random_dag(120, 0.03, 8);
    let base = write_bv(dir.path(), "r", &g);
    let w = Window { lo: 30, hi: 100 };
    let h = vec_graph(
        w.len(),
        arcs(&g)
            .into_iter()
            .filter(|&(u, v)| w.contains(u) && w.contains(v))
            .map(|(u, v)| (u - w.lo, v - w.lo)),
    );
    let shift = |a: Vec<u32>| -> Vec<u32> {
        a.into_iter()
            .map(|x| if x == NONE { NONE } else { x + w.lo as u32 })
            .collect()
    };
    let out = dir.path().join("w");
    // the random DAG is not a biclique digraph: `auto` falls back to the generic builders
    run_ok(&[
        "dom",
        "build",
        "--graph",
        &p(&base),
        "--window",
        "30..100",
        "--out",
        &p(&out),
        "--only",
        "post",
    ]);
    assert!(!array_path::<u32>(&out, "idom").exists());
    assert_eq!(arr(&out, "ipdom"), shift(brute_post(&h)));
    run_ok(&[
        "dom",
        "build",
        "--graph",
        &p(&base),
        "--window",
        "30..100",
        "--out",
        &p(&out),
        "--only",
        "dom",
        "--blocks",
        "none",
    ]);
    assert_eq!(arr(&out, "idom"), shift(brute(&h)));
    let rep = dir.path().join("v");
    run_ok(&[
        "dom",
        "verify",
        "--graph",
        &p(&base),
        "--window",
        "30..100",
        "--idom",
        &p(&array_path::<u32>(&out, "idom")),
        "--out",
        &p(&rep),
    ]);
    // the full-graph verifier rejects the window tree (wrong length: error)
    run_expect(
        &[
            "dom",
            "verify",
            "--graph",
            &p(&base),
            "--idom",
            &p(&array_path::<u32>(&out, "idom")),
            "--out",
            &p(&rep),
        ],
        2,
    );
    run_ok(&["dom", "stats", "--prefix", &p(&out)]);
    // over budget: exit 2
    run_expect(
        &[
            "--max-mem",
            "0.000001",
            "dom",
            "build",
            "--graph",
            &p(&base),
            "--blocks",
            "none",
            "--out",
            &p(&out),
        ],
        2,
    );
}

/// Regression: `dom stats --prefix P` reads only the arrays written by the last
/// `dom build --out P`, not the ones left at `P` by an earlier run on another window.
#[test]
fn stats_ignore_stale_arrays() {
    let dir = tempfile::tempdir().unwrap();
    let g = mini_pg(40, 3, &MiniPgOpts::default()).graph;
    let n = g.num_nodes();
    assert!(n >= 40, "mini-PG too small: {n}");
    let base = write_bv(dir.path(), "g", &g);
    let out = dir.path().join("p");
    let build = |extra: &[&str]| {
        let mut a = vec!["dom".into(), "build".into(), "--graph".into(), p(&base)];
        a.extend(["--out".into(), p(&out)]);
        a.extend(extra.iter().map(|s| s.to_string()));
        run_ok(&a);
    };
    let half = n / 2;
    let w1 = format!("0..{half}");
    let w2 = format!("{half}..{n}");
    // both trees and sizes on the first half, then only dominators without sizes on the second
    build(&["--window", &w1, "--sizes"]);
    build(&["--window", &w2, "--only", "dom"]);
    assert!(array_path::<u32>(&out, "ipdom").exists());
    assert!(array_path::<u32>(&out, "dom.size").exists());
    run_ok(&["dom", "stats", "--prefix", &p(&out)]);
    let s = read_json(&dir.path().join("p.stats.json"));
    assert_eq!(s["results"]["dom"]["root_children_size_sum"], n - half);
    assert!(s["results"].get("pdom").is_none());
    assert!(!dir.path().join("p.pdom.top.tsv").exists());
}
