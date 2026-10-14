//! Command-line tests of the aggregate measures: `--buckets FILE --top K` on every producing
//! subcommand, `dag aggregate` on their stored arrays (which must reproduce the in-run
//! distributions), the frontier curve, and the run metadata (`phases_s`, `mem`, `rss`).

#[path = "support/cli.rs"]
mod cli;

use std::path::{Path, PathBuf};

use cli::{p, read_json, run_ok};
use dag::blocks::BlockIndex;
use dag::testkit::{self, MiniPgOpts};
use dag::width::frontier_curve_blocks;
use serde_json::Value;
use webgraph::prelude::*;

fn json(prefix: &Path) -> Value {
    read_json(&dag::io::suffixed(prefix, "json"))
}

/// A `chunk_boundaries.tsv`-style file with three "half-years" covering `[0, n)`.
fn bucket_file(dir: &Path, n: usize) -> (PathBuf, Vec<(usize, usize)>) {
    let cuts = [0, n / 3, 2 * n / 3, n];
    let mut s = String::from("chunk_id\tstart_date\tfirst_node\tlast_node_exclusive\n");
    let mut ranges = Vec::new();
    for (i, w) in cuts.windows(2).enumerate() {
        let half = if i % 2 == 0 { "01" } else { "07" };
        s.push_str(&format!(
            "{}\t{}-{half}-01 00:00:00 Europe/Rome\t{}\t{}\n",
            i + 1,
            2009 + i / 2,
            w[0],
            w[1]
        ));
        ranges.push((w[0], w[1]));
    }
    let path = dir.join("chunks.tsv");
    std::fs::write(&path, s).unwrap();
    (path, ranges)
}

/// Checks that `d` (a serialized `DistReport` or `ValueSummary`) has one bucket per range whose
/// counts add up to `total`, and returns the buckets.
fn check_buckets(d: &Value, ranges: &[(usize, usize)], total: u64, count_key: &str) -> Vec<Value> {
    let b = d["by_bucket"].as_array().expect("by_bucket").clone();
    assert_eq!(b.len(), ranges.len());
    let mut sum = 0;
    for (x, &(lo, hi)) in b.iter().zip(ranges) {
        assert_eq!(
            (x["lo"].as_u64(), x["hi"].as_u64()),
            (Some(lo as u64), Some(hi as u64))
        );
        sum += x["count"].as_u64().unwrap() + x["absent"].as_u64().unwrap();
    }
    assert_eq!(
        sum + d["unbucketed"].as_u64().unwrap(),
        total,
        "{count_key}"
    );
    assert_eq!(b[0]["label"], "2009H1");
    assert_eq!(b[1]["label"], "2009H2");
    b
}

fn check_metadata(j: &Value, phases: &[&str]) {
    for ph in phases {
        assert!(
            j["phases_s"][ph].as_f64().is_some(),
            "phase {ph} missing in {}",
            j["phases_s"]
        );
    }
    assert!(j["mem"]["max_bytes"].is_u64());
    assert!(j["rss"].is_object());
}

#[test]
fn every_producer_reports_distributions_per_bucket() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let g = testkit::mini_pg(400, 21, &MiniPgOpts::default()).graph;
    let n = g.num_nodes();
    let base = testkit::write_bv(d, "pg", &g);
    let (buckets, ranges) = bucket_file(d, n);
    let bk = ["--buckets", &p(&buckets), "--top", "5"];
    let (idx, _) = BlockIndex::from_vec_graph(&g).unwrap();

    // blocks: one entry per block, bucketed by its first output
    let blk = d.join("b");
    run_ok(
        &[
            &["blocks", "--graph", &p(&base), "--out", &p(&blk)][..],
            &bk,
        ]
        .concat(),
    );
    let j = json(&blk);
    check_metadata(&j, &["open", "blocks", "save", "summary"]);
    let dl = &j["results"]["distributions"]["len"];
    assert_eq!(dl["count"].as_u64().unwrap() as usize, idx.nb);
    check_buckets(dl, &ranges, idx.nb as u64, "blocks len");
    assert_eq!(dl["top"].as_array().unwrap().len(), 5.min(idx.nb));
    let top0 = &dl["top"][0];
    let b0 = top0["id"].as_u64().unwrap() as usize;
    assert_eq!(
        top0["node"].as_u64().unwrap() as usize,
        idx.first[b0] as usize
    );
    assert_eq!(
        j["results"]["distributions"]["arcs"]["sum"]
            .as_f64()
            .unwrap() as u64,
        g.num_arcs()
    );

    // dp: the summary of the values, overall and per bucket; aggregate reproduces it
    let dp = d.join("dmax");
    run_ok(
        &[
            &[
                "dp",
                "--graph",
                &p(&base),
                "--semiring",
                "maxplus-u32",
                "--out",
                &p(&dp),
            ][..],
            &bk,
        ]
        .concat(),
    );
    let j = json(&dp);
    check_metadata(&j, &["open", "kernel", "summary", "write"]);
    let s = &j["results"]["summary"];
    check_buckets(s, &ranges, n as u64, "dp");
    let agg = d.join("dmax.agg");
    run_ok(
        &[
            &[
                "aggregate",
                "--values",
                &p(&dag::io::suffixed(&dp, "val.u32le")),
                "--absent",
                "max",
                "--out",
                &p(&agg),
            ][..],
            &bk,
        ]
        .concat(),
    );
    let a = &json(&agg)["results"];
    for (k_dp, k_agg) in [
        ("max", "max"),
        ("argmax", "argmax"),
        ("mean", "mean"),
        ("zero_nodes", "absent"),
        ("log2_hist", "log2_hist"),
        ("top", "top"),
        ("by_bucket", "by_bucket"),
    ] {
        assert_eq!(s[k_dp], a[k_agg], "dp {k_dp} vs aggregate {k_agg}");
    }

    // cpm: E/H/slack distributions; critical_e counts the critical nodes
    let cpm = d.join("cpm");
    run_ok(
        &[
            &[
                "cpm",
                "--graph",
                &p(&base),
                "--slack",
                "--path",
                "--out",
                &p(&cpm),
            ][..],
            &bk,
        ]
        .concat(),
    );
    let j = json(&cpm);
    check_metadata(&j, &["open", "kernel", "summary", "write", "compare"]);
    let r = &j["results"];
    let crit = &r["distributions"]["critical_e"];
    assert_eq!(crit["count"], r["summary"]["critical_nodes"]);
    check_buckets(crit, &ranges, n as u64, "cpm critical");
    assert!(r["summary"]["critical_width_log2_hist"].is_array());
    assert!(r["longest_path"]["nodes"].as_u64().unwrap() >= 1);
    // the slack distribution equals aggregate over the written slack array
    let agg = d.join("slack.agg");
    run_ok(
        &[
            &[
                "aggregate",
                "--values",
                &p(&dag::io::suffixed(&cpm, "slack.u32le")),
                "--out",
                &p(&agg),
            ][..],
            &bk,
        ]
        .concat(),
    );
    assert_eq!(json(&agg)["results"], r["distributions"]["slack"]);

    // count-paths --summary-only: the distributions survive without any array
    let cnt = d.join("count");
    run_ok(
        &[
            &[
                "count-paths",
                "--graph",
                &p(&base),
                "--summary-only",
                "--out",
                &p(&cnt),
            ][..],
            &bk,
        ]
        .concat(),
    );
    let j = json(&cnt);
    let dd = &j["results"]["distributions"];
    for k in ["forward_log2", "backward_log2", "through_log2"] {
        check_buckets(&dd[k], &ranges, n as u64, k);
    }
    assert_eq!(
        dd["forward_log2"]["max"], j["results"]["summary"]["forward"]["max_log2"],
        "the largest forward count"
    );

    // dom build and dom stats: the same distributions
    let dom = d.join("dom");
    run_ok(
        &[
            &[
                "dom",
                "build",
                "--graph",
                &p(&base),
                "--sizes",
                "--out",
                &p(&dom),
            ][..],
            &bk,
        ]
        .concat(),
    );
    let j = json(&dom);
    for tree in ["dom", "pdom"] {
        let t = &j["results"][tree];
        assert_eq!(
            t["distributions"]["root_children"]["count"],
            t["stats"]["root_children"]
        );
        check_buckets(&t["distributions"]["depth"], &ranges, n as u64, tree);
    }
    run_ok(&[
        "dom",
        "stats",
        "--prefix",
        &p(&dom),
        "--top",
        "5",
        "--buckets",
        &p(&buckets),
    ]);
    let st = read_json(&dag::io::suffixed(&dom, "stats.json"));
    for tree in ["dom", "pdom"] {
        assert_eq!(
            st["results"][tree]["distributions"], j["results"][tree]["distributions"],
            "{tree}: dom stats vs dom build"
        );
    }

    // reach-counts: absolute ids, block arrays with their first output
    let rc = d.join("rc");
    run_ok(
        &[
            &[
                "reach-counts",
                "--graph",
                &p(&base),
                "--window",
                &format!("{}..{n}", n / 4),
                "--out",
                &p(&rc),
            ][..],
            &bk,
        ]
        .concat(),
    );
    let j = json(&rc);
    let anc = &j["results"]["anc"];
    let am = anc["argmax"].as_u64().unwrap() as usize;
    assert!(am >= n / 4 && am < n, "argmax {am} is absolute");
    let ba = &j["results"]["blk_anc"];
    let bm = ba["argmax"].as_u64().unwrap() as usize;
    assert_eq!(
        ba["argmax_node"].as_u64().unwrap() as usize,
        idx.first[bm] as usize
    );
    assert!(anc["by_bucket"].is_array());

    // frontier: the curve per bucket equals the library's
    let fr = d.join("frontier");
    // cuts: every 50th node (a stand-in for the first node of every blockchain block)
    let cuts: Vec<u64> = (0..=n as u64).step_by(50).collect();
    let cuts_file = d.join("cuts.u64le");
    dag::io::write_array(&cuts_file, &cuts).unwrap();
    run_ok(&[
        "frontier",
        "--graph",
        &p(&base),
        "--buckets",
        &p(&buckets),
        "--profile-step",
        "100",
        "--cuts",
        &p(&cuts_file),
        "--out",
        &p(&fr),
    ]);
    let at: Vec<u64> = dag::io::read_array(&dag::io::suffixed(&fr, "cuts.u64le")).unwrap();
    let cuts_us: Vec<usize> = cuts.iter().map(|&c| c as usize).collect();
    assert_eq!(
        at,
        dag::width::frontier_at_cuts_blocks(&idx, dag::Window::full(n), &cuts_us).unwrap()
    );
    let j = json(&fr);
    let curve = &j["results"]["curve"];
    let want = frontier_curve_blocks(
        &idx,
        dag::Window::full(n),
        &ranges
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| (format!("x{i}"), a, b))
            .collect::<Vec<_>>(),
    );
    let got = curve["by_bucket"].as_array().unwrap();
    assert_eq!(got.len(), 3);
    for (g, w) in got.iter().zip(&want) {
        for k in ["at_lo", "at_hi", "max", "argmax"] {
            let wv = serde_json::to_value(w).unwrap();
            assert_eq!(g[k], wv[k], "frontier curve {k}");
        }
    }
    assert_eq!(curve["profile"].as_array().unwrap().len(), n.div_ceil(100));
    // the maximum over the profile is the frontier
    let pmax = curve["profile"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["max"].as_u64().unwrap())
        .max();
    assert_eq!(pmax, j["results"]["size"].as_u64());

    // path covers and antichains: chain-length and antichain-position statistics
    for (blocks, method) in [("auto", "blocks"), ("none", "hk")] {
        let pc = d.join(format!("pc-{method}"));
        run_ok(&[
            "path-cover",
            "--graph",
            &p(&base),
            "--blocks",
            blocks,
            "--explicit",
            "--out",
            &p(&pc),
        ]);
        let r = &json(&pc)["results"];
        assert_eq!(r["method"], method);
        assert_eq!(
            r["chains"]["count"], r["mpc_d"],
            "{method}: one head per path"
        );
        assert_eq!(r["chains"]["sum"].as_f64().unwrap() as usize, n);
    }
    let wd = d.join("width");
    run_ok(&[
        "width",
        "--graph",
        &p(&base),
        "--blocks",
        "none",
        "--method",
        "minflow",
        "--out",
        &p(&wd),
    ]);
    let r = &json(&wd)["results"];
    assert_eq!(r["chains"]["count"], r["width"]);
    assert_eq!(r["chains"]["sum"].as_f64().unwrap() as usize, n);
    assert_eq!(r["antichain_ids"]["count"], r["width"]);
    let wc = d.join("width-closure");
    run_ok(&[
        "width",
        "--graph",
        &p(&base),
        "--window",
        &format!("0..{}", n - 1),
        "--method",
        "closure",
        "--out",
        &p(&wc),
    ]);
    let r = &json(&wc)["results"];
    assert_eq!(r["antichain_ids"]["count"], r["width"]);
}
