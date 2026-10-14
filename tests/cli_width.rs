//! Command-line tests of `dag path-cover`, `frontier` and `width` on BV graphs
//! written by the test kit, compared with the library.

#[path = "support/cli.rs"]
mod cli;

use std::path::Path;

use cli::{p, read_json, run_expect, run_ok};
use dag::blocks::BlockIndex;
use dag::csr::Csr;
use dag::io::read_array;
use dag::testkit::{self, MiniPgOpts};
use dag::width::{
    check_antichain, check_antichain_blocks, frontier_blocks, min_path_cover_blocks, width_min_flow,
};
use dag::{NONE, Window};
use webgraph::prelude::*;

fn res(prefix: &Path) -> serde_json::Value {
    read_json(&dag::io::suffixed(prefix, "json"))["results"].clone()
}

fn window_arg(w: Window) -> String {
    format!("{}..{}", w.lo, w.hi)
}

#[test]
fn path_cover_blocks_and_hk_agree() {
    let dir = tempfile::tempdir().unwrap();
    let g = testkit::mini_pg(400, 3, &MiniPgOpts::default()).graph;
    let n = g.num_nodes();
    let base = testkit::write_bv(dir.path(), "pg", &g);
    let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
    for w in [
        Window::full(n),
        Window { lo: n / 4, hi: n },
        Window {
            lo: n / 3,
            hi: 2 * n / 3,
        },
    ] {
        let a = dir.path().join(format!("a{}", w.lo));
        let b = dir.path().join(format!("b{}", w.lo));
        run_ok(&[
            "path-cover",
            "--graph",
            &p(&base),
            "--window",
            &window_arg(w),
            "--explicit",
            "--out",
            &p(&a),
        ]);
        run_ok(&[
            "path-cover",
            "--graph",
            &p(&base),
            "--blocks",
            "none",
            "--window",
            &window_arg(w),
            "--explicit",
            "--out",
            &p(&b),
        ]);
        let (ra, rb) = (res(&a), res(&b));
        assert_eq!(ra["method"], "blocks");
        assert_eq!(rb["method"], "hk");
        assert_eq!(ra["mpc_d"], rb["mpc_d"]);
        assert_eq!(ra["cross_check"]["agree"], true);
        assert_eq!(rb["konig_ok"], true);
        let (stats, _) = min_path_cover_blocks(&blocks, w, false);
        assert_eq!(ra["mpc_d"].as_u64().unwrap(), stats.mpc_d);
        // both explicit covers are valid covers with mpc_d paths
        for prefix in [&a, &b] {
            let next: Vec<u32> = read_array(&dag::io::suffixed(prefix, "next.u32le")).unwrap();
            assert_eq!(next.len(), w.len());
            let mut has_pred = vec![false; w.len()];
            for (i, &x) in next.iter().enumerate() {
                if x != NONE {
                    let u = w.lo + i;
                    assert!(g.successors(u).into_iter().any(|v| v == x as usize));
                    assert!(!std::mem::replace(&mut has_pred[x as usize - w.lo], true));
                }
            }
            let paths = has_pred.iter().filter(|&&h| !h).count() as u64;
            assert_eq!(paths, stats.mpc_d);
        }
    }
}

#[test]
fn frontier_blocks_and_generic_agree() {
    let dir = tempfile::tempdir().unwrap();
    let g = testkit::mini_pg(300, 4, &MiniPgOpts::default()).graph;
    let n = g.num_nodes();
    let base = testkit::write_bv(dir.path(), "pg", &g);
    let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
    let w = Window { lo: n / 5, hi: n };
    let a = dir.path().join("fa");
    let b = dir.path().join("fb");
    for (prefix, blk) in [(&a, "auto"), (&b, "none")] {
        run_ok(&[
            "frontier",
            "--graph",
            &p(&base),
            "--blocks",
            blk,
            "--window",
            &window_arg(w),
            "--materialize",
            "--out",
            &p(prefix),
        ]);
    }
    let (ra, rb) = (res(&a), res(&b));
    assert_eq!(
        (ra["method"].as_str(), rb["method"].as_str()),
        (Some("blocks"), Some("generic"))
    );
    assert_eq!(ra["size"], rb["size"]);
    assert_eq!(ra["cut"], rb["cut"]);
    let f = frontier_blocks(&blocks, w, true);
    assert_eq!(ra["size"].as_u64().unwrap(), f.size);
    assert_eq!(ra["cut"].as_u64().unwrap() as usize, f.cut);
    let aa: Vec<u32> = read_array(&dag::io::suffixed(&a, "antichain.u32le")).unwrap();
    let ab: Vec<u32> = read_array(&dag::io::suffixed(&b, "antichain.u32le")).unwrap();
    assert_eq!(aa, ab);
    assert_eq!(Some(aa.clone()), f.members);
    check_antichain_blocks(&blocks, w, &aa).unwrap();
}

#[test]
fn width_closure_minflow_and_auto_agree() {
    let dir = tempfile::tempdir().unwrap();
    let g = testkit::mini_pg(250, 5, &MiniPgOpts::default()).graph;
    let n = g.num_nodes();
    let base = testkit::write_bv(dir.path(), "pg", &g);
    let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
    let w = Window {
        lo: n / 6,
        hi: n - n / 7,
    };
    let mut widths = Vec::new();
    for method in ["closure", "minflow", "auto"] {
        let prefix = dir.path().join(method);
        run_ok(&[
            "width",
            "--graph",
            &p(&base),
            "--window",
            &window_arg(w),
            "--method",
            method,
            "--out",
            &p(&prefix),
        ]);
        let r = res(&prefix);
        assert_eq!(r["certificate_ok"], true, "{method}");
        let width = r["width"].as_u64().unwrap();
        widths.push(width);
        let a: Vec<u32> = read_array(&dag::io::suffixed(&prefix, "antichain.u32le")).unwrap();
        assert_eq!(a.len() as u64, width);
        check_antichain_blocks(&blocks, w, &a).unwrap();
        if method == "minflow" {
            let nodes: Vec<u32> = read_array(&dag::io::suffixed(&prefix, "chains.u32le")).unwrap();
            let off: Vec<u64> =
                read_array(&dag::io::suffixed(&prefix, "chains_off.u64le")).unwrap();
            assert_eq!(off.len() as u64, width + 1);
            assert_eq!(*off.last().unwrap(), nodes.len() as u64);
            assert!(nodes.iter().all(|&x| w.contains(x as usize)));
        } else {
            assert_eq!(r["method"], "closure");
            assert_eq!(r["cross_check"]["agree"], true);
        }
    }
    for method in ["closure", "minflow"] {
        let prefix = dir.path().join(format!("{method}-dinic"));
        run_ok(&[
            "width",
            "--graph",
            &p(&base),
            "--window",
            &window_arg(w),
            "--method",
            method,
            "--flow",
            "dinic",
            "--out",
            &p(&prefix),
        ]);
        let r = res(&prefix);
        assert_eq!(r["certificate_ok"], true, "{method} dinic");
        widths.push(r["width"].as_u64().unwrap());
    }
    assert!(widths.windows(2).all(|x| x[0] == x[1]), "{widths:?}");
    let csr = Csr::from_vec_graph(&g);
    let full = width_min_flow(&csr);
    // the full graph: auto gives the bracket, closure needs --experimental-full
    let prefix = dir.path().join("full");
    run_ok(&["width", "--graph", &p(&base), "--out", &p(&prefix)]);
    let r = res(&prefix);
    assert_eq!(r["method"], "bracket");
    assert_eq!(r["certificate_ok"], true);
    assert!(
        r["lower"].as_u64().unwrap() <= full.width && full.width <= r["upper"].as_u64().unwrap()
    );
    let a: Vec<u32> = read_array(&dag::io::suffixed(&prefix, "antichain.u32le")).unwrap();
    check_antichain(&csr, &a).unwrap();
    run_expect(
        &[
            "width",
            "--graph",
            &p(&base),
            "--method",
            "closure",
            "--out",
            &p(&prefix),
        ],
        2,
    );
    run_ok(&[
        "width",
        "--graph",
        &p(&base),
        "--method",
        "closure",
        "--experimental-full",
        "--out",
        &p(&prefix),
    ]);
    assert_eq!(res(&prefix)["width"].as_u64().unwrap(), full.width);
}

#[test]
fn width_falls_back_to_minflow_on_generic_dags() {
    let dir = tempfile::tempdir().unwrap();
    let g = testkit::random_dag(60, 0.08, 11);
    let base = testkit::write_bv(dir.path(), "rnd", &g);
    let prefix = dir.path().join("w");
    run_ok(&["width", "--graph", &p(&base), "--out", &p(&prefix)]);
    let r = res(&prefix);
    assert_eq!(r["method"], "minflow");
    // the chains are a partition: exactly n ids are written
    assert_eq!(r["chain_nodes"], 60);
    let chains: Vec<u32> = read_array(&dag::io::suffixed(&prefix, "chains.u32le")).unwrap();
    let mut sorted = chains.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..60).collect::<Vec<u32>>());
    assert_eq!(
        r["width"].as_u64().unwrap(),
        width_min_flow(&Csr::from_vec_graph(&g)).width
    );
    let a: Vec<u32> = read_array(&dag::io::suffixed(&prefix, "antichain.u32le")).unwrap();
    check_antichain(&Csr::from_vec_graph(&g), &a).unwrap();
    // closure needs a BlockIndex: error exit
    run_expect(
        &[
            "width",
            "--graph",
            &p(&base),
            "--method",
            "closure",
            "--window",
            "0..30",
            "--out",
            &p(&prefix),
        ],
        2,
    );
    run_expect(
        &[
            "width",
            "--graph",
            &p(&base),
            "--blocks",
            "none",
            "--method",
            "closure",
            "--window",
            "0..30",
            "--out",
            &p(&prefix),
        ],
        2,
    );
    // path-cover and frontier fall back to the generic algorithms
    run_ok(&["path-cover", "--graph", &p(&base), "--out", &p(&prefix)]);
    assert_eq!(res(&prefix)["method"], "hk");
    run_ok(&["frontier", "--graph", &p(&base), "--out", &p(&prefix)]);
    assert_eq!(res(&prefix)["method"], "generic");
}

#[test]
fn saved_block_index_is_used() {
    let dir = tempfile::tempdir().unwrap();
    let g = testkit::mini_pg(200, 6, &MiniPgOpts::default()).graph;
    let base = testkit::write_bv(dir.path(), "pg", &g);
    let blk = dir.path().join("idx");
    run_ok(&["blocks", "--graph", &p(&base), "--out", &p(&blk)]);
    let prefix = dir.path().join("pc");
    run_ok(&[
        "path-cover",
        "--graph",
        &p(&base),
        "--blocks",
        &p(&blk),
        "--out",
        &p(&prefix),
    ]);
    let (blocks, _) = BlockIndex::from_vec_graph(&g).unwrap();
    let (stats, _) = min_path_cover_blocks(&blocks, Window::full(g.num_nodes()), false);
    assert_eq!(res(&prefix)["mpc_d"].as_u64().unwrap(), stats.mpc_d);
    assert_eq!(res(&prefix)["method"], "blocks");
}

/// Regression: `width --method minflow` checks the min-flow memory estimate
/// before decoding the window into a CSR (so a too-large window fails without the ≈ 52 GB
/// decode on the PG).
#[test]
fn minflow_budget_is_checked_before_decoding() {
    let dir = tempfile::tempdir().unwrap();
    let g = testkit::mini_pg(300, 6, &MiniPgOpts::default()).graph;
    let (n, m) = (g.num_nodes() as u64, g.num_arcs());
    let base = testkit::write_bv(dir.path(), "pg", &g);
    // enough for the CSR (8(n + 1) + 4m bytes), far too little for the flow network
    let csr_bytes = 8 * (n + 1) + 4 * m;
    let max_mem = format!("{}", (csr_bytes + 64) as f64 / 1e9);
    let out = run_expect(
        &[
            "--max-mem",
            &max_mem,
            "width",
            "--graph",
            &p(&base),
            "--method",
            "minflow",
            "--out",
            &p(&dir.path().join("w")),
        ],
        2,
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("min-flow network"), "{err}");
    assert!(
        !err.contains("into a CSR"),
        "decoded before the budget check:\n{err}"
    );
}
