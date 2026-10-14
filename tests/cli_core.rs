//! Command-line tests of the core subcommands: `topo-check`, `toposort`, `window`, `blocks`.

#[path = "support/cli.rs"]
mod cli;

use cli::{p, read_json, run_cli, run_expect, run_ok};
use dag::blocks::BlockIndex;
use dag::graph::{self, ChunkConf};
use dag::io::{self, LoadMode, NONE};
use dag::testkit::*;
use dag::topo;
use dsi_progress_logger::prelude::*;
use webgraph::prelude::*;

fn cquiet() -> Option<ConcurrentWrapper<ProgressLogger>> {
    None
}

#[test]
fn topo_check_agrees_with_library() {
    let dir = tempfile::tempdir().unwrap();
    let g = random_dag(300, 0.02, 1);
    let base = write_bv(dir.path(), "g", &g);
    let out = dir.path().join("tc");
    let o = run_ok(&[
        "--chunk-arcs",
        "17",
        "topo-check",
        "--graph",
        &p(&base),
        "--out",
        &p(&out),
    ]);
    let printed: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let lib = topo::check_identity(
        &graph::open(&base, ChunkConf::default()).unwrap(),
        Some(
            &graph::open(
                &graph::transpose_basename(&base).unwrap(),
                ChunkConf::default(),
            )
            .unwrap(),
        ),
        &mut cquiet(),
    )
    .unwrap();
    assert_eq!(printed, serde_json::to_value(&lib).unwrap());
    let json = read_json(&io::suffixed(&out, "json"));
    assert_eq!(json["results"]["identity_topological"], true);
    assert_eq!(json["results"]["sources"], lib.sources.unwrap());
    assert_eq!(json["m"], g.num_arcs());
    // a non-topological graph exits with 1 and reports the violation
    let bad = write_bv(dir.path(), "bad", &with_back_arc(&g));
    let o = run_expect(&["topo-check", "--graph", &p(&bad), "--no-transpose"], 1);
    let printed: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(printed["identity_topological"], false);
    assert!(printed["sources"].is_null());
    // regression: a self-loop, forward scan only and with the transpose
    let n = g.num_nodes();
    let sl = p(&write_bv(dir.path(), "sl", &with_self_loop(&g)));
    for extra in [&["--no-transpose"][..], &[][..]] {
        let mut args = vec!["topo-check", "--graph", &sl];
        args.extend_from_slice(extra);
        let o = run_expect(&args, 1);
        let printed: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
        assert_eq!(printed["identity_topological"], false);
        assert_eq!(
            printed["first_violation"],
            serde_json::json!([n / 2, n / 2])
        );
    }
    // regression: open_transpose rejects a transpose of another graph
    let other = write_bv(dir.path(), "other", &random_dag(301, 0.02, 2));
    let o = run_expect(
        &[
            "topo-check",
            "--graph",
            &p(&base),
            "--transpose",
            &p(&graph::transpose_basename(&other).unwrap()),
        ],
        2,
    );
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("nodes and"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

/// Regression: `window`, `toposort` and `blocks` honour `--max-mem`.
#[test]
fn max_mem_is_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let base = p(&write_bv(
        dir.path(),
        "g",
        &mini_pg(2000, 3, &MiniPgOpts::default()).graph,
    ));
    for (cmd, extra) in [
        ("window", &["--window", "0..100"][..]),
        ("toposort", &[][..]),
        ("blocks", &[][..]),
    ] {
        let out = dir.path().join(cmd);
        let mut args: Vec<&str> = vec!["--max-mem", "0.000001", cmd, "--graph", &base];
        args.extend_from_slice(extra);
        let o_out = p(&out);
        args.extend_from_slice(&["--out", &o_out]);
        let o = run_expect(&args, 2);
        assert!(
            String::from_utf8_lossy(&o.stderr).contains("exceeds --max-mem"),
            "{cmd}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
}

#[test]
fn toposort_relabels() {
    let dir = tempfile::tempdir().unwrap();
    let (g, _) = shuffled_dag(120, 0.05, 4);
    let base = write_bv(dir.path(), "s", &g);
    let out = dir.path().join("sorted");
    run_ok(&["toposort", "--graph", &p(&base), "--out", &p(&out)]);
    let bv = graph::load_ra(&out).unwrap();
    assert!((0..bv.num_nodes()).all(|u| bv.successors(u).into_iter().all(|v| v > u)));
    let rank = io::read_array::<u32>(&io::suffixed(&out, "rank.u32le")).unwrap();
    let rank: Vec<usize> = rank.into_iter().map(|r| r as usize).collect();
    webgraph::traits::graph::eq(&topo::relabel_vec(&g, &rank), &bv).unwrap();
    run_ok(&["topo-check", "--graph", &p(&out)]);
    assert!(
        read_json(&io::suffixed(&out, "json"))["files"]
            .as_array()
            .unwrap()
            .len()
            == 2
    );
}

#[test]
fn window_writes_the_induced_subgraph() {
    let dir = tempfile::tempdir().unwrap();
    let g = random_dag(200, 0.04, 2);
    let base = write_bv(dir.path(), "g", &g);
    let out = dir.path().join("w");
    run_ok(&[
        "window",
        "--graph",
        &p(&base),
        "--window",
        "50..150",
        "--out",
        &p(&out),
    ]);
    let bv = graph::load_ra(&out).unwrap();
    let want = vec_graph(
        100,
        arcs(&g)
            .into_iter()
            .filter(|&(u, v)| (50..150).contains(&u) && (50..150).contains(&v))
            .map(|(u, v)| (u - 50, v - 50)),
    );
    webgraph::traits::graph::eq(&want, &bv).unwrap();
    let bvt = graph::load_ra(&graph::transpose_basename(&out).unwrap()).unwrap();
    webgraph::traits::graph::eq(&transpose(&want), &bvt).unwrap();
    assert!(out.with_extension("dcf").exists());
    assert_eq!(
        read_json(&io::suffixed(&out, "json"))["results"]["arcs"],
        want.num_arcs()
    );
    run_expect(
        &[
            "window",
            "--graph",
            &p(&base),
            "--window",
            "150..50",
            "--out",
            &p(&out),
        ],
        2,
    );
}

#[test]
fn blocks_agrees_with_library_and_reference() {
    let dir = tempfile::tempdir().unwrap();
    let mp = mini_pg(2000, 5, &MiniPgOpts::default());
    let base = write_bv(dir.path(), "pg", &mp.graph);
    let out = dir.path().join("pg");
    run_ok(&[
        "--chunk-arcs",
        "64",
        "--threads",
        "3",
        "blocks",
        "--graph",
        &p(&base),
        "--out",
        &p(&out),
    ]);
    let (lib, rep) = BlockIndex::from_vec_graph(&mp.graph).unwrap();
    for mode in [LoadMode::Ram, LoadMode::Mmap] {
        let l = BlockIndex::load(&out, mode).unwrap();
        assert_eq!(&*l.create, &*lib.create);
        assert_eq!(&*l.spend, &*lib.spend);
        assert_eq!(&*l.first, &*lib.first);
        assert_eq!(&*l.len, &*lib.len);
        assert_eq!(&*l.nin, &*lib.nin);
    }
    let json = read_json(&io::suffixed(&out, "json"));
    assert_eq!(
        json["results"]["report"],
        serde_json::to_value(&rep).unwrap()
    );
    assert_eq!(json["files"].as_array().unwrap().len(), 5);

    // reference files in the format of the `bicliques` lens
    let refdir = dir.path().join("bicliques");
    std::fs::create_dir(&refdir).unwrap();
    let mut rec = Vec::new();
    for b in 0..lib.nb {
        rec.extend([lib.first[b], lib.nin[b], lib.len[b]]);
    }
    io::write_array(&refdir.join("blocks.u32x3"), &rec).unwrap();
    let start: Vec<u32> = (0..lib.n)
        .map(|v| match lib.create[v] {
            NONE => NONE,
            b => lib.first[b as usize],
        })
        .collect();
    io::write_array(&refdir.join("blk_start.u32"), &start).unwrap();
    run_ok(&[
        "blocks",
        "--graph",
        &p(&base),
        "--out",
        &p(&out),
        "--compare-bicliques",
        &p(&refdir),
    ]);
    assert_eq!(
        read_json(&io::suffixed(&out, "json"))["results"]["comparison"]["node_mismatches"],
        0
    );
    let mut start2 = start.clone();
    start2[7] ^= 1;
    io::write_array(&refdir.join("blk_start.u32"), &start2).unwrap();
    run_expect(
        &[
            "blocks",
            "--graph",
            &p(&base),
            "--out",
            &p(&out),
            "--compare-bicliques",
            &p(&refdir),
        ],
        1,
    );
    let c = &read_json(&io::suffixed(&out, "json"))["results"]["comparison"];
    assert_eq!(
        (
            c["node_mismatches"].as_u64(),
            c["first_node_mismatch"].as_u64()
        ),
        (Some(1), Some(7))
    );

    // a graph that is not a biclique digraph is an error (exit 2) naming the violation
    let bad = mini_pg(
        300,
        1,
        &MiniPgOpts {
            non_contiguous: true,
            ..Default::default()
        },
    );
    let bb = write_bv(dir.path(), "bad", &bad.graph);
    let o = run_expect(
        &[
            "blocks",
            "--graph",
            &p(&bb),
            "--out",
            &p(&dir.path().join("bad")),
        ],
        2,
    );
    assert!(String::from_utf8_lossy(&o.stderr).contains("not a biclique"));
}

#[test]
fn unimplemented_and_bad_invocations() {
    let dir = tempfile::tempdir().unwrap();
    let g = chain(5);
    let base = write_bv(dir.path(), "c", &g);
    let o = run_cli(&[
        "tr",
        "verify",
        "--graph",
        &p(&base),
        "--out",
        &p(&dir.path().join("x")),
    ]);
    // whatever the state of the subcommand, it must exit cleanly (0, 1 or 2), never panic
    assert!(
        matches!(o.status.code(), Some(0..=2)),
        "tr verify exited with {:?}: {}",
        o.status.code(),
        String::from_utf8_lossy(&o.stderr)
    );
    // missing .ef
    std::fs::remove_file(base.with_extension("ef")).unwrap();
    let o = run_expect(&["topo-check", "--graph", &p(&base)], 2);
    assert!(String::from_utf8_lossy(&o.stderr).contains("webgraph build ef"));
    // clap errors
    assert_eq!(run_cli(&["no-such-command"]).status.code(), Some(2));
    run_ok(&["--help"]);
}
