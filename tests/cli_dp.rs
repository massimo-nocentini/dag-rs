//! Command-line tests of `dag dp`, `dag certify` and `dag cpm`.

#[path = "support/cli.rs"]
mod cli;
#[path = "support/dp.rs"]
mod support;

use std::path::{Path, PathBuf};

use cli::{p, read_json, run_cli, run_expect, run_ok};
use dag::blocks::BlockIndex;
use dag::cpm::cpm;
use dag::dp::certify::bit_eq;
use dag::dp::solve::solve;
use dag::dp::{ArcWeights, Const, DpInput, DpResult, Init, Kernel, Node, One};
use dag::io::{self, Dump};
use dag::semiring::{Boolean, MaxPlus, MinPlus, Semiring};
use dag::testkit::*;
use dag::{Dir, Window};
use support::*;
use webgraph::prelude::*;

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    base: PathBuf,
    g: VecGraph,
}

fn fixture(g: VecGraph) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_owned();
    let base = write_bv(&root, "g", &g);
    Fixture {
        _dir: dir,
        root,
        base,
        g,
    }
}

impl Fixture {
    fn out(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Runs `dp` with `extra` arguments and returns the output prefix.
    fn dp(&self, name: &str, extra: &[&str], code: i32) -> PathBuf {
        let out = self.out(name);
        let mut args = vec![
            "--chunk-arcs".to_owned(),
            "5".to_owned(),
            "dp".to_owned(),
            "--graph".to_owned(),
            p(&self.base),
            "--out".to_owned(),
            p(&out),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        run_expect(&args, code);
        out
    }

    /// The library result over the in-memory graph (forward via pull, backward via pull).
    fn lib<S: Semiring, W: ArcWeights<S>>(
        &self,
        s: &S,
        dir: Dir,
        w: &W,
        init: &Init<S::Elem>,
        win: Window,
        pred: bool,
    ) -> DpResult<S::Elem> {
        let gt = transpose(&self.g);
        let (cg, cgt) = (chunked(&self.g, 3), chunked(&gt, 3));
        let c = conf(win, Kernel::Auto, pred, 2);
        solve(
            s,
            &DpInput::with_transpose(&cg, &cgt),
            dir,
            w,
            init,
            &c,
            &mut quiet(),
        )
        .unwrap()
    }
}

fn assert_file_eq<E: Dump + std::fmt::Debug>(path: &Path, want: &[E]) {
    let got = io::read_array::<E>(path).unwrap();
    assert_eq!(got.len(), want.len(), "{}", path.display());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            bit_eq(a, b),
            "{}: entry {i}: {a:?} vs {b:?}",
            path.display()
        );
    }
}

fn val_path<E: Dump>(out: &Path) -> PathBuf {
    io::array_path::<E>(out, "val")
}

#[test]
fn dp_matches_the_library_for_every_kernel() {
    let pg = mini_pg(120, 4, &MiniPgOpts::default()).graph;
    for (gi, g) in [pg, random_dag(150, 0.03, 2)].into_iter().enumerate() {
        let biclique = gi == 0;
        let f = fixture(g);
        let n = f.g.num_nodes();
        let full = Window::full(n);
        let s = MaxPlus::<u32>::new();
        let want_f = f.lib(
            &s,
            Dir::Forward,
            &Const(1u32),
            &Init::Sources(0),
            full,
            true,
        );
        let want_b = f.lib(
            &s,
            Dir::Backward,
            &Const(1u32),
            &Init::Sources(0),
            full,
            true,
        );
        for kernel in ["auto", "push", "pull", "blocks"] {
            let code = if kernel == "blocks" && !biclique {
                2
            } else {
                0
            };
            let out = f.dp(
                &format!("f-{kernel}"),
                &[
                    "--semiring",
                    "maxplus-u32",
                    "--kernel",
                    kernel,
                    "--pred",
                    "--certify",
                ],
                code,
            );
            if code != 0 {
                continue;
            }
            assert_file_eq(&val_path::<u32>(&out), &want_f.val);
            assert_file_eq(
                &io::array_path::<u32>(&out, "pred"),
                want_f.pred.as_ref().unwrap(),
            );
            let j = read_json(&io::suffixed(&out, "json"));
            let used = j["params"]["kernel"].as_str().unwrap();
            let expect = match (kernel, biclique) {
                ("auto", true) | ("blocks", _) => "Blocks",
                ("auto", false) | ("pull", _) => "Pull",
                _ => "Push",
            };
            assert_eq!(used, expect, "{kernel}");
            assert_eq!(j["results"]["certify"]["violations"], 0);
            assert_eq!(j["results"]["certify"]["checked"], n);
            assert_eq!(
                j["results"]["summary"]["max"],
                *want_f.val.iter().max().unwrap() as f64
            );
            assert_eq!(j["n"], n);
        }
        for kernel in ["auto", "pull", "blocks"] {
            let code = if kernel == "blocks" && !biclique {
                2
            } else {
                0
            };
            let out = f.dp(
                &format!("b-{kernel}"),
                &[
                    "--semiring",
                    "maxplus-u32",
                    "--dir",
                    "backward",
                    "--kernel",
                    kernel,
                    "--pred",
                    "--certify",
                ],
                code,
            );
            if code == 0 {
                assert_file_eq(&val_path::<u32>(&out), &want_b.val);
                assert_file_eq(
                    &io::array_path::<u32>(&out, "pred"),
                    want_b.pred.as_ref().unwrap(),
                );
            }
        }
        // there is no backward push kernel
        f.dp(
            "b-push",
            &[
                "--semiring",
                "maxplus-u32",
                "--dir",
                "backward",
                "--kernel",
                "push",
            ],
            2,
        );
        // --blocks none never uses the index
        let out = f.dp(
            "none",
            &["--semiring", "maxplus-u32", "--blocks", "none"],
            0,
        );
        assert_eq!(
            read_json(&io::suffixed(&out, "json"))["params"]["blocks"],
            false
        );
    }
}

#[test]
fn dp_other_semirings_inits_and_windows() {
    let f = fixture(mini_pg(100, 7, &MiniPgOpts::default()).graph);
    let n = f.g.num_nodes();
    let win = Window {
        lo: n / 4,
        hi: 3 * n / 4,
    };
    let ws = format!("{}..{}", win.lo, win.hi);
    for (dir, ds) in [(Dir::Forward, "forward"), (Dir::Backward, "backward")] {
        let a = ["--dir", ds, "--window", &ws, "--certify"];
        // min-plus i64, init all, unit weights
        let out = f.dp(
            &format!("mi-{ds}"),
            &[&["--semiring", "minplus-i64", "--init", "all"], &a[..]].concat(),
            0,
        );
        let s = MinPlus::<i64>::new();
        assert_file_eq(
            &val_path::<i64>(&out),
            &f.lib(&s, dir, &Const(1i64), &Init::All(0), win, false).val,
        );
        // max-plus f64 with one = 0-length arcs
        let out = f.dp(
            &format!("mf-{ds}"),
            &[&["--semiring", "maxplus-f64", "--weights", "one"], &a[..]].concat(),
            0,
        );
        let s = MaxPlus::<f64>::new();
        assert_file_eq(
            &val_path::<f64>(&out),
            &f.lib(&s, dir, &One, &Init::Sources(0.0), win, false).val,
        );
        // Boolean reachability from window sources
        let out = f.dp(
            &format!("bo-{ds}"),
            &[&["--semiring", "bool"], &a[..]].concat(),
            0,
        );
        let s = Boolean::new();
        assert_file_eq(
            &val_path::<bool>(&out),
            &f.lib(&s, dir, &One, &Init::Sources(true), win, false).val,
        );
        // min-plus u32
        let out = f.dp(
            &format!("mu-{ds}"),
            &[&["--semiring", "minplus-u32"], &a[..]].concat(),
            0,
        );
        let s = MinPlus::<u32>::new();
        assert_file_eq(
            &val_path::<u32>(&out),
            &f.lib(&s, dir, &Const(1u32), &Init::Sources(0), win, false)
                .val,
        );
    }
}

#[test]
fn dp_node_and_inv_outdeg_weights() {
    let f = fixture(mini_pg(80, 9, &MiniPgOpts::default()).graph);
    let n = f.g.num_nodes();
    let full = Window::full(n);
    let du: Vec<u32> = (0..n).map(|u| (mix(u as u64) % 7) as u32).collect();
    let df: Vec<f64> = (0..n)
        .map(|u| (mix(u as u64 + 99) % 1000) as f64 / 64.0)
        .collect();
    let fu = f.out("w.u32le");
    let ff = f.out("w.f64le");
    io::write_array(&fu, &du).unwrap();
    io::write_array(&ff, &df).unwrap();
    let s = MaxPlus::<u64>::new();
    let w = Node {
        alpha: |u: usize| du[u] as u64,
        beta: |_: usize| 0u64,
    };
    let out = f.dp(
        "nu",
        &[
            "--semiring",
            "maxplus-u64",
            "--weights",
            &format!("node-u32:{}", p(&fu)),
            "--certify",
        ],
        0,
    );
    assert_file_eq(
        &val_path::<u64>(&out),
        &f.lib(&s, Dir::Forward, &w, &Init::Sources(0), full, false)
            .val,
    );
    let s = MinPlus::<f64>::new();
    let w = Node {
        alpha: |u: usize| df[u],
        beta: |_: usize| 0.0,
    };
    for kernel in ["blocks", "push"] {
        let out = f.dp(
            &format!("nf-{kernel}"),
            &[
                "--semiring",
                "minplus-f64",
                "--kernel",
                kernel,
                "--weights",
                &format!("node-f64:{}", p(&ff)),
                "--pred",
                "--certify",
            ],
            0,
        );
        let want = f.lib(&s, Dir::Forward, &w, &Init::Sources(0.0), full, true);
        assert_file_eq(&val_path::<f64>(&out), &want.val);
        assert_file_eq(
            &io::array_path::<u32>(&out, "pred"),
            want.pred.as_ref().unwrap(),
        );
    }
    // inv-outdeg: α(u) = 1 / outdeg(u), from the BlockIndex or from the graph
    let s = MaxPlus::<f64>::new();
    let w = Node {
        alpha: |u: usize| match f.g.outdegree(u) {
            0 => 0.0,
            d => 1.0 / d as f64,
        },
        beta: |_: usize| 0.0,
    };
    let want = f.lib(&s, Dir::Backward, &w, &Init::Sources(0.0), full, false);
    for blocks in ["auto", "none"] {
        let out = f.dp(
            &format!("io-{blocks}"),
            &[
                "--semiring",
                "maxplus-f64",
                "--dir",
                "backward",
                "--weights",
                "inv-outdeg",
                "--blocks",
                blocks,
                "--certify",
            ],
            0,
        );
        assert_file_eq(&val_path::<f64>(&out), &want.val);
    }
    // integer semirings have no element for 1/d, and the files must have n entries
    f.dp(
        "bad1",
        &["--semiring", "maxplus-u32", "--weights", "inv-outdeg"],
        2,
    );
    let short = f.out("short.u32le");
    io::write_array(&short, &du[1..]).unwrap();
    f.dp(
        "bad2",
        &[
            "--semiring",
            "maxplus-u32",
            "--weights",
            &format!("node-u32:{}", p(&short)),
        ],
        2,
    );
    f.dp(
        "bad3",
        &["--semiring", "maxplus-u32", "--weights", "nonsense"],
        2,
    );
}

#[test]
fn certify_subcommand_and_compare() {
    let f = fixture(random_dag(200, 0.03, 5));
    let out = f.dp("v", &["--semiring", "maxplus-u32", "--pred"], 0);
    let vals = val_path::<u32>(&out);
    let preds = io::array_path::<u32>(&out, "pred");
    let certify = |name: &str, values: &Path, pred: Option<&Path>, dir: &str, code: i32| {
        let o = f.out(name);
        let mut args = vec![
            "certify".to_owned(),
            "--graph".to_owned(),
            p(&f.base),
            "--out".to_owned(),
            p(&o),
            "--semiring".to_owned(),
            "maxplus-u32".to_owned(),
            "--values".to_owned(),
            p(values),
            "--dir".to_owned(),
            dir.to_owned(),
        ];
        if let Some(pr) = pred {
            args.push("--pred".to_owned());
            args.push(p(pr));
        }
        run_expect(&args, code);
        // errors (exit 2) write no report
        (code != 2).then(|| read_json(&io::suffixed(&o, "json")))
    };
    let j = certify("c-ok", &vals, Some(&preds), "forward", 0).unwrap();
    assert_eq!(j["results"]["violations"], 0);
    assert_eq!(j["results"]["checked"], 200);
    // the same values are not the backward solution
    certify("c-bwd", &vals, None, "backward", 1);
    // one perturbed value
    let mut v = io::read_array::<u32>(&vals).unwrap();
    let x = v.iter().position(|&d| d > 0).unwrap();
    v[x] += 1;
    let bad = f.out("bad.val.u32le");
    io::write_array(&bad, &v).unwrap();
    let j = certify("c-bad", &bad, None, "forward", 1).unwrap();
    assert_eq!(j["results"]["first_violation"], x);
    // one perturbed predecessor
    let mut pr = io::read_array::<u32>(&preds).unwrap();
    pr[x] = x as u32;
    let badp = f.out("bad.pred.u32le");
    io::write_array(&badp, &pr).unwrap();
    let j = certify("c-badp", &vals, Some(&badp), "forward", 1).unwrap();
    assert_eq!(j["results"]["pred_violations"], 1);
    // a values file of the wrong length is an error
    io::write_array(&bad, &v[1..]).unwrap();
    certify("c-len", &bad, None, "forward", 2);
    // --compare: identical, a prefix window against the n-entry file, a file of the wrong
    // length (error), and a perturbed file
    f.dp(
        "cmp-ok",
        &["--semiring", "maxplus-u32", "--compare", &p(&vals)],
        0,
    );
    f.dp(
        "cmp-prefix",
        &[
            "--semiring",
            "maxplus-u32",
            "--window",
            "0..150",
            "--compare",
            &p(&vals),
        ],
        0,
    );
    f.dp(
        "cmp-len",
        &["--semiring", "maxplus-u32", "--compare", &p(&bad)],
        2,
    );
    io::write_array(&bad, &v).unwrap();
    let out = f.dp(
        "cmp-bad2",
        &["--semiring", "maxplus-u32", "--compare", &p(&bad)],
        1,
    );
    let j = read_json(&io::suffixed(&out, "json"));
    assert_eq!(j["results"]["comparison"]["mismatches"], 1);
    assert_eq!(j["results"]["comparison"]["first_mismatch"], x);
    // a window not starting at 0 against an n-entry file reads the file at offset lo: the
    // windowed values placed at [50, 150) match, the same values placed at [0, 100) do not
    let wout = f.dp(
        "win",
        &["--semiring", "maxplus-u32", "--window", "50..150"],
        0,
    );
    let wv = io::read_array::<u32>(&val_path::<u32>(&wout)).unwrap();
    assert_eq!(wv.len(), 100);
    let (aligned, shifted) = (f.out("aligned.u32le"), f.out("shifted.u32le"));
    let place = |at: usize, file: &Path| {
        let mut a = vec![u32::MAX - 1; 200];
        a[at..at + 100].copy_from_slice(&wv);
        io::write_array(file, &a).unwrap();
    };
    place(50, &aligned);
    place(0, &shifted);
    f.dp(
        "cmp-win",
        &[
            "--semiring",
            "maxplus-u32",
            "--window",
            "50..150",
            "--compare",
            &p(&aligned),
        ],
        0,
    );
    let out = f.dp(
        "cmp-win-bad",
        &[
            "--semiring",
            "maxplus-u32",
            "--window",
            "50..150",
            "--compare",
            &p(&shifted),
        ],
        1,
    );
    let j = read_json(&io::suffixed(&out, "json"));
    assert!(j["results"]["comparison"]["mismatches"].as_u64().unwrap() >= 50);
    assert!(
        j["results"]["comparison"]["first_mismatch"]
            .as_u64()
            .unwrap()
            >= 50
    );
}

#[test]
fn blocks_auto_falls_back_on_non_biclique_graphs() {
    let f = fixture(random_dag(100, 0.05, 1));
    let out = f.dp("auto", &["--semiring", "maxplus-u32", "--certify"], 0);
    let j = read_json(&io::suffixed(&out, "json"));
    assert_eq!(j["params"]["blocks"], false);
    assert_eq!(j["params"]["kernel"], "Pull");
    // a saved index is loaded with --blocks PREFIX
    let g = fixture(mini_pg(60, 3, &MiniPgOpts::default()).graph);
    let (idx, _) = BlockIndex::from_vec_graph(&g.g).unwrap();
    let prefix = g.out("idx");
    idx.save(&prefix).unwrap();
    for load in ["ram", "mmap"] {
        let out = g.out(&format!("pref-{load}"));
        run_ok(&[
            "--load",
            load,
            "dp",
            "--graph",
            &p(&g.base),
            "--out",
            &p(&out),
            "--semiring",
            "maxplus-u32",
            "--blocks",
            &p(&prefix),
            "--kernel",
            "blocks",
            "--certify",
        ]);
        let j = read_json(&io::suffixed(&out, "json"));
        assert_eq!(j["params"]["kernel"], "Blocks");
        assert_eq!(j["results"]["certify"]["violations"], 0);
    }
    // a RAM load is budgeted from <p>.blk.json before any array is read: with an array missing
    // and no memory, the run fails on the budget, not on the missing file
    let gone = io::suffixed(&prefix, "blk.create.u32le");
    std::fs::remove_file(&gone).unwrap();
    let out = g.out("over");
    let run = |max_mem: &str| {
        let o = run_cli(&[
            "--load",
            "ram",
            "--max-mem",
            max_mem,
            "dp",
            "--graph",
            &p(&g.base),
            "--out",
            &p(&out),
            "--semiring",
            "maxplus-u32",
            "--blocks",
            &p(&prefix),
            "--kernel",
            "blocks",
        ]);
        assert!(!o.status.success());
        String::from_utf8_lossy(&o.stderr).into_owned()
    };
    let err = run("0");
    assert!(
        err.contains("exceeds --max-mem") && err.contains("BlockIndex"),
        "{err}"
    );
    let err = run("100");
    assert!(err.contains("blk.create"), "{err}");
}

#[test]
fn cpm_command() {
    let f = fixture(mini_pg(150, 11, &MiniPgOpts::default()).graph);
    let n = f.g.num_nodes();
    let cg = chunked(&f.g, 4);
    let lib = cpm::<u32, _, _>(
        &DpInput::graph(&cg),
        &Const(1u32),
        &conf(Window::full(n), Kernel::Auto, true, 2),
        &mut quiet(),
    )
    .unwrap();
    let sum = lib.summary(true);
    let arcs = lib.critical_arcs(&cg, &Const(1u32), &mut cquiet()).unwrap();
    for blocks in ["auto", "none"] {
        let out = f.out(&format!("cpm-{blocks}"));
        run_ok(&[
            "cpm",
            "--graph",
            &p(&f.base),
            "--out",
            &p(&out),
            "--blocks",
            blocks,
            "--slack",
            "--path",
            "--critical-arcs",
        ]);
        assert_file_eq(&io::array_path::<u32>(&out, "E"), &lib.e);
        assert_file_eq(&io::array_path::<u32>(&out, "H"), &lib.h);
        assert_file_eq(&io::array_path::<u32>(&out, "slack"), &lib.slacks());
        assert_file_eq(
            &io::array_path::<u32>(&out, "path"),
            sum.longest_path.as_ref().unwrap(),
        );
        let j = read_json(&io::suffixed(&out, "json"));
        let r = &j["results"]["summary"];
        assert_eq!(r["t"], sum.t);
        assert_eq!(r["critical_nodes"], sum.critical_nodes);
        assert_eq!(r["critical_arcs"], arcs);
        assert_eq!(r["width1_layers"], sum.width1_layers.unwrap());
        assert_eq!(r["max_critical_width"], sum.max_critical_width);
        assert_eq!(
            j["params"]["kernel_forward"],
            if blocks == "auto" { "Blocks" } else { "Pull" }
        );
        // comparisons with the E/H just written pass; a perturbed one fails
        let e = io::array_path::<u32>(&out, "E");
        let h = io::array_path::<u32>(&out, "H");
        let o2 = f.out(&format!("cmp-{blocks}"));
        run_ok(&[
            "cpm",
            "--graph",
            &p(&f.base),
            "--out",
            &p(&o2),
            "--compare-e",
            &p(&e),
            "--compare-h",
            &p(&h),
        ]);
        let mut bad = lib.h.clone();
        bad[n / 2] += 1;
        let bh = f.out("bad.H.u32le");
        io::write_array(&bh, &bad).unwrap();
        run_expect(
            &[
                "cpm",
                "--graph",
                &p(&f.base),
                "--out",
                &p(&o2),
                "--compare-h",
                &p(&bh),
            ],
            1,
        );
    }
    // node durations
    let du: Vec<u32> = (0..n).map(|u| (mix(u as u64) % 5) as u32).collect();
    let fu = f.out("dur.u32le");
    io::write_array(&fu, &du).unwrap();
    let w = Node {
        alpha: |u: usize| du[u],
        beta: |_: usize| 0u32,
    };
    let lib = cpm::<u32, _, _>(
        &DpInput::graph(&cg),
        &w,
        &conf(Window::full(n), Kernel::Auto, false, 2),
        &mut quiet(),
    )
    .unwrap();
    let out = f.out("cpm-node");
    run_ok(&[
        "cpm",
        "--graph",
        &p(&f.base),
        "--out",
        &p(&out),
        "--weights",
        &format!("node-u32:{}", p(&fu)),
    ]);
    assert_file_eq(&io::array_path::<u32>(&out, "E"), &lib.e);
    let j = read_json(&io::suffixed(&out, "json"));
    assert_eq!(
        j["results"]["summary"]["width1_layers"],
        serde_json::Value::Null
    );
    // a window, and unsupported weights
    let out = f.out("cpm-win");
    run_ok(&[
        "cpm",
        "--graph",
        &p(&f.base),
        "--out",
        &p(&out),
        "--window",
        "10..90",
    ]);
    let a = cpm::<u32, _, _>(
        &DpInput::graph(&cg),
        &Const(1u32),
        &conf(Window { lo: 10, hi: 90 }, Kernel::Auto, false, 2),
        &mut quiet(),
    )
    .unwrap();
    assert_file_eq(&io::array_path::<u32>(&out, "E"), &a.e);
    // --compare-e / --compare-h with a window starting at 10 against n-entry files read them at
    // offset 10: aligned copies match, copies shifted to offset 0 do not
    for (flag, vals) in [("--compare-e", &a.e), ("--compare-h", &a.h)] {
        let (aligned, shifted) = (f.out("al.u32le"), f.out("sh.u32le"));
        for (at, file) in [(10, &aligned), (0, &shifted)] {
            let mut x = vec![u32::MAX - 1; n];
            x[at..at + 80].copy_from_slice(vals);
            io::write_array(file, &x).unwrap();
        }
        run_ok(&[
            "cpm",
            "--graph",
            &p(&f.base),
            "--out",
            &p(&out),
            "--window",
            "10..90",
            flag,
            &p(&aligned),
        ]);
        run_expect(
            &[
                "cpm",
                "--graph",
                &p(&f.base),
                "--out",
                &p(&out),
                "--window",
                "10..90",
                flag,
                &p(&shifted),
            ],
            1,
        );
        let j = read_json(&io::suffixed(&out, "json"));
        let c = &j["results"]["comparisons"][0]["result"];
        assert_eq!(
            j["results"]["comparisons"][0]["array"],
            if flag == "--compare-e" { "E" } else { "H" }
        );
        assert!(c["mismatches"].as_u64().unwrap() > 0, "{j}");
        assert!(c["first_mismatch"].as_u64().unwrap() >= 10, "{j}");
    }
    run_expect(
        &[
            "cpm",
            "--graph",
            &p(&f.base),
            "--out",
            &p(&out),
            "--weights",
            "inv-outdeg",
        ],
        2,
    );
}

#[test]
fn library_unit_weights_one_vs_const() {
    // `--weights unit` is arc length 1 for tropical semirings and 1̄ for the others
    let f = fixture(chain(5));
    let out = f.dp("u", &["--semiring", "maxplus-u32"], 0);
    assert_file_eq(&val_path::<u32>(&out), &[0u32, 1, 2, 3, 4]);
    let out = f.dp("o", &["--semiring", "maxplus-u32", "--weights", "one"], 0);
    assert_file_eq(&val_path::<u32>(&out), &[0u32; 5]);
    let out = f.dp("b", &["--semiring", "bool"], 0);
    assert_file_eq(&val_path::<bool>(&out), &[true; 5]);
}

fn params(out: &Path) -> serde_json::Value {
    read_json(&io::suffixed(out, "json"))["params"].clone()
}

#[test]
fn blocks_auto_is_not_built_for_a_strict_sub_window() {
    // regression: `--blocks auto` used to build the index over the whole graph for any window
    let f = fixture(mini_pg(120, 13, &MiniPgOpts::default()).graph);
    let n = f.g.num_nodes();
    let cg = chunked(&f.g, 4);
    let out = f.dp("full", &["--semiring", "maxplus-u32"], 0);
    assert_eq!(params(&out)["blocks"], true);
    assert_eq!(params(&out)["kernel"], "Blocks");
    let win = Window {
        lo: n / 4,
        hi: 3 * n / 4,
    };
    let ws = format!("{}..{}", win.lo, win.hi);
    let s = MaxPlus::<u32>::new();
    for (dir, ds) in [(Dir::Forward, "forward"), (Dir::Backward, "backward")] {
        let want = f.lib(&s, dir, &Const(1u32), &Init::Sources(0), win, true);
        let out = f.dp(
            &format!("sub-{ds}"),
            &[
                "--semiring",
                "maxplus-u32",
                "--dir",
                ds,
                "--window",
                &ws,
                "--pred",
            ],
            0,
        );
        let pa = params(&out);
        assert_eq!(pa["blocks"], false, "{ds}");
        assert_eq!(pa["kernel"], "Pull", "{ds}");
        assert_file_eq(&val_path::<u32>(&out), &want.val);
        assert_file_eq(
            &io::array_path::<u32>(&out, "pred"),
            want.pred.as_ref().unwrap(),
        );
        // an explicit --kernel blocks still builds it
        let out = f.dp(
            &format!("subk-{ds}"),
            &[
                "--semiring",
                "maxplus-u32",
                "--dir",
                ds,
                "--window",
                &ws,
                "--kernel",
                "blocks",
                "--pred",
                "--certify",
            ],
            0,
        );
        assert_eq!(params(&out)["blocks"], true, "{ds}");
        assert_eq!(params(&out)["kernel"], "Blocks", "{ds}");
        assert_file_eq(&val_path::<u32>(&out), &want.val);
    }
    // cpm: the same rule
    let want = cpm::<u32, _, _>(
        &DpInput::graph(&cg),
        &Const(1u32),
        &conf(win, Kernel::Auto, false, 2),
        &mut quiet(),
    )
    .unwrap();
    let out = f.out("cpm-sub");
    run_ok(&[
        "cpm",
        "--graph",
        &p(&f.base),
        "--out",
        &p(&out),
        "--window",
        &ws,
        "--slack",
    ]);
    assert_eq!(params(&out)["blocks"], false);
    assert_eq!(params(&out)["kernel_forward"], "Pull");
    assert_eq!(params(&out)["kernel_backward"], "Pull");
    assert_file_eq(&io::array_path::<u32>(&out, "E"), &want.e);
    assert_file_eq(&io::array_path::<u32>(&out, "H"), &want.h);
    assert_file_eq(&io::array_path::<u32>(&out, "slack"), &want.slacks());
    let out = f.out("cpm-full");
    run_ok(&["cpm", "--graph", &p(&f.base), "--out", &p(&out)]);
    assert_eq!(params(&out)["blocks"], true);
}

#[test]
fn a_saved_index_runs_without_the_graph() {
    // regression: block-kernel runs with --blocks PREFIX used to open the BV graph
    let f = fixture(mini_pg(100, 17, &MiniPgOpts::default()).graph);
    let n = f.g.num_nodes();
    let cg = chunked(&f.g, 4);
    let (idx, _) = BlockIndex::from_vec_graph(&f.g).unwrap();
    let prefix = f.out("idx");
    idx.save(&prefix).unwrap();
    let pre = p(&prefix);
    let s = MaxPlus::<u32>::new();
    let win = Window {
        lo: n / 3,
        hi: n - 5,
    };
    let ws = format!("{}..{}", win.lo, win.hi);
    let run_all = |tag: &str, graph_present: bool| {
        for (w, wargs) in [
            (Window::full(n), vec![]),
            (win, vec!["--window", ws.as_str()]),
        ] {
            for (dir, ds) in [(Dir::Forward, "forward"), (Dir::Backward, "backward")] {
                let want = f.lib(&s, dir, &Const(1u32), &Init::Sources(0), w, true);
                // the default --load ram; a strict sub-window is memory-mapped
                let args = [
                    &[
                        "--semiring",
                        "maxplus-u32",
                        "--dir",
                        ds,
                        "--blocks",
                        &pre,
                        "--pred",
                    ][..],
                    &wargs,
                ]
                .concat();
                let out = f.dp(&format!("{tag}-{ds}-{}", w.lo), &args, 0);
                let pa = params(&out);
                assert_eq!(pa["graph_opened"], false);
                assert_eq!(pa["kernel"], "Blocks");
                assert_file_eq(&val_path::<u32>(&out), &want.val);
                assert_file_eq(
                    &io::array_path::<u32>(&out, "pred"),
                    want.pred.as_ref().unwrap(),
                );
                // --certify needs the graph
                let args = [&args[..], &["--certify"]].concat();
                let out = f.dp(
                    &format!("{tag}-{ds}-{}-c", w.lo),
                    &args,
                    if graph_present { 0 } else { 2 },
                );
                if graph_present {
                    assert_eq!(params(&out)["graph_opened"], true);
                }
            }
            let want = cpm::<u32, _, _>(
                &DpInput::graph(&cg),
                &Const(1u32),
                &conf(w, Kernel::Auto, false, 2),
                &mut quiet(),
            )
            .unwrap();
            let out = f.out(&format!("{tag}-cpm-{}", w.lo));
            let (base, outs) = (p(&f.base), p(&out));
            run_ok(
                &[
                    &["cpm", "--graph", &base, "--out", &outs, "--blocks", &pre][..],
                    &wargs,
                ]
                .concat(),
            );
            assert_eq!(params(&out)["graph_opened"], false);
            assert_file_eq(&io::array_path::<u32>(&out, "E"), &want.e);
            assert_file_eq(&io::array_path::<u32>(&out, "H"), &want.h);
            let args = [
                &[
                    "cpm",
                    "--graph",
                    &base,
                    "--out",
                    &outs,
                    "--blocks",
                    &pre,
                    "--critical-arcs",
                ][..],
                &wargs,
            ]
            .concat();
            run_expect(&args, if graph_present { 0 } else { 2 });
        }
    };
    run_all("with", true);
    // without the graph files (only the properties) block-only runs still work
    for ext in ["graph", "ef", "offsets"] {
        let _ = std::fs::remove_file(f.base.with_extension(ext));
    }
    run_all("without", false);
}

#[test]
fn cpm_kernel_push_runs_h_with_pull() {
    // regression: `cpm --kernel push` always failed (no backward push kernel)
    let f = fixture(random_dag(150, 0.03, 8));
    let n = f.g.num_nodes();
    let cg = chunked(&f.g, 4);
    let want = cpm::<u32, _, _>(
        &DpInput::graph(&cg),
        &Const(1u32),
        &conf(Window::full(n), Kernel::Auto, true, 2),
        &mut quiet(),
    )
    .unwrap();
    let out = f.out("cpm-push");
    run_ok(&[
        "cpm",
        "--graph",
        &p(&f.base),
        "--out",
        &p(&out),
        "--kernel",
        "push",
        "--path",
    ]);
    assert_eq!(params(&out)["kernel_forward"], "Push");
    assert_eq!(params(&out)["kernel_backward"], "Pull");
    assert_file_eq(&io::array_path::<u32>(&out, "E"), &want.e);
    assert_file_eq(&io::array_path::<u32>(&out, "H"), &want.h);
}

#[test]
fn windowed_node_weights_and_negative_min_plus_weights() {
    // regression: per-node weights and outdegrees are materialised for the window only
    let f = fixture(mini_pg(90, 21, &MiniPgOpts::default()).graph);
    let n = f.g.num_nodes();
    let win = Window {
        lo: n / 5,
        hi: 4 * n / 5,
    };
    let ws = format!("{}..{}", win.lo, win.hi);
    let du: Vec<u32> = (0..n).map(|u| (mix(u as u64 + 7) % 9) as u32).collect();
    let fu = f.out("w.u32le");
    io::write_array(&fu, &du).unwrap();
    let s = MaxPlus::<u64>::new();
    let w = Node {
        alpha: |u: usize| du[u] as u64,
        beta: |_: usize| 0u64,
    };
    let wu = format!("node-u32:{}", p(&fu));
    for (dir, ds) in [(Dir::Forward, "forward"), (Dir::Backward, "backward")] {
        let want = f.lib(&s, dir, &w, &Init::Sources(0), win, false);
        for kernel in ["auto", "pull", "blocks"] {
            let out = f.dp(
                &format!("nu-{ds}-{kernel}"),
                &[
                    "--semiring",
                    "maxplus-u64",
                    "--dir",
                    ds,
                    "--window",
                    &ws,
                    "--kernel",
                    kernel,
                    "--weights",
                    &wu,
                    "--certify",
                ],
                0,
            );
            assert_file_eq(&val_path::<u64>(&out), &want.val);
        }
    }
    // inv-outdeg over a window, from a windowed outdegree pass or from the index
    let s = MaxPlus::<f64>::new();
    let w = Node {
        alpha: |u: usize| match f.g.outdegree(u) {
            0 => 0.0,
            d => 1.0 / d as f64,
        },
        beta: |_: usize| 0.0,
    };
    let want = f.lib(&s, Dir::Forward, &w, &Init::Sources(0.0), win, false);
    for (blocks, kernel) in [("none", "pull"), ("auto", "blocks")] {
        let out = f.dp(
            &format!("io-{blocks}"),
            &[
                "--semiring",
                "maxplus-f64",
                "--window",
                &ws,
                "--weights",
                "inv-outdeg",
                "--blocks",
                blocks,
                "--kernel",
                kernel,
                "--certify",
            ],
            0,
        );
        assert_file_eq(&val_path::<f64>(&out), &want.val);
    }
    // min-plus over i64 accepts negative integral weights (Bellman–Ford oracle)
    let di: Vec<f64> = (0..n)
        .map(|u| (mix(u as u64 + 3) % 11) as f64 - 6.0)
        .collect();
    let fi = f.out("w.f64le");
    io::write_array(&fi, &di).unwrap();
    let wi = format!("node-f64:{}", p(&fi));
    let out = f.dp(
        "neg",
        &[
            "--semiring",
            "minplus-i64",
            "--weights",
            &wi,
            "--pred",
            "--certify",
        ],
        0,
    );
    let sources: Vec<usize> = (0..n)
        .filter(|&v| transpose(&f.g).outdegree(v) == 0)
        .collect();
    let bf = bellman_ford(&f.g, |u, _| di[u] as i64, &sources);
    let want: Vec<i64> = bf.iter().map(|d| d.unwrap()).collect();
    assert!(want.iter().any(|&d| d < 0));
    assert_file_eq(&val_path::<i64>(&out), &want);
    // a windowed run with the same file agrees with the library
    let s = MinPlus::<i64>::new();
    let w = Node {
        alpha: |u: usize| di[u] as i64,
        beta: |_: usize| 0i64,
    };
    let out = f.dp(
        "neg-win",
        &[
            "--semiring",
            "minplus-i64",
            "--weights",
            &wi,
            "--window",
            &ws,
            "--certify",
        ],
        0,
    );
    assert_file_eq(
        &val_path::<i64>(&out),
        &f.lib(&s, Dir::Forward, &w, &Init::Sources(0), win, false)
            .val,
    );
    // unsigned instances still reject negative weights, every integer instance fractions
    f.dp("neg-u", &["--semiring", "minplus-u32", "--weights", &wi], 2);
    let frac: Vec<f64> = (0..n).map(|_| -1.5).collect();
    io::write_array(&fi, &frac).unwrap();
    f.dp(
        "neg-frac",
        &["--semiring", "minplus-i64", "--weights", &wi],
        2,
    );
}
