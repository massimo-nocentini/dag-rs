//! `dag dom build|verify|stats`: dominator and post-dominator trees.
//!
//! - `dom build` builds the dominator tree (`--only dom`), the post-dominator tree
//!   (`--only post`) or both (default; concurrently with `rayon::join` when `--max-mem` allows
//!   it). With a [`BlockIndex`] (`--blocks auto|PREFIX`) the builders are decode-free
//!   ([`dominators_blocks`], [`post_dominators_blocks`]); otherwise (`--blocks none`, or `auto`
//!   on a graph that is not a biclique digraph) they scan the forward graph
//!   ([`dominators_push`], [`post_dominators_pull`]). Outputs, indexed by `v − lo`:
//!   `P.idom.u32le` / `P.ipdom.u32le` (absolute ids, `u32::MAX` = virtual root/sink),
//!   `P.dom.depth.u32le` / `P.pdom.depth.u32le`, and with `--sizes` / `--pre`
//!   `P.{dom,pdom}.{size,pre}.u32le`. The JSON reports the [`NcaStats`](crate::dom::NcaStats),
//!   the timings, the [`DomStats`] summary, the
//!   [`DomDistributions`](crate::dom::DomDistributions) of depth, size and root children (per
//!   node range with `--buckets`) and, with `--depth-bound`, the check
//!   `depth ≤ dmin + 1` ([`DepthBound`](crate::dom::DepthBound)).
//! - `dom verify --idom FILE [--post]` runs the independent verifier
//!   ([`verify_dominators`] / [`verify_post_dominators`]) on the forward graph and exits with 1
//!   if the candidate is not the tree.
//! - `dom stats --prefix P` recomputes [`DomStats`] from the files written by the last
//!   `dom build --out P` (window, trees and `--sizes` read from `P.json`; arrays left at `P` by
//!   earlier runs are ignored), with the distributions (`--buckets`), and writes `P.stats.json`
//!   and one TSV per tree,
//!   `P.dom.top.tsv` / `P.pdom.top.tsv` (`node\tsize\tdepth\tidom`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use serde_json::{Value, json};

use super::common::{
    BlocksOpt, BucketsOpt, GlobalArgs, GraphOpt, OutOpt, WindowOpt, concurrent_logger, exit_status,
    logger,
};
use crate::blocks::BlockIndex;
use crate::dom::stats::{
    self, DomStats, dmin_blocks, dmin_sweep, dom_distributions, sizes_from_idom, tree_distributions,
};
use crate::dom::{
    DomConf, DomTree, check_depth_bound, dominators_blocks, dominators_push,
    post_dominators_blocks, post_dominators_pull, tree_stats, verify_dominators,
    verify_post_dominators,
};
use crate::graph::{BvRa, ChunkedGraph};
use crate::io::{FileInfo, MemBudget, RunInfo, U32Array, array_path, suffixed};
use crate::summary::{Buckets, DistConf};
use crate::window::Window;

/// `dag dom`.
#[derive(ClapArgs, Debug)]
pub struct Args {
    /// The dom subcommand.
    #[command(subcommand)]
    pub cmd: DomCmd,
}

/// Which tree(s) to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DomOnly {
    /// Dominators only.
    Dom,
    /// Post-dominators only.
    Post,
}

/// The dom subcommands.
#[derive(Subcommand, Debug)]
pub enum DomCmd {
    /// Builds the dominator and/or post-dominator tree.
    Build {
        /// Input graph.
        #[command(flatten)]
        graph: GraphOpt,
        /// Block index.
        #[command(flatten)]
        blocks: BlocksOpt,
        /// Window.
        #[command(flatten)]
        window: WindowOpt,
        /// Output prefix.
        #[command(flatten)]
        out: OutOpt,
        /// Build only one tree.
        #[arg(long, value_enum)]
        only: Option<DomOnly>,
        /// Budget of the NCA walk before the jump-pointer fallback.
        #[arg(long, default_value_t = 64)]
        budget: u32,
        /// Also write subtree sizes.
        #[arg(long)]
        sizes: bool,
        /// Also write preorder numbers.
        #[arg(long)]
        pre: bool,
        /// Also compute the shortest distance from the window sources (to the window sinks)
        /// and check depth ≤ dmin + 1.
        #[arg(long)]
        depth_bound: bool,
        /// Length of the top-k lists (by subtree size, and of the distributions) in the JSON
        /// summary.
        #[arg(long, default_value_t = 20)]
        top: usize,
        /// Buckets of the distributions.
        #[command(flatten)]
        buckets: BucketsOpt,
    },
    /// Verifies an immediate-dominator array.
    Verify {
        /// Input graph.
        #[command(flatten)]
        graph: GraphOpt,
        /// Window.
        #[command(flatten)]
        window: WindowOpt,
        /// Output prefix (JSON report).
        #[command(flatten)]
        out: OutOpt,
        /// The idom (or ipdom) array.
        #[arg(long)]
        idom: PathBuf,
        /// Verify post-dominators.
        #[arg(long)]
        post: bool,
    },
    /// Statistics of a built tree.
    Stats {
        /// Prefix of the tree files.
        #[arg(long)]
        prefix: PathBuf,
        /// Number of top nodes by subtree size.
        #[arg(long, default_value_t = 100)]
        top: usize,
        /// Buckets of the distributions.
        #[command(flatten)]
        buckets: BucketsOpt,
    },
}

/// Runs `dag dom`.
pub fn run(args: &Args, g: &GlobalArgs) -> Result<ExitCode> {
    match &args.cmd {
        DomCmd::Build {
            graph,
            blocks,
            window,
            out,
            only,
            budget,
            sizes,
            pre,
            depth_bound,
            top,
            buckets,
        } => {
            let n = crate::graph::read_properties(&graph.graph)?.0;
            let opts = BuildOpts {
                dom: *only != Some(DomOnly::Post),
                post: *only != Some(DomOnly::Dom),
                budget: *budget,
                sizes: *sizes,
                pre: *pre,
                depth_bound: *depth_bound,
                top: *top,
                buckets: buckets.load(n)?,
            };
            run_build(graph, blocks, window, out, &opts, g)
        }
        DomCmd::Verify {
            graph,
            window,
            out,
            idom,
            post,
        } => run_verify(graph, window, out, idom, *post, g),
        DomCmd::Stats {
            prefix,
            top,
            buckets,
        } => run_stats(prefix, *top, buckets, g),
    }
}

/// Options of `dom build`.
struct BuildOpts {
    dom: bool,
    post: bool,
    budget: u32,
    sizes: bool,
    pre: bool,
    depth_bound: bool,
    top: usize,
    buckets: Option<Buckets>,
}

/// The input of the builders.
enum Source<'a> {
    Blocks(&'a BlockIndex),
    Graph(&'a ChunkedGraph<BvRa>),
}

/// Outcome of one tree: JSON results and output files.
struct TreeOut {
    results: Value,
    files: Vec<FileInfo>,
}

/// Estimated bytes of one tree run (tree, accumulators, sizes, preorder, dmin, sweep buffers).
fn tree_bytes(n: usize, nblocks: usize, opts: &BuildOpts, sweep: u64) -> u64 {
    let node = n as u64 + 1;
    let mut b = DomTree::bytes(n) + 4 * nblocks as u64 + 4 * node + sweep;
    if opts.pre {
        b += 8 * node; // pre + temporary next
    }
    if opts.depth_bound {
        b += 4 * node;
    }
    b
}

fn build_one(
    src: &Source<'_>,
    w: Window,
    post: bool,
    opts: &BuildOpts,
    g: &GlobalArgs,
    out: &Path,
) -> Result<TreeOut> {
    let mut conf = DomConf::new(w);
    conf.budget = opts.budget;
    conf.sweep = g.sweep_conf();
    let mut pl = logger("node");
    let t0 = Instant::now();
    let tree = match (src, post) {
        (Source::Blocks(b), false) => dominators_blocks(b, &conf, &mut pl)?,
        (Source::Blocks(b), true) => post_dominators_blocks(b, &conf, &mut pl)?,
        (Source::Graph(cg), false) => dominators_push(cg, &conf, &mut pl)?,
        (Source::Graph(cg), true) => post_dominators_pull(cg, &conf, &mut pl)?,
    };
    let build_s = t0.elapsed().as_secs_f64();
    let (idom_name, tree_name) = tree.field_names();
    log::info!(
        "{tree_name}: built in {build_s:.2} s, {} NCA calls, {} top exits, {} jump fallbacks",
        tree.stats.calls,
        tree.stats.top_exits,
        tree.stats.jump_fallbacks
    );
    let t1 = Instant::now();
    let sizes = tree.sizes();
    let pre = opts.pre.then(|| tree.preorder(&sizes));
    let derived_s = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    tree.write(out, opts.sizes.then_some(&sizes[..]), pre.as_deref())?;
    let write_s = t2.elapsed().as_secs_f64();
    drop(pre);
    let t3 = Instant::now();
    let summary = tree_stats(&tree, &sizes, opts.top);
    let dists = tree_distributions(
        &tree,
        &sizes,
        &DistConf {
            top: opts.top,
            buckets: opts.buckets.as_ref(),
        },
    );
    let summary_s = t3.elapsed().as_secs_f64();
    drop(sizes);
    log::info!(
        "{tree_name}: {} root children, {} cut nodes, max depth {} (node {:?}), mean depth {:.3}",
        summary.root_children,
        summary.cut_nodes,
        summary.max_depth,
        summary.argmax_depth,
        summary.mean_depth
    );
    let bound = if opts.depth_bound {
        let mut pl = logger("node");
        let d = match src {
            Source::Blocks(b) => dmin_blocks(b, w, post, &mut pl),
            Source::Graph(cg) => dmin_sweep(cg, w, &conf.sweep, post, &mut pl)?,
        };
        let r = check_depth_bound(&tree, &d);
        log::info!("{tree_name}: depth bound {r:?}");
        Some(r)
    } else {
        None
    };
    let n = w.len();
    let sentinel = Some("u32::MAX = NONE (virtual root/sink)");
    let mut files = vec![
        FileInfo::new::<u32>(
            array_path::<u32>(out, idom_name),
            n,
            sentinel,
            if post {
                "immediate post-dominator (absolute id), indexed by v - lo"
            } else {
                "immediate dominator (absolute id), indexed by v - lo"
            },
        ),
        FileInfo::new::<u32>(
            array_path::<u32>(out, &format!("{tree_name}.depth")),
            n,
            None,
            "depth in the tree (root children 1)",
        ),
    ];
    if opts.sizes {
        files.push(FileInfo::new::<u32>(
            array_path::<u32>(out, &format!("{tree_name}.size")),
            n,
            None,
            "number of nodes in the subtree, itself included",
        ));
    }
    if opts.pre {
        files.push(FileInfo::new::<u32>(
            array_path::<u32>(out, &format!("{tree_name}.pre")),
            n,
            None,
            "preorder number (root 0): d dominates v iff pre[d] <= pre[v] < pre[d] + size[d]",
        ));
    }
    let results = json!({
        "kernel": match src { Source::Blocks(_) => "blocks", Source::Graph(_) => "generic" },
        "build_s": build_s,
        "sizes_pre_s": derived_s,
        "write_s": write_s,
        "summary_s": summary_s,
        "nca": tree.stats,
        "stats": summary,
        "distributions": dists,
        "depth_bound": bound,
    });
    Ok(TreeOut { results, files })
}

fn run_build(
    graph: &GraphOpt,
    blocks: &BlocksOpt,
    window: &WindowOpt,
    out: &OutOpt,
    opts: &BuildOpts,
    g: &GlobalArgs,
) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = out.prepare()?.to_owned();
    let cg = graph.open(g)?;
    let n = cg.num_nodes();
    let w = window.window(n)?;
    let mut budget: MemBudget = g.budget();
    let idx = blocks.resolve(&cg, g, &mut budget)?;
    let src = match &idx {
        Some(b) => Source::Blocks(b),
        None => Source::Graph(&cg),
    };
    let (nblocks, sweep) = match &idx {
        Some(b) => (b.block_bound(w.hi) - b.block_bound(w.lo), 0),
        None => {
            let c = g.chunk_conf();
            let d = g.sweep_conf().decoders as u64;
            (0, 2 * d * (4 * c.chunk_arcs + 4 * c.max_chunk_nodes as u64))
        }
    };
    let one = tree_bytes(w.len(), nblocks, opts, sweep);
    let concurrent = opts.dom && opts.post && budget.available() >= 2 * one;
    let open_s = t0.elapsed().as_secs_f64();
    let tk = Instant::now();
    let mut results = serde_json::Map::new();
    let mut files = Vec::new();
    let mut collect = |key: &str, r: TreeOut| {
        results.insert(key.to_owned(), r.results);
        files.extend(r.files);
    };
    if concurrent {
        budget.need(2 * one, "dominator and post-dominator trees")?;
        log::info!("building both trees concurrently");
        let (d, p) = rayon::join(
            || build_one(&src, w, false, opts, g, &out),
            || build_one(&src, w, true, opts, g, &out),
        );
        collect("dom", d?);
        collect("pdom", p?);
        budget.release(2 * one);
    } else {
        for (key, post, on) in [("dom", false, opts.dom), ("pdom", true, opts.post)] {
            if on {
                budget.need(
                    one,
                    if post {
                        "post-dominator tree"
                    } else {
                        "dominator tree"
                    },
                )?;
                collect(key, build_one(&src, w, post, opts, g, &out)?);
                budget.release(one);
            }
        }
    }
    let mut info = RunInfo::new(Some(graph.graph.clone()), n, cg.num_arcs, w);
    info.phases_s.add("open", open_s);
    // the trees (build, sizes, write, summary of each; concurrent trees overlap)
    info.phases_s.add("kernel", tk.elapsed().as_secs_f64());
    info.files = files;
    info.params = json!({
        "dom": opts.dom,
        "post": opts.post,
        "budget": opts.budget,
        "sizes": opts.sizes,
        "pre": opts.pre,
        "depth_bound": opts.depth_bound,
        "top": opts.top,
        "buckets": opts.buckets.as_ref().and_then(|b| b.source.clone()),
        "blocks": idx.is_some(),
        "concurrent": concurrent,
    });
    let mut ok = true;
    for r in results.values() {
        if let Some(v) = r["depth_bound"]["violations"].as_u64() {
            ok &= v == 0;
        }
    }
    info.results = Value::Object(results);
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}

fn run_verify(
    graph: &GraphOpt,
    window: &WindowOpt,
    out: &OutOpt,
    idom: &Path,
    post: bool,
    g: &GlobalArgs,
) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = out.prepare()?.to_owned();
    let cg = graph.open(g)?;
    let n = cg.num_nodes();
    let w = window.window(n)?;
    let mut budget = g.budget();
    budget.need(
        crate::dom::verify::verify_bytes(w.len()) + 4 * w.len() as u64,
        "dominator verifier",
    )?;
    let t = U32Array::load(idom, g.load)?;
    let open_s = t0.elapsed().as_secs_f64();
    let tv = Instant::now();
    let mut pl = concurrent_logger("node");
    let report = if post {
        verify_post_dominators(&cg, &t, w, &mut pl)?
    } else {
        verify_dominators(&cg, &t, w, &mut pl)?
    };
    drop(pl);
    if report.ok {
        log::info!("verification passed: {report:?}");
    } else {
        log::warn!("verification FAILED: {report:?}");
    }
    let mut info = RunInfo::new(Some(graph.graph.clone()), n, cg.num_arcs, w);
    info.phases_s.add("open", open_s);
    info.phases_s.add("verify", tv.elapsed().as_secs_f64());
    info.params = json!({ "idom": idom, "post": post });
    info.results = json!({ "report": report });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(report.ok))
}

/// What `dom stats` reads from the `<prefix>.json` written by `dom build`.
struct BuildJson {
    window: Window,
    n: usize,
    m: u64,
    graph: Option<PathBuf>,
    /// Trees written by that run (`params.dom`, `params.post`).
    trees: [bool; 2],
    /// Whether that run wrote the subtree sizes (`params.sizes`).
    sizes: bool,
}

/// Reads the window, the graph size and the build parameters from the `<prefix>.json` written
/// by `dom build`. Only the arrays written by that run are read afterwards: files left at the
/// same prefix by an earlier run (another `--only`, `--sizes` or window) are ignored.
fn read_build_json(prefix: &Path) -> Result<BuildJson> {
    let path = suffixed(prefix, "json");
    let s = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let v: Value =
        serde_json::from_str(&s).with_context(|| format!("cannot parse {}", path.display()))?;
    let get = |k: &str| -> Result<u64> {
        v["window"][k]
            .as_u64()
            .with_context(|| format!("{}: missing window.{k}", path.display()))
    };
    let w = Window {
        lo: get("lo")? as usize,
        hi: get("hi")? as usize,
    };
    let n = v["n"].as_u64().unwrap_or(w.hi as u64) as usize;
    let m = v["m"].as_u64().unwrap_or(0);
    let graph = v["graph"].as_str().map(PathBuf::from);
    let flag = |k: &str| -> Result<bool> {
        v["params"][k]
            .as_bool()
            .with_context(|| format!("{}: missing params.{k}", path.display()))
    };
    Ok(BuildJson {
        window: w,
        n,
        m,
        graph,
        trees: [flag("dom")?, flag("post")?],
        sizes: flag("sizes")?,
    })
}

fn run_stats(prefix: &Path, top: usize, buckets: &BucketsOpt, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let BuildJson {
        window: w,
        n,
        m,
        graph,
        trees,
        sizes,
    } = read_build_json(prefix)?;
    let buckets = buckets.load(n)?;
    let conf = DistConf {
        top,
        buckets: buckets.as_ref(),
    };
    let mut results = serde_json::Map::new();
    let mut files = Vec::new();
    let mut budget = g.budget();
    let names = [("idom", "dom", false), ("ipdom", "pdom", true)];
    for ((idom_name, tree_name, mirrored), built) in names.into_iter().zip(trees) {
        if !built {
            continue;
        }
        let idom_path = array_path::<u32>(prefix, idom_name);
        budget.need(12 * w.len() as u64, "tree arrays")?;
        let idom = U32Array::load(&idom_path, g.load)?;
        anyhow::ensure!(
            idom.len() == w.len(),
            "{} has {} entries, the window {} nodes",
            idom_path.display(),
            idom.len(),
            w.len()
        );
        let depth = U32Array::load(
            &array_path::<u32>(prefix, &format!("{tree_name}.depth")),
            g.load,
        )?;
        let size_path = array_path::<u32>(prefix, &format!("{tree_name}.size"));
        let size = if sizes {
            U32Array::load(&size_path, g.load)?
        } else {
            U32Array::from_vec(sizes_from_idom(&idom, w, mirrored))
        };
        let st: DomStats = stats::dom_stats(&idom, &depth, &size, w, top);
        let tsv = suffixed(prefix, &format!("{tree_name}.top.tsv"));
        stats::write_top_tsv(&tsv, &st)?;
        files.push(FileInfo::new::<u32>(
            tsv,
            st.top.len(),
            None,
            "TSV node, size, depth, idom ('-' = root) of the largest subtrees",
        ));
        log::info!(
            "{tree_name}: {} root children, {} cut nodes, max depth {}",
            st.root_children,
            st.cut_nodes,
            st.max_depth
        );
        let roots_ok = st.root_children_size_sum == w.len() as u64;
        anyhow::ensure!(
            roots_ok,
            "{tree_name}: the root children cover {} nodes, the window has {}",
            st.root_children_size_sum,
            w.len()
        );
        let mut v = serde_json::to_value(&st)?;
        v["distributions"] =
            serde_json::to_value(dom_distributions(&idom, &depth, &size, w, &conf))?;
        results.insert(tree_name.to_owned(), v);
        budget.release(12 * w.len() as u64);
    }
    anyhow::ensure!(
        !results.is_empty(),
        "{}.json lists no tree",
        prefix.display()
    );
    let mut info = RunInfo::new(graph, n, m, w);
    info.files = files;
    info.params = json!({
        "prefix": prefix,
        "top": top,
        "buckets": buckets.as_ref().and_then(|b| b.source.clone()),
    });
    info.results = Value::Object(results);
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&suffixed(prefix, "stats"))?;
    Ok(ExitCode::SUCCESS)
}
