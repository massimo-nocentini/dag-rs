//! `dag reach-index build|query|bench`: the exact reachability index ChronoChain.
//!
//! - `build` builds the index of [`crate::reach::index`] on the block quotient Q when a
//!   [`BlockIndex`] is available (`--blocks auto|PREFIX`; `auto` falls back to the graph itself
//!   if the graph is not a biclique digraph), else on the graph (`--blocks none`), restricted
//!   to `--window`. Writes `<out>.reach.*` and `<out>.json`.
//! - `query` answers `--pair U V` (one JSON line on stdout) or `--pairs FILE` (little-endian
//!   `u32` pairs of absolute node ids): with `--out P` it writes `P.answers.u8` (1 = reachable)
//!   and `P.json` (decider histogram), otherwise one line `u v answer decider visited` per pair.
//! - `bench` runs [`crate::reach::bench`] and exits with 1 unless every answer agrees with the
//!   cone ground truth.
//!
//! An index built on Q needs the same `BlockIndex` at query time (`--blocks PREFIX` avoids
//! rebuilding it). Query threads are capped by `--max-mem`: each needs at most
//! [`Scratch::max_bytes`] of scratch space (bitset, dirty-word list, DFS stack). The label
//! arrays are charged to `--max-mem` before they are loaded.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Subcommand};
use serde_json::json;

use super::common::{BlocksOpt, GlobalArgs, GraphOpt, OutOpt, WindowOpt, exit_status, logger};
use crate::blocks::BlockIndex;
use crate::graph::{BvRa, ChunkedGraph};
use crate::io::{self, FileInfo, MemBudget, NONE, RunInfo};
use crate::reach::bench::{self, BenchConf};
use crate::reach::dagview::{BlockQuotient, GraphDag, Mode, SuccDag};
use crate::reach::index::{Decider, QueryConf, ReachIndex, Scratch, ScratchPool, reach};
use crate::window::Window;

/// `dag reach-index`.
#[derive(ClapArgs, Debug)]
pub struct Args {
    /// The reach-index subcommand.
    #[command(subcommand)]
    pub cmd: ReachIndexCmd,
}

/// The reach-index subcommands.
#[derive(Subcommand, Debug)]
pub enum ReachIndexCmd {
    /// Builds the index.
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
        /// Number of chains.
        #[arg(long, default_value_t = 4)]
        chains: usize,
    },
    /// Answers reachability queries.
    Query {
        /// Input graph.
        #[command(flatten)]
        graph: GraphOpt,
        /// Block index.
        #[command(flatten)]
        blocks: BlocksOpt,
        /// Index prefix.
        #[arg(long)]
        index: PathBuf,
        /// One query `U V`.
        #[arg(long, num_args = 2, value_names = ["U", "V"])]
        pair: Option<Vec<usize>>,
        /// Queries as little-endian u32 pairs.
        #[arg(long)]
        pairs: Option<PathBuf>,
        /// Visit budget of the pruned DFS.
        #[arg(long, default_value_t = 1 << 20)]
        budget: u32,
        /// Output prefix (u8 answers + JSON stats).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Benchmarks the index against cone labels.
    Bench {
        /// Input graph.
        #[command(flatten)]
        graph: GraphOpt,
        /// Block index.
        #[command(flatten)]
        blocks: BlocksOpt,
        /// Index prefix.
        #[arg(long)]
        index: PathBuf,
        /// Number of cone sources.
        #[arg(long, default_value_t = 64)]
        sources: usize,
        /// Number of queries.
        #[arg(long, default_value_t = 100_000)]
        queries: usize,
        /// Random seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Abort if a single query takes longer.
        #[arg(long, default_value_t = 60.0)]
        max_query_secs: f64,
        /// Visit budget of the pruned DFS (0 forces the window scan on every undecided query).
        #[arg(long, default_value_t = 1 << 20)]
        budget: u32,
        /// Output prefix (JSON metrics).
        #[command(flatten)]
        out: OutOpt,
    },
}

/// Runs `dag reach-index`.
pub fn run(args: &Args, g: &GlobalArgs) -> Result<ExitCode> {
    match &args.cmd {
        ReachIndexCmd::Build {
            graph,
            blocks,
            window,
            out,
            chains,
        } => run_build(graph, blocks, window, out, *chains, g),
        ReachIndexCmd::Query {
            graph,
            blocks,
            index,
            pair,
            pairs,
            budget,
            out,
        } => run_query(
            graph,
            blocks,
            index,
            pair.as_deref(),
            pairs.as_deref(),
            *budget,
            out.as_deref(),
            g,
        ),
        ReachIndexCmd::Bench {
            graph,
            blocks,
            index,
            sources,
            queries,
            seed,
            max_query_secs,
            budget,
            out,
        } => {
            let conf = BenchConf {
                sources: *sources,
                queries: *queries,
                seed: *seed,
                max_query_secs: *max_query_secs,
                query: QueryConf { budget: *budget },
                threads: 0,
            };
            run_bench(graph, blocks, index, conf, out, g)
        }
    }
}

/// The DAG of an index: Q of `blocks` or the graph, restricted to `w`.
fn make_dag<'a>(
    cg: &'a ChunkedGraph<BvRa>,
    blocks: Option<&'a BlockIndex>,
    w: Window,
    g: &GlobalArgs,
) -> Box<dyn SuccDag + 'a> {
    match blocks {
        Some(b) => Box::new(BlockQuotient::with_window(b, w)),
        None => Box::new(GraphDag::with_window(cg, g.sweep_conf(), w)),
    }
}

fn run_build(
    graph: &GraphOpt,
    blocks: &BlocksOpt,
    window: &WindowOpt,
    out: &OutOpt,
    chains: usize,
    g: &GlobalArgs,
) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = out.prepare()?.to_owned();
    let cg = graph.open(g)?;
    let n = cg.num_nodes();
    let w = window.window(n)?;
    let mut budget = g.budget();
    let bi = blocks.resolve(&cg, g, &mut budget)?;
    let dag = make_dag(&cg, bi.as_ref(), w, g);
    let n_dag = dag.num_nodes();
    budget.need(ReachIndex::build_bytes(n_dag, chains), "reach index build")?;
    log::info!(
        "building ChronoChain on {} with {n_dag} DAG nodes, {chains} chains, window {}..{}",
        if bi.is_some() {
            "the block quotient"
        } else {
            "the graph"
        },
        w.lo,
        w.hi
    );
    let mut pl = logger(if bi.is_some() { "block" } else { "node" });
    let (idx, stats) = ReachIndex::build_with_stats(&*dag, chains, &mut pl)?;
    let build_s = t0.elapsed().as_secs_f64();
    let t = Instant::now();
    idx.save(&out)?;
    let save_s = t.elapsed().as_secs_f64();
    log::info!(
        "index: {} chains of lengths {:?}, max lvl {}, {:.2} GB of labels; build {build_s:.1} s, save {save_s:.1} s",
        idx.k(),
        stats.chain_lens,
        stats.max_lvl,
        stats.label_bytes as f64 / io::GB
    );
    let mut info = RunInfo::new(Some(graph.graph.clone()), n, cg.num_arcs, w);
    let k = idx.k();
    for (i, path) in ReachIndex::files(&out, k).into_iter().enumerate() {
        let (len, sentinel, meaning) = match i {
            0 => (
                n_dag,
                None,
                "longest path (arcs) from a DAG source".to_owned(),
            ),
            1 => (n_dag, None, "longest path (arcs) to a DAG sink".to_owned()),
            2 => (
                n_dag,
                None,
                "largest DAG node reachable (reflexive)".to_owned(),
            ),
            3 => (
                n_dag,
                None,
                "smallest DAG node reaching (reflexive)".to_owned(),
            ),
            _ => {
                let c = (i - 4) / 3;
                match (i - 4) % 3 {
                    0 => (
                        idx.chains[c].nodes.len(),
                        None,
                        format!("chain {c}: DAG nodes along the path"),
                    ),
                    1 => (
                        n_dag,
                        Some("u32::MAX = +inf"),
                        format!("chain {c}: smallest position reachable"),
                    ),
                    _ => (
                        n_dag,
                        Some("0 = -inf"),
                        format!("chain {c}: 1 + largest position reaching"),
                    ),
                }
            }
        };
        info.files
            .push(FileInfo::new::<u32>(path, len, sentinel, meaning));
    }
    info.params = json!({ "chains": chains, "mode": idx.mode, "dag": idx.info });
    info.results = json!({ "stats": stats, "build_s": build_s, "save_s": save_s });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(true))
}

/// Loads an index and the matching DAG inputs (graph, and the BlockIndex for a Q index).
fn open_index(
    graph: &GraphOpt,
    blocks: &BlocksOpt,
    index: &Path,
    g: &GlobalArgs,
    budget: &mut MemBudget,
) -> Result<(ReachIndex, ChunkedGraph<BvRa>, Option<BlockIndex>)> {
    let meta = ReachIndex::load_meta(index)?;
    budget.need(
        ReachIndex::label_bytes(meta.n_dag, meta.chain_lens.len()),
        "reach index",
    )?;
    let idx = ReachIndex::load(index, g.load)?;
    let cg = graph.open(g)?;
    anyhow::ensure!(
        cg.num_nodes() == idx.info.n,
        "the index was built on a graph with {} nodes, {} has {}",
        idx.info.n,
        graph.graph.display(),
        cg.num_nodes()
    );
    let bi = match idx.mode {
        Mode::Graph => None,
        Mode::Blocks => {
            let bi = blocks.resolve(&cg, g, budget)?;
            anyhow::ensure!(
                bi.is_some(),
                "{} was built on the block quotient: pass --blocks auto or --blocks PREFIX",
                index.display()
            );
            bi
        }
    };
    Ok((idx, cg, bi))
}

/// Checks that `dag` is the DAG `idx` was built on.
fn check_dag(idx: &ReachIndex, dag: &dyn SuccDag) -> Result<()> {
    anyhow::ensure!(
        dag.info() == idx.info && dag.num_nodes() == idx.n_dag,
        "the index was built on {:?} ({} DAG nodes), the inputs give {:?} ({} DAG nodes)",
        idx.info,
        idx.n_dag,
        dag.info(),
        dag.num_nodes()
    );
    Ok(())
}

/// Largest successor list assumed by the per-thread scratch estimate (a larger list only grows
/// one reusable buffer).
const MAX_SUCC_ESTIMATE: usize = 1 << 17;

/// Query threads allowed by the memory budget: each needs at most [`Scratch::max_bytes`].
fn query_threads(n_dag: usize, visit_budget: u32, budget: &MemBudget) -> usize {
    let per = Scratch::max_bytes(n_dag, visit_budget, MAX_SUCC_ESTIMATE).max(1);
    let by_mem = (budget.available() / per).max(1) as usize;
    rayon::current_num_threads().min(by_mem).max(1)
}

#[allow(clippy::too_many_arguments)]
fn run_query(
    graph: &GraphOpt,
    blocks: &BlocksOpt,
    index: &Path,
    pair: Option<&[usize]>,
    pairs: Option<&Path>,
    budget_visits: u32,
    out: Option<&Path>,
    g: &GlobalArgs,
) -> Result<ExitCode> {
    let t0 = Instant::now();
    anyhow::ensure!(
        pair.is_some() != pairs.is_some(),
        "pass exactly one of --pair U V and --pairs FILE"
    );
    let mut budget = g.budget();
    let (idx, cg, bi) = open_index(graph, blocks, index, g, &mut budget)?;
    let w = Window {
        lo: idx.info.lo,
        hi: idx.info.hi,
    };
    let dag = make_dag(&cg, bi.as_ref(), w, g);
    check_dag(&idx, &*dag)?;
    let qs: Vec<(u32, u32)> = match (pair, pairs) {
        (Some(p), _) => {
            anyhow::ensure!(p.len() == 2, "--pair takes two node ids");
            anyhow::ensure!(
                p[0] < NONE as usize && p[1] < NONE as usize,
                "node ids must fit u32"
            );
            vec![(p[0] as u32, p[1] as u32)]
        }
        (None, Some(f)) => io::read_array::<(u32, u32)>(f)
            .with_context(|| format!("cannot read pairs from {}", f.display()))?,
        (None, None) => unreachable!(),
    };
    if let Some(&(u, v)) = qs
        .iter()
        .find(|&&(u, v)| !w.contains(u as usize) || !w.contains(v as usize))
    {
        anyhow::bail!(
            "query ({u}, {v}) lies outside the index window {}..{}",
            w.lo,
            w.hi
        );
    }
    let q = QueryConf {
        budget: budget_visits,
    };
    let threads = query_threads(idx.n_dag, budget_visits, &budget);
    let pool = ScratchPool::new(threads)?;
    let tq = Instant::now();
    let answers: Vec<(bool, Decider, u32)> = pool.map(&qs, |&(u, v), s| {
        reach(&idx, &*dag, bi.as_ref(), u as usize, v as usize, &q, s)
    });
    let query_s = tq.elapsed().as_secs_f64();
    let mut hist = std::collections::BTreeMap::<String, u64>::new();
    for (_, d, _) in &answers {
        *hist.entry(d.label()).or_default() += 1;
    }
    let positives = answers.iter().filter(|a| a.0).count();
    if pair.is_some() {
        let (r, d, vis) = answers[0];
        println!(
            "{}",
            json!({ "u": qs[0].0, "v": qs[0].1, "reachable": r, "decider": d.label(), "visited": vis })
        );
    } else if out.is_none() {
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut lock = std::io::BufWriter::new(stdout.lock());
        for (&(u, v), (r, d, vis)) in qs.iter().zip(&answers) {
            writeln!(lock, "{u} {v} {} {} {vis}", u8::from(*r), d.label())?;
        }
        lock.flush()?;
    }
    if let Some(out) = out {
        if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let ans: Vec<u8> = answers.iter().map(|a| u8::from(a.0)).collect();
        let path = io::suffixed(out, "answers.u8");
        io::write_array(&path, &ans)?;
        let mut info = RunInfo::new(Some(graph.graph.clone()), cg.num_nodes(), cg.num_arcs, w);
        info.files.push(FileInfo::new::<u8>(
            path,
            ans.len(),
            None,
            "1 if the pair's target is reachable from its source",
        ));
        info.params =
            json!({ "index": index, "budget": budget_visits, "threads": threads, "pairs": pairs });
        info.results = json!({
            "queries": qs.len(), "reachable": positives, "deciders": hist,
            "query_s": query_s, "qps": qs.len() as f64 / query_s.max(1e-9),
        });
        info.elapsed_s = t0.elapsed().as_secs_f64();
        info.write(out)?;
    }
    log::info!(
        "{} queries, {positives} reachable, {query_s:.3} s on {threads} threads; deciders {hist:?}",
        qs.len()
    );
    Ok(exit_status(true))
}

fn run_bench(
    graph: &GraphOpt,
    blocks: &BlocksOpt,
    index: &PathBuf,
    mut conf: BenchConf,
    out: &OutOpt,
    g: &GlobalArgs,
) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = out.prepare()?.to_owned();
    let mut budget = g.budget();
    let (idx, cg, bi) = open_index(graph, blocks, index, g, &mut budget)?;
    let w = Window {
        lo: idx.info.lo,
        hi: idx.info.hi,
    };
    let dag = make_dag(&cg, bi.as_ref(), w, g);
    check_dag(&idx, &*dag)?;
    // cone masks: 8 B per DAG node
    budget.need(idx.n_dag as u64 * 8, "cone masks")?;
    conf.threads = query_threads(idx.n_dag, conf.query.budget, &budget);
    let mut pl = logger(if bi.is_some() { "block" } else { "node" });
    let rep = bench::run_bench(&idx, &*dag, &conf, &mut pl)?;
    let ok = rep.disagreements() == 0;
    if !ok {
        log::error!("{} answers disagree with the cones", rep.disagreements());
    }
    let mut info = RunInfo::new(Some(graph.graph.clone()), cg.num_nodes(), cg.num_arcs, w);
    info.params = json!({ "index": index, "conf": conf, "dag": idx.info, "chains": idx.k() });
    info.results = serde_json::to_value(&rep)?;
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}
