//! `dag topo-check`, `dag toposort` and `dag window`.
//!
//! - `topo-check` scans the graph (and the transpose, when available) in parallel and reports
//!   whether the identity is a topological order, plus sinks and sources (exit 1 if not);
//! - `toposort` relabels a graph in topological order ([`crate::topo::relabel_bv`]);
//! - `window` writes the subgraph induced by `--window a..b` with ids shifted by `−a`.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use serde_json::json;

use super::common::{GlobalArgs, GraphOpt, OutOpt, concurrent_logger, exit_status, logger};
use crate::graph;
use crate::io::{FileInfo, RunInfo, suffixed};
use crate::topo;
use crate::window::Window;

/// `dag topo-check`.
#[derive(Args, Debug)]
pub struct TopoCheckArgs {
    /// Input graph (the transpose, when present, is scanned too and gives the sources).
    #[command(flatten)]
    pub graph: GraphOpt,
    /// Do not scan the transpose even if it exists.
    #[arg(long)]
    pub no_transpose: bool,
    /// Output prefix of the JSON report (the report is also printed on stdout).
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// `dag toposort`.
#[derive(Args, Debug)]
pub struct ToposortArgs {
    /// Input graph.
    #[command(flatten)]
    pub graph: GraphOpt,
    /// Output basename of the relabelled graph.
    #[command(flatten)]
    pub out: OutOpt,
    /// Directory for temporary sort batches (default: `<out>.tmp`).
    #[arg(long)]
    pub tmp: Option<PathBuf>,
}

/// `dag window`.
#[derive(Args, Debug)]
pub struct WindowArgs {
    /// Input graph.
    #[command(flatten)]
    pub graph: GraphOpt,
    /// The window `a..b` to extract.
    #[arg(long)]
    pub window: String,
    /// Output basename of the induced subgraph.
    #[command(flatten)]
    pub out: OutOpt,
    /// Directory for temporary sort batches (default: `<out>.tmp`).
    #[arg(long)]
    pub tmp: Option<PathBuf>,
}

/// Runs `dag topo-check`.
pub fn run_topo_check(args: &TopoCheckArgs, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let cg = args.graph.open(g)?;
    let ct = if args.no_transpose {
        None
    } else {
        args.graph.open_transpose(g)?
    };
    let mut pl = concurrent_logger("node");
    let rep = topo::check_identity(&cg, ct.as_ref(), &mut pl)?;
    let m_prop = cg.num_arcs;
    let ok = rep.identity_topological && rep.m == m_prop;
    if rep.m != m_prop {
        log::error!("scan counted {} arcs, the graph declares {m_prop}", rep.m);
    }
    log::info!(
        "identity topological: {}; n = {}, m = {}, sinks = {}, sources = {:?}",
        rep.identity_topological,
        rep.n,
        rep.m,
        rep.sinks,
        rep.sources
    );
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if let Some(out) = &args.out {
        let mut info = RunInfo::new(
            Some(args.graph.graph.clone()),
            rep.n,
            m_prop,
            Window::full(rep.n),
        );
        info.params = json!({ "transpose": ct.as_ref().map(|_| args.graph.transpose_path()) });
        info.results = serde_json::to_value(&rep)?;
        info.elapsed_s = t0.elapsed().as_secs_f64();
        info.write(out)?;
    }
    Ok(exit_status(ok))
}

/// Runs `dag toposort`.
pub fn run_toposort(args: &ToposortArgs, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = args.out.prepare()?.to_owned();
    let tmp = args.tmp.clone().unwrap_or_else(|| suffixed(&out, "tmp"));
    let bv = graph::load_ra(&args.graph.graph)?;
    let n = bv.num_nodes();
    let m = bv.num_arcs();
    let n64 = n as u64;
    let mut budget = g.budget();
    // order + rank (8 B/node each), the DFS visited bitset (n/8), then the u32 inverse
    // permutation of relabel_bv (4 B/node); the DFS stack grows with the depth of the visit
    budget.need(
        16 * n64 + n64.div_ceil(8) + 4 * n64,
        "toposort order + rank + visited + inverse",
    )?;
    let mem = topo::sort_memory(&mut budget, "toposort external-sort buffers")?;
    let mut pl = logger("node");
    let topo::TopoOrder { order, rank } = topo::topo_order(&bv, &mut pl)?;
    drop(order);
    drop(bv);
    topo::relabel_bv_with(&args.graph.graph, &out, &rank, &tmp, mem, &mut pl)?;
    let _ = std::fs::remove_dir(&tmp);
    let mut info = RunInfo::new(Some(args.graph.graph.clone()), n, m, Window::full(n));
    info.files = vec![
        FileInfo::new::<u32>(suffixed(&out, "order.u32le"), n, None, "new id -> old id"),
        FileInfo::new::<u32>(suffixed(&out, "rank.u32le"), n, None, "old id -> new id"),
    ];
    info.results = json!({ "output": out, "transpose": suffixed(&out, "t") });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(true))
}

/// Runs `dag window`.
pub fn run_window(args: &WindowArgs, g: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = args.out.prepare()?.to_owned();
    let tmp = args.tmp.clone().unwrap_or_else(|| suffixed(&out, "tmp"));
    let cg = args.graph.open(g)?;
    let w = Window::parse(&args.window, cg.num_nodes())?;
    let mut budget = g.budget();
    // per decoding thread, the arcs of one piece as (usize, usize) pairs: chunk_arcs /
    // INDUCED_SPLIT on average, reserved with a factor 2 of slack
    budget.need(
        rayon::current_num_threads() as u64 * 16 * 2 * g.chunk_conf().chunk_arcs
            / topo::INDUCED_SPLIT as u64,
        "window: per-thread chunk arc buffers",
    )?;
    let mem = topo::sort_memory(&mut budget, "window external-sort buffers")?;
    let mut pl = logger("arc");
    let (n, m) = topo::write_induced(&cg, w, &out, &tmp, mem, &mut pl)?;
    let _ = std::fs::remove_dir(&tmp);
    let mut info = RunInfo::new(
        Some(args.graph.graph.clone()),
        cg.num_nodes(),
        cg.num_arcs,
        w,
    );
    info.results = json!({ "output": out, "nodes": n, "arcs": m });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(true))
}

use webgraph::prelude::{RandomAccessLabeling, SequentialLabeling};
