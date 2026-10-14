//! The `dag` command line (clap derive).
//!
//! One binary with one subcommand per analysis. Every subcommand logs with
//! `dsi-progress-logger`, writes `<out>.json` ([`RunInfo`](crate::io::RunInfo), including the
//! peak RSS) and returns an exit code: 0 ok, 1 verification failed, 2 error. The producing
//! subcommands take `--buckets FILE --top K` ([`common::StatsOpt`]) and report the
//! distributions of their outputs per node range ([`crate::summary`]).
//!
//! | Subcommand | Module |
//! |---|---|
//! | `topo-check`, `toposort`, `window` | [`topo`] |
//! | `blocks` | [`blocks`] |
//! | `dp`, `certify` | [`dp`] |
//! | `cpm` | [`cpm`] |
//! | `count-paths` | [`count`] |
//! | `dom build\|verify\|stats` | [`dom`] |
//! | `reach-counts` | [`reach_counts`] |
//! | `reach-index build\|query\|bench` | [`reach_index`] |
//! | `tr verify\|reduce` | [`tr`] |
//! | `path-cover`, `frontier`, `width` | [`width`] |
//! | `aggregate` | [`aggregate`] |

use std::process::ExitCode;

use clap::{Parser, Subcommand};

pub mod aggregate;
pub mod blocks;
pub mod common;
pub mod count;
pub mod cpm;
pub mod dom;
pub mod dp;
pub mod reach_counts;
pub mod reach_index;
pub mod topo;
pub mod tr;
pub mod width;

pub use common::GlobalArgs;

/// DAG algorithms for the Bitcoin Payment Graph and any webgraph DAG.
#[derive(Parser, Debug)]
#[command(name = "dag", version, about, long_about = None)]
pub struct Cli {
    /// Global options.
    #[command(flatten)]
    pub global: GlobalArgs,
    /// The subcommand.
    #[command(subcommand)]
    pub cmd: Command,
}

/// The subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Checks that the identity is a topological order; counts sinks (and sources).
    TopoCheck(topo::TopoCheckArgs),
    /// Relabels a graph in topological order (BV + transpose + .ef + .dcf + order/rank).
    Toposort(topo::ToposortArgs),
    /// Writes the subgraph induced by a window (ids shifted to start at 0) and its transpose.
    Window(topo::WindowArgs),
    /// Builds and verifies the BlockIndex (biclique decomposition).
    Blocks(blocks::Args),
    /// Semiring dynamic programming.
    Dp(dp::DpArgs),
    /// Certifies DP values against the fixpoint equations.
    Certify(dp::CertifyArgs),
    /// Critical path method (earliest/latest times, slack, critical nodes and arcs).
    Cpm(cpm::Args),
    /// Path counting (ExtFloat magnitudes and exact residues).
    CountPaths(count::Args),
    /// Dominator and post-dominator trees.
    Dom(dom::Args),
    /// Approximate ancestor/descendant counts (HyperLogLog).
    ReachCounts(reach_counts::Args),
    /// Exact reachability index.
    ReachIndex(reach_index::Args),
    /// Transitive reduction.
    Tr(tr::Args),
    /// Minimum path cover (MPC_d) of a window.
    PathCover(width::PathCoverArgs),
    /// Largest prefix antichain (frontier) of a window.
    Frontier(width::FrontierArgs),
    /// Width (maximum antichain) of a window.
    Width(width::WidthArgs),
    /// Distribution of a stored array, overall and per node-range bucket (e.g. half-years).
    Aggregate(aggregate::Args),
}

/// Runs a parsed command line: sets up logging and the rayon pool, dispatches, and maps errors
/// to exit code 2.
pub fn run(cli: Cli) -> ExitCode {
    let g = &cli.global;
    let _ = env_logger::Builder::new()
        .filter_level(g.log_level())
        .format_timestamp_millis()
        .try_init();
    if let Some(t) = g.threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(t.max(1))
            .build_global()
        {
            log::warn!("cannot configure the rayon pool: {e}");
        }
    }
    let res = match &cli.cmd {
        Command::TopoCheck(a) => topo::run_topo_check(a, g),
        Command::Toposort(a) => topo::run_toposort(a, g),
        Command::Window(a) => topo::run_window(a, g),
        Command::Blocks(a) => blocks::run(a, g),
        Command::Dp(a) => dp::run_dp(a, g),
        Command::Certify(a) => dp::run_certify(a, g),
        Command::Cpm(a) => cpm::run(a, g),
        Command::CountPaths(a) => count::run(a, g),
        Command::Dom(a) => dom::run(a, g),
        Command::ReachCounts(a) => reach_counts::run(a, g),
        Command::ReachIndex(a) => reach_index::run(a, g),
        Command::Tr(a) => tr::run(a, g),
        Command::PathCover(a) => width::run_path_cover(a, g),
        Command::Frontier(a) => width::run_frontier(a, g),
        Command::Width(a) => width::run_width(a, g),
        Command::Aggregate(a) => aggregate::run(a),
    };
    match res {
        Ok(code) => code,
        Err(e) => {
            log::error!("{e:#}");
            eprintln!("error: {e:#}");
            ExitCode::from(common::EXIT_ERROR)
        }
    }
}
