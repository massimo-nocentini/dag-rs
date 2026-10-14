//! Global options and argument groups shared by every subcommand.
//!
//! Global options (before the subcommand): `--threads N` (rayon global pool, default all),
//! `--decoders D` (default `min(threads, 32)`), `--chunk-arcs A` (2²⁴), `--max-chunk-nodes`
//! (2²²), `--max-mem GB` (400), `--load ram|mmap` (ram), `-v`/`-q`.
//!
//! Argument groups: [`GraphOpt`] (`--graph B`, `--transpose BT` defaulting to `B-t` if it
//! exists), [`BlocksOpt`] (`--blocks auto|none|PREFIX`), [`WindowOpt`] (`--window a..b`),
//! [`OutOpt`] (`--out PREFIX`), [`StatsOpt`] (`--buckets FILE --top K`, aggregate measures per
//! node range) and the `--weights` specification [`WeightsSpec`].
//!
//! Exit codes: [`EXIT_OK`] (0), [`EXIT_FAILED`] (1, a verification failed), [`EXIT_ERROR`] (2).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;

use anyhow::Result;
use clap::Args;
use dsi_progress_logger::prelude::*;

use crate::blocks::BlockIndex;
use crate::error::DagError;
use crate::graph::{self, BvRa, ChunkConf, ChunkedGraph};
use crate::io::{LoadMode, MemBudget};
use crate::summary::Buckets;
use crate::sweep::SweepConf;
use crate::window::Window;

/// Exit code of a successful run.
pub const EXIT_OK: u8 = 0;
/// Exit code of a run whose verification failed.
pub const EXIT_FAILED: u8 = 1;
/// Exit code of a run that ended with an error.
pub const EXIT_ERROR: u8 = 2;

/// `ExitCode` 0 if `ok`, else 1.
pub fn exit_status(ok: bool) -> ExitCode {
    ExitCode::from(if ok { EXIT_OK } else { EXIT_FAILED })
}

/// Options that precede the subcommand.
#[derive(Args, Debug, Clone)]
pub struct GlobalArgs {
    /// Size of the rayon global pool (default: all cores).
    #[arg(long, global = true)]
    pub threads: Option<usize>,
    /// Chunks decoded in parallel by the sweep engine (default: min(threads, 32)).
    #[arg(long, global = true)]
    pub decoders: Option<usize>,
    /// Target number of arcs per chunk.
    #[arg(long, global = true, default_value_t = 1 << 24)]
    pub chunk_arcs: u64,
    /// Maximum number of nodes per chunk.
    #[arg(long, global = true, default_value_t = 1 << 22)]
    pub max_chunk_nodes: usize,
    /// Memory cap in GB (10⁹ bytes) checked against each subcommand's estimate.
    #[arg(long, global = true, default_value_t = 400.0)]
    pub max_mem: f64,
    /// How persisted arrays are loaded.
    #[arg(long, global = true, value_enum, default_value_t = LoadMode::Ram)]
    pub load: LoadMode,
    /// More logging (-v debug, -vv trace).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
    /// Less logging (-q warnings only, -qq errors only).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub quiet: u8,
}

impl GlobalArgs {
    /// The chunk geometry.
    pub fn chunk_conf(&self) -> ChunkConf {
        ChunkConf {
            chunk_arcs: self.chunk_arcs.max(1),
            max_chunk_nodes: self.max_chunk_nodes.max(1),
        }
    }

    /// The sweep configuration (`--decoders`, default `min(threads, 32)`).
    pub fn sweep_conf(&self) -> SweepConf {
        match self.decoders {
            Some(d) => SweepConf { decoders: d.max(1) },
            None => SweepConf::default(),
        }
    }

    /// A fresh memory budget of `--max-mem` GB.
    pub fn budget(&self) -> MemBudget {
        MemBudget::new(self.max_mem)
    }

    /// The log level filter implied by `-v`/`-q`.
    pub fn log_level(&self) -> log::LevelFilter {
        match self.verbose as i32 - self.quiet as i32 {
            i32::MIN..=-2 => log::LevelFilter::Error,
            -1 => log::LevelFilter::Warn,
            0 => log::LevelFilter::Info,
            1 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        }
    }
}

/// `--graph B [--transpose BT]`.
#[derive(Args, Debug, Clone)]
pub struct GraphOpt {
    /// Basename of the forward BV graph (needs `.ef`; `.dcf` is used when present).
    #[arg(long)]
    pub graph: PathBuf,
    /// Basename of the transpose (default: `<graph>-t` if it exists).
    #[arg(long)]
    pub transpose: Option<PathBuf>,
}

impl GraphOpt {
    /// Opens the forward graph.
    pub fn open(&self, g: &GlobalArgs) -> Result<ChunkedGraph<BvRa>> {
        graph::open(&self.graph, g.chunk_conf())
    }

    /// The transpose basename: `--transpose`, else `<graph>-t` if it exists.
    pub fn transpose_path(&self) -> Option<PathBuf> {
        self.transpose
            .clone()
            .or_else(|| graph::transpose_basename(&self.graph))
    }

    /// Opens the transpose if available, checking that it has the same number of nodes and arcs
    /// as the forward graph (read from `<graph>.properties`).
    pub fn open_transpose(&self, g: &GlobalArgs) -> Result<Option<ChunkedGraph<BvRa>>> {
        let Some(t) = self.transpose_path() else {
            return Ok(None);
        };
        let (n, m) = graph::read_properties(&self.graph)?;
        let gt = graph::open(&t, g.chunk_conf())?;
        anyhow::ensure!(
            gt.num_nodes() == n && gt.num_arcs == m,
            "transpose {} has {} nodes and {} arcs, graph {} has {n} and {m}",
            t.display(),
            gt.num_nodes(),
            gt.num_arcs,
            self.graph.display()
        );
        Ok(Some(gt))
    }
}

/// The value of `--blocks`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlocksSpec {
    /// Build the index in memory when the subcommand benefits and the graph qualifies.
    Auto,
    /// Never use a block index.
    None,
    /// Load `PREFIX.blk.*`.
    Prefix(PathBuf),
}

impl FromStr for BlocksSpec {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "auto" => Self::Auto,
            "none" => Self::None,
            p => Self::Prefix(PathBuf::from(p)),
        })
    }
}

/// `--blocks auto|none|PREFIX`.
#[derive(Args, Debug, Clone)]
pub struct BlocksOpt {
    /// `auto` builds the BlockIndex in memory when useful (falling back to generic kernels if the
    /// graph is not a biclique digraph), `none` disables it, a PREFIX loads `PREFIX.blk.*`.
    #[arg(long, default_value = "auto")]
    pub blocks: BlocksSpec,
}

impl BlocksOpt {
    /// Resolves the option: loads `PREFIX.blk.*`, or (for `auto`) builds the index from `g`
    /// with [`BlockIndex::build_budgeted`], reserving 8 B/node + 12 B/block in `budget`. A
    /// `NotBiclique` failure in `auto` mode logs a warning, releases the reservation and
    /// returns `None`.
    pub fn resolve(
        &self,
        g: &ChunkedGraph<BvRa>,
        global: &GlobalArgs,
        budget: &mut MemBudget,
    ) -> Result<Option<BlockIndex>> {
        match &self.blocks {
            BlocksSpec::None => Ok(None),
            BlocksSpec::Prefix(p) => {
                let idx = BlockIndex::load(p, global.load)?;
                anyhow::ensure!(
                    idx.n == g.num_nodes(),
                    "{}.blk.* has {} nodes, the graph has {}",
                    p.display(),
                    idx.n,
                    g.num_nodes()
                );
                budget.need(BlockIndex::bytes(idx.n, idx.nb), "BlockIndex")?;
                Ok(Some(idx))
            }
            BlocksSpec::Auto => {
                // build_budgeted reserves 8 B/node before pass 1 and 12 B/block once the
                // number of blocks is known, and releases both on failure
                let mut pl = concurrent_logger("node");
                match BlockIndex::build_budgeted(g, &mut pl, budget) {
                    Ok((idx, report)) => {
                        log::info!(
                            "BlockIndex: {} blocks, Σ len·nin = {}",
                            report.nb,
                            report.sum_len_times_nin
                        );
                        Ok(Some(idx))
                    }
                    Err(e)
                        if matches!(
                            e.downcast_ref::<DagError>(),
                            Some(DagError::NotBiclique { .. })
                        ) =>
                    {
                        log::warn!("{e}: falling back to the generic kernels");
                        Ok(None)
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }
}

/// `--window a..b` (default: the full graph).
#[derive(Args, Debug, Clone, Default)]
pub struct WindowOpt {
    /// Node window `a..b` (half open; `a..`, `..b` and `..` allowed; `_` separators allowed).
    #[arg(long, default_value = "")]
    pub window: String,
}

impl WindowOpt {
    /// Parses the window for a graph with `n` nodes.
    pub fn window(&self, n: usize) -> Result<Window> {
        Window::parse(&self.window, n)
    }
}

/// `--buckets FILE`: node-range buckets for the aggregate measures.
#[derive(Args, Debug, Clone, Default)]
pub struct BucketsOpt {
    /// Node-range buckets (TSV/CSV with a header, e.g. the PG's `chunk_boundaries.tsv`: one
    /// bucket per half-year): every distribution in the JSON is repeated for each bucket
    /// (`by_bucket`), see [`crate::summary`].
    #[arg(long)]
    pub buckets: Option<PathBuf>,
}

impl BucketsOpt {
    /// Loads the buckets for a graph of `n` nodes (`None` without `--buckets`).
    pub fn load(&self, n: usize) -> Result<Option<Buckets>> {
        crate::summary::load_buckets(self.buckets.as_deref(), n)
    }
}

/// `--buckets FILE --top K`: the options of the aggregate measures ([`crate::summary`]).
#[derive(Args, Debug, Clone)]
pub struct StatsOpt {
    /// Buckets.
    #[command(flatten)]
    pub buckets: BucketsOpt,
    /// Length of the top-k lists of the distributions in the JSON (0 disables them).
    #[arg(long, default_value_t = 20)]
    pub top: usize,
}

impl Default for StatsOpt {
    fn default() -> Self {
        Self {
            buckets: BucketsOpt::default(),
            top: 20,
        }
    }
}

/// `--out PREFIX` (required for producers).
#[derive(Args, Debug, Clone)]
pub struct OutOpt {
    /// Output prefix: files are `<PREFIX>.<field>.<dtype>` plus `<PREFIX>.json`.
    #[arg(long)]
    pub out: PathBuf,
}

impl OutOpt {
    /// Creates the parent directory of the prefix if needed.
    pub fn prepare(&self) -> Result<&Path> {
        if let Some(dir) = self.out.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        Ok(&self.out)
    }
}

/// The value of `--weights`.
///
/// Mapping (fixed here for every subcommand):
/// - `unit`: arc length 1 for `maxplus`/`minplus` semirings (`Const(1)`, which reproduces
///   `longest.u32le`), and `One` for every other semiring;
/// - `one`: `One` always;
/// - `inv-outdeg`: `Node { alpha: 1/outdeg(u), beta: 1̄ }` (outdegree from the BlockIndex, else
///   a parallel outdegree pass); with `prob` this is the uniform-split ("haircut") provenance
///   mass, with `viterbi` the most probable random-walk path;
/// - `node-u32:FILE` / `node-f64:FILE`: `Node { alpha: FILE[u], beta: 1̄ }` from a per-node
///   little-endian array (absolute ids).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightsSpec {
    /// See the type documentation.
    Unit,
    /// See the type documentation.
    One,
    /// See the type documentation.
    InvOutdeg,
    /// Per-node `u32` tail weights from a file.
    NodeU32(PathBuf),
    /// Per-node `f64` tail weights from a file.
    NodeF64(PathBuf),
}

impl FromStr for WeightsSpec {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Ok(match s {
            "unit" => Self::Unit,
            "one" => Self::One,
            "inv-outdeg" => Self::InvOutdeg,
            _ => {
                if let Some(f) = s.strip_prefix("node-u32:") {
                    Self::NodeU32(PathBuf::from(f))
                } else if let Some(f) = s.strip_prefix("node-f64:") {
                    Self::NodeF64(PathBuf::from(f))
                } else {
                    return Err(format!(
                        "invalid weights {s:?}: expected unit, one, inv-outdeg, node-u32:FILE or node-f64:FILE"
                    ));
                }
            }
        })
    }
}

/// A sequential progress logger (`item_name`, `display_memory = true`).
pub fn logger(item_name: &str) -> ProgressLogger {
    let mut pl = ProgressLogger::default();
    pl.item_name(item_name).display_memory(true);
    pl
}

/// A concurrent progress logger (`item_name`, `display_memory = true`).
pub fn concurrent_logger(item_name: &str) -> ConcurrentWrapper<ProgressLogger> {
    let mut pl = ConcurrentWrapper::wrap(logger(item_name));
    pl.item_name(item_name);
    pl
}
