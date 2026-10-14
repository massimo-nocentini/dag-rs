//! `dag dp` and `dag certify`: semiring DP and fixpoint certification from the command
//! line.
//!
//! `dp` writes `<out>.val.<dtype>` (indexed by `v − lo`), with `--pred` also
//! `<out>.pred.u32le` (absolute ids, `u32::MAX` = none), and `<out>.json` with the kernel used,
//! timings, a value summary (number of 0̄ nodes, max/argmax, min/argmin and mean of the other
//! values), and optionally the [`CertReport`] (`--certify`) and a comparison with a reference
//! array (`--compare FILE`). `certify` re-checks stored values (and predecessors) against the
//! fixpoint equations ([`certify`]). Both exit with 1 when a check fails.
//!
//! `--weights` follows [`WeightsSpec`]: `unit` is arc length 1 for the max-plus/min-plus
//! semirings and 1̄ otherwise; `inv-outdeg` and `node-*` weights are materialised as one
//! element per node (`α(u)`, `β = 1̄`) after converting the real weight `x` into the semiring:
//! `x` itself for tropical, probability and counting semirings, `log2 x` for `log2sum`,
//! `x ≠ 0` for `bool`. Integer semirings reject fractional (`inv-outdeg`, `node-f64`) weights;
//! `minplus-i64` also accepts negative integral weights, the others only non-negative ones.
//!
//! # Resources
//! The cost of a run follows its window `n_W`, not the graph:
//! - `--blocks auto` builds the [`BlockIndex`] (a pass over all `n` nodes, `8n + 12·nb` bytes)
//!   only when the window is the whole graph or `--kernel blocks` is given; a strict sub-window
//!   otherwise runs the generic kernels ([`window_blocks`]);
//! - `--blocks PREFIX` is memory-mapped for a strict sub-window (whatever `--load`), and a
//!   block-kernel run without `--certify` does not open the BV graph ([`block_only`]), saving
//!   its in-RAM metadata (≈ 1.3 GiB on PG);
//! - per-node weights and outdegrees are materialised for the window only (`|E|·n_W` and
//!   `4·n_W` bytes, from memory-mapped files).
//!
//! The value summary is computed over fixed chunks reduced in order, so it is bit-reproducible
//! across runs and thread counts.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use dsi_progress_logger::prelude::*;
use rayon::prelude::*;
use serde_json::json;
use webgraph::prelude::RandomAccessLabeling;

use super::common::{
    BlocksOpt, BlocksSpec, GlobalArgs, GraphOpt, OutOpt, StatsOpt, WeightsSpec, WindowOpt,
    concurrent_logger, exit_status, logger,
};
use crate::blocks::{BlockIndex, BlockReport};
use crate::dp::certify::{CertReport, RelErr, Tolerance, bit_eq, certify};
use crate::dp::solve::{estimate_bytes, resolve_kernel, solve, sweep_buffer_bytes};
use crate::dp::{ArcWeights, Const, DpConf, DpInput, DpResult, Init, Kernel, Node, One};
use crate::graph::{self, BvRa, ChunkedGraph};
use crate::io::{self, Dump, FileInfo, LoadMode, MemBudget, RunInfo};
use crate::semiring::bottleneck::MaxMin;
use crate::semiring::counting::{SatU64, SatU128};
use crate::semiring::extfloat::{Ext, ExtFloat};
use crate::semiring::logsum::Log2Sum;
use crate::semiring::modular::M61;
use crate::semiring::prob::{Probability, Viterbi};
use crate::semiring::{Boolean, MaxPlus, MinPlus, Semiring};
use crate::summary::{BucketDist, Buckets, DistConf, Log2Hist, Top, distribution};
use crate::window::{Dir, Window};

/// The semirings selectable with `--semiring`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SemiringName {
    /// (max, +) over u32.
    MaxplusU32,
    /// (max, +) over u64.
    MaxplusU64,
    /// (max, +) over f64.
    MaxplusF64,
    /// (min, +) over u32.
    MinplusU32,
    /// (min, +) over i64.
    MinplusI64,
    /// (min, +) over f64.
    MinplusF64,
    /// (max, min) over u32.
    MaxminU32,
    /// Boolean (∨, ∧).
    Bool,
    /// Saturating u64 counting.
    Sat64,
    /// Saturating u128 counting.
    Sat128,
    /// Counting modulo 2⁶¹ − 1.
    Mod61,
    /// Counting with an extended-exponent float.
    Extfloat,
    /// log2-sum-exp.
    Log2sum,
    /// Probabilities (+, ×).
    Prob,
    /// Viterbi (max, ×).
    Viterbi,
}

/// The value of `--init`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum InitKind {
    /// 1̄ on window sources (forward) or sinks (backward).
    Sources,
    /// 1̄ on every node.
    All,
}

/// `dag dp`.
#[derive(Args, Debug)]
pub struct DpArgs {
    /// Input graph.
    #[command(flatten)]
    pub graph: GraphOpt,
    /// Block index.
    #[command(flatten)]
    pub blocks: BlocksOpt,
    /// Window.
    #[command(flatten)]
    pub window: WindowOpt,
    /// Output prefix.
    #[command(flatten)]
    pub out: OutOpt,
    /// Semiring.
    #[arg(long, value_enum)]
    pub semiring: SemiringName,
    /// Direction.
    #[arg(long, value_enum, default_value = "forward")]
    pub dir: Dir,
    /// Initial terms.
    #[arg(long, value_enum, default_value = "sources")]
    pub init: InitKind,
    /// Arc weights: unit, one, inv-outdeg, node-u32:FILE or node-f64:FILE.
    #[arg(long, default_value = "unit")]
    pub weights: WeightsSpec,
    /// Kernel.
    #[arg(long, value_enum, default_value = "auto")]
    pub kernel: Kernel,
    /// Also compute argbest predecessors (selective semirings).
    #[arg(long)]
    pub pred: bool,
    /// Certify the result afterwards.
    #[arg(long)]
    pub certify: bool,
    /// Compare the values with this array (same dtype; `n` entries, of which the window is
    /// compared, or one entry per window node); exit 1 on any difference.
    #[arg(long)]
    pub compare: Option<PathBuf>,
    /// Aggregate measures (`--buckets`, `--top`).
    #[command(flatten)]
    pub stats: StatsOpt,
}

/// `dag certify`.
#[derive(Args, Debug)]
pub struct CertifyArgs {
    /// Input graph.
    #[command(flatten)]
    pub graph: GraphOpt,
    /// Window.
    #[command(flatten)]
    pub window: WindowOpt,
    /// Output prefix (JSON report).
    #[command(flatten)]
    pub out: OutOpt,
    /// Values to certify.
    #[arg(long)]
    pub values: PathBuf,
    /// Semiring.
    #[arg(long, value_enum)]
    pub semiring: SemiringName,
    /// Direction.
    #[arg(long, value_enum, default_value = "forward")]
    pub dir: Dir,
    /// Initial terms.
    #[arg(long, value_enum, default_value = "sources")]
    pub init: InitKind,
    /// Arc weights.
    #[arg(long, default_value = "unit")]
    pub weights: WeightsSpec,
    /// Predecessor array to certify.
    #[arg(long)]
    pub pred: Option<PathBuf>,
}

// ---------------------------------------------------------------------------------------------
// Per-semiring glue
// ---------------------------------------------------------------------------------------------

/// What the command line needs to know about a semiring.
trait CliSemiring: Semiring<Elem: RelErr> + Copy + 'static {
    /// Whether the elements are floating point (then non-selective instances are certified
    /// with [`Tolerance::PerTerm`]).
    const FLOAT: bool;
    /// The element "arc length 1" of `--weights unit`, for the tropical semirings.
    fn unit_length(&self) -> Option<Self::Elem> {
        None
    }
    /// The element of an integer weight.
    fn of_u32(&self, x: u32) -> Self::Elem;
    /// The element of a real weight, `None` if the semiring has no such element.
    fn of_f64(&self, x: f64) -> Option<Self::Elem>;
    /// A real summary key of a (non-0̄) value.
    fn key_f64(&self, x: Self::Elem) -> f64;

    fn tolerance(&self) -> Tolerance {
        if Self::FLOAT && !Self::SELECTIVE {
            Tolerance::DEFAULT_PER_TERM
        } else {
            Tolerance::Exact
        }
    }
}

/// `x` if it is an integer in `[lo, hi)` (`None` for fractions, non-finite values and values
/// out of range).
fn integral(x: f64, lo: f64, hi: f64) -> Option<f64> {
    (x.fract() == 0.0 && lo <= x && x < hi).then_some(x)
}

macro_rules! cli_int {
    // unsigned elements: integral weights in [0, MAX)
    ($s:ty, $t:ty, unit = $unit:expr) => {
        cli_int!($s, $t, unit = $unit, lo = 0.0);
    };
    // `lo` is the smallest accepted weight; `MAX` (a sentinel of every integer instance) is
    // always excluded
    ($s:ty, $t:ty, unit = $unit:expr, lo = $lo:expr) => {
        impl CliSemiring for $s {
            const FLOAT: bool = false;
            fn unit_length(&self) -> Option<$t> {
                $unit
            }
            fn of_u32(&self, x: u32) -> $t {
                (x as $t).min(<$t>::MAX - 1)
            }
            fn of_f64(&self, x: f64) -> Option<$t> {
                integral(x, $lo, <$t>::MAX as f64).map(|x| x as $t)
            }
            fn key_f64(&self, x: $t) -> f64 {
                x as f64
            }
        }
    };
}
cli_int!(MaxPlus<u32>, u32, unit = Some(1));
cli_int!(MaxPlus<u64>, u64, unit = Some(1));
cli_int!(MinPlus<u32>, u32, unit = Some(1));
// min-plus over i64 is the instance for negative weights (Bellman–Ford style shortest paths);
// the smallest accepted weight is the f64 just above `i64::MIN`, and `times` saturates
cli_int!(
    MinPlus<i64>,
    i64,
    unit = Some(1),
    lo = -(i64::MAX as f64) + 1024.0
);
cli_int!(MaxMin<u32>, u32, unit = None);
cli_int!(SatU64, u64, unit = None);
cli_int!(SatU128, u128, unit = None);
cli_int!(M61, u64, unit = None);

macro_rules! cli_float {
    ($s:ty, unit = $unit:expr, from = $from:expr) => {
        impl CliSemiring for $s {
            const FLOAT: bool = true;
            fn unit_length(&self) -> Option<f64> {
                $unit
            }
            fn of_u32(&self, x: u32) -> f64 {
                ($from)(x as f64)
            }
            fn of_f64(&self, x: f64) -> Option<f64> {
                Some(($from)(x))
            }
            fn key_f64(&self, x: f64) -> f64 {
                x
            }
        }
    };
}
cli_float!(MaxPlus<f64>, unit = Some(1.0), from = |x: f64| x);
cli_float!(MinPlus<f64>, unit = Some(1.0), from = |x: f64| x);
cli_float!(Probability, unit = None, from = |x: f64| x);
cli_float!(Viterbi, unit = None, from = |x: f64| x);
cli_float!(Log2Sum, unit = None, from = |x: f64| x.log2());

impl CliSemiring for Boolean {
    const FLOAT: bool = false;
    fn of_u32(&self, x: u32) -> bool {
        x != 0
    }
    fn of_f64(&self, x: f64) -> Option<bool> {
        Some(x != 0.0)
    }
    fn key_f64(&self, x: bool) -> f64 {
        f64::from(u8::from(x))
    }
}

impl CliSemiring for ExtFloat {
    const FLOAT: bool = true;
    fn of_u32(&self, x: u32) -> Ext {
        self.of_f64(x as f64).unwrap_or(self.zero())
    }
    /// `x = m·2^e` with `m ∈ [1, 2)`; 0̄ for `x = 0`; `None` for negative or non-finite `x`.
    fn of_f64(&self, x: f64) -> Option<Ext> {
        if x == 0.0 {
            return Some(self.zero());
        }
        if !(x.is_finite() && x > 0.0) {
            return None;
        }
        let mut e = x.log2().floor() as i64;
        let mut m = x / (e as f64).exp2();
        if m >= 2.0 {
            m /= 2.0;
            e += 1;
        } else if m < 1.0 {
            m *= 2.0;
            e -= 1;
        }
        Some(Ext { m, e })
    }
    /// `log2` of the value.
    fn key_f64(&self, x: Ext) -> f64 {
        x.m.log2() + x.e as f64
    }
}

/// Runs `$body` with `$s` bound to the instance named by `$name`.
macro_rules! with_semiring {
    ($name:expr, $s:ident => $body:expr) => {
        match $name {
            SemiringName::MaxplusU32 => {
                let $s = MaxPlus::<u32>::new();
                $body
            }
            SemiringName::MaxplusU64 => {
                let $s = MaxPlus::<u64>::new();
                $body
            }
            SemiringName::MaxplusF64 => {
                let $s = MaxPlus::<f64>::new();
                $body
            }
            SemiringName::MinplusU32 => {
                let $s = MinPlus::<u32>::new();
                $body
            }
            SemiringName::MinplusI64 => {
                let $s = MinPlus::<i64>::new();
                $body
            }
            SemiringName::MinplusF64 => {
                let $s = MinPlus::<f64>::new();
                $body
            }
            SemiringName::MaxminU32 => {
                let $s = MaxMin::<u32>::new();
                $body
            }
            SemiringName::Bool => {
                let $s = Boolean::new();
                $body
            }
            SemiringName::Sat64 => {
                let $s = SatU64::new();
                $body
            }
            SemiringName::Sat128 => {
                let $s = SatU128::new();
                $body
            }
            SemiringName::Mod61 => {
                let $s = M61::new();
                $body
            }
            SemiringName::Extfloat => {
                let $s = ExtFloat::new();
                $body
            }
            SemiringName::Log2sum => {
                let $s = Log2Sum::new();
                $body
            }
            SemiringName::Prob => {
                let $s = Probability::new();
                $body
            }
            SemiringName::Viterbi => {
                let $s = Viterbi::new();
                $body
            }
        }
    };
}

/// The `--weights` of a run: unit/one, or one tail factor per window node (indexed by
/// `u − lo`).
enum Weights<E> {
    One,
    Const(E),
    Node { alpha: Vec<E>, lo: usize },
}

impl<E: Copy> Weights<E> {
    fn describe(&self) -> &'static str {
        match self {
            Self::One => "One (every arc 1̄)",
            Self::Const(_) => "Const (arc length 1)",
            Self::Node { .. } => "Node (α per node, β = 1̄)",
        }
    }
}

/// A read-only memory map of a little-endian array of `E` with `n` entries (absolute ids).
struct MappedArray<E> {
    map: Option<webgraph::utils::MmapHelper<u8>>,
    _elem: std::marker::PhantomData<E>,
}

impl<E: Dump> MappedArray<E> {
    /// Maps `file`, checking that it has `n` entries; `need` = `false` skips the map (empty
    /// window).
    fn open(file: &Path, n: usize, need: bool) -> Result<Self> {
        let bytes = std::fs::metadata(file)
            .with_context(|| format!("cannot stat {}", file.display()))?
            .len() as usize;
        ensure!(
            bytes == n * E::BYTES,
            "{} has {} bytes: expected {n} entries of {} ({})",
            file.display(),
            bytes,
            E::BYTES,
            E::DTYPE
        );
        let map = if need && bytes > 0 {
            Some(
                webgraph::utils::MmapHelper::<u8>::mmap(file, mmap_rs::MmapFlags::SEQUENTIAL)
                    .with_context(|| format!("cannot map {}", file.display()))?,
            )
        } else {
            None
        };
        Ok(Self {
            map,
            _elem: std::marker::PhantomData,
        })
    }

    /// Entry `u` (only valid if the array was mapped).
    fn get(&self, u: usize) -> E {
        let raw: &[u8] = self.map.as_ref().expect("array not mapped").as_ref();
        E::get(&raw[u * E::BYTES..(u + 1) * E::BYTES])
    }
}

/// Allocates (with [`io::huge_vec`]) and fills in parallel one element per node of `win`
/// (indexed by `u − lo`).
fn per_window_node<E: Copy + Send + Sync>(
    win: Window,
    fill: E,
    f: &(dyn Fn(usize) -> Result<E> + Sync),
) -> Result<Vec<E>> {
    let mut v = io::huge_vec(win.len(), fill);
    v.par_iter_mut()
        .with_min_len(1 << 16)
        .enumerate()
        .try_for_each(|(i, x)| -> Result<()> {
            *x = f(win.lo + i)?;
            Ok(())
        })?;
    Ok(v)
}

/// Builds the weights of `spec` for the window `win` of a graph of `n` nodes; `outdeg` gives
/// the outdegree (absolute ids) for `inv-outdeg`.
///
/// Every kernel and [`certify`] read `α(u)` only for window nodes `u` (arcs leaving the window
/// are ignored), so only `win` is materialised: `|E|·n_W` bytes, from memory-mapped files.
fn weights<S: CliSemiring>(
    s: &S,
    spec: &WeightsSpec,
    n: usize,
    win: Window,
    outdeg: impl Fn(usize) -> usize + Sync,
    budget: &mut MemBudget,
) -> Result<Weights<S::Elem>> {
    let mut node = |f: &(dyn Fn(usize) -> Result<S::Elem> + Sync)| -> Result<Weights<S::Elem>> {
        budget.need(
            (win.len() * std::mem::size_of::<S::Elem>()) as u64,
            "per-node weights",
        )?;
        Ok(Weights::Node {
            alpha: per_window_node(win, s.one(), f)?,
            lo: win.lo,
        })
    };
    match spec {
        WeightsSpec::Unit => Ok(s.unit_length().map_or(Weights::One, Weights::Const)),
        WeightsSpec::One => Ok(Weights::One),
        WeightsSpec::InvOutdeg => node(&|u| match outdeg(u) {
            // sinks never push: their α is never used
            0 => Ok(s.one()),
            d => s
                .of_f64(1.0 / d as f64)
                .with_context(|| format!("{} has no element for the weight 1/{d}", s.name())),
        }),
        WeightsSpec::NodeU32(f) => {
            let a = MappedArray::<u32>::open(f, n, !win.is_empty())?;
            node(&|u| Ok(s.of_u32(a.get(u))))
        }
        WeightsSpec::NodeF64(f) => {
            let a = MappedArray::<f64>::open(f, n, !win.is_empty())?;
            node(&|u| {
                let x = a.get(u);
                s.of_f64(x)
                    .with_context(|| format!("{} has no element for the weight {x}", s.name()))
            })
        }
    }
}

/// Calls `$body` with `$w` bound to a concrete [`ArcWeights`] value for `$weights`.
macro_rules! with_weights {
    ($s:expr, $weights:expr, $w:ident => $body:expr) => {
        match $weights {
            Weights::One => {
                let $w = One;
                $body
            }
            Weights::Const(e) => {
                let $w = Const(*e);
                $body
            }
            Weights::Node { alpha, lo } => {
                let (one, lo) = ($s.one(), *lo);
                let $w = Node {
                    alpha: |u: usize| alpha[u - lo],
                    beta: move |_: usize| one,
                };
                $body
            }
        }
    };
}

fn init_of<E: Copy>(kind: InitKind, one: E) -> Init<E> {
    match kind {
        InitKind::Sources => Init::Sources(one),
        InitKind::All => Init::All(one),
    }
}

// ---------------------------------------------------------------------------------------------
// dp
// ---------------------------------------------------------------------------------------------

/// Summary of a value array: the [`Dist`](crate::summary::Dist) of the real keys of the
/// non-0̄ values (for `extfloat` the key is `log2` of the value), under the historical field
/// names of the first version (`nodes`, `zero_nodes`, `max`, …) plus the log2 histogram, the
/// top-k nodes and, with `--buckets`, one distribution per bucket.
#[derive(Debug, Default, serde::Serialize)]
pub struct ValueSummary {
    /// Values summarised.
    pub nodes: usize,
    /// Nodes whose value is 0̄ (unreached).
    pub zero_nodes: u64,
    /// Largest real key of the other values.
    pub max: Option<f64>,
    /// Smallest node (absolute id) attaining `max`.
    pub argmax: Option<usize>,
    /// Smallest real key.
    pub min: Option<f64>,
    /// Smallest node (absolute id) attaining `min`.
    pub argmin: Option<usize>,
    /// Mean of the real keys of the non-0̄ values.
    pub mean: Option<f64>,
    /// Log2 histogram of the real keys ([`crate::summary`] bins: 0, then `[2^(k−1), 2^k)`).
    pub log2_hist: Log2Hist,
    /// The `--top` nodes with the largest keys (absolute ids; key descending, id ascending).
    pub top: Vec<Top>,
    /// The same statistics per `--buckets` bucket.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub by_bucket: Option<Vec<BucketDist>>,
    /// Window nodes outside every bucket.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unbucketed: Option<u64>,
}

/// The value summary of `val` (window starting at `lo`). The partial sums are taken over fixed
/// chunks of [`crate::summary::CHUNK`] values and added in chunk order, so the `mean` does not
/// depend on the thread count or on rayon's splitting.
fn summarize<S: CliSemiring>(
    s: &S,
    val: &[S::Elem],
    lo: usize,
    top: usize,
    buckets: Option<&Buckets>,
) -> ValueSummary {
    let z = s.zero();
    let r = distribution(
        val.len(),
        |i| (!bit_eq(&val[i], &z)).then(|| s.key_f64(val[i])),
        lo,
        &DistConf { top, buckets },
    );
    let a = r.all;
    ValueSummary {
        nodes: val.len(),
        zero_nodes: a.absent,
        max: a.max,
        argmax: a.argmax,
        min: a.min,
        argmin: a.argmin,
        mean: a.mean,
        log2_hist: a.log2_hist,
        top: a.top,
        by_bucket: r.by_bucket,
        unbucketed: r.unbucketed,
    }
}

/// Result of `--compare`.
#[derive(Debug, Default, serde::Serialize)]
pub struct Comparison {
    /// The reference file.
    pub file: PathBuf,
    /// Entries compared.
    pub compared: usize,
    /// Entries that differ (bitwise).
    pub mismatches: u64,
    /// Smallest differing node (absolute id).
    pub first_mismatch: Option<usize>,
}

/// Compares `val` (window `w` of a graph with `n` nodes) with the array in `file`.
fn compare_with<E: Dump>(val: &[E], w: Window, n: usize, file: &Path) -> Result<Comparison> {
    let bytes = std::fs::metadata(file)
        .with_context(|| format!("cannot stat {}", file.display()))?
        .len() as usize;
    ensure!(
        bytes % E::BYTES == 0,
        "{}: size {bytes} is not a multiple of {} ({})",
        file.display(),
        E::BYTES,
        E::DTYPE
    );
    let len = bytes / E::BYTES;
    let off = if len == n {
        w.lo
    } else if len == w.len() {
        0
    } else {
        bail!(
            "{} has {len} entries: expected {n} (graph) or {} (window)",
            file.display(),
            w.len()
        )
    };
    let map = webgraph::utils::MmapHelper::<u8>::mmap(file, mmap_rs::MmapFlags::SEQUENTIAL)
        .with_context(|| format!("cannot map {}", file.display()))?;
    let raw: &[u8] = map.as_ref();
    let (mismatches, first) = val
        .par_iter()
        .enumerate()
        .with_min_len(1 << 16)
        .fold(
            || (0u64, None::<usize>),
            |(c, f), (i, x)| {
                let at = (off + i) * E::BYTES;
                if bit_eq(x, &E::get(&raw[at..at + E::BYTES])) {
                    (c, f)
                } else {
                    (c + 1, f.or(Some(i + w.lo)))
                }
            },
        )
        .reduce(
            || (0, None),
            |a, b| {
                (
                    a.0 + b.0,
                    match (a.1, b.1) {
                        (Some(x), Some(y)) => Some(x.min(y)),
                        (x, y) => x.or(y),
                    },
                )
            },
        );
    Ok(Comparison {
        file: file.to_owned(),
        compared: val.len(),
        mismatches,
        first_mismatch: first,
    })
}

/// Outdegrees of the window nodes of `g` (one parallel pass over `win`), indexed by `u − lo`.
fn outdegrees(g: &ChunkedGraph<BvRa>, win: Window, budget: &mut MemBudget) -> Result<Vec<u32>> {
    budget.need(4 * win.len() as u64, "outdegree array")?;
    let mut pl = logger("node");
    pl.start(format!("computing outdegrees over {win}"));
    let d = per_window_node(win, 0u32, &|u| Ok(g.graph.outdegree(u) as u32))?;
    pl.done_with_count(win.len());
    Ok(d)
}

/// Whether a `dp`/`cpm` run can do without the BV graph: a saved index (`--blocks PREFIX`)
/// whose block kernels (`--kernel auto|blocks`, every command-line weight is separable) are
/// the only consumers. `graph_needed` covers the other consumers (`--certify`,
/// `--critical-arcs`). Not opening the graph saves its in-RAM metadata (≈ 1.3 GiB on PG).
pub fn block_only(opt: &BlocksOpt, kernel: Kernel, graph_needed: bool) -> bool {
    matches!(opt.blocks, BlocksSpec::Prefix(_))
        && matches!(kernel, Kernel::Auto | Kernel::Blocks)
        && !graph_needed
}

/// The block index of a `dp`/`cpm` run over `win` (graph of `n` nodes), window-aware:
///
/// - `--kernel push|pull` or `--blocks none`: no index;
/// - `--blocks PREFIX`: loaded with `--load` for the full graph; a strict sub-window loads it
///   with [`LoadMode::Mmap`] (the block kernels touch only the window's slice of
///   `create`/`spend`/blocks, while `--load ram` would read all `8n + 12·nb` bytes) and
///   reserves nothing (file-backed pages);
/// - `--blocks auto`: built from `g` with [`BlocksOpt::resolve`] only for the full window or
///   when `--kernel blocks` asks for it explicitly; for a strict sub-window under
///   `--kernel auto` it is skipped (with a log line) and the generic kernels run, because
///   building it is a pass over the whole graph reserving `8n + 12·nb` bytes, whatever the
///   window ([`BlockIndex::build_budgeted`] has no window).
pub fn window_blocks(
    opt: &BlocksOpt,
    kernel: Kernel,
    g: Option<&ChunkedGraph<BvRa>>,
    n: usize,
    win: Window,
    global: &GlobalArgs,
    budget: &mut MemBudget,
) -> Result<Option<BlockIndex>> {
    if !matches!(kernel, Kernel::Auto | Kernel::Blocks) {
        return Ok(None);
    }
    let full = win.is_full(n);
    match &opt.blocks {
        BlocksSpec::None => Ok(None),
        BlocksSpec::Prefix(p) => {
            let mode = if full { global.load } else { LoadMode::Mmap };
            if mode != global.load {
                log::info!(
                    "{win} is a strict sub-window: memory-mapping {}.blk.*",
                    p.display()
                );
            }
            // the sizes come from `<p>.blk.json`, so a RAM load is budgeted before any array is
            // read (§8)
            let json = crate::io::suffixed(p, "blk.json");
            let report: BlockReport = serde_json::from_str(
                &std::fs::read_to_string(&json)
                    .with_context(|| format!("cannot read {}", json.display()))?,
            )
            .with_context(|| format!("cannot parse {}", json.display()))?;
            ensure!(
                report.n == n,
                "{}.blk.* has {} nodes, the graph has {n}",
                p.display(),
                report.n
            );
            if mode == LoadMode::Ram {
                budget.need(BlockIndex::bytes(report.n, report.nb), "BlockIndex")?;
            }
            let idx = BlockIndex::load(p, mode)?;
            ensure!(
                (idx.n, idx.nb) == (report.n, report.nb),
                "{} changed while loading",
                json.display()
            );
            Ok(Some(idx))
        }
        BlocksSpec::Auto if !full && kernel == Kernel::Auto => {
            log::info!(
                "--blocks auto: not building the BlockIndex for the strict sub-window {win} \
                 (a pass over all {n} nodes); using the generic kernels (pass --blocks PREFIX \
                 or --kernel blocks to use the block kernels)"
            );
            Ok(None)
        }
        BlocksSpec::Auto => {
            let g = g.context("--blocks auto needs the graph")?;
            opt.resolve(g, global, budget)
        }
    }
}

/// Runs `dag dp`.
pub fn run_dp(args: &DpArgs, g: &GlobalArgs) -> Result<ExitCode> {
    with_semiring!(args.semiring, s => dp_with(s, args, g))
}

fn dp_with<S: CliSemiring>(s: S, args: &DpArgs, global: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = args.out.prepare()?.to_owned();
    let mut budget = global.budget();
    let (n, m) = graph::read_properties(&args.graph.graph)?;
    let win = args.window.window(n)?;
    let buckets = args.stats.buckets.load(n)?;
    ensure!(
        !args.pred || S::SELECTIVE,
        "--pred needs a selective semiring ({} is not)",
        s.name()
    );
    if args.kernel == Kernel::Blocks && args.blocks.blocks == BlocksSpec::None {
        bail!("--kernel blocks needs --blocks auto or --blocks PREFIX");
    }
    let cg = if block_only(&args.blocks, args.kernel, args.certify) {
        log::info!("block-only run: the BV graph is not opened");
        None
    } else {
        let cg = args.graph.open(global)?;
        ensure!(
            cg.num_nodes() == n,
            "{} changed while opening it",
            args.graph.graph.display()
        );
        Some(cg)
    };
    let blocks = window_blocks(
        &args.blocks,
        args.kernel,
        cg.as_ref(),
        n,
        win,
        global,
        &mut budget,
    )?;
    let forward = args.dir == Dir::Forward;
    let needs_gt = forward
        && (args.certify
            || args.kernel == Kernel::Pull
            || (args.kernel == Kernel::Auto && blocks.is_none()));
    let gt = if needs_gt {
        let gt = args.graph.open_transpose(global)?;
        if gt.is_none() && args.kernel == Kernel::Pull {
            bail!("--kernel pull --dir forward needs the transpose (--transpose or <graph>-t)");
        }
        if gt.is_none() && args.certify {
            bail!("certifying a forward DP needs the transpose (--transpose or <graph>-t)");
        }
        gt
    } else {
        None
    };
    let mut input = match (&cg, &gt, &blocks) {
        (Some(cg), Some(gt), _) => DpInput::with_transpose(cg, gt),
        (Some(cg), None, _) => DpInput::graph(cg),
        (None, _, Some(b)) => DpInput::blocks(b),
        (None, _, None) => bail!("a block-only run needs the BlockIndex"),
    };
    if let Some(b) = &blocks {
        input = input.and_blocks(b);
    }
    let kernel = resolve_kernel(&input, args.dir, true, args.kernel)?;
    if kernel == Kernel::Blocks && blocks.is_none() {
        bail!("the graph is not a biclique digraph: the blocks kernel is not available");
    }
    let outdeg_vec;
    let w = {
        let outdeg: Box<dyn Fn(usize) -> usize + Sync> = match (&args.weights, &blocks, &cg) {
            (WeightsSpec::InvOutdeg, Some(b), _) => Box::new(|u| b.outdegree(u)),
            (WeightsSpec::InvOutdeg, None, Some(cg)) => {
                outdeg_vec = outdegrees(cg, win, &mut budget)?;
                Box::new(|u| outdeg_vec[u - win.lo] as usize)
            }
            _ => Box::new(|_| 0),
        };
        weights(&s, &args.weights, n, win, outdeg, &mut budget)?
    };
    let sweep = global.sweep_conf();
    let n_blocks = blocks
        .as_ref()
        .map_or(0, |b| b.block_bound(win.hi) - b.block_bound(win.lo));
    budget.need(
        estimate_bytes::<S::Elem>(
            win.len(),
            args.dir,
            kernel,
            args.pred,
            n_blocks,
            sweep_buffer_bytes(&sweep, global.chunk_arcs, global.max_chunk_nodes),
        ),
        "dp arrays",
    )?;
    let conf = DpConf {
        window: win,
        kernel,
        pred: args.pred,
        sweep,
    };
    let init = init_of(args.init, s.one());

    let mut info = RunInfo::new(Some(args.graph.graph.clone()), n, m, win);
    // everything before the kernel: opening the graph, loading or building the blocks, weights
    info.phases_s.add("open", t0.elapsed().as_secs_f64());
    info.params = json!({
        "semiring": s.name(),
        "dir": args.dir,
        "init": format!("{:?}", args.init),
        "weights": format!("{:?}", args.weights),
        "weights_kind": w.describe(),
        "graph_opened": cg.is_some(),
        "kernel_requested": args.kernel,
        "kernel": kernel,
        "pred": args.pred,
        "blocks": blocks.is_some(),
        "buckets": args.stats.buckets.buckets,
        "top": args.stats.top,
    });
    let (res, kernel_s) = with_weights!(s, &w, ww => {
        let tk = Instant::now();
        let mut pl = logger("node");
        let res = solve(&s, &input, args.dir, &ww, &init, &conf, &mut pl)?;
        (res, tk.elapsed().as_secs_f64())
    });
    log::info!("dp kernel {kernel:?}: {kernel_s:.3} s");
    info.phases_s.add("kernel", kernel_s);
    let summary = info.phases_s.time("summary", || {
        summarize(&s, &res.val, win.lo, args.stats.top, buckets.as_ref())
    });
    log::info!(
        "values: {} nodes, {} 0̄, max {:?} at {:?}, min {:?} at {:?}, mean {:?}",
        summary.nodes,
        summary.zero_nodes,
        summary.max,
        summary.argmax,
        summary.min,
        summary.argmin,
        summary.mean
    );

    let tw = Instant::now();
    let vpath = io::array_path::<S::Elem>(&out, "val");
    io::write_array(&vpath, &res.val)?;
    info.files.push(FileInfo::new::<S::Elem>(
        &vpath,
        res.val.len(),
        Some("0̄ of the semiring = unreached"),
        format!("{} {:?} DP value of v − lo", s.name(), args.dir),
    ));
    if let Some(p) = &res.pred {
        let ppath = io::array_path::<u32>(&out, "pred");
        io::write_array(&ppath, p)?;
        info.files.push(FileInfo::new::<u32>(
            &ppath,
            p.len(),
            Some("u32::MAX = NONE"),
            "argbest predecessor (forward) or successor (backward), absolute id",
        ));
    }
    let write_s = tw.elapsed().as_secs_f64();
    info.phases_s.add("write", write_s);

    let mut ok = true;
    let mut cert: Option<(CertReport, f64)> = None;
    if args.certify {
        let tc = Instant::now();
        let cg = cg.as_ref().context("certification needs the graph")?;
        let rep = with_weights!(s, &w, ww => certify_with(&s, cg, gt.as_ref(), args.dir, &ww, &init, &res)?);
        log::info!("certification: {rep:?}");
        ok &= rep.ok();
        cert = Some((rep, tc.elapsed().as_secs_f64()));
        info.phases_s.add("certify", tc.elapsed().as_secs_f64());
    }
    let comparison = match &args.compare {
        Some(f) => {
            let c = info
                .phases_s
                .time("compare", || compare_with(&res.val, win, n, f))?;
            log::info!("comparison: {c:?}");
            ok &= c.mismatches == 0;
            Some(c)
        }
        None => None,
    };
    info.results = json!({
        "kernel_s": kernel_s,
        "write_s": write_s,
        "summary": summary,
        "certify": cert.as_ref().map(|c| &c.0),
        "certify_s": cert.as_ref().map(|c| c.1),
        "comparison": comparison,
    });
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(ok))
}

/// Certifies `res` with the semiring's default tolerance, reading the transpose (forward) or
/// the graph (backward).
fn certify_with<S: CliSemiring, W: ArcWeights<S>>(
    s: &S,
    g: &ChunkedGraph<BvRa>,
    gt: Option<&ChunkedGraph<BvRa>>,
    dir: Dir,
    w: &W,
    init: &Init<S::Elem>,
    res: &DpResult<S::Elem>,
) -> Result<CertReport> {
    let on = match dir {
        Dir::Forward => gt.context("certifying a forward DP needs the transpose")?,
        Dir::Backward => g,
    };
    let mut pl = concurrent_logger("node");
    certify(s, on, dir, w, init, res, s.tolerance(), &mut pl)
}

// ---------------------------------------------------------------------------------------------
// certify
// ---------------------------------------------------------------------------------------------

/// Runs `dag certify`.
pub fn run_certify(args: &CertifyArgs, g: &GlobalArgs) -> Result<ExitCode> {
    with_semiring!(args.semiring, s => certify_cli(s, args, g))
}

fn certify_cli<S: CliSemiring>(s: S, args: &CertifyArgs, global: &GlobalArgs) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = args.out.prepare()?.to_owned();
    let mut budget = global.budget();
    let cg = args.graph.open(global)?;
    let n = cg.num_nodes();
    let win = args.window.window(n)?;
    let gt =
        match args.dir {
            Dir::Forward => Some(args.graph.open_transpose(global)?.context(
                "certifying a forward DP needs the transpose (--transpose or <graph>-t)",
            )?),
            Dir::Backward => None,
        };
    budget.need(
        ((std::mem::size_of::<S::Elem>() + if args.pred.is_some() { 4 } else { 0 }) * win.len())
            as u64,
        "values to certify",
    )?;
    let val = io::read_array::<S::Elem>(&args.values)?;
    ensure!(
        val.len() == win.len(),
        "{} has {} values, the window {win} has {} nodes",
        args.values.display(),
        val.len(),
        win.len()
    );
    let pred = match &args.pred {
        Some(f) => {
            ensure!(S::SELECTIVE, "predecessors need a selective semiring");
            let p = io::read_array::<u32>(f)?;
            ensure!(
                p.len() == win.len(),
                "{} has {} entries, the window {win} has {} nodes",
                f.display(),
                p.len(),
                win.len()
            );
            Some(p)
        }
        None => None,
    };
    let res = DpResult {
        window: win,
        val,
        pred,
    };
    let outdeg_vec;
    let w = {
        let outdeg: Box<dyn Fn(usize) -> usize + Sync> = match &args.weights {
            WeightsSpec::InvOutdeg => {
                outdeg_vec = outdegrees(&cg, win, &mut budget)?;
                Box::new(|u| outdeg_vec[u - win.lo] as usize)
            }
            _ => Box::new(|_| 0),
        };
        weights(&s, &args.weights, n, win, outdeg, &mut budget)?
    };
    let init = init_of(args.init, s.one());
    let open_s = t0.elapsed().as_secs_f64();
    let tc = Instant::now();
    let rep =
        with_weights!(s, &w, ww => certify_with(&s, &cg, gt.as_ref(), args.dir, &ww, &init, &res)?);
    let certify_s = tc.elapsed().as_secs_f64();
    log::info!("certification: {rep:?}");
    println!("{}", serde_json::to_string_pretty(&rep)?);
    let mut info = RunInfo::new(Some(args.graph.graph.clone()), n, cg.num_arcs, win);
    info.phases_s.add("open", open_s);
    info.phases_s.add("certify", certify_s);
    info.params = json!({
        "semiring": s.name(),
        "dir": args.dir,
        "init": format!("{:?}", args.init),
        "weights": format!("{:?}", args.weights),
        "values": args.values,
        "pred": args.pred,
        "tolerance": s.tolerance(),
    });
    info.results = serde_json::to_value(&rep)?;
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(exit_status(rep.ok()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extfloat_conversion_normalises() {
        // only the field layout is used, so this holds whatever the arithmetic of ExtFloat
        let s = ExtFloat::new();
        for x in [1.0, 1.5, 3.0, 0.25, 1e300, 7.0 / 3.0] {
            let e = s.of_f64(x).unwrap();
            assert!((1.0..2.0).contains(&e.m), "{x}: {e:?}");
            assert!((e.m * (e.e as f64).exp2() - x).abs() <= x * 1e-15);
        }
        assert!(s.of_f64(-1.0).is_none());
    }

    #[test]
    fn integer_semirings_reject_fractions() {
        assert_eq!(MaxPlus::<u32>::new().of_f64(0.5), None);
        assert_eq!(MaxPlus::<u32>::new().of_f64(3.0), Some(3));
        assert_eq!(MaxPlus::<u32>::new().of_u32(u32::MAX), u32::MAX - 1);
    }

    #[test]
    fn min_plus_i64_accepts_negative_integral_weights() {
        let s = MinPlus::<i64>::new();
        assert_eq!(s.of_f64(-3.0), Some(-3));
        assert_eq!(s.of_f64(-0.5), None);
        assert_eq!(s.of_f64(-1e19), None);
        assert_eq!(s.of_f64(1e19), None);
        assert_eq!(s.of_f64(f64::NAN), None);
        assert_eq!(s.of_f64(f64::NEG_INFINITY), None);
        assert!(
            s.of_f64(-(i64::MAX as f64) + 1024.0)
                .is_some_and(|x| x > i64::MIN)
        );
        assert_eq!(MaxPlus::<u32>::new().of_f64(-1.0), None);
        assert_eq!(MinPlus::<u32>::new().of_f64(-1.0), None);
    }

    const SUMMARY_CHUNK: usize = crate::summary::CHUNK;

    #[test]
    fn summary_is_independent_of_the_thread_count() {
        // regression: the mean was a rayon fold/reduce, whose summation order varies
        let s = MaxPlus::<f64>::new();
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let val: Vec<f64> = (0..3 * SUMMARY_CHUNK + 12345)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                if i % 97 == 0 {
                    s.zero()
                } else {
                    (x >> 11) as f64 * 1e-3 + 1e12 * f64::from(u8::from(i % 3 == 0))
                }
            })
            .collect();
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| summarize(&s, &val, 5, 20, None))
        };
        let want = run(1);
        for threads in [2, 3, 8] {
            let got = run(threads);
            assert_eq!(
                got.mean.map(f64::to_bits),
                want.mean.map(f64::to_bits),
                "{threads} threads"
            );
            assert_eq!(
                (got.argmax, got.argmin, got.zero_nodes),
                (want.argmax, want.argmin, want.zero_nodes)
            );
        }
        // the sequential definition, chunk by chunk
        let mut sum = 0.0;
        for c in val.chunks(SUMMARY_CHUNK) {
            sum += c
                .iter()
                .filter(|v| !bit_eq(*v, &s.zero()))
                .fold(0.0, |a, v| a + v);
        }
        let cnt = val.iter().filter(|v| !bit_eq(*v, &s.zero())).count();
        assert_eq!(want.mean.unwrap().to_bits(), (sum / cnt as f64).to_bits());
    }
}
