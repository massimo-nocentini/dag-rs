//! `dag aggregate`: the distribution of a stored per-node or per-block array, overall and
//! per node-range bucket ([`crate::summary`]).
//!
//! Every producing subcommand already reports the distributions of its outputs; `aggregate`
//! computes the same [`DistReport`] for any array written earlier (or by another tool) without
//! re-running the analysis: for example, the dominator depths and subtree sizes per half-year of
//! the Payment Graph with
//!
//! ```text
//! dag aggregate --values P.dom.depth.u32le --buckets chunk_boundaries.tsv --out P.dom.depth.agg
//! ```
//!
//! The element type comes from the file suffix (`u8`, `u32le`, `u64le`, `i64le`, `f32le`,
//! `f64le`) unless `--dtype` is given. Entry `i` is node `--lo + i`; with `--blocks PREFIX` the
//! entries are blocks `--lo + i` of that BlockIndex, bucketed by their first output (as the
//! `blk.*` arrays of `reach-counts`). `--absent max` treats the largest value of the type
//! (`u32::MAX`, the 0̄ of the unsigned tropical semirings and the `NONE` sentinel) as no value,
//! `--absent zero` the value 0; NaN is never a value. The array is memory-mapped and scanned
//! once in parallel. Output: `<out>.json` whose `results` is the [`DistReport`].

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args as ClapArgs, ValueEnum};
use serde_json::json;

use super::common::{OutOpt, StatsOpt};
use crate::io::{Dump, LoadMode, RunInfo, U32Array, suffixed};
use crate::summary::{DistConf, DistReport, distribution_with};
use crate::window::Window;

/// Element types readable by `aggregate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, serde::Serialize)]
pub enum Dtype {
    /// `u8`.
    U8,
    /// `u32le`.
    U32le,
    /// `u64le`.
    U64le,
    /// `i64le`.
    I64le,
    /// `f32le`.
    F32le,
    /// `f64le`.
    F64le,
}

impl Dtype {
    /// The type named by the last suffix of `path` (`run.dom.depth.u32le` gives `U32le`).
    pub fn of_path(path: &Path) -> Option<Self> {
        Some(match path.extension()?.to_str()? {
            "u8" => Self::U8,
            "u32le" => Self::U32le,
            "u64le" => Self::U64le,
            "i64le" => Self::I64le,
            "f32le" => Self::F32le,
            "f64le" => Self::F64le,
            _ => return None,
        })
    }

    fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32le | Self::F32le => 4,
            Self::U64le | Self::I64le | Self::F64le => 8,
        }
    }
}

/// Which stored value means "no value".
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, serde::Serialize)]
pub enum Absent {
    /// Every value counts (NaN excepted).
    None,
    /// The largest value of the type (`u32::MAX`, …; `+∞` for floats).
    Max,
    /// The value 0.
    Zero,
}

/// `dag aggregate`.
#[derive(ClapArgs, Debug)]
pub struct Args {
    /// The array to summarise.
    #[arg(long)]
    pub values: PathBuf,
    /// Element type (default: from the file suffix).
    #[arg(long, value_enum)]
    pub dtype: Option<Dtype>,
    /// Absolute id of entry 0 (the window start of a windowed run, or the first block).
    #[arg(long, default_value_t = 0)]
    pub lo: usize,
    /// The entries are blocks of this BlockIndex (`PREFIX.blk.first.u32le` gives their nodes).
    #[arg(long)]
    pub blocks: Option<PathBuf>,
    /// Which stored value means "no value".
    #[arg(long, value_enum, default_value = "none")]
    pub absent: Absent,
    /// Aggregate measures (`--buckets`, `--top`).
    #[command(flatten)]
    pub stats: StatsOpt,
    /// Output prefix (`<out>.json`).
    #[command(flatten)]
    pub out: OutOpt,
}

/// The [`DistReport`] of the little-endian array `raw` of type `dtype` (see the module
/// documentation): entry `i` has id `lo + i` and, with `first`, node `first[lo + i]`.
pub fn aggregate_bytes(
    raw: &[u8],
    dtype: Dtype,
    absent: Absent,
    lo: usize,
    first: Option<&[u32]>,
    conf: &DistConf<'_>,
) -> DistReport {
    fn key<T: Dump + Into<f64>>(raw: &[u8], i: usize, max: T, absent: Absent) -> Option<f64> {
        let x: f64 = T::get(&raw[i * T::BYTES..]).into();
        match absent {
            Absent::None => Some(x),
            Absent::Max => (x != max.into()).then_some(x),
            Absent::Zero => (x != 0.0).then_some(x),
        }
    }
    let len = raw.len() / dtype.bytes();
    let k = |i: usize| match dtype {
        Dtype::U8 => key(raw, i, u8::MAX, absent),
        Dtype::U32le => key(raw, i, u32::MAX, absent),
        Dtype::F32le => key(raw, i, f32::INFINITY, absent),
        Dtype::F64le => key(raw, i, f64::INFINITY, absent),
        // 64-bit integers: the f64 conversion rounds above 2^53, the sentinel test is exact
        Dtype::U64le => {
            let x = u64::get(&raw[i * 8..]);
            match absent {
                Absent::Max if x == u64::MAX => None,
                Absent::Zero if x == 0 => None,
                _ => Some(x as f64),
            }
        }
        Dtype::I64le => {
            let x = i64::get(&raw[i * 8..]);
            match absent {
                Absent::Max if x == i64::MAX => None,
                Absent::Zero if x == 0 => None,
                _ => Some(x as f64),
            }
        }
    };
    let id = |i: usize| lo + i;
    match first {
        Some(f) => {
            let node = |i: usize| f[lo + i] as usize;
            distribution_with(len, k, &id, Some(&node), conf)
        }
        None => distribution_with(len, k, &id, None, conf),
    }
}

/// Runs `dag aggregate`.
pub fn run(args: &Args) -> Result<ExitCode> {
    let t0 = Instant::now();
    let out = args.out.prepare()?.to_owned();
    let dtype = match args.dtype.or_else(|| Dtype::of_path(&args.values)) {
        Some(d) => d,
        None => bail!(
            "cannot infer the element type of {}: pass --dtype",
            args.values.display()
        ),
    };
    let bytes = std::fs::metadata(&args.values)
        .with_context(|| format!("cannot stat {}", args.values.display()))?
        .len() as usize;
    ensure!(
        bytes % dtype.bytes() == 0,
        "{}: size {bytes} is not a multiple of {} ({dtype:?})",
        args.values.display(),
        dtype.bytes()
    );
    let len = bytes / dtype.bytes();
    let first = match &args.blocks {
        Some(p) => {
            let f = U32Array::load(&suffixed(p, "blk.first.u32le"), LoadMode::Mmap)?;
            ensure!(
                args.lo + len <= f.len(),
                "{} entries from block {} exceed the {} blocks of {}",
                len,
                args.lo,
                f.len(),
                p.display()
            );
            Some(f)
        }
        None => None,
    };
    // nodes covered: [lo, lo + len), or the nodes of the blocks
    let n_hint = match &first {
        Some(f) if len > 0 => f[args.lo + len - 1] as usize + 1,
        Some(_) => 0,
        None => args.lo + len,
    };
    let buckets = args.stats.buckets.load(n_hint.max(1))?;
    let conf = DistConf {
        top: args.stats.top,
        buckets: buckets.as_ref(),
    };
    let map = (bytes > 0)
        .then(|| {
            webgraph::utils::MmapHelper::<u8>::mmap(&args.values, mmap_rs::MmapFlags::SEQUENTIAL)
                .with_context(|| format!("cannot map {}", args.values.display()))
        })
        .transpose()?;
    let raw: &[u8] = map.as_ref().map_or(&[], |m| m.as_ref());
    let open_s = t0.elapsed().as_secs_f64();
    let ts = Instant::now();
    let report = aggregate_bytes(raw, dtype, args.absent, args.lo, first.as_deref(), &conf);
    log::info!(
        "{}: {} values, {} absent, mean {:?}, max {:?} at {:?}",
        args.values.display(),
        report.all.count,
        report.all.absent,
        report.all.mean,
        report.all.max,
        report.all.argmax
    );
    let window = Window {
        lo: args.lo,
        hi: args.lo + len,
    };
    let mut info = RunInfo::new(None, len, 0, window);
    info.phases_s.add("open", open_s);
    info.phases_s.add("summary", ts.elapsed().as_secs_f64());
    info.params = json!({
        "values": args.values,
        "dtype": dtype,
        "lo": args.lo,
        "blocks": args.blocks,
        "absent": args.absent,
        "buckets": args.stats.buckets.buckets,
        "top": args.stats.top,
    });
    info.results = serde_json::to_value(&report)?;
    info.elapsed_s = t0.elapsed().as_secs_f64();
    info.write(&out)?;
    Ok(ExitCode::SUCCESS)
}
