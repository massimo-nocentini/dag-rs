//! Input/output of per-node arrays, large-array allocation, memory budget and run metadata.
//!
//! # File format
//! Every per-node or per-block output is a **raw little-endian array**
//! `<out>.<field>.<dtype>` (for example `run.E.u32le`) plus one JSON sidecar `<out>.json`
//! ([`RunInfo`]). The sentinel for "no node" is [`NONE`] = `u32::MAX`. This matches the
//! existing tools of the repository (`analysis_path`, `pg_analysis/results/*.u32le`), so outputs
//! can be compared byte for byte.
//!
//! # Memory policy
//! - Large arrays come from [`huge_vec`]: `madvise(MADV_HUGEPAGE)` on the page-aligned interior
//!   (the host has THP in `madvise` mode) and a **parallel first touch**, which also spreads the
//!   pages over both NUMA nodes. Every PG kernel does random accesses into multi-GB arrays, and
//!   the STLB covers only a few MB with 4 KiB pages.
//! - Persisted arrays are loaded into RAM by default ([`LoadMode::Ram`]); [`LoadMode::Mmap`] maps
//!   them with sequential advice, never `MADV_RANDOM`.
//! - Every subcommand checks its estimated footprint with [`MemBudget::need`] before allocating.
//!
//! # Unsafe code
//! The allowed sites in this module are [`huge_vec`] (`libc::madvise`, advisory only) and
//! [`as_atomic_u32`]/[`as_atomic_u64`] (layout-compatible reinterpretation).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64};

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::error::DagError;
use crate::window::Window;

/// The "no node" sentinel stored in `u32` arrays (`u32::MAX`). Node ids are always smaller, see
/// [`graph::open`](crate::graph::open).
pub const NONE: u32 = u32::MAX;

/// Size of the buffers used by the array readers and writers (1 MiB).
const IO_BUF: usize = 1 << 20;

/// Bytes converted per parallel batch by [`ArrayWriter::extend`] (128 MiB).
///
/// Each batch is one short parallel phase followed by a sequential `write`. Between phases the
/// idle rayon workers spin (`sched_yield`) before they sleep, so small batches cost far more
/// system time than conversion time: with 4 Mi-element batches and 112 threads, `strace -c`
/// counted 6·10⁶ `sched_yield` calls (15,000 s of system time) while loading 26 GB. Large
/// batches keep the number of phases, and hence the spinning, small.
const WRITE_BATCH_BYTES: usize = 1 << 27;

/// Concurrent sequential streams of [`read_array`]: few enough to keep a rotational RAID
/// streaming, enough to read the page cache at memory bandwidth.
const READ_STREAMS: usize = 8;

/// Bytes read per `pread` by each stream of [`read_array`] (8 MiB).
const READ_PIECE_BYTES: usize = 1 << 23;

/// Threshold above which [`huge_vec`] asks for transparent huge pages (2 MiB).
const HUGE_PAGE: usize = 1 << 21;

// ---------------------------------------------------------------------------------------------
// Dump: fixed-size little-endian elements
// ---------------------------------------------------------------------------------------------

/// A fixed-size element stored in little-endian order in raw array files.
///
/// `DTYPE` is the file suffix (`u32le`, `f64le`, ...), `BYTES` the element size.
pub trait Dump: Copy + Send + Sync + 'static {
    /// Name used as the file suffix, e.g. `"u32le"`.
    const DTYPE: &'static str;
    /// Size of one element in bytes.
    const BYTES: usize;
    /// Writes `self` into `out[..Self::BYTES]`.
    fn put(&self, out: &mut [u8]);
    /// Reads an element from `inp[..Self::BYTES]`.
    fn get(inp: &[u8]) -> Self;
}

macro_rules! dump_le {
    ($t:ty, $name:literal) => {
        impl Dump for $t {
            const DTYPE: &'static str = $name;
            const BYTES: usize = std::mem::size_of::<$t>();
            #[inline(always)]
            fn put(&self, out: &mut [u8]) {
                out[..Self::BYTES].copy_from_slice(&self.to_le_bytes());
            }
            #[inline(always)]
            fn get(inp: &[u8]) -> Self {
                <$t>::from_le_bytes(inp[..Self::BYTES].try_into().unwrap())
            }
        }
    };
}

dump_le!(u8, "u8");
dump_le!(u32, "u32le");
dump_le!(u64, "u64le");
dump_le!(i64, "i64le");
dump_le!(u128, "u128le");
dump_le!(f32, "f32le");
dump_le!(f64, "f64le");

/// Booleans are stored as one byte (`0` or `1`); any non-zero byte reads as `true`.
impl Dump for bool {
    const DTYPE: &'static str = "u8";
    const BYTES: usize = 1;
    #[inline(always)]
    fn put(&self, out: &mut [u8]) {
        out[0] = *self as u8;
    }
    #[inline(always)]
    fn get(inp: &[u8]) -> Self {
        inp[0] != 0
    }
}

/// Pairs are stored as the concatenation of their components (`DTYPE` = `"pair"`).
impl<A: Dump, B: Dump> Dump for (A, B) {
    const DTYPE: &'static str = "pair";
    const BYTES: usize = A::BYTES + B::BYTES;
    #[inline(always)]
    fn put(&self, out: &mut [u8]) {
        self.0.put(&mut out[..A::BYTES]);
        self.1.put(&mut out[A::BYTES..A::BYTES + B::BYTES]);
    }
    #[inline(always)]
    fn get(inp: &[u8]) -> Self {
        (
            A::get(&inp[..A::BYTES]),
            B::get(&inp[A::BYTES..A::BYTES + B::BYTES]),
        )
    }
}

/// Appends `.suffix` to the file name of `prefix` (unlike [`Path::with_extension`], it never
/// replaces an existing extension: `run.v1` + `E.u32le` gives `run.v1.E.u32le`).
pub fn suffixed(prefix: &Path, suffix: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(suffix);
    PathBuf::from(s)
}

/// The conventional path `<prefix>.<field>.<T::DTYPE>` of a per-node array.
pub fn array_path<T: Dump>(prefix: &Path, field: &str) -> PathBuf {
    suffixed(prefix, &format!("{field}.{}", T::DTYPE))
}

/// Writes `data` as a raw little-endian array (1 MiB buffered, conversion in parallel).
pub fn write_array<T: Dump>(path: &Path, data: &[T]) -> Result<()> {
    let mut w = array_writer::<T>(path)?;
    w.extend(data)?;
    w.finish()?;
    Ok(())
}

/// Opens a streaming writer of a raw little-endian array.
///
/// An existing file at `path` is **unlinked** first and a new one created, rather than
/// truncated in place: on ext4 (the `/data` RAID of the PG host) rewriting a truncated
/// multi-GB file ran at 360 MB/s against 1.5 GB/s for a new file (`dd` of 6 GiB, measured), and
/// re-running a subcommand over its own outputs made writing 52 GB of dominator arrays take
/// 232 s instead of 26 s. Hard links and permissions of the old file are not preserved.
pub fn array_writer<T: Dump>(path: &Path) -> Result<ArrayWriter<T>> {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(e).with_context(|| format!("cannot replace {}", path.display()));
        }
    }
    let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    Ok(ArrayWriter {
        out: BufWriter::with_capacity(IO_BUF, file),
        path: path.to_owned(),
        buf: Vec::new(),
        len: 0,
        _marker: PhantomData,
    })
}

/// A streaming writer of raw little-endian arrays, see [`array_writer`].
pub struct ArrayWriter<T: Dump> {
    out: BufWriter<File>,
    path: PathBuf,
    buf: Vec<u8>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Dump> ArrayWriter<T> {
    /// Appends one element.
    #[inline]
    pub fn push(&mut self, x: T) -> Result<()> {
        let mut tmp = [0u8; 64];
        if T::BYTES <= tmp.len() {
            x.put(&mut tmp[..T::BYTES]);
            self.out.write_all(&tmp[..T::BYTES])?;
        } else {
            let mut v = vec![0u8; T::BYTES];
            x.put(&mut v);
            self.out.write_all(&v)?;
        }
        self.len += 1;
        Ok(())
    }

    /// Appends a slice of elements (converted in parallel batches of 128 MiB).
    pub fn extend(&mut self, data: &[T]) -> Result<()> {
        for batch in data.chunks((WRITE_BATCH_BYTES / T::BYTES).max(1)) {
            self.buf.resize(batch.len() * T::BYTES, 0);
            self.buf
                .par_chunks_mut(T::BYTES * 4096)
                .zip(batch.par_chunks(4096))
                .for_each(|(out, inp)| {
                    for (o, x) in out.chunks_exact_mut(T::BYTES).zip(inp) {
                        x.put(o);
                    }
                });
            self.out
                .write_all(&self.buf)
                .with_context(|| format!("cannot write {}", self.path.display()))?;
            self.len += batch.len();
        }
        Ok(())
    }

    /// Number of elements written so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing was written yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Flushes the file and returns the number of elements written.
    pub fn finish(mut self) -> Result<usize> {
        self.out
            .flush()
            .with_context(|| format!("cannot flush {}", self.path.display()))?;
        Ok(self.len)
    }
}

/// Reads a whole raw little-endian array into memory (storage from [`huge_vec`]).
///
/// The file is split into 8 contiguous parts read concurrently, each sequentially with
/// positioned reads of 8 MiB, so the whole load is **one** parallel phase: between short
/// parallel phases the idle rayon workers spin (`sched_yield`), which on 112 threads cost more
/// system time than the load itself. Transient memory: 8 MiB per stream.
pub fn read_array<T: Dump>(path: &Path) -> Result<Vec<T>> {
    use std::os::unix::fs::FileExt;
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let bytes = file.metadata()?.len() as usize;
    anyhow::ensure!(
        bytes % T::BYTES == 0,
        "{}: size {bytes} is not a multiple of {} ({})",
        path.display(),
        T::BYTES,
        T::DTYPE
    );
    let n = bytes / T::BYTES;
    let mut out = huge_vec(n, T::get(&vec![0u8; T::BYTES]));
    let part = n.div_ceil(READ_STREAMS).max(1);
    let piece = (READ_PIECE_BYTES / T::BYTES).max(1);
    out.par_chunks_mut(part)
        .enumerate()
        .with_max_len(1)
        .try_for_each(|(p, dst)| -> Result<()> {
            let mut buf = vec![0u8; piece.min(dst.len()) * T::BYTES];
            for (q, d) in dst.chunks_mut(piece).enumerate() {
                let b = &mut buf[..d.len() * T::BYTES];
                let offset = ((p * part + q * piece) * T::BYTES) as u64;
                file.read_exact_at(b, offset)
                    .with_context(|| format!("cannot read {}", path.display()))?;
                for (x, e) in d.iter_mut().zip(b.chunks_exact(T::BYTES)) {
                    *x = T::get(e);
                }
            }
            Ok(())
        })?;
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Persisted u32 arrays
// ---------------------------------------------------------------------------------------------

/// How persisted arrays are brought into memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum LoadMode {
    /// One sequential read into a [`huge_vec`]-style buffer (default).
    Ram,
    /// A read-only memory mapping with sequential advice.
    Mmap,
}

/// A `u32` array that is either owned or memory mapped. It dereferences to `[u32]`.
pub enum U32Array {
    /// Owned storage.
    Owned(Vec<u32>),
    /// Read-only memory mapping (little-endian host only).
    Mapped(webgraph::utils::MmapHelper<u32>),
}

#[cfg(not(target_endian = "little"))]
compile_error!("dag maps little-endian arrays directly and needs a little-endian host");

impl U32Array {
    /// Loads a `.u32le` file. `Ram` reads it sequentially into memory; `Mmap` maps it with
    /// `MmapFlags::SEQUENTIAL` (never random-access advice).
    pub fn load(path: &Path, mode: LoadMode) -> Result<Self> {
        match mode {
            LoadMode::Ram => Ok(Self::Owned(read_array::<u32>(path)?)),
            LoadMode::Mmap => {
                let len = std::fs::metadata(path)
                    .with_context(|| format!("cannot stat {}", path.display()))?
                    .len();
                anyhow::ensure!(
                    len % 4 == 0,
                    "{}: size {len} is not a multiple of 4",
                    path.display()
                );
                if len == 0 {
                    return Ok(Self::Owned(Vec::new()));
                }
                let m =
                    webgraph::utils::MmapHelper::<u32>::mmap(path, mmap_rs::MmapFlags::SEQUENTIAL)
                        .with_context(|| format!("cannot mmap {}", path.display()))?;
                Ok(Self::Mapped(m))
            }
        }
    }

    /// Wraps an owned vector.
    pub fn from_vec(v: Vec<u32>) -> Self {
        Self::Owned(v)
    }

    /// Returns the owned vector, copying a mapped array.
    pub fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Owned(v) => v,
            Self::Mapped(m) => m.as_ref().to_vec(),
        }
    }
}

impl Deref for U32Array {
    type Target = [u32];
    #[inline(always)]
    fn deref(&self) -> &[u32] {
        match self {
            Self::Owned(v) => v,
            Self::Mapped(m) => m.as_ref(),
        }
    }
}

impl std::fmt::Debug for U32Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Self::Owned(_) => "Owned",
            Self::Mapped(_) => "Mapped",
        };
        write!(f, "U32Array::{kind}(len = {})", self.len())
    }
}

// ---------------------------------------------------------------------------------------------
// Large arrays
// ---------------------------------------------------------------------------------------------

/// Whether [`huge_vec`] and [`huge_capacity`] ask for transparent huge pages: yes unless the
/// environment variable `DAG_BIN_NO_THP` is set (to anything).
///
/// With THP in `madvise` mode and `defrag = madvise`, every huge-page fault of an advised range
/// may compact memory synchronously. On a host whose RAM is full of page cache this direct
/// compaction can dominate the system time of large allocations (measured on the PG host:
/// `count-paths` allocating 164 GB spent 6,950 s of system time in 207 s of wall time, with
/// hundreds of thousands of `compact_stall` events); `DAG_BIN_NO_THP` trades the TLB benefit of
/// huge pages for plain 4 KiB faults.
pub fn thp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DAG_BIN_NO_THP").is_none())
}

/// Asks for transparent huge pages on the page-aligned interior of `[ptr, ptr + bytes)`.
fn advise_huge(ptr: *const u8, bytes: usize) {
    if bytes < 2 * HUGE_PAGE || !thp_enabled() {
        return;
    }
    let start = (ptr as usize).next_multiple_of(HUGE_PAGE);
    let end = (ptr as usize + bytes) / HUGE_PAGE * HUGE_PAGE;
    if end > start {
        // SAFETY: madvise(MADV_HUGEPAGE) is advisory: it does not change the contents or the
        // validity of the range, which lies inside an allocation owned by the caller. Errors
        // (e.g. THP disabled) are deliberately ignored.
        unsafe {
            libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
        }
    }
}

/// An empty vector with capacity `n`, advised for transparent huge pages but not touched.
pub fn huge_capacity<T>(n: usize) -> Vec<T> {
    let v = Vec::<T>::with_capacity(n);
    advise_huge(v.as_ptr() as *const u8, n * std::mem::size_of::<T>());
    v
}

/// Large-array allocator: `Vec::with_capacity(n)`, `madvise(MADV_HUGEPAGE)` on the page-aligned
/// interior (best effort, errors ignored), then a parallel first touch with
/// `par_extend(repeatn(fill, n))`.
///
/// ```
/// let v = dag::io::huge_vec(5, 7u32);
/// assert_eq!(v, [7; 5]);
/// ```
pub fn huge_vec<T: Copy + Send + Sync>(n: usize, fill: T) -> Vec<T> {
    let mut v = huge_capacity::<T>(n);
    v.par_extend(rayon::iter::repeatn(fill, n));
    v
}

/// Views a `u32` slice as atomics, for parallel scatter updates.
pub fn as_atomic_u32(s: &mut [u32]) -> &[AtomicU32] {
    // SAFETY: AtomicU32 has the same size, alignment and in-memory representation as u32
    // (documented in std), and the exclusive borrow guarantees no non-atomic access to the
    // memory while the returned shared slice lives.
    unsafe { &*(s as *mut [u32] as *const [AtomicU32]) }
}

/// Views a `u64` slice as atomics, for parallel scatter updates.
pub fn as_atomic_u64(s: &mut [u64]) -> &[AtomicU64] {
    // SAFETY: AtomicU64 has the same size and in-memory representation as u64, and on the
    // supported 64-bit targets the same alignment (checked at compile time below); the
    // exclusive borrow rules out concurrent non-atomic accesses.
    const _: () = assert!(std::mem::align_of::<AtomicU64>() == std::mem::align_of::<u64>());
    unsafe { &*(s as *mut [u64] as *const [AtomicU64]) }
}

// ---------------------------------------------------------------------------------------------
// Memory budget
// ---------------------------------------------------------------------------------------------

/// Bytes per GB as used by `--max-mem` (decimal, 10⁹).
pub const GB: f64 = 1e9;

/// Memory guard: [`need`](MemBudget::need) fails with [`DagError::OverBudget`] when the running
/// estimate would exceed the cap.
///
/// Every reservation is also recorded in a process-wide log ([`budget_log`]): the reservations
/// in order and the high-water mark of the reserved bytes, which [`RunInfo::write`] stores in
/// the JSON sidecar (`mem`), so a run records its *estimated* footprint next to the measured
/// RSS.
#[derive(Clone, Debug)]
pub struct MemBudget {
    /// The cap in bytes.
    pub max_bytes: u64,
    used: u64,
    peak: u64,
}

/// One reservation of a [`MemBudget`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Reservation {
    /// What the bytes are for.
    pub what: String,
    /// Bytes reserved.
    pub bytes: u64,
    /// Bytes reserved by the budget after this reservation.
    pub total: u64,
}

/// The process-wide log of [`MemBudget`] reservations (the `mem` field of [`RunInfo`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct BudgetLog {
    /// The cap of the last budget created (`--max-mem`).
    pub max_bytes: u64,
    /// The largest number of bytes reserved at once by one budget: the estimated peak heap.
    pub estimate_peak_bytes: u64,
    /// The first [`BUDGET_LOG_LEN`] reservations, in order.
    pub reservations: Vec<Reservation>,
    /// Reservations not listed because the list was full.
    pub dropped: u64,
}

/// Maximum number of reservations kept in [`BudgetLog::reservations`].
pub const BUDGET_LOG_LEN: usize = 256;

static BUDGET_LOG: std::sync::Mutex<BudgetLog> = std::sync::Mutex::new(BudgetLog {
    max_bytes: 0,
    estimate_peak_bytes: 0,
    reservations: Vec::new(),
    dropped: 0,
});

/// A snapshot of the process-wide reservation log. Tests that run in parallel in one process
/// share it, so it is meaningful for single-run processes (the binary).
pub fn budget_log() -> BudgetLog {
    BUDGET_LOG.lock().map(|l| l.clone()).unwrap_or_default()
}

fn log_budget(f: impl FnOnce(&mut BudgetLog)) {
    if let Ok(mut l) = BUDGET_LOG.lock() {
        f(&mut l);
    }
}

impl MemBudget {
    /// A budget of `max_gb` GB (10⁹ bytes).
    pub fn new(max_gb: f64) -> Self {
        let max_bytes = (max_gb.max(0.0) * GB) as u64;
        log_budget(|l| l.max_bytes = max_bytes);
        Self {
            max_bytes,
            used: 0,
            peak: 0,
        }
    }

    /// An unlimited budget (tests).
    pub fn unlimited() -> Self {
        Self {
            max_bytes: u64::MAX,
            used: 0,
            peak: 0,
        }
    }

    /// Reserves `bytes` for `what`; fails, without reserving, if the cap would be exceeded.
    pub fn need(&mut self, bytes: u64, what: &str) -> Result<()> {
        let total = self.used.saturating_add(bytes);
        if total > self.max_bytes {
            return Err(DagError::OverBudget {
                what: what.to_owned(),
                need_gb: total as f64 / GB,
                max_gb: self.max_bytes as f64 / GB,
            }
            .into());
        }
        self.used = total;
        self.peak = self.peak.max(total);
        log_budget(|l| {
            l.estimate_peak_bytes = l.estimate_peak_bytes.max(total);
            if l.reservations.len() < BUDGET_LOG_LEN {
                l.reservations.push(Reservation {
                    what: what.to_owned(),
                    bytes,
                    total,
                });
            } else {
                l.dropped += 1;
            }
        });
        log::debug!(
            "memory budget: +{:.2} GB for {what}, total {:.2} GB",
            bytes as f64 / GB,
            total as f64 / GB
        );
        Ok(())
    }

    /// Returns `bytes` to the budget.
    pub fn release(&mut self, bytes: u64) {
        self.used = self.used.saturating_sub(bytes);
    }

    /// Bytes currently reserved.
    pub fn used(&self) -> u64 {
        self.used
    }

    /// The largest number of bytes reserved at once so far.
    pub fn peak(&self) -> u64 {
        self.peak
    }

    /// Bytes still available.
    pub fn available(&self) -> u64 {
        self.max_bytes.saturating_sub(self.used)
    }
}

// ---------------------------------------------------------------------------------------------
// Run metadata
// ---------------------------------------------------------------------------------------------

/// A `kB` field of `/proc/self/status`, in bytes.
fn status_bytes(status: &str, field: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with(field))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Peak resident set size of this process (`VmHWM` of `/proc/self/status`), in bytes. It
/// includes the resident pages of memory-mapped files (the BV graph, mapped arrays), so it is
/// an upper bound on the heap; see [`RssInfo`].
pub fn peak_rss_bytes() -> Option<u64> {
    status_bytes(
        &std::fs::read_to_string("/proc/self/status").ok()?,
        "VmHWM:",
    )
}

/// The memory counters of `/proc/self/status` when the sidecar is written, in bytes.
///
/// `vm_hwm` (the kernel's peak RSS) mixes heap and file-backed pages; `rss_anon` is the
/// anonymous (heap) part and `rss_file` the mapped-file part *at the time of writing*. The
/// kernel keeps no anonymous peak: the peak heap of a run is measured from outside, by sampling
/// `RssAnon` (the `peak_rss_anon_kB` of the measurement wrapper), and estimated from inside by
/// [`BudgetLog::estimate_peak_bytes`].
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct RssInfo {
    /// `VmHWM`: peak resident set size.
    pub vm_hwm: Option<u64>,
    /// `VmRSS`: current resident set size.
    pub vm_rss: Option<u64>,
    /// `RssAnon`: current anonymous (heap) pages.
    pub rss_anon: Option<u64>,
    /// `RssFile`: current file-backed pages.
    pub rss_file: Option<u64>,
}

impl RssInfo {
    /// Reads the counters of this process (all `None` if `/proc` is not available).
    pub fn current() -> Self {
        let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
            return Self::default();
        };
        Self {
            vm_hwm: status_bytes(&s, "VmHWM:"),
            vm_rss: status_bytes(&s, "VmRSS:"),
            rss_anon: status_bytes(&s, "RssAnon:"),
            rss_file: status_bytes(&s, "RssFile:"),
        }
    }
}

/// Wall-clock time of the named phases of a run, in seconds ([`RunInfo::phases_s`]).
///
/// The key names are shared by every subcommand: `open` (graph and index loading), `blocks`
/// (building the BlockIndex), `kernel` (the main computation), `summary` (aggregate measures),
/// `write` (output arrays), `certify`, `compare`, `verify`, `save`.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Phases(std::collections::BTreeMap<String, f64>);

impl Phases {
    /// Runs `f`, adds its wall time to phase `name` and returns its result.
    pub fn time<R>(&mut self, name: &str, f: impl FnOnce() -> R) -> R {
        let t = std::time::Instant::now();
        let r = f();
        self.add(name, t.elapsed().as_secs_f64());
        r
    }

    /// Adds `secs` to phase `name`.
    pub fn add(&mut self, name: &str, secs: f64) {
        *self.0.entry(name.to_owned()).or_insert(0.0) += secs;
    }

    /// The time of phase `name`, if recorded.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.0.get(name).copied()
    }

    /// Sum of all phases.
    pub fn total(&self) -> f64 {
        self.0.values().sum()
    }
}

/// The JSON sidecar `<out>.json` written by every subcommand.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RunInfo {
    /// Always `"dag"`.
    pub tool: &'static str,
    /// Crate version.
    pub version: &'static str,
    /// Command line.
    pub argv: Vec<String>,
    /// Graph basename, if any.
    pub graph: Option<PathBuf>,
    /// Number of nodes of the graph.
    pub n: usize,
    /// Number of arcs of the graph.
    pub m: u64,
    /// The window of the run.
    pub window: Window,
    /// Size of the rayon pool.
    pub threads: usize,
    /// Wall time of the run in seconds.
    pub elapsed_s: f64,
    /// Peak RSS in bytes (`VmHWM`, which includes mapped graph pages); filled by
    /// [`write`](RunInfo::write) when `None`.
    pub peak_rss_bytes: Option<u64>,
    /// The memory counters at the end of the run; filled by [`write`](RunInfo::write).
    pub rss: Option<RssInfo>,
    /// The [`MemBudget`] reservations of the run (estimated footprint); filled by
    /// [`write`](RunInfo::write).
    pub mem: Option<BudgetLog>,
    /// Wall time per phase (see [`Phases`]); `elapsed_s` minus their sum is the untimed rest
    /// (argument parsing, small allocations, the sidecar itself).
    pub phases_s: Phases,
    /// Output files.
    pub files: Vec<FileInfo>,
    /// Parameters of the run.
    pub params: serde_json::Value,
    /// Results (scalars, reports).
    pub results: serde_json::Value,
}

/// Description of one output array in [`RunInfo::files`].
#[derive(Clone, Debug, serde::Serialize)]
pub struct FileInfo {
    /// Path of the file.
    pub path: PathBuf,
    /// Element type ([`Dump::DTYPE`]).
    pub dtype: &'static str,
    /// Number of elements.
    pub len: usize,
    /// Sentinel value, if any (e.g. `"u32::MAX = NONE"`).
    pub sentinel: Option<String>,
    /// What the values mean.
    pub meaning: String,
}

impl FileInfo {
    /// Describes an array of `T` with `len` elements.
    pub fn new<T: Dump>(
        path: impl Into<PathBuf>,
        len: usize,
        sentinel: Option<&str>,
        meaning: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            dtype: T::DTYPE,
            len,
            sentinel: sentinel.map(str::to_owned),
            meaning: meaning.into(),
        }
    }
}

impl RunInfo {
    /// A record with the command line, version and thread count filled in.
    pub fn new(graph: Option<PathBuf>, n: usize, m: u64, window: Window) -> Self {
        Self {
            tool: "dag",
            version: env!("CARGO_PKG_VERSION"),
            argv: std::env::args().collect(),
            graph,
            n,
            m,
            window,
            threads: rayon::current_num_threads(),
            elapsed_s: 0.0,
            peak_rss_bytes: None,
            rss: None,
            mem: None,
            phases_s: Phases::default(),
            files: Vec::new(),
            params: serde_json::Value::Null,
            results: serde_json::Value::Null,
        }
    }

    /// Writes `<out_prefix>.json` (pretty printed). The peak RSS, the [`RssInfo`] and the
    /// [`budget_log`] are filled in if not set.
    pub fn write(&self, out_prefix: &Path) -> Result<()> {
        let path = suffixed(out_prefix, "json");
        let mut me = self.clone();
        if me.peak_rss_bytes.is_none() {
            me.peak_rss_bytes = peak_rss_bytes();
        }
        if me.rss.is_none() {
            me.rss = Some(RssInfo::current());
        }
        if me.mem.is_none() {
            me.mem = Some(budget_log());
        }
        let file =
            File::create(&path).with_context(|| format!("cannot create {}", path.display()))?;
        let mut w = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut w, &me)?;
        w.write_all(b"\n")?;
        w.flush()?;
        Ok(())
    }
}
