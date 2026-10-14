//! The ordered parallel-decode engine: every streaming pass over a BV graph.
//!
//! On the Payment Graph a sequential BV decode costs ~110 s, a parallel one 2.6 s, while the
//! DP consumers are inherently sequential (depth 3.56·10⁶, median layer 224 nodes). The engine
//! therefore **decodes in parallel and consumes in order**:
//!
//! ```text
//! chunks of w (cut points of the ChunkedGraph) grouped in batches of `decoders` chunks
//! decode(batch_0)
//! for i in 0..: rayon::join(|| consume(batch_i), || decode(batch_{i+1}))
//!               pl.update_with_count(nodes of batch_i)
//! ```
//!
//! - `decode` is a `par_iter` over the batch; each chunk does `graph.iter_from(start).take(len)`
//!   into a [`Chunk`] taken from a pool of `2 × decoders` recycled buffers.
//! - Successors `≥ w.hi` are dropped; consumers filter `v ≥ lo` where relevant.
//! - While decoding, the engine checks the topological order: an arc `(u, v)` with `v ≤ u`
//!   yields [`DagError::NotTopological`] for the smallest such `u` of the batch (for
//!   [`sweep_forward`], the smallest of the whole window), so every sweep doubles as an
//!   identity-order check.
//! - Peak memory: `2 × decoders × (4·chunk_arcs + 4·chunk_nodes)` bytes, about 5 GB at the
//!   defaults.
//!
//! **Progress.** The engine owns progress updates: after each batch it calls
//! `pl.update_with_count`. Consumers must never capture `pl` (a `ProgressLog` is not `Send`);
//! callers call `pl.start`/`pl.done` around a sweep. [`par_chunks`] gives every task a clone of a
//! `ConcurrentProgressLog` and updates it once per chunk.
//!
//! Complexity: O(n_w + m_w) decode work, spread over `decoders` threads, plus the consumer.

use anyhow::Result;
use dsi_progress_logger::prelude::*;
use lender::prelude::*;
use rayon::prelude::*;
use webgraph::prelude::*;

use crate::error::DagError;
use crate::graph::ChunkedGraph;
use crate::window::Window;

/// Configuration of the sweep engine.
#[derive(Clone, Debug)]
pub struct SweepConf {
    /// Number of chunks decoded in parallel per batch (default `min(threads, 32)`).
    pub decoders: usize,
}

impl Default for SweepConf {
    fn default() -> Self {
        Self {
            decoders: rayon::current_num_threads().clamp(1, 32),
        }
    }
}

/// A decoded chunk: nodes `[start, end)`, successors as `u32`, CSR offsets relative to the chunk.
#[derive(Clone, Debug, Default)]
pub struct Chunk {
    /// First node of the chunk.
    pub start: usize,
    /// One past the last node of the chunk.
    pub end: usize,
    offsets: Vec<u32>,
    succ: Vec<u32>,
}

impl Chunk {
    /// Successors of `u ∈ [start, end)`, ascending (BV lists are sorted), all `< w.hi`.
    #[inline]
    pub fn succ(&self, u: usize) -> &[u32] {
        let i = u - self.start;
        &self.succ[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    /// `(u, succ(u))` for ascending `u`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[u32])> {
        (self.start..self.end).map(move |u| (u, self.succ(u)))
    }

    /// `(u, succ(u))`, double ended; `.rev()` gives descending `u`.
    pub fn iter_rev(&self) -> impl DoubleEndedIterator<Item = (usize, &[u32])> {
        (self.start..self.end).map(move |u| (u, self.succ(u))).rev()
    }

    /// Number of (kept) arcs of the chunk.
    #[inline]
    pub fn num_arcs(&self) -> usize {
        self.succ.len()
    }

    /// Number of nodes of the chunk.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.end - self.start
    }

    /// Decodes `[start, end)` of `g` into this (recycled) buffer, keeping the neighbours that
    /// `keep` accepts and checking the orientation of every arc.
    fn decode<G: RandomAccessGraph>(
        &mut self,
        g: &G,
        start: usize,
        end: usize,
        keep: Keep,
    ) -> Result<()> {
        self.start = start;
        self.end = end;
        self.offsets.clear();
        self.succ.clear();
        self.offsets.push(0);
        if start == end {
            return Ok(());
        }
        let mut it = g.iter_from(start).take(end - start);
        while let Some((x, s)) = it.next() {
            for y in s {
                match keep {
                    Keep::SuccBelow(hi) => {
                        if y <= x {
                            return Err(DagError::NotTopological { u: x, v: y }.into());
                        }
                        if y < hi {
                            self.succ.push(y as u32);
                        }
                    }
                    Keep::PredFrom(lo) => {
                        if y >= x {
                            return Err(DagError::NotTopological { u: y, v: x }.into());
                        }
                        if y >= lo {
                            self.succ.push(y as u32);
                        }
                    }
                }
            }
            anyhow::ensure!(
                self.succ.len() <= u32::MAX as usize,
                "chunk [{start}, {end}) has more than 2^32 arcs: lower --chunk-arcs"
            );
            self.offsets.push(self.succ.len() as u32);
        }
        anyhow::ensure!(
            self.offsets.len() == end - start + 1,
            "graph ended before node {end}"
        );
        Ok(())
    }
}

/// Which neighbours a decoded chunk keeps, and which orientation every arc must have.
#[derive(Clone, Copy, Debug)]
enum Keep {
    /// Forward lists: every successor must be larger than its node; keep those `< hi`.
    SuccBelow(usize),
    /// Transpose lists: every predecessor must be smaller than its node; keep those `≥ lo`.
    PredFrom(usize),
}

/// Decodes `ranges` into `bufs[..ranges.len()]` in parallel; returns the error of the chunk with
/// the smallest start.
fn decode_batch<G: RandomAccessGraph + Sync>(
    g: &G,
    bufs: &mut [Chunk],
    ranges: &[(usize, usize)],
    keep: Keep,
) -> Result<()> {
    let errs: Vec<(usize, anyhow::Error)> = bufs[..ranges.len()]
        .par_iter_mut()
        .zip(ranges.par_iter())
        .with_max_len(1)
        .filter_map(|(b, &(s, e))| b.decode(g, s, e, keep).err().map(|err| (s, err)))
        .collect();
    match errs.into_iter().min_by_key(|(s, _)| *s) {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// The shared engine: `ranges` are the chunks in consumption order.
fn sweep<G, F>(
    g: &ChunkedGraph<G>,
    keep: Keep,
    ranges: Vec<(usize, usize)>,
    conf: &SweepConf,
    pl: &mut impl ProgressLog,
    mut f: F,
) -> Result<()>
where
    G: RandomAccessGraph + Sync,
    F: FnMut(&Chunk) -> Result<()> + Send,
{
    if ranges.is_empty() {
        return Ok(());
    }
    let d = conf.decoders.max(1);
    let batches: Vec<&[(usize, usize)]> = ranges.chunks(d).collect();
    let graph = &g.graph;
    let mut cur: Vec<Chunk> = vec![Chunk::default(); d];
    let mut next: Vec<Chunk> = vec![Chunk::default(); d];
    decode_batch(graph, &mut cur, batches[0], keep)?;
    for i in 0..batches.len() {
        let k = batches[i].len();
        let nodes: usize = batches[i].iter().map(|(s, e)| e - s).sum();
        let following = batches.get(i + 1).copied();
        let (rc, rd) = rayon::join(
            || -> Result<()> {
                for c in &cur[..k] {
                    f(c)?;
                }
                Ok(())
            },
            || match following {
                Some(r) => decode_batch(graph, &mut next, r, keep),
                None => Ok(()),
            },
        );
        rc?;
        pl.update_with_count(nodes);
        rd?;
        std::mem::swap(&mut cur, &mut next);
    }
    Ok(())
}

/// Calls `f` on every chunk of `w`, in **ascending** order (nodes ascending inside a chunk).
///
/// Fails with [`DagError::NotTopological`] at the smallest offending node if the identity is not
/// a topological order of the window's nodes. Consumers must not capture `pl`.
pub fn sweep_forward<G, F>(
    g: &ChunkedGraph<G>,
    w: Window,
    conf: &SweepConf,
    pl: &mut impl ProgressLog,
    f: F,
) -> Result<()>
where
    G: RandomAccessGraph + Sync,
    F: FnMut(&Chunk) -> Result<()> + Send,
{
    let ranges = g.chunks_in(w);
    sweep(g, Keep::SuccBelow(w.hi), ranges, conf, pl, f)
}

/// Calls `f` on every chunk of `w`, in **descending** order; use [`Chunk::iter_rev`] inside a
/// chunk. Decoding is still forward inside each chunk (block-buffered reverse).
pub fn sweep_backward<G, F>(
    g: &ChunkedGraph<G>,
    w: Window,
    conf: &SweepConf,
    pl: &mut impl ProgressLog,
    f: F,
) -> Result<()>
where
    G: RandomAccessGraph + Sync,
    F: FnMut(&Chunk) -> Result<()> + Send,
{
    let mut ranges = g.chunks_in(w);
    ranges.reverse();
    sweep(g, Keep::SuccBelow(w.hi), ranges, conf, pl, f)
}

/// Calls `f` on every chunk of the **transpose** `gt` restricted to `w`, in ascending order;
/// inside a chunk, [`Chunk::succ`] (and [`Chunk::iter`]) then yield the **predecessors** of each
/// node that lie in `[w.lo, node)`, ascending.
///
/// This is the decode of the forward pull kernels (F1 §6.2): the same batching and double
/// buffering as [`sweep_forward`], but the orientation check is mirrored, because transpose
/// arcs go from larger to smaller ids. Fails with [`DagError::NotTopological`] (reported as
/// the arc `(u, v)` of the forward graph, `u ≥ v`) at the smallest offending chunk.
pub fn sweep_forward_preds<G, F>(
    gt: &ChunkedGraph<G>,
    w: Window,
    conf: &SweepConf,
    pl: &mut impl ProgressLog,
    f: F,
) -> Result<()>
where
    G: RandomAccessGraph + Sync,
    F: FnMut(&Chunk) -> Result<()> + Send,
{
    let ranges = gt.chunks_in(w);
    sweep(gt, Keep::PredFrom(w.lo), ranges, conf, pl, f)
}

/// An object-safe iterator over `(node, successors)` with a reused buffer, handed to the
/// `map` closure of [`par_chunks`]. Successors are ascending and `< w.hi`.
pub trait NodeSuccIter {
    /// The next node of the chunk and its successors, or `None` at the end of the chunk.
    fn next_node(&mut self) -> Option<(usize, &[u32])>;
}

struct LenderSucc<L> {
    lender: L,
    left: usize,
    hi: usize,
    buf: Vec<u32>,
}

impl<L> NodeSuccIter for LenderSucc<L>
where
    L: for<'next> NodeLabelsLender<'next, Label = usize>,
{
    fn next_node(&mut self) -> Option<(usize, &[u32])> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        let hi = self.hi;
        let buf = &mut self.buf;
        buf.clear();
        let u = {
            let (u, s) = self.lender.next()?;
            buf.extend(s.into_iter().filter(|&v| v < hi).map(|v| v as u32));
            u
        };
        Some((u, &self.buf))
    }
}

/// Runs `map` on every chunk of `w` in parallel (rayon) and folds the results with `reduce`.
///
/// Each task decodes its chunk with `iter_from(start).take(len)`, works on a clone of `pl` and
/// calls `update_with_count(chunk nodes)` once. Errors are deterministic: the error of the
/// **lowest-index failing chunk** is returned. Unlike the sweeps, no topology check is done.
/// Used for checks, degree counts, `BlockIndex` pass 1, certification and verifiers.
pub fn par_chunks<G, A, M, R>(
    g: &ChunkedGraph<G>,
    w: Window,
    pl: &mut impl ConcurrentProgressLog,
    map: M,
    reduce: R,
) -> Result<A>
where
    G: RandomAccessGraph + Sync,
    M: Fn(&mut dyn NodeSuccIter) -> Result<A> + Sync,
    A: Send + Default,
    R: Fn(A, A) -> A + Sync,
{
    let ranges = g.chunks_in(w);
    let graph = &g.graph;
    let hi = w.hi;
    let res: std::result::Result<A, (usize, anyhow::Error)> = ranges
        .into_par_iter()
        .enumerate()
        .with_max_len(1)
        .map_with((pl.clone(), Vec::<u32>::new()), |(pl, buf), (i, (s, e))| {
            let mut it = LenderSucc {
                lender: graph.iter_from(s),
                left: e - s,
                hi,
                buf: std::mem::take(buf),
            };
            let r = map(&mut it);
            *buf = it.buf;
            pl.update_with_count(e - s);
            r.map_err(|err| (i, err))
        })
        .reduce(
            || Ok(A::default()),
            |a, b| match (a, b) {
                (Ok(a), Ok(b)) => Ok(reduce(a, b)),
                (Err(x), Err(y)) => Err(if x.0 <= y.0 { x } else { y }),
                (Err(x), _) | (_, Err(x)) => Err(x),
            },
        );
    res.map_err(|(_, e)| e)
}

/// An object-safe subset of [`ProgressLog`] (which is not dyn compatible), for APIs that take
/// trait objects such as `reach::dagview::SuccDag`. Every `ProgressLog` implements it; wrap a
/// `&mut dyn DynProgressLog` in [`DynLog`] to pass it where `impl ProgressLog` is expected.
pub trait DynProgressLog {
    /// See [`ProgressLog::start`].
    fn dyn_start(&mut self, msg: &str);
    /// See [`ProgressLog::update_with_count`].
    fn dyn_update_with_count(&mut self, count: usize);
    /// See [`ProgressLog::light_update`].
    fn dyn_light_update(&mut self);
    /// See [`ProgressLog::done`].
    fn dyn_done(&mut self);
    /// See [`ProgressLog::item_name`].
    fn dyn_item_name(&mut self, name: &str);
    /// See [`ProgressLog::expected_updates`].
    fn dyn_expected_updates(&mut self, n: Option<usize>);
    /// See [`ProgressLog::info`].
    fn dyn_info(&self, args: std::fmt::Arguments<'_>);
}

impl<P: ProgressLog> DynProgressLog for P {
    fn dyn_start(&mut self, msg: &str) {
        self.start(msg);
    }
    fn dyn_update_with_count(&mut self, count: usize) {
        self.update_with_count(count);
    }
    fn dyn_light_update(&mut self) {
        self.light_update();
    }
    fn dyn_done(&mut self) {
        self.done();
    }
    fn dyn_item_name(&mut self, name: &str) {
        self.item_name(name);
    }
    fn dyn_expected_updates(&mut self, n: Option<usize>) {
        self.expected_updates(n);
    }
    fn dyn_info(&self, args: std::fmt::Arguments<'_>) {
        self.info(args);
    }
}

/// Adapter that turns a `&mut dyn DynProgressLog` back into a [`ProgressLog`].
///
/// Counting, start/done and messages are forwarded; display settings other than the item name
/// and the expected updates are ignored, and [`ProgressLog::concurrent`] returns a disabled
/// logger.
pub struct DynLog<'a>(pub &'a mut dyn DynProgressLog);

impl ProgressLog for DynLog<'_> {
    type Concurrent = Option<ConcurrentWrapper<ProgressLogger>>;
    fn log(&mut self, _now: std::time::Instant) {}
    fn log_if(&mut self, _now: std::time::Instant) {}
    fn display_memory(&mut self, _display_memory: bool) -> &mut Self {
        self
    }
    fn item_name(&mut self, item_name: impl AsRef<str>) -> &mut Self {
        self.0.dyn_item_name(item_name.as_ref());
        self
    }
    fn log_interval(&mut self, _log_interval: std::time::Duration) -> &mut Self {
        self
    }
    fn expected_updates(&mut self, expected_updates: impl Into<Option<usize>>) -> &mut Self {
        self.0.dyn_expected_updates(expected_updates.into());
        self
    }
    fn time_unit(&mut self, _time_unit: impl Into<Option<TimeUnit>>) -> &mut Self {
        self
    }
    fn local_speed(&mut self, _local_speed: bool) -> &mut Self {
        self
    }
    fn log_target(&mut self, _target: impl AsRef<str>) -> &mut Self {
        self
    }
    fn push_log_target(&mut self, _suffix: impl AsRef<str>) -> &mut Self {
        self
    }
    fn pop_log_target(&mut self) -> &mut Self {
        self
    }
    fn log_level(&mut self, _log_level: Level) -> &mut Self {
        self
    }
    fn add_to_count(&mut self, count: usize) {
        self.0.dyn_update_with_count(count);
    }
    fn start(&mut self, msg: impl AsRef<str>) {
        self.0.dyn_start(msg.as_ref());
    }
    fn update(&mut self) {
        self.0.dyn_update_with_count(1);
    }
    fn update_with_count_and_time(&mut self, count: usize, _now: std::time::Instant) {
        self.0.dyn_update_with_count(count);
    }
    fn light_update(&mut self) {
        self.0.dyn_light_update();
    }
    fn update_and_display(&mut self) {
        self.0.dyn_update_with_count(1);
    }
    fn stop(&mut self) {}
    fn done(&mut self) {
        self.0.dyn_done();
    }
    fn done_with_count(&mut self, count: usize) {
        self.0.dyn_update_with_count(count);
        self.0.dyn_done();
    }
    fn elapsed(&self) -> Option<std::time::Duration> {
        None
    }
    fn count(&self) -> usize {
        0
    }
    fn refresh(&mut self) {}
    fn trace(&self, args: std::fmt::Arguments<'_>) {
        log::trace!("{args}");
    }
    fn debug(&self, args: std::fmt::Arguments<'_>) {
        log::debug!("{args}");
    }
    fn info(&self, args: std::fmt::Arguments<'_>) {
        self.0.dyn_info(args);
    }
    fn warn(&self, args: std::fmt::Arguments<'_>) {
        log::warn!("{args}");
    }
    fn error(&self, args: std::fmt::Arguments<'_>) {
        log::error!("{args}");
    }
    fn concurrent(&self) -> Self::Concurrent {
        None
    }
}
