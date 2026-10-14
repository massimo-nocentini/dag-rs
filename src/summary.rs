//! Distributions of per-node (or per-block) values, overall and per **time bucket**.
//!
//! Every subcommand that produces a per-node array also reports its distribution in the JSON
//! sidecar through [`distribution`]: count, sum, mean, min/argmin, max/argmax, a log2
//! histogram ([`Log2Hist`]) and the top-k entries; with `--buckets FILE` ([`Buckets`]) the same
//! statistics are repeated for every node range of the file (on the Payment Graph, the 28
//! half-year chunks of `chunk_boundaries.tsv`, so the report can show how a measure evolves over
//! time without re-reading multi-GB arrays). `dag aggregate` computes the same report from a
//! stored array.
//!
//! # Conventions
//! - **Ids.** Entry `i` of an array has an absolute *id* (`lo + i` for per-node arrays of the
//!   window `[lo, hi)`, the absolute block id for per-block arrays) and an absolute *node* used
//!   for bucketing (the id itself, or the first output of the block). `argmin`, `argmax` and the
//!   top-k list report ids, plus the node when it differs (`argmax_node`, `top[].node`).
//! - **Absent values.** The `key` closure returns `None` for entries without a value (0̄ of a
//!   semiring, a sentinel, a node outside the selection); NaN keys count as absent too. Absent
//!   entries contribute only to `absent`.
//! - **Histogram.** Bin 0 counts the value 0, bin `k ≥ 1` the values in `[2^(k−1), 2^k)` (for
//!   integers: bin `k` holds the values with `k` significant bits), the last bin
//!   ([`LOG2_BINS`]` − 1`) everything `≥ 2^(LOG2_BINS − 2)`; values in `(0, 1)` and negative
//!   values are counted in `below_one` and `negative`. This is the convention of
//!   [`DomStats`](crate::dom::DomStats) and of the dominator-tree histograms. Trailing empty
//!   bins are dropped in the JSON.
//! - **Ties.** `argmin`/`argmax` are the smallest id attaining the extreme; the top-k list is
//!   ordered by value descending, then id ascending.
//!
//! # Determinism and complexity
//! The array is cut into fixed chunks of [`CHUNK`] entries processed in parallel; each chunk
//! yields one accumulator per bucket it intersects, and the accumulators are merged
//! sequentially in chunk order. Counts, extremes, histograms and top-k lists are exact; the
//! floating-point sums are added in a fixed order (element order within a chunk, chunk order
//! across chunks, and the overall sum never depends on the buckets), so the report is
//! bit-reproducible across runs and thread counts. Time `O(len · log k)` in parallel; memory
//! `O((len / CHUNK + #buckets) · (LOG2_BINS + k))`. Top-k selection keeps a buffer of at most
//! `2k` candidates and compacts it by sorting, i.e. the bounded-buffer variant of selection
//! [\[Knu98\]](crate::references#knu98) (Sec. 5.3.3).
//!
//! # Time buckets of the Payment Graph
//! PG node ids are chronological: block `b` of the blockchain owns the ids
//! `[first_node[b], first_node[b + 1])`, and the half-year chunks used to build the graph are
//! contiguous id ranges. [`Buckets::load`] reads such ranges from a TSV/CSV file with a header,
//! recognising the columns of `chunk_boundaries.tsv` (`chunk_id`, `start_date`, `first_node`,
//! `last_node_exclusive`) or the generic `label`, `lo`, `hi`
//! ([\[TMAP\]](crate::references#tmap)).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

/// Entries per parallel chunk of [`distribution`] (2²⁰). Fixed, so that the floating-point
/// sums do not depend on the thread count.
pub const CHUNK: usize = 1 << 20;

/// Number of bins of [`Log2Hist::bins`]: value 0, the 64 binary orders of magnitude
/// `[2^(k−1), 2^k)` for `k = 1..=64`, and one overflow bin `≥ 2^64`.
pub const LOG2_BINS: usize = 66;

// ---------------------------------------------------------------------------------------------
// Buckets
// ---------------------------------------------------------------------------------------------

/// A named range `[lo, hi)` of absolute node ids.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Bucket {
    /// Label (for `chunk_boundaries.tsv`: `2009H1`, `2009H2`, …).
    pub label: String,
    /// First node.
    pub lo: usize,
    /// One past the last node.
    pub hi: usize,
}

/// Sorted, pairwise disjoint node ranges (see the module documentation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buckets {
    /// The file the buckets were read from, if any.
    pub source: Option<PathBuf>,
    buckets: Vec<Bucket>,
}

/// The label of a half-year chunk from its start date `YYYY-MM-DD…`: `YYYYH1` for January 1,
/// `YYYYH2` for July 1, else the date itself.
fn date_label(start: &str) -> String {
    let date = start.split_whitespace().next().unwrap_or(start);
    match (date.get(..4), date.get(4..)) {
        (Some(y), Some("-01-01")) => format!("{y}H1"),
        (Some(y), Some("-07-01")) => format!("{y}H2"),
        _ => date.to_owned(),
    }
}

impl Buckets {
    /// Buckets from explicit ranges; fails unless every range has `lo ≤ hi` and the ranges are
    /// sorted and pairwise disjoint.
    ///
    /// ```
    /// use dag::summary::{Bucket, Buckets};
    /// let b = Buckets::new(vec![
    ///     Bucket { label: "a".into(), lo: 0, hi: 10 },
    ///     Bucket { label: "b".into(), lo: 10, hi: 25 },
    /// ]).unwrap();
    /// assert_eq!(b.locate(9), Some(0));
    /// assert_eq!(b.locate(10), Some(1));
    /// assert_eq!(b.locate(25), None);
    /// ```
    pub fn new(buckets: Vec<Bucket>) -> Result<Self> {
        for (i, b) in buckets.iter().enumerate() {
            ensure!(
                b.lo <= b.hi,
                "bucket {:?}: lo {} > hi {}",
                b.label,
                b.lo,
                b.hi
            );
            if i > 0 {
                let p = &buckets[i - 1];
                ensure!(
                    p.hi <= b.lo,
                    "buckets {:?} [{}, {}) and {:?} [{}, {}) overlap or are not sorted",
                    p.label,
                    p.lo,
                    p.hi,
                    b.label,
                    b.lo,
                    b.hi
                );
            }
        }
        Ok(Self {
            source: None,
            buckets,
        })
    }

    /// Parses a header plus one bucket per line, tab- or comma-separated (lines starting with
    /// `#` and empty lines are skipped). Range columns: `first_node` and `last_node_exclusive`,
    /// or `lo` and `hi`. Label: the `label` column, else derived from `start_date`
    /// (`2009-07-01 …` gives `2009H2`), else `chunk_id`, else the line number.
    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty() && !l.starts_with('#'));
        let (_, header) = lines.next().context("empty bucket file")?;
        let sep = if header.contains('\t') { '\t' } else { ',' };
        let cols: Vec<&str> = header.split(sep).map(str::trim).collect();
        let col = |names: &[&str]| names.iter().find_map(|n| cols.iter().position(|c| c == n));
        let lo_c = col(&["first_node", "lo"]).context("no first_node or lo column")?;
        let hi_c =
            col(&["last_node_exclusive", "hi"]).context("no last_node_exclusive or hi column")?;
        let label_c = col(&["label"]);
        let date_c = col(&["start_date"]);
        let id_c = col(&["chunk_id"]);
        let mut out = Vec::new();
        for (ln, line) in lines {
            let f: Vec<&str> = line.split(sep).map(str::trim).collect();
            let field = |c: usize| {
                f.get(c)
                    .copied()
                    .with_context(|| format!("line {}: missing column {}", ln + 1, cols[c]))
            };
            let num = |c: usize| -> Result<usize> {
                let s = field(c)?;
                s.replace('_', "").parse().with_context(|| {
                    format!("line {}: {} = {s:?} is not a node id", ln + 1, cols[c])
                })
            };
            let label = if let Some(c) = label_c {
                field(c)?.to_owned()
            } else if let Some(c) = date_c {
                date_label(field(c)?)
            } else if let Some(c) = id_c {
                field(c)?.to_owned()
            } else {
                format!("{}", ln + 1)
            };
            out.push(Bucket {
                label,
                lo: num(lo_c)?,
                hi: num(hi_c)?,
            });
        }
        ensure!(!out.is_empty(), "the bucket file lists no bucket");
        Self::new(out)
    }

    /// Reads and parses a bucket file ([`parse`](Self::parse)).
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let mut b = Self::parse(&text).with_context(|| format!("in {}", path.display()))?;
        b.source = Some(path.to_owned());
        Ok(b)
    }

    /// The buckets, sorted.
    pub fn as_slice(&self) -> &[Bucket] {
        &self.buckets
    }

    /// Number of buckets.
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// Whether there is no bucket.
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// The index of the bucket containing node `v` (binary search).
    pub fn locate(&self, v: usize) -> Option<usize> {
        let i = self.buckets.partition_point(|b| b.hi <= v);
        (i < self.buckets.len() && self.buckets[i].lo <= v).then_some(i)
    }

    /// Logs a warning when the buckets do not cover `[0, n)` exactly.
    pub fn check_cover(&self, n: usize) {
        let gaps = self
            .buckets
            .windows(2)
            .filter(|w| w[0].hi != w[1].lo)
            .count();
        let (lo, hi) = (
            self.buckets.first().map_or(0, |b| b.lo),
            self.buckets.last().map_or(0, |b| b.hi),
        );
        if lo != 0 || hi != n || gaps > 0 {
            log::warn!(
                "buckets cover [{lo}, {hi}) with {gaps} gaps, the graph has {n} nodes: \
                 nodes outside every bucket are counted in `unbucketed`"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Histogram and top-k
// ---------------------------------------------------------------------------------------------

/// The log2 histogram of a distribution (see the module documentation for the bins).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Log2Hist {
    /// Negative values.
    pub negative: u64,
    /// Values in `(0, 1)`.
    pub below_one: u64,
    /// Bin 0: value 0; bin `k ≥ 1`: `[2^(k−1), 2^k)`; the last of the [`LOG2_BINS`] bins also
    /// holds every larger value. Trailing zero bins are omitted.
    pub bins: Vec<u64>,
}

/// The [`Log2Hist`] bin of a finite value `x ≥ 0` that is 0 or at least 1.
///
/// ```
/// use dag::summary::log2_bin;
/// assert_eq!(log2_bin(0.0), 0);
/// assert_eq!(log2_bin(1.0), 1);
/// assert_eq!(log2_bin(3.0), 2);
/// assert_eq!(log2_bin(4.0), 3);
/// assert_eq!(log2_bin(1e300), 65);
/// ```
#[inline]
pub fn log2_bin(x: f64) -> usize {
    if x == 0.0 {
        return 0;
    }
    // x ≥ 1 is normal: floor(log2 x) is the unbiased exponent
    let e = ((x.to_bits() >> 52) & 0x7ff) as usize - 1023;
    (e + 1).min(LOG2_BINS - 1)
}

/// `a` ranks before `b` in a top-k list: larger value, then smaller index.
#[inline]
fn ranks_before(a: (f64, usize), b: (f64, usize)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// A bounded buffer of the `k` best `(value, index)` pairs.
#[derive(Clone, Debug)]
struct TopK {
    k: usize,
    buf: Vec<(f64, usize)>,
    /// The k-th best pair after the last compaction: anything ranking after it is dropped.
    floor: Option<(f64, usize)>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            buf: Vec::new(),
            floor: None,
        }
    }

    #[inline]
    fn offer(&mut self, x: f64, i: usize) {
        if self.k == 0 || self.floor.is_some_and(|f| !ranks_before((x, i), f)) {
            return;
        }
        self.buf.push((x, i));
        if self.buf.len() >= 2 * self.k.max(8) {
            self.compact();
        }
    }

    fn compact(&mut self) {
        self.buf
            .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        if self.buf.len() >= self.k {
            self.buf.truncate(self.k);
            self.floor = self.buf.last().copied();
        }
    }

    fn merge(&mut self, other: &Self) {
        for &(x, i) in &other.buf {
            self.offer(x, i);
        }
    }

    fn finish(mut self) -> Vec<(f64, usize)> {
        self.compact();
        self.buf
    }
}

// ---------------------------------------------------------------------------------------------
// Accumulator
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Acc {
    count: u64,
    absent: u64,
    sum: f64,
    min: (f64, usize),
    max: (f64, usize),
    negative: u64,
    below_one: u64,
    bins: [u64; LOG2_BINS],
    top: TopK,
}

impl Acc {
    fn new(k: usize) -> Self {
        Self {
            count: 0,
            absent: 0,
            sum: 0.0,
            min: (f64::INFINITY, usize::MAX),
            max: (f64::NEG_INFINITY, usize::MAX),
            negative: 0,
            below_one: 0,
            bins: [0; LOG2_BINS],
            top: TopK::new(k),
        }
    }

    #[inline]
    fn add(&mut self, i: usize, key: Option<f64>) {
        let Some(x) = key.filter(|x| !x.is_nan()) else {
            self.absent += 1;
            return;
        };
        self.count += 1;
        self.sum += x;
        if x > self.max.0 || (x == self.max.0 && i < self.max.1) {
            self.max = (x, i);
        }
        if x < self.min.0 || (x == self.min.0 && i < self.min.1) {
            self.min = (x, i);
        }
        if x < 0.0 {
            self.negative += 1;
        } else if x > 0.0 && x < 1.0 {
            self.below_one += 1;
        } else {
            self.bins[log2_bin(x)] += 1;
        }
        self.top.offer(x, i);
    }

    /// Merges `o` into `self`; the sum is added after `self`'s (callers merge in index order).
    fn merge(&mut self, o: &Self) {
        self.count += o.count;
        self.absent += o.absent;
        self.sum += o.sum;
        if ranks_before(o.max, self.max) {
            self.max = o.max;
        }
        if o.min.0 < self.min.0 || (o.min.0 == self.min.0 && o.min.1 < self.min.1) {
            self.min = o.min;
        }
        self.negative += o.negative;
        self.below_one += o.below_one;
        for (a, b) in self.bins.iter_mut().zip(o.bins) {
            *a += b;
        }
        self.top.merge(&o.top);
    }

    fn finish(self, id: &(dyn Fn(usize) -> usize + Sync), node: Option<&NodeOf<'_>>) -> Dist {
        let some = self.count > 0;
        let last = self.bins.iter().rposition(|&c| c > 0).map_or(0, |p| p + 1);
        let top = self
            .top
            .finish()
            .into_iter()
            .map(|(value, i)| Top {
                id: id(i),
                value,
                node: node.map(|f| f(i)),
            })
            .collect();
        Dist {
            count: self.count,
            absent: self.absent,
            sum: self.sum,
            mean: some.then(|| self.sum / self.count as f64),
            min: some.then_some(self.min.0),
            argmin: some.then(|| id(self.min.1)),
            max: some.then_some(self.max.0),
            argmax: some.then(|| id(self.max.1)),
            argmax_node: node.filter(|_| some).map(|f| f(self.max.1)),
            log2_hist: Log2Hist {
                negative: self.negative,
                below_one: self.below_one,
                bins: self.bins[..last].to_vec(),
            },
            top,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------------------------

/// One entry of a top-k list.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Top {
    /// Absolute id (node or block).
    pub id: usize,
    /// The value.
    pub value: f64,
    /// The node of a block id (its first output), for per-block arrays.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<usize>,
}

/// The distribution of the present values of an array (or of one bucket of it).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Dist {
    /// Entries with a value.
    pub count: u64,
    /// Entries without a value (see the module documentation).
    pub absent: u64,
    /// Sum of the values.
    pub sum: f64,
    /// Mean of the values (`None` if there is none).
    pub mean: Option<f64>,
    /// Smallest value.
    pub min: Option<f64>,
    /// Smallest id attaining `min`.
    pub argmin: Option<usize>,
    /// Largest value.
    pub max: Option<f64>,
    /// Smallest id attaining `max`.
    pub argmax: Option<usize>,
    /// The node of `argmax`, for per-block arrays.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argmax_node: Option<usize>,
    /// Log2 histogram of the values.
    pub log2_hist: Log2Hist,
    /// The largest values (value descending, id ascending).
    pub top: Vec<Top>,
}

/// The [`Dist`] of one bucket.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BucketDist {
    /// The bucket's label.
    pub label: String,
    /// Its first node.
    pub lo: usize,
    /// One past its last node.
    pub hi: usize,
    /// The distribution of the entries whose node lies in `[lo, hi)`.
    #[serde(flatten)]
    pub dist: Dist,
}

/// The output of [`distribution`]: the overall [`Dist`] and, with buckets, one per bucket.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct DistReport {
    /// The overall distribution.
    #[serde(flatten)]
    pub all: Dist,
    /// One distribution per bucket (all buckets, in order, including empty ones).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub by_bucket: Option<Vec<BucketDist>>,
    /// Entries whose node lies in no bucket (with buckets only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unbucketed: Option<u64>,
}

/// Options of [`distribution`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DistConf<'a> {
    /// Length of the top-k lists (0 disables them).
    pub top: usize,
    /// Optional buckets.
    pub buckets: Option<&'a Buckets>,
}

/// One chunk of [`distribution_with`]: the overall sum of its values in element order, and one
/// accumulator per bucket segment (bucket index, or `None` outside every bucket).
type ChunkPart = (f64, Vec<(Option<usize>, Acc)>);

/// The node of entry `i` (for per-block arrays).
pub type NodeOf<'a> = dyn Fn(usize) -> usize + Sync + 'a;

/// [`distribution_with`] for a per-node array of the window starting at `lo`: entry `i` is
/// node `lo + i`.
///
/// ```
/// use dag::summary::{DistConf, distribution};
/// let v = [0u32, 5, u32::MAX, 3, 5];
/// let r = distribution(v.len(), |i| (v[i] != u32::MAX).then(|| v[i] as f64), 100,
///                      &DistConf { top: 2, buckets: None });
/// assert_eq!((r.all.count, r.all.absent), (4, 1));
/// assert_eq!((r.all.max, r.all.argmax), (Some(5.0), Some(101)));
/// assert_eq!(r.all.log2_hist.bins, [1, 0, 1, 2]); // 0 | 1 | 2..3 | 4..7
/// assert_eq!(r.all.top.iter().map(|t| t.id).collect::<Vec<_>>(), [101, 104]);
/// ```
pub fn distribution(
    len: usize,
    key: impl Fn(usize) -> Option<f64> + Sync,
    lo: usize,
    conf: &DistConf<'_>,
) -> DistReport {
    distribution_with(len, key, &|i| lo + i, None, conf)
}

/// The distribution of `key(i)` for `i < len` (see the module documentation): `id(i)` is the
/// absolute id reported for entry `i`, `node(i)` (default: `id(i)`) the node used for
/// bucketing; both must be non-decreasing in `i`.
pub fn distribution_with(
    len: usize,
    key: impl Fn(usize) -> Option<f64> + Sync,
    id: &(dyn Fn(usize) -> usize + Sync),
    node: Option<&NodeOf<'_>>,
    conf: &DistConf<'_>,
) -> DistReport {
    let node_of = |i: usize| node.map_or_else(|| id(i), |f| f(i));
    let k = conf.top;
    // per chunk: the overall sum in element order, and one accumulator per bucket segment
    // (bucket index, or None outside every bucket)
    let parts: Vec<ChunkPart> = (0..len.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let (a, b) = (c * CHUNK, ((c + 1) * CHUNK).min(len));
            let mut total = 0.0;
            let mut segs: Vec<(Option<usize>, Acc)> = Vec::new();
            match conf.buckets {
                None => {
                    let mut acc = Acc::new(k);
                    for i in a..b {
                        let x = key(i);
                        if let Some(x) = x.filter(|x| !x.is_nan()) {
                            total += x;
                        }
                        acc.add(i, x);
                    }
                    segs.push((None, acc));
                }
                Some(bk) => {
                    let mut cur: Option<(Option<usize>, Acc)> = None;
                    let mut range = 0..0;
                    for i in a..b {
                        let v = node_of(i);
                        if cur.is_none() || !range.contains(&v) {
                            let at = bk.locate(v);
                            range = match at {
                                Some(j) => bk.buckets[j].lo..bk.buckets[j].hi,
                                // the gap containing v, up to the next bucket
                                None => {
                                    let j = bk.buckets.partition_point(|b| b.hi <= v);
                                    let lo = if j == 0 { 0 } else { bk.buckets[j - 1].hi };
                                    let hi = bk.buckets.get(j).map_or(usize::MAX, |b| b.lo);
                                    lo..hi
                                }
                            };
                            if cur.as_ref().is_none_or(|c| c.0 != at) {
                                if let Some(s) = cur.take() {
                                    segs.push(s);
                                }
                                cur = Some((at, Acc::new(k)));
                            }
                        }
                        let x = key(i);
                        if let Some(x) = x.filter(|x| !x.is_nan()) {
                            total += x;
                        }
                        cur.as_mut().unwrap().1.add(i, x);
                    }
                    segs.extend(cur);
                }
            }
            (total, segs)
        })
        .collect();
    let mut all = Acc::new(k);
    let mut sum = 0.0;
    let nb = conf.buckets.map_or(0, Buckets::len);
    let mut per: Vec<Acc> = (0..nb).map(|_| Acc::new(k)).collect();
    let mut unbucketed = 0u64;
    for (total, segs) in &parts {
        sum += total;
        for (b, acc) in segs {
            all.merge(acc);
            match b {
                Some(j) => per[*j].merge(acc),
                None if conf.buckets.is_some() => unbucketed += acc.count + acc.absent,
                None => {}
            }
        }
    }
    all.sum = sum;
    let by_bucket = conf.buckets.map(|bk| {
        per.into_iter()
            .zip(&bk.buckets)
            .map(|(acc, b)| BucketDist {
                label: b.label.clone(),
                lo: b.lo,
                hi: b.hi,
                dist: acc.finish(id, node),
            })
            .collect()
    });
    DistReport {
        all: all.finish(id, node),
        by_bucket,
        unbucketed: conf.buckets.map(|_| unbucketed),
    }
}

/// The buckets of `--buckets FILE`, or `None`; warns when they do not cover `[0, n)`.
pub fn load_buckets(path: Option<&Path>, n: usize) -> Result<Option<Buckets>> {
    let Some(p) = path else { return Ok(None) };
    let b = Buckets::load(p)?;
    b.check_cover(n);
    if b.as_slice().iter().all(|x| x.lo >= n) {
        bail!("{}: no bucket intersects [0, {n})", p.display());
    }
    Ok(Some(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_keeps_the_best() {
        let mut t = TopK::new(3);
        for (i, x) in [5.0, 1.0, 9.0, 9.0, 2.0, 7.0, 9.0].into_iter().enumerate() {
            t.offer(x, i);
        }
        assert_eq!(t.finish(), [(9.0, 2), (9.0, 3), (9.0, 6)]);
    }

    #[test]
    fn date_labels() {
        assert_eq!(date_label("2009-01-01 00:00:00 Europe/Rome"), "2009H1");
        assert_eq!(date_label("2022-07-01 00:00:00 Europe/Rome"), "2022H2");
        assert_eq!(date_label("2022-03-05"), "2022-03-05");
    }
}
