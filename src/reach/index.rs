//! The exact reachability index "ChronoChain" and its bounded exact fallback.
//!
//! # Index
//! Per DAG node ([`SuccDag`]: the block quotient Q on the PG, the graph itself otherwise) the
//! index stores the reflexive labels of [`crate::reach::labels`]: `lvl`, `lo` (forward), `ht`,
//! `hi` (backward) and, for each of `k` chains, `inn_c` (forward max, stored `+1`, 0 = −∞) and
//! `out_c` (backward min, [`NONE`] = +∞). Chain labels are the partial chain cover of
//! [\[Jag90\]](crate::references#jag90) / [\[Sim88\]](crate::references#sim88); the level and
//! order filters follow O'Reach [\[HST22\]](crate::references#hst22); the bounded search with
//! label pruning is the GRAIL-style fallback [\[YCZ10\]](crate::references#ycz10) (see also
//! the survey [\[ZBO23\]](crate::references#zbo23)).
//!
//! # Label test ([`ReachIndex::test`]) for X < Y
//! ```text
//! if Y > hi(X) or lo(Y) > X:                         No  (HiLo)
//! if lvl(X) >= lvl(Y) or ht(X) <= ht(Y):             No  (Level)
//! for c: if out_c(X) ≠ NONE ∧ inn_c(Y) ≠ 0 ∧ out_c(X) < inn_c(Y):   Yes (ChainPositive)
//! for c: if on_c(Y): return out_c(X) ≠ NONE ∧ out_c(X) ≤ out_c(Y)          (ChainExact)
//!        if on_c(X): return inn_c(Y) ≠ 0 ∧ out_c(X) < inn_c(Y)              (ChainExact)
//! for c: if out_c(X) > out_c(Y) or inn_c(X) > inn_c(Y):   No (ChainNegative)
//! if hi(Y) > hi(X) or lo(X) < lo(Y):                 No  (HiLo)
//! Unknown
//! ```
//! with `on_c(Z) ⟺ out_c(Z) ≠ NONE ∧ nodes_c[out_c(Z)] = Z`.
//!
//! **Soundness** (F3 Lemma 5). If `X ⇝ Y` with `X ≠ Y`, then `desc(Y) ⊊ desc(X)` and
//! `anc(X) ⊊ anc(Y)`, hence `hi(Y) ≤ hi(X)`, `lo(Y) ≤ lo(X)`, `out_c(X) ≤ out_c(Y)`,
//! `inn_c(X) ≤ inn_c(Y)`, `lvl(X) < lvl(Y)`, `ht(X) > ht(Y)`, `Y ≤ hi(X)` and `lo(Y) ≤ X`:
//! every "No" is sound (with `NONE` = +∞ and `0` = −∞ the inequalities hold for nodes that
//! reach, or are reached by, no chain node). If `out_c(X) ≤ inn_c(Y) − 1`, then
//! `X ⇝ P_c[out_c(X)] ⇝ P_c[inn_c(Y) − 1] ⇝ Y` along the chain: "Yes" is sound. A chain node's
//! `out_c` is its own position (a chain node of smaller position reachable from it would close
//! a cycle), which gives the two exact rules.
//!
//! **ChainExact adds no decisive power.** For a chain node `Z`, `out_c(Z) = pos_c(Z)` and
//! `inn_c(Z) = pos_c(Z) + 1`. If `Y` is on `c`, its positive case `out_c(X) ≤ out_c(Y)` is
//! `out_c(X) < inn_c(Y)` (ChainPositive) and its negative case `out_c(X) > out_c(Y)` is
//! ChainNegative; if `X` is on `c`, its negative case gives `inn_c(X) = out_c(X) + 1 >
//! inn_c(Y)` (ChainNegative). So every ChainExact verdict is also reached by ChainPositive or
//! ChainNegative of the same chain; the rule is kept only to attribute a decision to a chain
//! one endpoint lies on, and its share in a deciding-filter histogram depends on the order of
//! the checks, not on an independent filter's effectiveness.
//!
//! # Queries ([`ReachIndex::reach_dag`], [`reach`])
//! When the test is undecided:
//! 1. **Search**: DFS from X visiting only `W ≤ Y` with `test(W, Y) ≠ No`, stopping at `W = Y`
//!    or `test(W, Y) = Yes`; the visited set is a [`DirtyBitset`] over `[X, Y]` (reset in
//!    O(previously visited)). Abandoned after `budget` visited nodes.
//! 2. **WindowScan** (exact, bounded): mark X; for Z ascending over the marked bits of
//!    `[X, Y]`, for `W ∈ succ(Z)` with `W ≤ Y`: `W = Y` or `test(W, Y) = Yes` → true;
//!    `test(W, Y) ≠ No` → mark W. False at the end. Cost ≤ one sequential pass over `[X, Y]`
//!    and their successor lists.
//!
//! The G-level query [`reach`] on the PG first applies O(1) tests on the
//! [`BlockIndex`] maps (F3 Lemma 1): `u = v` → true; `u > v` → false (Chrono); `spend[u] =
//! NONE` or `create[v] = NONE` → false (SinkSource); `spend[u] = create[v]` → true
//! (DirectArc); `spend[u] > create[v]` → false (Chrono); otherwise the DAG query on Q.
//!
//! Memory: `16 + 8k` bytes per DAG node (37.3 GB on the PG's Q with `k = 4`), plus at most
//! [`Scratch::max_bytes`] per query thread (`n_dag / 8` bytes of bitset, `n_dag / 16` bytes of
//! dirty-word list and `4 · budget` bytes of DFS stack, each up to twice that in capacity
//! because of amortized `Vec` growth).
//!
//! # Example
//! ```
//! # fn main() -> anyhow::Result<()> {
//! use dsi_progress_logger::prelude::*;
//! use dag::graph::{ChunkConf, ChunkedGraph};
//! use dag::reach::dagview::GraphDag;
//! use dag::reach::index::{Decider, QueryConf, ReachIndex, Scratch};
//! use dag::sweep::SweepConf;
//!
//! // 3 diamonds in series: 0 → {1, 2} → 3 → {4, 5} → 6 → {7, 8} → 9
//! let cg = ChunkedGraph::new(dag::testkit::diamonds(3), ChunkConf::default())?;
//! let dag = GraphDag::new(&cg, SweepConf::default());
//! let idx = ReachIndex::build(&dag, 1, &mut Option::<ProgressLogger>::None)?;
//! assert_eq!(idx.chains[0].nodes.len(), 7); // a longest path, e.g. 0 1 3 4 6 7 9
//! let mut s = Scratch::new();
//! assert_eq!(idx.test(1, 2), Some((false, Decider::Level)));
//! assert!(idx.reach_dag(&dag, 2, 8, &QueryConf::default(), &mut s).0);
//! # Ok(())
//! # }
//! ```
//!
//! Files: `<p>.reach.{lvl,ht,hi,lo}.u32le`, `<p>.reach.chain{c}.{nodes,out,in}.u32le` and
//! `<p>.reach.json` ([`IndexMeta`]).

use std::path::Path;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::bitset::DirtyBitset;
use crate::blocks::BlockIndex;
use crate::io::{self, LoadMode, NONE, U32Array};
pub use crate::reach::dagview::Mode;
use crate::reach::dagview::{DagInfo, SuccDag};
use crate::reach::labels::{self, ChainSelection};
use crate::sweep::DynProgressLog;

/// The ChronoChain index (see the module documentation).
pub struct ReachIndex {
    /// The DAG the index was built on.
    pub mode: Mode,
    /// Number of DAG nodes.
    pub n_dag: usize,
    /// Id mapping of the DAG (graph size, node window, id offset).
    pub info: DagInfo,
    /// Longest path (arcs) from a DAG source.
    pub lvl: U32Array,
    /// Longest path (arcs) to a DAG sink.
    pub ht: U32Array,
    /// Largest DAG node reachable (reflexive).
    pub hi: U32Array,
    /// Smallest DAG node reaching (reflexive).
    pub lo: U32Array,
    /// The chains and their labels.
    pub chains: Vec<Chain>,
}

/// One chain of the index and its labels.
pub struct Chain {
    /// The chain: a path of the DAG, `nodes[pos]`.
    pub nodes: U32Array,
    /// Per DAG node: the smallest position reachable (reflexive); [`NONE`] = +∞.
    pub out: U32Array,
    /// Per DAG node: `1 +` the largest position reaching it (reflexive); `0` = −∞.
    pub inn: U32Array,
}

/// What decided a query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub enum Decider {
    /// `u = v`.
    Trivial,
    /// The source comes after the target in the topological order.
    Chrono,
    /// The source is a sink or the target a source (G-level, blocks mode).
    SinkSource,
    /// The target is an output of the block spending the source (G-level, blocks mode).
    DirectArc,
    /// `hi`/`lo` bounds.
    HiLo,
    /// Levels (`lvl`, `ht`).
    Level,
    /// Chain `c` proves reachability.
    ChainPositive(u8),
    /// One endpoint lies on chain `c`, which decides exactly. Every such verdict is also
    /// reached by [`ChainPositive`](Self::ChainPositive) or
    /// [`ChainNegative`](Self::ChainNegative) of the same chain (see the module
    /// documentation): this variant only attributes the decision to that chain.
    ChainExact(u8),
    /// Chain `c` proves non-reachability.
    ChainNegative(u8),
    /// The budgeted pruned DFS.
    Search,
    /// The exact window scan.
    WindowScan,
}

impl Decider {
    /// A short label for histograms, e.g. `ChainPositive(0)`.
    pub fn label(&self) -> String {
        format!("{self:?}")
    }

    /// Whether the decision came from the labels alone (no search).
    pub fn is_label(&self) -> bool {
        !matches!(self, Self::Search | Self::WindowScan)
    }
}

/// Query configuration.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct QueryConf {
    /// Visited nodes after which the pruned DFS gives up for the window scan (default 2²⁰;
    /// 0 forces the window scan, `u32::MAX` practically never uses it).
    pub budget: u32,
}

impl Default for QueryConf {
    fn default() -> Self {
        Self { budget: 1 << 20 }
    }
}

/// Per-thread scratch space of the fallback searches: a [`DirtyBitset`] over the query window
/// (`n_dag / 8` bytes of words plus 4 bytes per touched word), the DFS stack (at most `budget`
/// entries) and one successor buffer; see [`max_bytes`](Self::max_bytes).
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    bits: DirtyBitset,
    stack: Vec<u32>,
    buf: Vec<u32>,
}

impl Scratch {
    /// Empty scratch space (grows on demand).
    pub fn new() -> Self {
        Self::default()
    }

    /// Heap bytes in use.
    pub fn bytes(&self) -> usize {
        self.bits.bytes() + 4 * (self.stack.capacity() + self.buf.capacity())
    }

    /// Upper bound on [`bytes`](Self::bytes) for queries on a DAG of `n_dag` nodes with visit
    /// budget `budget`, given successor lists of at most `max_succ` entries: twice (amortized
    /// `Vec` growth) the `⌈n_dag / 64⌉` bitset words (8 B), dirty-word entries (4 B), stack
    /// entries (4 B, at most `min(budget, n_dag)`) and successor-buffer entries (4 B).
    pub fn max_bytes(n_dag: usize, budget: u32, max_succ: usize) -> u64 {
        let words = n_dag.div_ceil(64) as u64;
        let stack = u64::from(budget).min(n_dag as u64);
        2 * (12 * words + 4 * stack + 4 * max_succ as u64)
    }
}

/// A dedicated rayon pool with exactly one [`Scratch`] per worker thread
/// (one scratch per query thread, reset in O(touched words)).
///
/// Rayon's `map_init` calls its initializer once per job split, not once per thread, so a
/// fresh `Scratch` would be zero-filled to `n_dag / 64` words thousands of times per batch
///. Here worker `i` (`rayon::current_thread_index`) always uses
/// scratch `i`; its lock is never contended, and the closure runs no rayon work, so it is
/// never re-entered. Memory: at most `threads ·` [`Scratch::max_bytes`].
#[derive(Debug)]
pub struct ScratchPool {
    pool: rayon::ThreadPool,
    scratches: Vec<Mutex<Scratch>>,
}

impl ScratchPool {
    /// A pool of `threads` (at least 1) workers, each with an empty scratch.
    pub fn new(threads: usize) -> Result<Self> {
        let threads = threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        let scratches = (0..threads).map(|_| Mutex::new(Scratch::new())).collect();
        Ok(Self { pool, scratches })
    }

    /// Number of worker threads (= number of scratches).
    pub fn threads(&self) -> usize {
        self.scratches.len()
    }

    /// Total heap bytes of the scratches.
    pub fn bytes(&self) -> usize {
        self.scratches
            .iter()
            .map(|s| s.lock().unwrap_or_else(PoisonError::into_inner).bytes())
            .sum()
    }

    /// `items.map(|t| f(t, scratch))` in parallel on the pool, in order. `f` must not run
    /// rayon work itself.
    pub fn map<T: Sync, R: Send>(
        &self,
        items: &[T],
        f: impl Fn(&T, &mut Scratch) -> R + Sync,
    ) -> Vec<R> {
        self.pool.install(|| {
            items
                .par_iter()
                .map(|t| {
                    let i = rayon::current_thread_index()
                        .expect("ScratchPool::map runs on its own pool");
                    let mut s = self.scratches[i]
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    f(t, &mut s)
                })
                .collect()
        })
    }
}

/// Timings and sizes of a build.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BuildStats {
    /// Number of DAG nodes.
    pub n_dag: usize,
    /// Chains requested.
    pub k_requested: usize,
    /// Chain lengths (nodes).
    pub chain_lens: Vec<usize>,
    /// Newly covered nodes per chain.
    pub chain_gains: Vec<u64>,
    /// Nodes on at least one chain.
    pub covered: u64,
    /// Largest `lvl` (the DAG's longest path, arcs).
    pub max_lvl: u32,
    /// Seconds spent selecting chains.
    pub chains_s: f64,
    /// Seconds of the fused forward pass.
    pub forward_s: f64,
    /// Seconds of the fused backward pass.
    pub backward_s: f64,
    /// Bytes of the labels.
    pub label_bytes: u64,
}

/// The `<p>.reach.json` sidecar.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IndexMeta {
    /// Format tag, `"chronochain/1"`.
    pub format: String,
    /// Number of DAG nodes.
    pub n_dag: usize,
    /// Id mapping.
    pub info: DagInfo,
    /// Chain lengths (nodes).
    pub chain_lens: Vec<usize>,
}

const FORMAT: &str = "chronochain/1";

impl ReachIndex {
    /// Builds the index of `dag` with up to `k ≤ 255` chains (fewer if the chains already
    /// cover every node). Costs `k + 2` streaming passes.
    pub fn build(dag: &dyn SuccDag, k: usize, pl: &mut dyn DynProgressLog) -> Result<Self> {
        Ok(Self::build_with_stats(dag, k, pl)?.0)
    }

    /// Like [`build`](Self::build), also returning timings and sizes.
    pub fn build_with_stats(
        dag: &dyn SuccDag,
        k: usize,
        pl: &mut dyn DynProgressLog,
    ) -> Result<(Self, BuildStats)> {
        anyhow::ensure!(k <= 255, "at most 255 chains (got {k})");
        let n = dag.num_nodes();
        anyhow::ensure!(
            n < NONE as usize,
            "{n} DAG nodes do not fit u32 labels with a NONE sentinel"
        );
        let t = Instant::now();
        let sel: ChainSelection = labels::select_chains(dag, k, pl)?;
        let chains_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let f = labels::forward_labels(dag, &sel.chains, pl)?;
        let forward_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let b = labels::backward_labels(dag, &sel.chains, pl)?;
        let backward_s = t.elapsed().as_secs_f64();
        let stats = BuildStats {
            n_dag: n,
            k_requested: k,
            chain_lens: sel.chains.iter().map(Vec::len).collect(),
            covered: sel.gains.iter().sum(),
            chain_gains: sel.gains,
            max_lvl: f.lvl.iter().copied().max().unwrap_or(0),
            chains_s,
            forward_s,
            backward_s,
            label_bytes: Self::label_bytes(n, sel.chains.len()),
        };
        let chains = sel
            .chains
            .into_iter()
            .zip(f.inn)
            .zip(b.out)
            .map(|((nodes, inn), out)| Chain {
                nodes: U32Array::from_vec(nodes),
                out: U32Array::from_vec(out),
                inn: U32Array::from_vec(inn),
            })
            .collect();
        let info = dag.info();
        let idx = Self {
            mode: info.mode,
            n_dag: n,
            info,
            lvl: U32Array::from_vec(f.lvl),
            ht: U32Array::from_vec(b.ht),
            hi: U32Array::from_vec(b.hi),
            lo: U32Array::from_vec(f.lo),
            chains,
        };
        Ok((idx, stats))
    }

    /// Bytes of the labels of an index with `n_dag` nodes and `k` chains (`16 + 8k` per node,
    /// chain node lists excluded).
    pub fn label_bytes(n_dag: usize, k: usize) -> u64 {
        n_dag as u64 * (16 + 8 * k as u64)
    }

    /// Peak bytes of a build: the labels plus the transient chain-selection arrays (8 B/node
    /// and a bitmap).
    pub fn build_bytes(n_dag: usize, k: usize) -> u64 {
        Self::label_bytes(n_dag, k) + n_dag as u64 * 8 + n_dag as u64 / 8
    }

    /// Number of chains.
    pub fn k(&self) -> usize {
        self.chains.len()
    }

    /// Writes `<p>.reach.*` (see the module documentation).
    pub fn save(&self, prefix: &Path) -> Result<()> {
        let arr = |name: &str| io::suffixed(prefix, &format!("reach.{name}.u32le"));
        io::write_array(&arr("lvl"), &self.lvl)?;
        io::write_array(&arr("ht"), &self.ht)?;
        io::write_array(&arr("hi"), &self.hi)?;
        io::write_array(&arr("lo"), &self.lo)?;
        for (c, ch) in self.chains.iter().enumerate() {
            io::write_array(&arr(&format!("chain{c}.nodes")), &ch.nodes)?;
            io::write_array(&arr(&format!("chain{c}.out")), &ch.out)?;
            io::write_array(&arr(&format!("chain{c}.in")), &ch.inn)?;
        }
        let meta = IndexMeta {
            format: FORMAT.to_owned(),
            n_dag: self.n_dag,
            info: self.info,
            chain_lens: self.chains.iter().map(|c| c.nodes.len()).collect(),
        };
        let json = io::suffixed(prefix, "reach.json");
        std::fs::write(&json, serde_json::to_string_pretty(&meta)? + "\n")
            .with_context(|| format!("cannot write {}", json.display()))?;
        Ok(())
    }

    /// The paths of the files written by [`save`](Self::save) for `k` chains.
    pub fn files(prefix: &Path, k: usize) -> Vec<std::path::PathBuf> {
        let mut v: Vec<_> = ["lvl", "ht", "hi", "lo"]
            .iter()
            .map(|n| io::suffixed(prefix, &format!("reach.{n}.u32le")))
            .collect();
        for c in 0..k {
            for f in ["nodes", "out", "in"] {
                v.push(io::suffixed(prefix, &format!("reach.chain{c}.{f}.u32le")));
            }
        }
        v
    }

    /// Reads the sidecar `<p>.reach.json`.
    pub fn load_meta(prefix: &Path) -> Result<IndexMeta> {
        let json = io::suffixed(prefix, "reach.json");
        let meta: IndexMeta = serde_json::from_str(
            &std::fs::read_to_string(&json)
                .with_context(|| format!("cannot read {}", json.display()))?,
        )
        .with_context(|| format!("cannot parse {}", json.display()))?;
        anyhow::ensure!(
            meta.format == FORMAT,
            "{}: unknown format {:?}",
            json.display(),
            meta.format
        );
        Ok(meta)
    }

    /// Loads an index written by [`save`](Self::save).
    pub fn load(prefix: &Path, mode: LoadMode) -> Result<Self> {
        let meta = Self::load_meta(prefix)?;
        let n = meta.n_dag;
        let ld = |name: &str, len: usize| -> Result<U32Array> {
            let p = io::suffixed(prefix, &format!("reach.{name}.u32le"));
            let a = U32Array::load(&p, mode)?;
            anyhow::ensure!(
                a.len() == len,
                "{} has {} entries, expected {len}",
                p.display(),
                a.len()
            );
            Ok(a)
        };
        let mut chains = Vec::with_capacity(meta.chain_lens.len());
        for (c, &len) in meta.chain_lens.iter().enumerate() {
            chains.push(Chain {
                nodes: ld(&format!("chain{c}.nodes"), len)?,
                out: ld(&format!("chain{c}.out"), n)?,
                inn: ld(&format!("chain{c}.in"), n)?,
            });
        }
        Ok(Self {
            mode: meta.info.mode,
            n_dag: n,
            info: meta.info,
            lvl: ld("lvl", n)?,
            ht: ld("ht", n)?,
            hi: ld("hi", n)?,
            lo: ld("lo", n)?,
            chains,
        })
    }

    // ---- individual filters (each sound on its own; X < Y required) -------------------------

    /// First hi/lo filter: `Some(false)` if `Y > hi(X)` or `lo(Y) > X`. It is implied by
    /// [`filter_hilo_dominance`](Self::filter_hilo_dominance) (`hi(Y) ≥ Y` and `lo(X) ≤ X`) and
    /// runs first only as a cheap early exit.
    #[inline]
    pub fn filter_hilo_bounds(&self, x: usize, y: usize) -> Option<bool> {
        (y > self.hi[x] as usize || self.lo[y] as usize > x).then_some(false)
    }

    /// Level filter: `Some(false)` if `lvl(X) ≥ lvl(Y)` or `ht(X) ≤ ht(Y)`.
    #[inline]
    pub fn filter_level(&self, x: usize, y: usize) -> Option<bool> {
        (self.lvl[x] >= self.lvl[y] || self.ht[x] <= self.ht[y]).then_some(false)
    }

    /// Positive filter of chain `c`: `Some(true)` if `out_c(X) < inn_c(Y)` (both defined).
    #[inline]
    pub fn filter_chain_positive(&self, c: usize, x: usize, y: usize) -> Option<bool> {
        let ch = &self.chains[c];
        let (o, i) = (ch.out[x], ch.inn[y]);
        (o != NONE && i != 0 && o < i).then_some(true)
    }

    /// Whether DAG node `z` lies on chain `c`.
    #[inline]
    pub fn on_chain(&self, c: usize, z: usize) -> bool {
        let ch = &self.chains[c];
        let o = ch.out[z];
        o != NONE && ch.nodes[o as usize] as usize == z
    }

    /// Exact filter of chain `c`: decides when `Y` or `X` lies on the chain.
    #[inline]
    pub fn filter_chain_exact(&self, c: usize, x: usize, y: usize) -> Option<bool> {
        let ch = &self.chains[c];
        if self.on_chain(c, y) {
            let o = ch.out[x];
            return Some(o != NONE && o <= ch.out[y]);
        }
        if self.on_chain(c, x) {
            let i = ch.inn[y];
            return Some(i != 0 && ch.out[x] < i);
        }
        None
    }

    /// Negative filter of chain `c`: `Some(false)` if `out_c(X) > out_c(Y)` or
    /// `inn_c(X) > inn_c(Y)`.
    #[inline]
    pub fn filter_chain_negative(&self, c: usize, x: usize, y: usize) -> Option<bool> {
        let ch = &self.chains[c];
        (ch.out[x] > ch.out[y] || ch.inn[x] > ch.inn[y]).then_some(false)
    }

    /// Second hi/lo filter: `Some(false)` if `hi(Y) > hi(X)` or `lo(X) < lo(Y)`.
    #[inline]
    pub fn filter_hilo_dominance(&self, x: usize, y: usize) -> Option<bool> {
        (self.hi[y] > self.hi[x] || self.lo[x] < self.lo[y]).then_some(false)
    }

    /// The label test (no search): `Some((answer, decider))`, or `None` if undecided. Also
    /// handles `x == y` (Trivial) and `x > y` (Chrono).
    pub fn test(&self, x: usize, y: usize) -> Option<(bool, Decider)> {
        if x == y {
            return Some((true, Decider::Trivial));
        }
        if x > y {
            return Some((false, Decider::Chrono));
        }
        if let Some(r) = self.filter_hilo_bounds(x, y) {
            return Some((r, Decider::HiLo));
        }
        if let Some(r) = self.filter_level(x, y) {
            return Some((r, Decider::Level));
        }
        for c in 0..self.chains.len() {
            if let Some(r) = self.filter_chain_positive(c, x, y) {
                return Some((r, Decider::ChainPositive(c as u8)));
            }
        }
        for c in 0..self.chains.len() {
            if let Some(r) = self.filter_chain_exact(c, x, y) {
                return Some((r, Decider::ChainExact(c as u8)));
            }
        }
        for c in 0..self.chains.len() {
            if let Some(r) = self.filter_chain_negative(c, x, y) {
                return Some((r, Decider::ChainNegative(c as u8)));
            }
        }
        if let Some(r) = self.filter_hilo_dominance(x, y) {
            return Some((r, Decider::HiLo));
        }
        None
    }

    /// Exact DAG-level query `x ⇝* y`: the label test, then the budgeted pruned DFS, then the
    /// window scan. Returns the answer, the decider and the number of visited (marked) nodes.
    pub fn reach_dag(
        &self,
        dag: &dyn SuccDag,
        x: usize,
        y: usize,
        q: &QueryConf,
        s: &mut Scratch,
    ) -> (bool, Decider, u32) {
        debug_assert_eq!(dag.num_nodes(), self.n_dag, "index built on another DAG");
        if let Some((r, d)) = self.test(x, y) {
            return (r, d, 0);
        }
        if let Some((r, v)) = self.search(dag, x, y, q.budget, s) {
            return (r, Decider::Search, v);
        }
        let (r, v) = self.window_scan(dag, x, y, s);
        (r, Decider::WindowScan, v)
    }

    /// Pruned DFS; `None` if the budget ran out.
    fn search(
        &self,
        dag: &dyn SuccDag,
        x: usize,
        y: usize,
        budget: u32,
        s: &mut Scratch,
    ) -> Option<(bool, u32)> {
        if budget == 0 {
            return None;
        }
        let Scratch { bits, stack, buf } = s;
        bits.reset(x, y - x + 1);
        stack.clear();
        bits.insert(x);
        stack.push(x as u32);
        let mut visited = 1u32;
        while let Some(z) = stack.pop() {
            dag.succ_into(z as usize, buf);
            for &w in buf.iter() {
                let w = w as usize;
                if w > y {
                    break;
                }
                if w == y {
                    return Some((true, visited));
                }
                if !bits.insert(w) {
                    continue;
                }
                visited += 1;
                match self.test(w, y) {
                    Some((true, _)) => return Some((true, visited)),
                    Some((false, _)) => {}
                    None => stack.push(w as u32),
                }
                if visited >= budget {
                    return None;
                }
            }
        }
        Some((false, visited))
    }

    /// Exact scan over the marked nodes of `[x, y]` in ascending order.
    fn window_scan(&self, dag: &dyn SuccDag, x: usize, y: usize, s: &mut Scratch) -> (bool, u32) {
        let Scratch { bits, buf, .. } = s;
        bits.reset(x, y - x + 1);
        bits.insert(x);
        let mut visited = 1u32;
        let mut cur = x;
        loop {
            // the iterator borrows the bitset only for this statement
            let Some(z) = bits.iter_from(cur).next() else {
                break;
            };
            cur = z + 1;
            dag.succ_into(z, buf);
            for &w in buf.iter() {
                let w = w as usize;
                if w > y {
                    break;
                }
                if w == y {
                    return (true, visited);
                }
                if bits.contains(w) {
                    continue;
                }
                match self.test(w, y) {
                    Some((true, _)) => return (true, visited),
                    Some((false, _)) => {}
                    None => {
                        bits.insert(w);
                        visited = visited.saturating_add(1);
                    }
                }
            }
        }
        (false, visited)
    }
}

/// Exact G-level query `u ⇝* v` (absolute node ids inside the index's window).
///
/// In [`Mode::Blocks`] the O(1) map tests of the module documentation run first and `blocks`
/// must be the [`BlockIndex`] the index was built from; in [`Mode::Graph`] `blocks` is
/// ignored. `dag` must be the DAG the index was built on.
///
/// # Panics
/// If `u` or `v` lies outside the index's window, or `blocks` is `None` in blocks mode.
pub fn reach(
    idx: &ReachIndex,
    dag: &dyn SuccDag,
    blocks: Option<&BlockIndex>,
    u: usize,
    v: usize,
    q: &QueryConf,
    s: &mut Scratch,
) -> (bool, Decider, u32) {
    let (lo, hi) = (idx.info.lo, idx.info.hi);
    assert!(
        (lo..hi).contains(&u) && (lo..hi).contains(&v),
        "query ({u}, {v}) outside the index window [{lo}, {hi})"
    );
    if u == v {
        return (true, Decider::Trivial, 0);
    }
    if u > v {
        return (false, Decider::Chrono, 0);
    }
    match idx.mode {
        Mode::Graph => idx.reach_dag(dag, u - lo, v - lo, q, s),
        Mode::Blocks => {
            let b = blocks.expect("a blocks-mode index needs the BlockIndex");
            let (xb, yb) = (b.spend[u], b.create[v]);
            if xb == NONE || yb == NONE {
                return (false, Decider::SinkSource, 0);
            }
            if xb == yb {
                return (true, Decider::DirectArc, 0);
            }
            if xb > yb {
                return (false, Decider::Chrono, 0);
            }
            // first[xb] > u ≥ lo gives xb ≥ blo = base, and first[yb] ≤ v < hi gives yb < bhi,
            // so both are DAG nodes of the windowed quotient
            let base = idx.info.base;
            idx.reach_dag(dag, xb as usize - base, yb as usize - base, q, s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ChunkConf;
    use crate::graph::ChunkedGraph;
    use crate::reach::dagview::GraphDag;
    use crate::sweep::SweepConf;
    use crate::testkit::{bit, closure, ladder};
    use dsi_progress_logger::prelude::*;

    #[test]
    fn ladder_queries_are_exact_with_every_budget() {
        let g = ladder(12);
        let cg = ChunkedGraph::new(g.clone(), ChunkConf::default()).unwrap();
        let dag = GraphDag::new(&cg, SweepConf::default());
        let cl = closure(&g);
        let idx = ReachIndex::build(&dag, 1, &mut Option::<ProgressLogger>::None).unwrap();
        let mut s = Scratch::new();
        for budget in [0, 1, 2, u32::MAX] {
            let q = QueryConf { budget };
            for (x, row) in cl.iter().enumerate() {
                for y in 0..dag.num_nodes() {
                    let (r, _, _) = idx.reach_dag(&dag, x, y, &q, &mut s);
                    assert_eq!(r, x == y || bit(row, y), "({x}, {y}) budget {budget}");
                }
            }
        }
    }
}
