//! Benchmark of the reachability index against exact cone labels.
//!
//! **Ground truth** ([`cones64`]): for up to 64 sources, one forward pass over the DAG
//! propagates a `u64` bitmask per DAG node (bit `i` set iff the node is reachable, reflexively,
//! from source `i`). This is a bit-parallel cone pass (64 sources per pass); `⌈sources / 64⌉` passes are run.
//!
//! **Workloads** (fixed seed): sources are sampled uniformly among the DAG nodes with at least
//! one successor; for each source `X`, targets `Y > X` are drawn
//! - *uniform*: uniformly from `(X, n_dag)`, i.e. the natural positive rate;
//! - *balanced*: half from the reachable and half from the unreachable targets in `(X, n_dag)`
//!   (reservoir samples over the cone row; a source with too few targets of one class
//!   contributes fewer queries).
//!
//! **Metrics** per workload ([`WorkloadReport`]): the deciding-filter histogram (with the
//! `Search` and `WindowScan` shares), latency percentiles on one thread, visited-node
//! percentiles, and throughput with rayon over the whole batch. Every answer is compared with
//! the cone label: the benchmark requires **100 % agreement** (reported as `disagreements`)
//! and fails as soon as a single query takes longer than `max_query_secs` (checked after the
//! query: a running query is not interrupted).
//!
//! Queries are DAG-level (Q on the PG). The G-level reduction of
//! [`reach`](crate::reach::index::reach) adds only O(1) map tests (F3 Lemma 1), which the
//! tests check exhaustively on small graphs. References:
//! [\[HST22\]](crate::references#hst22) for the deciding-filter methodology.
//!
//! **Complexity and memory.** `⌈s / 64⌉` forward passes over the DAG for `s` sources, each
//! O(n_dag + m_dag) time with 8 B per DAG node of cone masks; building the balanced workload
//! scans the cone row of every source, O(n_dag) time per source. Each query thread keeps a
//! [`Scratch`] of at most [`Scratch::max_bytes`] (`n_dag / 8 + n_dag / 16` bytes of dirty
//! bitset plus `4 · budget` bytes of DFS stack).

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::io::huge_vec;
use crate::reach::dagview::SuccDag;
use crate::reach::index::{QueryConf, ReachIndex, Scratch, ScratchPool};
use crate::sweep::DynProgressLog;

/// Exact cones of up to 64 `sources`: bit `i` of the result at `y` is set iff
/// `sources[i] ⇝* y` (reflexive). One forward pass; 8 B per DAG node.
pub fn cones64(
    dag: &dyn SuccDag,
    sources: &[u32],
    pl: &mut dyn DynProgressLog,
) -> Result<Vec<u64>> {
    anyhow::ensure!(
        sources.len() <= 64,
        "cones64 takes at most 64 sources, got {}",
        sources.len()
    );
    let n = dag.num_nodes();
    let mut mask: Vec<u64> = huge_vec(n, 0);
    for (i, &s) in sources.iter().enumerate() {
        anyhow::ensure!((s as usize) < n, "source {s} out of range ({n} nodes)");
        mask[s as usize] |= 1 << i;
    }
    pl.dyn_expected_updates(Some(n));
    pl.dyn_start(&format!("cones of {} sources", sources.len()));
    {
        let mask = &mut mask;
        dag.for_each_forward(
            &mut |x, succ| {
                let m = mask[x];
                if m != 0 {
                    for &w in succ {
                        mask[w as usize] |= m;
                    }
                }
            },
            pl,
        )?;
    }
    pl.dyn_done();
    Ok(mask)
}

/// Benchmark parameters.
#[derive(Clone, Debug, serde::Serialize)]
pub struct BenchConf {
    /// Number of cone sources (one forward pass per 64).
    pub sources: usize,
    /// Queries per workload.
    pub queries: usize,
    /// Random seed.
    pub seed: u64,
    /// Fail if a single query takes longer (seconds).
    pub max_query_secs: f64,
    /// Query configuration.
    pub query: QueryConf,
    /// Threads of the throughput run.
    pub threads: usize,
}

/// Summary of a sample.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Percentiles {
    /// Sample size.
    pub count: usize,
    /// Mean.
    pub mean: f64,
    /// Median.
    pub p50: u64,
    /// 90th percentile.
    pub p90: u64,
    /// 99th percentile.
    pub p99: u64,
    /// 99.9th percentile.
    pub p999: u64,
    /// Maximum.
    pub max: u64,
}

impl Percentiles {
    /// Percentiles of `v` (sorted in place; nearest-rank).
    pub fn of(v: &mut [u64]) -> Self {
        if v.is_empty() {
            return Self::default();
        }
        v.sort_unstable();
        let at = |q: f64| v[((q * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
        Self {
            count: v.len(),
            mean: v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64,
            p50: at(0.5),
            p90: at(0.9),
            p99: at(0.99),
            p999: at(0.999),
            max: v[v.len() - 1],
        }
    }
}

/// Results of one workload.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct WorkloadReport {
    /// `uniform` or `balanced`.
    pub name: String,
    /// Number of queries.
    pub queries: usize,
    /// Reachable pairs (ground truth).
    pub positives: usize,
    /// Answers that differ from the ground truth (must be 0).
    pub disagreements: usize,
    /// First disagreeing query `(X, Y, truth)`.
    pub first_disagreement: Option<(u32, u32, bool)>,
    /// Deciding filter → number of queries.
    pub deciders: BTreeMap<String, u64>,
    /// Share decided by the labels alone.
    pub label_share: f64,
    /// Share decided by the pruned DFS.
    pub search_share: f64,
    /// Share decided by the window scan.
    pub window_scan_share: f64,
    /// Latency (ns) on one thread.
    pub latency_ns: Percentiles,
    /// Visited (marked) nodes per query.
    pub visited: Percentiles,
    /// Wall time of the single-thread run (s).
    pub seq_s: f64,
    /// Threads of the throughput run.
    pub par_threads: usize,
    /// Wall time of the throughput run (s).
    pub par_s: f64,
    /// Queries per second of the throughput run.
    pub throughput_qps: f64,
    /// Slowest query of the throughput run (s).
    pub par_max_query_s: f64,
    /// Heap bytes of the throughput run's scratches after the run (one per thread, at most
    /// `par_threads ·` [`Scratch::max_bytes`]).
    pub par_scratch_bytes: usize,
}

/// The benchmark result.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BenchReport {
    /// Number of DAG nodes.
    pub n_dag: usize,
    /// The cone sources.
    pub sources: Vec<u32>,
    /// Seconds spent in the cone passes.
    pub cones_s: f64,
    /// The workloads.
    pub workloads: Vec<WorkloadReport>,
}

impl BenchReport {
    /// Total disagreements over all workloads.
    pub fn disagreements(&self) -> usize {
        self.workloads.iter().map(|w| w.disagreements).sum()
    }
}

/// A labelled DAG-level query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Query {
    /// Source.
    pub x: u32,
    /// Target.
    pub y: u32,
    /// Ground truth.
    pub truth: bool,
}

fn sample_sources(dag: &dyn SuccDag, k: usize, rng: &mut SmallRng) -> Vec<u32> {
    let n = dag.num_nodes();
    if n < 2 {
        return Vec::new();
    }
    let mut buf = Vec::new();
    (0..k)
        .map(|_| {
            let mut x = rng.random_range(0..n - 1);
            for _ in 0..1000 {
                dag.succ_into(x, &mut buf);
                if !buf.is_empty() {
                    break;
                }
                x = rng.random_range(0..n - 1);
            }
            x as u32
        })
        .collect()
}

/// Builds the uniform and balanced workloads (`per_source` queries per source each) from the
/// cones of one batch of ≤ 64 sources.
fn batch_queries(
    cones: &[u64],
    sources: &[u32],
    per_source: usize,
    seed: u64,
) -> (Vec<Query>, Vec<Query>) {
    let n = cones.len();
    let per: Vec<(Vec<Query>, Vec<Query>)> = sources
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let mut rng = SmallRng::seed_from_u64(seed ^ (u64::from(x) << 8 | i as u64));
            let truth = |y: usize| cones[y] >> i & 1 != 0;
            let lo = x as usize + 1;
            let uni: Vec<Query> = (0..per_source)
                .map(|_| {
                    let y = rng.random_range(lo..n);
                    Query {
                        x,
                        y: y as u32,
                        truth: truth(y),
                    }
                })
                .collect();
            // reservoir samples of each class
            let half = per_source.div_ceil(2);
            let mut res: [Vec<u32>; 2] = [Vec::with_capacity(half), Vec::with_capacity(half)];
            let mut seen = [0usize; 2];
            for y in lo..n {
                let c = usize::from(truth(y));
                seen[c] += 1;
                if res[c].len() < half {
                    res[c].push(y as u32);
                } else {
                    let j = rng.random_range(0..seen[c]);
                    if j < half {
                        res[c][j] = y as u32;
                    }
                }
            }
            let bal = res
                .iter()
                .enumerate()
                .flat_map(|(c, ys)| {
                    ys.iter().map(move |&y| Query {
                        x,
                        y,
                        truth: c == 1,
                    })
                })
                .collect();
            (uni, bal)
        })
        .collect();
    let mut uni = Vec::new();
    let mut bal = Vec::new();
    for (u, b) in per {
        uni.extend(u);
        bal.extend(b);
    }
    (uni, bal)
}

/// The generated workloads.
#[derive(Clone, Debug, Default)]
pub struct Workloads {
    /// The cone sources.
    pub sources: Vec<u32>,
    /// Uniform targets.
    pub uniform: Vec<Query>,
    /// Balanced targets.
    pub balanced: Vec<Query>,
    /// Seconds spent in the cone passes.
    pub cones_s: f64,
}

/// Generates the workloads (see the module documentation).
pub fn workloads(
    dag: &dyn SuccDag,
    conf: &BenchConf,
    pl: &mut dyn DynProgressLog,
) -> Result<Workloads> {
    let mut rng = SmallRng::seed_from_u64(conf.seed);
    let sources = sample_sources(dag, conf.sources, &mut rng);
    let mut uni = Vec::new();
    let mut bal = Vec::new();
    let mut cones_s = 0.0;
    if sources.is_empty() {
        return Ok(Workloads::default());
    }
    let per_source = conf.queries.div_ceil(sources.len());
    for (b, batch) in sources.chunks(64).enumerate() {
        let t = Instant::now();
        let cones = cones64(dag, batch, pl)?;
        cones_s += t.elapsed().as_secs_f64();
        let (u, bq) = batch_queries(&cones, batch, per_source, conf.seed ^ (b as u64) << 32);
        uni.extend(u);
        bal.extend(bq);
    }
    uni.shuffle(&mut rng);
    bal.shuffle(&mut rng);
    uni.truncate(conf.queries);
    bal.truncate(conf.queries);
    Ok(Workloads {
        sources,
        uniform: uni,
        balanced: bal,
        cones_s,
    })
}

/// Runs one workload: a single-thread latency run, then a throughput run on `conf.threads`
/// threads. Fails if a query exceeds `conf.max_query_secs`.
pub fn run_workload(
    idx: &ReachIndex,
    dag: &dyn SuccDag,
    name: &str,
    queries: &[Query],
    conf: &BenchConf,
) -> Result<WorkloadReport> {
    let mut rep = WorkloadReport {
        name: name.to_owned(),
        queries: queries.len(),
        positives: queries.iter().filter(|q| q.truth).count(),
        ..Default::default()
    };
    let mut lat = Vec::with_capacity(queries.len());
    let mut vis = Vec::with_capacity(queries.len());
    let mut s = Scratch::new();
    let t0 = Instant::now();
    for q in queries {
        let t = Instant::now();
        let (r, d, v) = idx.reach_dag(dag, q.x as usize, q.y as usize, &conf.query, &mut s);
        let dt = t.elapsed();
        anyhow::ensure!(
            dt.as_secs_f64() <= conf.max_query_secs,
            "query ({}, {}) took {:.3} s > --max-query-secs {} (decider {d:?}, {v} visited)",
            q.x,
            q.y,
            dt.as_secs_f64(),
            conf.max_query_secs
        );
        lat.push(dt.as_nanos() as u64);
        vis.push(u64::from(v));
        *rep.deciders.entry(d.label()).or_default() += 1;
        if r != q.truth {
            rep.disagreements += 1;
            rep.first_disagreement.get_or_insert((q.x, q.y, q.truth));
        }
    }
    rep.seq_s = t0.elapsed().as_secs_f64();
    let total = queries.len().max(1) as f64;
    let share = |f: &dyn Fn(&str) -> bool| {
        rep.deciders
            .iter()
            .filter(|(k, _)| f(k))
            .map(|(_, &c)| c)
            .sum::<u64>() as f64
            / total
    };
    rep.search_share = share(&|k| k == "Search");
    rep.window_scan_share = share(&|k| k == "WindowScan");
    rep.label_share = share(&|k| k != "Search" && k != "WindowScan");
    rep.latency_ns = Percentiles::of(&mut lat);
    rep.visited = Percentiles::of(&mut vis);

    let pool = ScratchPool::new(conf.threads)?;
    let t0 = Instant::now();
    let (bad, max_s) = pool
        .map(queries, |q, s| {
            let t = Instant::now();
            let (r, _, _) = idx.reach_dag(dag, q.x as usize, q.y as usize, &conf.query, s);
            (usize::from(r != q.truth), t.elapsed().as_secs_f64())
        })
        .into_iter()
        .fold((0, 0.0f64), |a, b| (a.0 + b.0, a.1.max(b.1)));
    rep.par_s = t0.elapsed().as_secs_f64();
    rep.par_threads = pool.threads();
    rep.par_scratch_bytes = pool.bytes();
    rep.throughput_qps = queries.len() as f64 / rep.par_s.max(1e-9);
    rep.par_max_query_s = max_s;
    anyhow::ensure!(
        bad == rep.disagreements,
        "parallel run disagrees with the sequential run ({bad} vs {} wrong answers)",
        rep.disagreements
    );
    anyhow::ensure!(
        max_s <= conf.max_query_secs,
        "a query of the throughput run took {max_s:.3} s > --max-query-secs {}",
        conf.max_query_secs
    );
    Ok(rep)
}

/// Runs the whole benchmark (workload generation, then both workloads).
pub fn run_bench(
    idx: &ReachIndex,
    dag: &dyn SuccDag,
    conf: &BenchConf,
    pl: &mut dyn DynProgressLog,
) -> Result<BenchReport> {
    anyhow::ensure!(
        dag.num_nodes() == idx.n_dag,
        "the index has {} DAG nodes, the DAG {}",
        idx.n_dag,
        dag.num_nodes()
    );
    let wl = workloads(dag, conf, pl)?;
    let mut rep = BenchReport {
        n_dag: idx.n_dag,
        sources: wl.sources,
        cones_s: wl.cones_s,
        workloads: Vec::new(),
    };
    for (name, qs) in [("uniform", &wl.uniform), ("balanced", &wl.balanced)] {
        let w = run_workload(idx, dag, name, qs, conf)?;
        log::info!(
            "{name}: {} queries, {} positive, {} disagreements, labels {:.2} %, search {:.2} %, window scan {:.2} %, p50 {} ns, p99 {} ns, {:.0} q/s on {} threads",
            w.queries,
            w.positives,
            w.disagreements,
            100.0 * w.label_share,
            100.0 * w.search_share,
            100.0 * w.window_scan_share,
            w.latency_ns.p50,
            w.latency_ns.p99,
            w.throughput_qps,
            w.par_threads
        );
        rep.workloads.push(w);
    }
    Ok(rep)
}
