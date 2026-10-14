# dag

DAG algorithms (semiring DP, dominators, reachability, transitive reduction, width) for any DAG
in [webgraph](https://github.com/vigna/webgraph-rs) format, engineered and validated at the scale
of the Bitcoin Payment Graph (2.18·10⁹ nodes, 8.64·10⁹ arcs): one library (`dag`) and one binary
(`dag`).

The Payment Graph (PG) has one node per transaction output and an arc from every input of a
transaction to each of its outputs. Ids are chronological, so **the identity is a topological
order**, and the arc set is a disjoint union of complete bicliques, one per spending transaction.
The crate exploits both facts: every kernel is a single pass in id order, and on the PG most
passes run on a decode-free index of the bicliques (`BlockIndex`). The theory (the PG model, the
transitive-reduction lemma, and the theorems behind each family) is in the crate documentation
(`cargo doc --open`, front page), and the bibliography with DOIs is in the `references` module.

| Family | Subcommands | Library modules |
|---|---|---|
| core | `topo-check`, `toposort`, `window`, `blocks`, `aggregate` | `topo`, `window`, `blocks`, `sweep`, `graph`, `io`, `summary` |
| 1. semiring DP | `dp`, `certify`, `cpm`, `count-paths` | `semiring`, `dp`, `cpm`, `paths` |
| 2. dominators | `dom build`, `dom verify`, `dom stats` | `dom` |
| 3. reachability | `reach-counts`, `reach-index build`, `reach-index query`, `reach-index bench` | `reach` |
| 4. TR, covers, width | `tr verify`, `tr reduce`, `path-cover`, `frontier`, `width` | `reduce`, `width` |

## Build

Requirements: Rust ≥ 1.85 (edition 2024). `webgraph` and `webgraph-algo` come from
[webgraph-rs](https://github.com/vigna/webgraph-rs) on GitHub, pinned in `Cargo.toml` to the
commit this crate was developed and validated against; every other dependency is pinned in
`Cargo.lock`, so after a first `cargo fetch` the build also works offline:

```sh
cargo build --offline --release          # target/release/dag (fat LTO, codegen-units = 1)
cargo doc --offline --no-deps --open     # crate documentation and bibliography
```

Profiles:

- `release` builds production binaries, with fat LTO and line tables for profiling.
- `release-test` is meant for tests: release optimisation without LTO, incremental, and debug
  assertions and overflow checks in this crate only. Use it for every `cargo test`.
- `dev` builds dependencies at `opt-level = 3`, so BV decoding stays usable in debug builds.

## Input graphs

A graph is given by the **basename** of a BV graph (`B.graph`, `B.properties`, `B.offsets`) and
needs its Elias–Fano offsets `B.ef`. The degree cumulative function `B.dcf` is optional and makes
chunking free. The transpose is looked up as `B-t` unless `--transpose` says otherwise; the
subcommands that need it (pull kernels, dominator builds) say so in their errors. With the
`webgraph` CLI:

```sh
webgraph build ef  B        # B.ef
webgraph build dcf B        # B.dcf (optional)
webgraph transform transpose B B-t && webgraph build ef B-t
```

Node ids must be a topological order. Check it with `dag topo-check`. If it fails, relabel
once with `dag toposort` and run everything on the relabelled graph: results come back in the
new ids, and `B2.order.u32le` (new → old) maps them back. The number of nodes must be at most
`u32::MAX − 1`.

## Command line

```text
dag [GLOBAL OPTIONS] <SUBCOMMAND> [OPTIONS]
```

Global options (they can also follow the subcommand):

| Option | Default | Meaning |
|---|---|---|
| `--threads N` | all cores | size of the rayon pool |
| `--decoders D` | min(threads, 32) | chunks decoded in parallel by the sweep engine |
| `--chunk-arcs A` | 16777216 (2²⁴) | target arcs per chunk |
| `--max-chunk-nodes K` | 4194304 (2²²) | maximum nodes per chunk |
| `--max-mem GB` | 400 | memory cap (10⁹ bytes) checked against each subcommand's estimate before allocating |
| `--load ram\|mmap` | `ram` | how persisted arrays (block index, trees, labels) are loaded |
| `-v`, `-q` | info | more or less logging (`-vv` trace, `-qq` errors only) |

Common options of the subcommands:

| Option | Meaning |
|---|---|
| `--graph B` | forward BV basename (required by every graph subcommand) |
| `--transpose BT` | transpose basename (default `B-t` if it exists) |
| `--blocks auto\|none\|PREFIX` | `auto` (default) builds the `BlockIndex` in memory when the subcommand benefits and falls back to the generic kernels if the graph is not a disjoint union of bicliques; `none` disables it; `PREFIX` loads `PREFIX.blk.*` written by `dag blocks` |
| `--window a..b` | half-open node window; `a..`, `..b`, `..` and `_` digit separators are accepted; default: the whole graph |
| `--out P` | output prefix: files are `P.<field>.<dtype>` plus the sidecar `P.json` |
| `--buckets FILE` | node-range buckets (for the PG, `data/chunk_boundaries.tsv`: one bucket per half-year); every distribution in the JSON is repeated per bucket, see *Aggregate measures* |
| `--top K` | length of the top-k lists of the distributions (default 20; `dom` keeps its own `--top`) |

`dag <subcommand> --help` prints the full option list. Exit codes: **0** success, **1** a
verification failed (the JSON says which check), **2** error (bad input, memory estimate above
`--max-mem`, I/O).

Environment: `DAG_BIN_NO_THP` (any value) stops the large arrays from asking for transparent
huge pages. Huge pages cut TLB misses in the random-access kernels, but with THP
`defrag = madvise` each huge-page fault may compact memory synchronously, which on a host whose
RAM is full of page cache can cost more system time than the computation (see *Memory and
time*). Output arrays are written to new files: an existing file is unlinked, not truncated,
because rewriting a truncated multi-GB file on ext4 is about 4× slower.

### Core

```sh
dag topo-check --graph B [--transpose BT | --no-transpose] [--out P]
```
Checks that every arc goes from a smaller to a larger id (on the graph, and on the transpose
unless disabled) and counts sources and sinks. The report is printed on stdout and, with `--out`,
written to `P.json`. Exit 1 if the order is not topological.

```sh
dag toposort --graph B --out B2 [--tmp DIR]
```
Relabels in topological order. It writes the BV graph `B2` and its transpose `B2-t` (with `.ef`
and `.dcf`), `B2.order.u32le` (new id → old id) and `B2.rank.u32le` (old id → new id), then
re-checks `B2` with `topo-check`. `--tmp` holds the external-sort batches (default `B2.tmp`); their
size follows `--max-mem`.

```sh
dag window --graph B --window a..b --out B2 [--tmp DIR]
```
Writes the subgraph induced by `[a, b)` with ids shifted by `−a`, and its transpose. Every window
of a topologically numbered graph is convex, so this is the right input for experiments on a time
range.

```sh
dag blocks --graph B --out P [--compare-bicliques DIR]
```
Builds and verifies the biclique decomposition. It writes `P.blk.create.u32le` and
`P.blk.spend.u32le` (per node: the creating and the spending block, `NONE` for sources and sinks),
`P.blk.first.u32le`, `P.blk.len.u32le` and `P.blk.nin.u32le` (per block: first output, number of
outputs, number of inputs), and `P.blk.json` (totals: nb, sources, sinks, Σ len, Σ nin,
Σ len·nin = m, maximum block sizes). It fails with a precise violation (node and kind) if the
graph is not a disjoint union of bicliques. `--compare-bicliques` cross-checks against the
repository's `bicliques/blocks.u32x3` and `blk_start.u32`. Pass `--blocks P` to later
subcommands to skip the rebuild (26.8 GB on the PG). Building the index is a pass over the whole
graph whatever the `--window`: on a strict sub-window `dp`, `cpm` and `count-paths` skip it under
`--blocks auto` (generic kernels) and memory-map a saved one, and `tr verify` checks the
line-digraph property of the window alone; the other subcommands build the full index, so for
windowed runs on the PG prefer `--blocks P --load mmap`, which pages in only what the window
touches.

### Family 1: semiring dynamic programming

```sh
dag dp --graph B --out P --semiring S [--dir forward|backward] [--init sources|all]
          [--weights W] [--kernel auto|push|pull|blocks] [--pred] [--certify] [--window a..b]
```
Computes `F(v) = b(v) ⊕ ⨁_{u→v} F(u) ⊗ w(u,v)` forward, or the symmetric backward value.

- `--semiring`: `maxplus-u32|maxplus-u64|maxplus-f64` (longest paths), `minplus-u32|minplus-i64|minplus-f64`
  (shortest paths), `maxmin-u32` (bottleneck), `bool` (reachability from the initial set),
  `sat64|sat128` (saturating path counts), `mod61` (path counts modulo 2⁶¹ − 1), `extfloat`
  (path counts as an extended-exponent float), `log2sum` (log2 of sums), `prob` (probabilities),
  `viterbi` (most probable path).
- `--init sources` puts 1̄ on window sources (forward) or window sinks (backward); `all` puts it
  on every node.
- `--weights`: `unit` is arc length 1 for the tropical semirings and 1̄ otherwise; `one` is 1̄
  always; `inv-outdeg` is `1/outdeg(u)` (with `prob` this splits mass uniformly over the outputs,
  with `viterbi` it gives the most probable random-walk path); `node-u32:FILE` and `node-f64:FILE`
  read a per-node weight array.
- `--kernel auto` picks, forward, `blocks` when a block index is available and the weights are
  separable, else `pull` when the transpose exists, else `push`; backward, `blocks` under the same
  condition, else `pull` over the graph itself (no transpose needed). All kernels give bit-identical values (with
  per-arc weights only push and pull agree).
- Outputs: `P.val.<dtype>` (indexed by `v − a` in a window), with `--pred` also `P.pred.u32le`
  (absolute id of the best predecessor, or successor backward, for selective semirings; `NONE` if
  none).

With `--certify` the result is then checked as by `certify`.

```sh
dag certify --graph B --out P --values FILE --semiring S [--dir] [--init] [--weights] [--pred FILE] [--window a..b]
```
Re-evaluates every fixpoint equation in an independent parallel scan, exactly for exact and
selective semirings and within a per-node rounding bound for float sums. With `--pred` it also
checks that each predecessor realises the value. It writes `P.json` (checked, violations,
pred_violations, max_rel_err, first_violation). Exit 1 on any violation.

```sh
dag cpm --graph B --out P [--weights unit|node-u32:FILE] [--slack] [--path] [--critical-arcs]
           [--compare-e FILE] [--compare-h FILE] [--window a..b]
```
Critical path method. It writes `P.E.u32le` (earliest start = longest path from a source) and
`P.H.u32le` (longest path to a sink), plus, on request, `P.slack.u32le`
(`T − E − H`) and `P.path.u32le` (one longest path). The JSON reports T, the number of critical
nodes (and arcs with `--critical-arcs`), the critical-layer profile and the end of a longest path.
`--compare-e` and `--compare-h` compare byte by byte with existing arrays (for example
`chains/longest.u32le` and `layers/height.u32le`).

```sh
dag count-paths --graph B --out P [--dir forward|backward|both] [--kinds extfloat,mod61[,sat64,sat128]] [--window a..b]
```
Counts the paths from sources to each node (`fwd`) and from each node to sinks (`bwd`). Outputs:
`P.fwd.log2.f64le` (log2 of the count), `P.fwd.mod61.u64le` (count mod 2⁶¹ − 1), the `bwd`
equivalents, and `sat64` and `sat128` arrays on request. The JSON checks that the forward and
backward totals agree (exactly mod 2⁶¹ − 1, with the observed relative difference for the float)
and counts saturated nodes.

### Family 2: dominators

```sh
dag dom build --graph B --out P [--only dom|post] [--budget 64] [--sizes] [--pre] [--depth-bound] [--window a..b]
```
Builds the dominator tree (virtual root above the sources) and the post-dominator tree (virtual
sink below the sinks), concurrently when memory allows. Outputs: `P.idom.u32le` and
`P.ipdom.u32le` (absolute parent ids, `NONE` for children of the virtual root/sink),
`P.dom.depth.u32le` and `P.pdom.depth.u32le`, and with `--sizes` and `--pre` also
`P.{dom,pdom}.size.u32le` and `P.{dom,pdom}.pre.u32le`. `--budget` bounds the nearest-common-ancestor
walk before the jump-pointer fallback; it changes speed only, never the result. The JSON includes
the NCA statistics and, with `--depth-bound`, the check `depth ≤ dmin + 1`.

```sh
dag dom verify --graph B --out P --idom FILE [--post] [--window a..b]
```
Checks a parent array with a verifier that shares no code path with the builders (one parallel
scan of the forward graph, conditions V0–V2). It writes `P.json` (ok, v0_fail, v1_fail, v2_fail,
first_failure). Exit 1 on failure.

```sh
dag dom stats --prefix P [--top 100]
```
Summarises a built tree: root children, cut nodes, log2 histograms of depth and subtree size. It
writes `P.stats.json` and `P.{dom,pdom}.top.tsv` (`node<TAB>size<TAB>depth<TAB>idom`, the largest
subtrees).

### Family 3: reachability

```sh
dag reach-counts --graph B --out P [--which forward|backward|both] [--log2m 10] [--seed S] [--window a..b]
```
Estimates the number of proper ancestors (`forward`) and descendants (`backward`) of every node
with HyperLogLog sketches, in one pass per direction. The relative standard deviation is about
`1.04/√2^log2m`, i.e. 3.3 % at the default. Outputs: `P.anc.f32le` and `P.desc.f32le` per node,
and on the PG also `P.blk.anc.f32le` and `P.blk.desc.f32le` per block. Errors are correlated across
nodes for one seed, so aggregate statistics should average several `--seed` values.

```sh
dag reach-index build --graph B --out P [--chains 4] [--window a..b]
dag reach-index query --graph B --index P (--pair U V | --pairs FILE.u32x2le) [--budget 1048576] [--out Q]
dag reach-index bench --graph B --index P --out Q [--sources 64] [--queries 100000] [--seed S] [--max-query-secs 60]
```
The exact reachability index "ChronoChain". It is built on the block quotient on the PG and on the
graph itself otherwise. `build` writes `P.reach.{lvl,ht,hi,lo}.u32le`,
`P.reach.chain<c>.{nodes,out,in}.u32le` and `P.reach.json`. `query` answers one pair on stdout;
with a file of little-endian `u32` pairs and `--out Q` it writes the answers as a `u8` array
(1 = reachable) and statistics in `Q.json`, including how many queries each filter decided. Every answer is exact; `--budget` only
bounds the pruned search before the exact window scan. `bench` compares against ground-truth cone
labels from `--sources` random sources and reports filter shares, latency percentiles and
throughput. It requires 100 % agreement and aborts if one query exceeds `--max-query-secs`.

### Family 4: transitive reduction, path covers, width

```sh
dag tr verify --graph B --out P [--blocks auto|none|PREFIX] [--window a..b]
dag tr reduce --graph B --out B2 [--algo auto|bitset|search] [--window a..b]
```
`verify` decides whether the graph (or window) is transitively reduced. The certificate is
`blocks` (a successful block index, which proves it), `H3` (an order-based sufficient condition)
or `exact` (a bounded search over the only arcs that can be redundant). It writes `P.json` and
`P.redundant.u32x2le` (the redundant arcs). Exit 1 if the graph is not reduced. `reduce` writes
the transitive reduction as a BV graph.

```sh
dag path-cover --graph B --out P [--explicit] [--window a..b]
dag frontier   --graph B --out P [--materialize] [--window a..b]
dag width      --graph B --out P [--method auto|minflow|closure] [--experimental-full] [--window a..b]
```
- `path-cover` computes the minimum vertex-disjoint path cover `MPC_d = n − ν`. On the PG it uses
  the closed form over blocks; otherwise it runs Hopcroft–Karp with a Kőnig certificate. With
  `--explicit` it also writes `P.next.u32le` (successor on the cover path, `NONE` at path ends)
  and reports the distribution of the path lengths (`chains`: count = MPC_d, longest path and its
  head, log2 histogram).
- `frontier` computes the largest prefix antichain `max_c |{u < c : succ(u) ⊆ [c, n)}|`, a lower
  bound on the width (on the PG: outputs created before `c` and still unspent). With
  `--materialize` it also writes `P.antichain.u32le`.
- `width` computes the exact width (maximum antichain = minimum chain cover, Dilworth), either by
  minimum flow on the split network or, on the PG, by a maximum-weight closure on the transaction
  graph. It writes `P.antichain.u32le`, and with `minflow` also `P.chains.u32le` and
  `P.chains_off.u64le` (CSR chains). The antichain is always re-checked. The full-PG closure needs
  `--experimental-full` (100+ GB); without it the full-PG answer is the bracket
  `[frontier, MPC_d]`. The JSON reports where the antichain lies (`antichain_ids`: size,
  smallest, largest and mean id) and, with `minflow`, the chain lengths (`chains`).

`frontier` also takes `--buckets FILE` and `--profile-step S`: the JSON then holds the curve
`c ↦ |A_c|` (on the PG, the number of outputs created and not yet spent at cut `c`, i.e. the UTXO
set), as one record per bucket or per range of `S` nodes: the value at the start and end of the
range, and its maximum inside with the cut where it occurs (`curve.by_bucket`, `curve.profile`).
`--cuts FILE` (sorted little-endian `u64` node ids, such as the time mapping's
`block_first_node.u64`) writes `P.cuts.u64le`, the value of `|A_c|` at each cut inside the
window: on the PG, the UTXO-set size at every blockchain block.

### Aggregate measures and time buckets

Every producing subcommand reports, in its JSON sidecar, the **distribution** of each per-node
(or per-block) array it computes, so a report never has to re-read multi-GB arrays:

| Field | Meaning |
|---|---|
| `count`, `absent` | entries with a value, and entries without one (0̄ of the semiring, a sentinel, a node outside the selection, NaN) |
| `sum`, `mean`, `min`, `argmin`, `max`, `argmax` | over the values; `arg*` is the smallest id attaining the extreme |
| `log2_hist` | `bins[0]` counts the value 0, `bins[k]` (k ≥ 1) the values in `[2^(k−1), 2^k)`, the last of 66 bins every value ≥ 2⁶⁴; `below_one` counts values in (0, 1) and `negative` the negative ones; trailing empty bins are omitted. The same convention as the dominator-tree histograms |
| `top` | the `--top K` largest values as `{id, value}` (value descending, then id ascending); per-block arrays add `node`, the block's first output |
| `by_bucket` | with `--buckets FILE`: the same fields for every bucket `{label, lo, hi, …}`, in file order, empty buckets included |
| `unbucketed` | with `--buckets`: entries whose node lies in no bucket |

**Ids** are absolute everywhere: node ids for per-node arrays (also in windowed runs), block ids
for per-block arrays (`argmax_node` and `top[].node` give the block's first output, the node used
for bucketing). The array is reduced over fixed chunks of 2²⁰ entries in a fixed order, so every
figure is bit-reproducible across runs and thread counts.

**Buckets.** `--buckets FILE` reads a TSV or CSV file with a header and one node range per line:
either the columns of `data/chunk_boundaries.tsv` (`chunk_id`, `start_date`,
`first_node`, `last_node_exclusive`; labels `2009H1` … `2022H2` are derived from `start_date`) or
the generic `label`, `lo`, `hi`. PG node ids are chronological, so the 28 half-year chunks used to
build the graph are contiguous id ranges and a node's half-year is its bucket. The chunk
boundaries are Europe/Rome midnights (as the graph was built); `chunk_boundaries_utc.tsv` gives UTC
boundaries, and any finer partition (months, blocks) can be written in the generic format from
`block_first_node.u64` and `block_time.csv`.

Which distributions each subcommand reports:

| Subcommand | JSON key | Arrays summarised |
|---|---|---|
| `blocks` | `results.distributions.{len,nin,arcs}` | per block: outputs, inputs, arcs `len·nin` |
| `dp` | `results.summary` | the values (key `log2` of the value for `extfloat`) |
| `cpm` | `results.distributions.{e,h,slack,critical_e}`; `results.summary.critical_width_log2_hist`, `max_critical_layer`, `max_critical_layer_nodes` | E, H, slack, E of the critical nodes (per bucket: critical nodes per half-year); widths of the critical layers |
| `count-paths` | `results.distributions.{forward_log2,backward_log2,through_log2}` | `log2` of the path counts (also with `--summary-only`) |
| `dom build`, `dom stats` | `results.{dom,pdom}.distributions.{depth,size,root_children}` | depth, subtree size, subtree size of the root children (top-k distinct dominating structures; per bucket: root children per half-year) |
| `reach-counts` | `results.{anc,desc,blk_anc,blk_desc}` | the HyperLogLog estimates |
| `frontier` | `results.curve.{by_bucket,profile}`, `results.antichain_ids` | `|A_c|` per range of cuts; members' ids with `--materialize` |
| `path-cover --explicit`, `width` | `results.chains`, `results.antichain_ids` | path or chain lengths; antichain ids |

```sh
dag aggregate --values FILE [--dtype D] [--lo A] [--blocks PREFIX] [--absent none|max|zero]
                 [--buckets FILE] [--top K] --out P
```
Computes the same distribution for any stored array, for example a dominator-depth array per
half-year, or an array written by another tool. The element type comes from the suffix (`u8`,
`u32le`, `u64le`, `i64le`, `f32le`, `f64le`) unless `--dtype` is given; entry `i` is node
`A + i`, or block `A + i` of `PREFIX.blk.*` with `--blocks`; `--absent max` ignores the largest
value of the type (`u32::MAX`: `NONE` and the 0̄ of `maxplus-u32`/`minplus-u32`). On the stored
arrays of a run it reproduces the run's own `by_bucket` figures exactly.

## Output formats

- **Arrays**: headerless raw little-endian arrays named `<prefix>.<field>.<dtype>`, with
  `<dtype>` one of `u8`, `u32le`, `u64le`, `i64le`, `u128le`, `f32le`, `f64le` and `extf` (an
  extended-exponent float `m·2^e`: 16 bytes, the f64 mantissa `m ∈ [1, 2)` followed by the i64
  exponent `e`; zero is `m = 0, e = 0`). Arrays of id pairs are named explicitly, for example
  `P.redundant.u32x2le` (`u`, `v` interleaved).
  Read them with `numpy.fromfile(path, dtype='<u4')` and so on. Windowed runs index per-node
  arrays by `v − a`; ids stored as values (predecessors, parents, chains, antichains) are absolute.
- **Sentinel**: `NONE = u32::MAX = 4294967295` means "no node" in id arrays. For the unsigned
  tropical semirings `u32::MAX` / `u64::MAX` is 0̄ ("unreached"), as in `longest.u32le`. In
  `maxmin-u32` outputs `u32::MAX` is the value 1̄ (unbounded capacity), not `NONE`.
- **JSON sidecar** `<prefix>.json`, written by every subcommand:
  `{ tool, version, argv, graph, n, m, window, threads, elapsed_s, peak_rss_bytes, rss, mem,
  phases_s, files: [{path, dtype, len, sentinel, meaning}], params, results }`.
  - `peak_rss_bytes` is the kernel's `VmHWM`. It **includes the resident pages of memory-mapped
    files** (the BV graph, mapped arrays), so on the PG it overstates the heap by up to the size of
    the mapped graph. `rss` gives `vm_hwm`, `vm_rss`, `rss_anon` (heap) and `rss_file` (mapped
    files) at the end of the run; the kernel keeps no peak of the heap alone. The heap figures in
    the tables below are the peak `RssAnon` sampled by the measurement wrapper.
  - `mem` is the memory-budget log: `max_bytes` (`--max-mem`), `estimate_peak_bytes` (the largest
    amount reserved at once, i.e. the subcommand's own estimate of its peak heap) and the
    `reservations` in order (`what`, `bytes`, `total`).
  - `phases_s` is the wall time per phase, with the same keys everywhere: `open` (graph and
    index loading, argument checks), `blocks` (building the BlockIndex), `kernel`, `summary`
    (aggregate measures), `write`, `save`, `certify`, `compare`, `verify`. Recorded by `blocks`,
    `dp`, `certify`, `cpm`, `count-paths`, `dom build`, `dom verify`, `reach-counts`,
    `path-cover`, `frontier` and `aggregate`.
  - `results` holds the subcommand's report; the table below lists its main keys.

| Subcommand | `results` keys (units; ids absolute unless noted) |
|---|---|
| `topo-check` | `n`, `m`, `identity_topological`, `sources`, `sinks`, `first_violation` |
| `blocks` | `report` (`BlockReport`: `nb`, `sources`, `sinks`, `covered`, `sum_len`, `sum_nin`, `sum_len_times_nin`, `max_len`, `max_nin`), `distributions`, `build_s`, `save_s`, `comparison` (mismatch counts, first mismatching block/node) |
| `dp` | `summary` (`ValueSummary`: `nodes`, `zero_nodes`, `max`/`argmax`, `min`/`argmin`, `mean` of the real keys, `log2_hist`, `top`, `by_bucket`), `kernel_s`, `write_s`, `certify` (`CertReport`), `certify_s`, `comparison` (`compared`, `mismatches`, `first_mismatch`) |
| `certify` | `CertReport`: `checked`, `violations`, `pred_violations`, `max_rel_err`, `first_violation` |
| `cpm` | `summary` (`CpmSummary`: `t` in arcs or duration units, `critical_nodes`, `critical_arcs`, `width1_layers`, `max_critical_width`, `max_critical_layer`, `max_critical_layer_nodes`, `critical_width_log2_hist` over the layers `E = 0..=T` with bin 0 = layers without a critical node, `longest_path_end`), `distributions`, `cpm_s`, `longest_path` (`nodes`, `start`, `end`), `comparisons` |
| `count-paths` | `summary` (`PathSummary`: per direction `total_log2`, `total_mod61`, `max_log2`, `argmax`, `mean_log2`, `log2_hist` of `log2` counts with bin 0 = count 1 and bin k = `log2` count in `[2^(k−1), 2^k)`, `above_2_64`, `above_2_128`; `duality`; `through`: `argmax`, `max_log2`, `max_share` = fraction of all source-to-sink paths through the argmax), `distributions`, `total_log2`, `total_mod61`, `saturated_nodes`, `count_s`, `write_s` |
| `dom build` | per tree `dom`/`pdom`: `stats` (`DomStats`: `root_children`, `root_children_cut`, `root_children_dominated`, `cut_nodes`, `max_depth`, `argmax_depth`, `mean_depth`, `depth_log2_hist`, `size_log2_hist`, `top` by subtree size), `distributions`, `nca` (`NcaStats`: `calls`, `top_exits`, `jump_fallbacks`, `walk_steps_log2_hist` in the same bins), `build_s`, `sizes_pre_s`, `write_s`, `summary_s`, `depth_bound` |
| `dom stats` | per tree: the `DomStats` fields plus `distributions` (file `P.stats.json`) |
| `dom verify` | `report` (`VerifyReport`: `ok`, `v0_fail`, `v1_fail`, `v2_fail`, `first_failure`) |
| `reach-counts` | `anc`, `desc`, `blk_anc`, `blk_desc` (`Summary`: `len`, `sum`, `mean`, `mean_over_window` = mean / window size, `max`, `argmax` (node or block id), `argmax_node`, `zeros`, `log2_hist`, `top`, `by_bucket`), `sources_check`, `mode`, `peak_live`, `blocks` (block window), `passes`, `timings_s` |
| `reach-index build` | `stats` (`n_dag`, `max_lvl` = longest path of the indexed DAG in arcs, `chain_lens`, `chain_gains`, `covered`, `label_bytes`, pass times), `build_s`, `save_s` |
| `reach-index query` | `queries`, `reachable`, `deciders` (queries decided by each filter), `query_s`, `qps` |
| `reach-index bench` | `n_dag`, `sources`, `cones_s`, `workloads[]`: `name`, `queries`, `positives`, `disagreements`, `deciders`, `label_share`, `search_share`, `window_scan_share`, `latency_ns` (sequential latencies: percentiles and `max`), `par_max_query_s` (slowest query under parallel load), `throughput_qps`, `seq_s`, `par_s` |
| `tr verify` / `tr reduce` | `report` (`ReductionReport`: `is_transitively_reduced`, `certificate`, `line_digraph`, `arcs`, `redundant`, `h1_violations`, `h3_failing_nodes`, `exact_checked_arcs`), `line_digraph_check`; reduce adds `output`, `nodes`, `arcs`, `redundant` |
| `path-cover` | `method`, `n`, `matching`, `mpc_d`, `closed_form_s` or `hk_s`, `cross_check`, `chains` (with `--explicit`) |
| `frontier` | `method`, `n`, `size`, `cut`, `frontier_s`, `certificate_ok`, `antichain_ids`, `curve` |
| `width` | `method`, `exact`, `width` (or `lower`/`upper` for the bracket), `certificate_ok`, `cross_check`, `antichain_ids`, `chains` (min-flow) |
| `aggregate` | the distribution itself (`count` … `top`, `by_bucket`, `unbucketed`) |

These conventions match the repository's other tools (`analysis_path`,
`pg_analysis/results/*.u32le`), so arrays can be compared with `cmp`.

## Validation

**Unit and property tests.** Every test runs seeded generators (random DAGs, chains, stars,
ladders, diamonds, and `mini_pg`, a small UTXO simulation with the PG's structure) against
brute-force oracles (bitset closures, u128 path enumeration, vertex-removal dominators, exhaustive
width and matching), both in memory and after a BV round trip:

```sh
cargo test --offline --profile release-test            # everything
cargo test --offline --profile release-test --test dp  # one suite
cargo test --offline --profile release-test --doc
```

| Suite | Checks |
|---|---|
| `core`, `cli_core` | sweep engine (every chunk size and window), identity check, relabelling, `BlockIndex` against ground truth and anomaly injection, I/O round trips |
| `semiring` | semiring axioms on random triples, exact identities, saturation and modular arithmetic against u128 |
| `dp`, `cli_dp`, `dp_instances` | push == pull == blocks bit-identity, path enumeration, Bellman–Ford, duality, certification accepts every result and rejects perturbations |
| `cpm`, `paths` | CPM invariants and critical sets; path counts against the u128 oracle |
| `dom`, `cli_dom` | brute-force dominators on 500 DAGs, blocks == generic, budget independence, verifier rejects mutated trees |
| `reach_counts` | sketches bit-identical to sketches of the exact sets; accuracy over 30 seeds |
| `reach_index`, `cli_reach` | `reach == closure` on all pairs, forced fallbacks, save/load |
| `reduce` | bitset == search == Floyd–Warshall, `closure(TR) == closure(G)`, minimality |
| `width`, `cli_width` | Hopcroft–Karp and Kőnig, MPC_d closed form, frontier ≤ width ≤ MPC_d, min flow == closure == brute force |
| `cnr2000` | two DAG-ifications of `cnr-2000` from a webgraph-rs checkout next to this repository, or `CNR2000_BASENAME` (skipped if absent) |
| `full_pg` | full-scale checks (ignored by default, see below) |

**Gate.** Before committing a change, run in order (CI runs the same steps, see
`.github/workflows/ci.yml`):

```sh
cargo fmt    --all --check
cargo check  --offline --all-targets
cargo test   --offline --profile release-test
cargo clippy --offline --all-targets -- -D warnings
RUSTDOCFLAGS=-Dwarnings cargo doc --offline --no-deps
```

**Full Payment Graph.** `scripts/validate_pg.sh` runs the release binary on
`/data/bitcoin/2022/pg/pg`, one subcommand at a time, and checks every JSON report (and, where the
table asks for byte identity, the per-node arrays) against the exact results already in
`/data/bitcoin/2022/pg_analysis/results`. `tests/full_pg.rs` (ignored by default) checks the same
rows through the library API instead of the CLI:

```sh
cargo build --offline --release
VALIDATE_OUT=/scratch/validation scripts/validate_pg.sh          # every step, then the checks
scripts/validate_pg.sh --list                                    # step names
VALIDATE_OUT=/scratch/validation scripts/validate_pg.sh dom check # one step, then re-check all
PG_BLOCKS=/scratch/validation/blocks \
  cargo test --offline --release --test full_pg -- --ignored --test-threads 1
```

The script reads `PG_BASENAME` (default `/data/bitcoin/2022/pg/pg`), `PG_ANALYSIS_DIR`, `DAG_BIN`
(the binary, default `target/release/dag`), `VALIDATE_OUT` (default `./validate_pg.out`),
`DAG_BIN_OPTS` (extra global options), `PG_BUCKETS` (optional: `data/chunk_boundaries.tsv`, passed as `--buckets` to every producing step, which adds the
per-half-year rows to the checks) and `MEASURE`, an optional wrapper prefix to which it appends
`-n <step> -- <command>` (for example a resource-measurement script). The first step writes the
`BlockIndex` to `$VALIDATE_OUT/blocks.blk.*` and the later steps load it with `--blocks PREFIX`, so
the index is built once. The `check` step prints one TSV row per check (check, expected, observed,
PASS/FAIL, wall seconds, peak RSS), also written to `$VALIDATE_OUT/validation.tsv`, and exits 1 if
any check fails. `full_pg` reads `PG_BASENAME`, `PG_ANALYSIS_DIR` and the optional `PG_BLOCKS`
(a saved index; otherwise the tests build it).

Results of the last full run (2026-10-13, 112 threads, shared host, `PG_BUCKETS` = the 28
half-year chunks and `PG_CUTS` = the 762,262 blockchain-block starts of the time mapping; `wall` and
`heap` = peak anonymous RSS as sampled by the measurement wrapper; all 94 checks passed). Rows not
affected by the last changes (`topo-check`, `tr verify`, `dom verify`, `reach-index build` and
`bench`) are from the previous round of the same day.

| Check | Expected on the 2022 PG | Wall | Heap |
|---|---|---|---|
| `topo-check` | n = 2,181,021,971, m = 8,639,773,499; identity is topological; 2,883,482 sources; 136,124,643 sinks | 8.9 s | 2.3 GiB |
| `blocks` | nb = 777,851,179; Σ len = 2,178,138,489; Σ nin = 2,044,897,328; Σ len·nin = m; max len 13,107; max nin 20,000; `first`, `len`, `nin`, `create` identical to `bicliques/blocks.u32x3` and `blk_start.u32`; block-size distributions cover all blocks, 28 half-year buckets | 88 s | 26.1 GiB |
| `dp maxplus-u32` forward, unit | max 3,559,224 at 2,181,015,947; mean 2,397,482.094; byte-identical to `chains/longest.u32le`; the pred chain of 2,181,015,947 ends at source 84,564 after 3,559,224 arcs; histogram covers all n nodes, top-1 = argmax | 97 s | 49.3 GiB |
| `certify` (standalone, values + preds) | 0 violations | 9.6 s | 18.4 GiB |
| `dp minplus-u32` forward, unit | max 25,348 at 479,412,379; mean 76.34 | 65 s | 41.1 GiB |
| `dp maxplus-u32` backward, unit | byte-identical to `layers/height.u32le`; max 3,559,224 at 84,564 | 71 s | 37.2 GiB |
| `cpm --critical-arcs --slack --path` | T = 3,559,224; 5,459,477 critical nodes (= nodes of slack 0); E and H identical to the two arrays above; one longest path of 3,559,225 nodes from 84,564 to 2,181,015,947; widest critical layer 102 nodes at E = 970,289 | 95 s | 59.3 GiB |
| `count-paths` (both directions) | totals equal mod 2⁶¹ − 1; ExtFloat totals within 1e-6 (observed 1.40e-12; 2^1,942,147.13 paths); top-k max = summary max | 183 s | 163.8 GiB |
| `tr verify` | reduced, certificate `blocks` | 5.0 s | 26.2 GiB |
| `tr verify --blocks none` | 0 H3-failing nodes | 9.4 s | 17.3 GiB |
| `tr reduce --window 0..10⁸` | 0 redundant arcs | 50 s | 26.2 GiB |
| `path-cover` | matching 1,016,828,057; MPC_d 1,164,193,914 | 6.0 s | 29.1 GiB |
| `frontier` | 136,188,373 at cut 2,180,649,454; the maximum of the curve over 10⁷-node ranges is the frontier; `|A_n|` at the end of 2022H2 = 136,124,643 sinks; at the 762,262 blockchain-block starts the largest UTXO set is 136,187,726 | 12 s | 29.1 GiB |
| `dom build --sizes --depth-bound` | root children ≥ sources (642,535,984) and ≥ sinks for the post-dominators (1,584,028,054); max depth 25,344 ≤ 25,349; root-child distributions sum to n | 196 s | 113.2 GiB |
| `dom stats --top 100` | same root children as `dom build`; 100-row TSVs | 12.5 s | 24.7 GiB |
| `dom verify` (each tree) | V0 = V1 = V2 = 0 | 68 s / 61 s | 74.5 GiB |
| `reach-counts --log2m 10` | peak live 5,139,210 ≤ 5,139,209 + 1 per pass; anc = 0 on all 2,883,482 sources; mean desc/N 0.4217 (0.4195 ± 3σ); absolute argmax ids | 1,183 s | 53.2 GiB |
| `reach-index build --chains 4` | longest path of the quotient 3,559,223 = 3,559,224 − 1; spine of 3,559,224 blocks | 357 s | 61.1 GiB |
| `reach-index bench` | 19,960 queries (uniform + balanced) agree with exact cone labels; slowest query 0.217 s sequential (0.084 s under parallel load) | 44 s | 67.2 GiB |
| `reach-index query` (15,001 pairs) | 5,000 arcs answered 1, the same pairs reversed answered 0, the spine (84,564 → 2,181,015,947) answered 1; 304,000 queries/s (the wall time is dominated by loading the index) | 123 s | 61.6 GiB |
| `width --window 0..1000000` | closure width == min-flow width (427,262), both certified | 15.8 s | 26.5 GiB |
| `path-cover --explicit` on 3 windows of 10⁶ nodes | Hopcroft–Karp == closed form, König and explicit cover checked; number of chains = MPC_d | 4.2 s each | 26.2 GiB |
| per-half-year rows (`dp` ×3, `cpm`, `count-paths`, `dom`, `reach-counts`, `blocks`) | 28 buckets 2009H1 … 2022H2 whose counts add up to n (or nb), 0 unbucketed | — | — |

For a quick check on a prefix, add `--window 0..100000000` (about 4.6 % of the nodes). Windows
are convex, so the results are exact for the induced subgraph. Every JSON sidecar records wall
time and peak RSS.

## Memory and time

Measured peak heap at full PG scale (besides the graph files in the page cache) is in the table
above. In terms of the components: `BlockIndex` 26.8 GB; a `u32` DP over blocks about 15 GB plus
blocks (49 GiB with `--pred --certify`); CPM about 27 GB plus blocks; dominator trees about 38 GB
per tree (TNode 34.9 GB plus 3.1 GB of block memo), about 55 GB with `--sizes --pre`, and twice
that when both trees build concurrently (110 GiB measured); `count-paths` 32 B/node per direction
at peak and 16 B/node kept (164 GiB measured for `--dir both`); dominator verification about
75 GiB; HyperLogLog counts about 27 GB plus blocks; the reachability index with 4 chains 37.3 GB
plus blocks, and `reach-index` queries add `Scratch::max_bytes` of scratch per query thread
(about `3·n_dag/8` bytes in the worst case; the query threads are capped by `--max-mem`). Every
subcommand checks its own estimate against `--max-mem` (`MemBudget`) before allocating. A
parallel decode of the full PG takes 2.6 s (forward) and 5.4 s (transpose) on 112 threads. On the
PG, `--blocks auto` avoids decoding entirely for most subcommands.

System time. Short parallel phases separated by sequential work make idle rayon workers spin
(`sched_yield`); persisted arrays are therefore read in one parallel phase (8 concurrent streams)
and written in 128 MiB conversion batches, which cut loading the 26.8 GB `BlockIndex` from 25 s
to 5 s (and its system time from 1,330 s to 90 s). The remaining large system time of
`count-paths` (3,100–5,200 s) comes from page reclaim and synchronous THP compaction while it
allocates 164 GB on a host whose RAM is full of page cache: with `DAG_BIN_NO_THP=1` the same run
had no compaction stalls and 1,680 s of system time, at the same wall time (174 s).

## Deferred work

These items are deliberately left out, each for a stated reason:

- **DOACROSS / synchronisation-free pipelines and level-synchronous scheduling** for the DP: the PG
  has 3.56 M longest-path levels with a median of 224 nodes, so the speedup is unproven; a
  makespan simulation should come first.
- **Forest contraction and parallel prefix formulations** of the DP: research-grade and not needed
  at current speeds.
- **Low-high order certificates** and **chunk-local wavefronts** for dominators: the independent
  V0–V2 verifier already certifies the trees.
- **GRAIL / O'Reach DFS orders, HyperLogLog dominance filters, supportive masks, bidirectional
  search** for the reachability index: to add if `reach-index bench` shows a high share of
  `Search` or `WindowScan` decisions.
- **Chain-compressed transitive reduction** (Simon / Jagadish), **parallel Hopcroft–Karp**,
  **pseudoflow**: the PG needs none of them (it is already reduced, and the cover has a closed
  form).
- A dedicated **kernel benchmark** subcommand: prefix windows (`--window 0..10^8`) serve the same
  purpose.
- **Counting linear extensions or antichains** (#P-complete, Brightwell and Winkler 1991), **DAG
  LCA sets**, **BDD/ZDD** encodings.

## Repository layout

```text
Cargo.toml, Cargo.lock    package; webgraph-rs pinned by commit, everything else by the lock
src/lib.rs                module tree; crate documentation = docs/crate.md
src/main.rs, src/cli/     the dag binary (one file per subcommand group)
src/*.rs, src/*/          core (sweep, graph, topo, blocks, io, ...) and the four families
docs/crate.md             crate-level documentation (PG model, theorems, conventions)
docs/references.md        bibliography (rendered as the `references` module)
tests/                    integration and property tests, one suite per area
scripts/validate_pg.sh    full-scale validation on the 2022 Payment Graph
data/chunk_boundaries.tsv the 28 half-year node-id buckets of the 2022 Payment Graph (--buckets)
.github/workflows/ci.yml  format, clippy, tests and docs on every push and pull request
```

## License

MIT, see [LICENSE](LICENSE).
