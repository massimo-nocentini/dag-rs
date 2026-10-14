DAG algorithms for the Bitcoin Payment Graph and any webgraph DAG.

`dag` computes, on directed acyclic graphs stored in the BV format of WebGraph
([\[BV04\]](crate::references#bv04)), four families of exact (or certified approximate) analyses:

1. **semiring dynamic programming**: longest and shortest paths, critical path method, path
   counting, bottleneck, probability and Viterbi paths ([`semiring`](crate::semiring),
   [`dp`](crate::dp), [`cpm`](crate::cpm), [`paths`](crate::paths));
2. **dominator and post-dominator trees** ([`dom`](crate::dom));
3. **reachability**: approximate ancestor and descendant counts, and an exact reachability index
   ([`reach`](crate::reach));
4. **transitive reduction, path covers and width** ([`reduce`](crate::reduce),
   [`width`](crate::width)).

The crate was built for the 2022 **Payment Graph** (PG, 2,181,021,971 nodes and 8,639,773,499
arcs), so every pass streams the graph and every large array is a flat array indexed by node or by
block. All the algorithms also run on any DAG whose ids are a topological order, and windows of it.
The command-line front end is the `dag` binary ([`cli`](crate::cli)); the bibliography is in
[`references`](crate::references).

# The Payment Graph model

The PG is built by the repository's builder ([\[PGB\]](crate::references#pgb)) from the Bitcoin
block chain:

- a **node** is a transaction output; outputs are numbered in chain order, and the outputs of one
  transaction get consecutive ids;
- an **arc** `(u, v)` means that output `u` is an input of the transaction that created `v`. A
  transaction `T` with `i_T` resolved inputs and `o_T` outputs therefore contributes the complete
  bipartite graph (biclique) `inputs(T) × outputs(T)`.

Call a spending transaction with at least one output a **block** `B`: its outputs are an id
interval `O(B) = [first(B), first(B) + len(B))` and its inputs are `I(B)`. Since an output is spent
at most once (Bitcoin consensus, [\[CVE-2018-17144\]](crate::references#cve-2018-17144)), the arc
set is the **disjoint union of the bicliques** `I(B) × O(B)`. Equivalently, the PG is the *line
digraph* of the transaction multigraph TG (vertices are transactions, plus one fresh sink vertex per
unspent output; one edge per output from its creator to its spender, or to its sink vertex if it
is unspent) ([\[HN60\]](crate::references#hn60)). By Heuchenne's characterization
([\[Heu64\]](crate::references#heu64), [\[BB21\]](crate::references#bb21)) a digraph is a line
digraph if and only if any two successor sets are identical or disjoint.

[`BlockIndex`](crate::blocks::BlockIndex) stores this decomposition as five flat `u32` arrays:
`create[v]` (the block that created `v`) and `spend[u]` (the block that spends `u`) per node, and
`first`, `len` and `nin` (number of inputs) per block. Building it verifies, exactly, with one
parallel decoding pass of the graph (plus two array scans), the three properties that every PG fast path relies on:

- **P1** every successor list is an interval of consecutive ids;
- **P2** any two successor lists are identical or disjoint;
- **P3** every successor list starts after its node (`first(spend(u)) > u`).

Measured on the 2022 PG: 777,851,179 blocks, 2,883,482 sources (coinbase outputs and outputs of
transactions without resolved inputs), 136,124,643 sinks (outputs unspent within the data),
maximum block size 13,107 outputs, maximum in-degree 20,000, and `Σ len·nin = m`. With the index,
the block DP, both dominator trees, the HyperLogLog counts, the reachability labels, the minimum
path cover, the frontier and the closure width are **array scans with no BV decoding**. A graph
that is not a disjoint union of bicliques (in general, any graph that is not a line digraph with
interval successor sets) makes the build fail with
[`DagError::NotBiclique`](crate::DagError::NotBiclique), and the subcommands fall back to the generic kernels.

# The topological-id assumption

**Every kernel assumes that the identity is a topological order**: `u < v` for every arc
`(u, v)`. On the PG this holds by construction (ids are chronological and a transaction can only
spend earlier outputs), and [`topo::check_identity`](crate::topo::check_identity) verifies it in
one parallel scan (zero violations on the 2022 PG). A graph whose ids are not topological is
relabelled once with `dag toposort` ([`topo`](crate::topo); topological sorting is
[\[Kah62\]](crate::references#kah62), the depth-first variant
[\[CLRS22\]](crate::references#clrs22)). The relabelling writes the permuted graph together with
the permutation in both directions, and results are in the new ids and can be mapped back
with the stored order `<out>.order.u32le` (new → old).

The assumption has three consequences used throughout the crate:

- **One pass is enough.** A forward pass in increasing id order sees every predecessor of `v`
  before `v`, and a backward pass in decreasing order sees every successor of `u` before `u`.
  There is no queue, no in-degree counter and no permutation in any kernel.
- **Every sweep is also a check.** The sweep engine ([`sweep`](crate::sweep)) decodes chunks of
  the BV graph in parallel and hands them to a sequential consumer in order; while decoding it
  rejects any arc `(u, v)` with `v ≤ u` as [`DagError::NotTopological`](crate::DagError::NotTopological).
- **Windows are convex.** Ids strictly increase along paths, so a path between two nodes of a half-open
  id range `[lo, hi)` never leaves the range. On the PG such a range is a **time window**, and
  every result on a window ([`Window`](crate::Window), `--window a..b`) is an exact statement about
  the induced subgraph. Per-node outputs of a windowed run are indexed by `v − lo`; node ids
  stored as values (predecessors, dominators, chains, antichains) are absolute.

Parallelism comes from decoding, not from the dependency structure: the PG has a longest path of
3,559,224 arcs and a median longest-path layer of 224 nodes, so level-synchronous schedules do not
pay off. The engine decodes up to `--decoders` chunks in parallel (2.6 s for the full PG instead
of about 110 s sequentially) while one thread consumes them. Independent passes (the two CPM
passes, the two count directions, dominators and post-dominators) run concurrently.

# The PG transitive-reduction lemma

**Lemma.** Let `G` be a digraph on `{0, …, n − 1}` such that (1) `u < v` for every arc
`(u, v)`, (2) every non-empty successor set is an integer interval `N⁺(u) = [a_u, b_u]`, and
(3) any two successor sets are identical or disjoint. Then `G` is transitively reduced: no arc is
implied by a path of length at least 2.

*Proof.* Suppose `(u, v)` is an arc and `u = x₀ → x₁ → … → x_k = v` with `k ≥ 2`. Let
`R = N⁺(u) = [a, b]`. Then `x₁, v ∈ R`, and by (1) ids strictly increase along the path, so
`a ≤ x₁ ≤ x_{k−1} < v ≤ b`, hence `x_{k−1} ∈ R`. Now `v ∈ N⁺(x_{k−1}) ∩ R`, so by (3)
`N⁺(x_{k−1}) = R` and `a_{x_{k−1}} = a ≤ x_{k−1}`. By (1) every successor of `x_{k−1}` is larger
than `x_{k−1}`, so `a_{x_{k−1}} > x_{k−1}`, a contradiction. ∎

Contiguity (2) is not needed: (1) and (3) alone suffice. In the same setting,
`v ∈ N⁺(x_{k−1}) ∩ N⁺(u)`, so (3) gives `N⁺(x_{k−1}) = N⁺(u) ∋ x₁`. Then (1) gives
`x₁ > x_{k−1}`, while `x₁` comes no later than `x_{k−1}` on the path, so `x₁ ≤ x_{k−1}`. This is
the form the crate relies on, and it is the line-digraph lemma in disguise: the line digraph of an
acyclic multigraph is transitively reduced, because a redundant arc would close a cycle in the
underlying graph. Transitive reductions of DAGs are unique
([\[AGU72\]](crate::references#agu72)).

All three hypotheses were verified on all 2.18·10⁹ nodes and 8.64·10⁹ arcs. Hence **the 2022 PG
equals its own transitive reduction**, and so does every time window, because induced subgraphs
inherit (1)–(3). A successful [`BlockIndex`](crate::blocks::BlockIndex) build is therefore a
certificate of transitive reduction ([`reduce`](crate::reduce) reports it as certificate
`"blocks"`). Two corollaries are used below:

- the Hasse diagram (cover relation) of the reachability order is `G` itself: every arc is a
  cover pair, and maximal chains of the order can be taken as paths in `G`;
- every non-source node of a block has the same predecessor set, so every forward quantity that
  depends only on the predecessor set (DP values, `idom`, ancestor sets) is **constant on a
  block**, and the backward quantities of the inputs of a block are folds over one interval.

Independently of the reduction, sinks of any DAG are pairwise incomparable; the 136,124,643 sinks
therefore give the PG **width at least 136,124,643**, so every chain cover is huge and chain
covers are computed on windows only.

# Family 1: semiring dynamic programming

A semiring `(E, ⊕, ⊗, 0̄, 1̄)` turns a path problem into one topological pass
([\[Leh77\]](crate::references#leh77), [\[Moh02\]](crate::references#moh02),
[\[GM08\]](crate::references#gm08), [\[Goo99\]](crate::references#goo99),
[\[Hua08\]](crate::references#hua08)). With arc weights `w(u, v)` and initial terms `b(v)`, the
forward value

```text
F(v) = b(v) ⊕ ⨁_{u ∈ pred(v)} F(u) ⊗ w(u, v)
```

is the ⊕ over all paths `s ⇝ v` (including the empty path) of `b(s)` ⊗ the product of the
path's arc weights. On a DAG the system has
a unique solution, and evaluating it in topological order computes it in `O(n + m)` semiring
operations; no closure (Kleene star) is needed, so only the semiring axioms are used. The backward
value `B(u) = c(u) ⊕ ⨁_{v ∈ succ(u)} w(u, v) ⊗ B(v)` is symmetric, and the duality
`⨁_v F(v) ⊗ c(v) = ⨁_u b(u) ⊗ B(u)` holds without commutativity of ⊗. The instances in
[`semiring`](crate::semiring) include (max, +) longest paths (Bellman's recurrence on a DAG,
[\[Bel58\]](crate::references#bel58), [\[CLRS22\]](crate::references#clrs22)), (min, +)
shortest paths, Boolean reachability, saturating and modular path counting, an extended-exponent
float, log-sum-exp, probability, Viterbi ([\[Vit67\]](crate::references#vit67)) and bottleneck
semirings, and componentwise products.

**Canonical evaluation order.** Every kernel folds predecessors (successors) in ascending id order
starting from 0̄, applies the node factor, and adds the initial term last ([`dp`](crate::dp)).
Hence the push, pull and block kernels are **bit-identical for every semiring, floats included**,
and the test suite checks equality, not tolerance.

**Block factorization (PG fast path).** For separable weights `w(u, v) = α(u) ⊗ β(v)`,
right-distributivity gives, for every output `v` of a block `B`,

```text
F(v) = (⨁_{u ∈ I(B)} F(u) ⊗ α(u)) ⊗ β(v) ⊕ b(v),
```

and every input of `B` precedes `first(B)` (P3). One accumulator per block, finalized when the
sweep reaches `first(B)`, replaces the `m` arc relaxations by `n` node steps: about
4.2·10⁹ operations instead of 1.7·10¹⁰ on the PG, with no BV decoding. The backward pass folds each
output interval once.

**Critical path method** ([`cpm`](crate::cpm); [\[KW59\]](crate::references#kw59),
[\[MRCF59\]](crate::references#mrcf59)). With `E` the forward (max, +) value (earliest start) and
`H` the backward one (tail length), `T = max E = max H`, the slack is `T − E(v) − H(v)`, and `v`
is critical iff its slack is 0. With unit arc lengths, `E` reproduces the repository's
longest-path arrays byte for byte (`T = 3,559,224` on the 2022 PG).

**Path counting** ([`paths`](crate::paths)). Path counts on the PG overflow every machine
integer, so they are computed in one pass per direction as a product of an extended-exponent float
(the magnitude) and residues modulo the Mersenne prime 2⁶¹ − 1 (an exact fingerprint). The forward
and backward totals must agree exactly modulo 2⁶¹ − 1 and to a stated relative tolerance in
floating point ([\[Hig93\]](crate::references#hig93)).

**Certification.** `dag certify` re-evaluates every fixpoint equation in an independent parallel
scan of the transpose (or of the graph, backward), exactly for exact and selective semirings and
within a per-node bound `c·(k + 2)·u` for float sums of `k` terms. It never uses the block index,
so it does not share a data path with the block kernel.

# Family 2: dominators and post-dominators

Add a virtual root `r` with an arc to every source. A node `d` dominates `v` if every `r`–`v` path
contains `d` ([\[LM69\]](crate::references#lm69), [\[Pro59\]](crate::references#pro59),
[\[All70\]](crate::references#all70)); post-dominators are dominators of the reversed graph with a
virtual sink ([\[CFRWZ91\]](crate::references#cfrwz91)). General-graph algorithms are
[\[LT79\]](crate::references#lt79), [\[AHLT99\]](crate::references#ahlt99) and
[\[BGKRTW08\]](crate::references#bgkrtw08). On a DAG the problem is simpler:

**Theorem (DAG dominators).** If the flow graph is acyclic and all predecessors of `v ≠ r` are
already in the dominator tree `D`, then `idom(v) = NCA_D(pred(v))` (with `pred(v) = {r}` for a
source).

*Proof.* `Dom(v) = {v} ∪ ⋂_{p ∈ pred(v)} Dom(p)`, the data-flow characterization, has a unique
solution on a DAG and is evaluated in topological order. `Dom(p)` is the root path of `p` in `D`,
and the intersection of root paths is the root path of their nearest common ancestor. Finally `v`
is in no `Dom(p)`, because `v` dominating a predecessor would close a cycle. ∎

This is the iterative algorithm of Cooper, Harvey and Kennedy
([\[CHK01\]](crate::references#chk01)) after a single sweep, the acyclic case of
[\[RR94\]](crate::references#rr94) and [\[Och83\]](crate::references#och83); see
[\[GTW06\]](crate::references#gtw06) for an experimental comparison. Nearest common ancestors use
a root-child test (two reads, which settles most PG queries at the root), then a bounded walk
guided by depth, then the skew-binary jump pointers of Myers
([\[Mye83\]](crate::references#mye83); see also [\[BF00\]](crate::references#bf00)).

**PG fast path.** `idom` is constant on a block (identical predecessor sets), so the builder folds
one NCA per input into a per-block accumulator and copies the result to the other outputs. Every
strict dominator of `v` lies on every root path, including a shortest one, so the tree depth is at
most `dmin(v) + 1 ≤ 25,349`. Post-dominators fold the NCA over the output interval of `spend(u)`,
once per block.

**Verification.** `dag dom verify` checks a tree independently of the builder in one parallel
scan of the forward graph. Every parent must precede its child (V0); `idom(v)` must be an ancestor
of every predecessor (V1); and the predecessors must not all lie in one child subtree of
`idom(v)` (V2), tested with preorder intervals. V0–V2 state that `t(v) = NCA_T(pred(v))` for all
`v`, and an induction in topological order then gives `T = D`. Low-high orders
([\[GT15\]](crate::references#gt15)) are background only. Dominator-tree subtree sizes measure
"retained" mass, as in [\[Mit06\]](crate::references#mit06) and
[\[AB04\]](crate::references#ab04); on the PG the subtree of a coinbase output counts the outputs
whose entire history comes from it.

# Family 3: reachability

**Approximate counts** ([\[FFGM07\]](crate::references#ffgm07),
[\[QKT16\]](crate::references#qkt16), [\[HNH13\]](crate::references#hnh13)). Each node gets a
HyperLogLog sketch with `2^p` registers. Following Cohen's size-estimation framework
([\[Coh97\]](crate::references#coh97)) and HyperANF ([\[BRV11\]](crate::references#brv11),
[\[BV13\]](crate::references#bv13)), but in a single topological pass instead of one iteration
per distance:

**Lemma (exact propagation).** With a fixed hash function `h`, the recurrence
`S(v) = ⨆_{u ∈ pred(v)} (add(u) ⊔ S(u))`, evaluated in topological order, gives exactly the
sketch of the ancestor set of `v`. The proof is an induction, since the register-wise maximum is
associative, commutative and idempotent. The same holds for descendants in reverse order.

So each estimate has the error of a single sketch (relative standard deviation about
`1.04/√(2^p)`). Errors are correlated across nodes because they share `h`, so aggregate statistics
average independent seeds. Estimates use the LogLog-β estimator of `card-est-array`. On the PG the
pass runs on blocks: a block's sketch is opened at its first input and released after its last
use, so only the live sketches stay in memory (peak about 5.1 M of 7.8·10⁸).

**Block quotient.** Let `Q` have one vertex per block and an arc `X → Y` when some output of `X`
is an input of `Y`. Then, for `u ≠ v`, `u ⇝ v` in the PG iff `spend(u)` and `create(v)` exist and
`spend(u) ⇝ create(v)` in `Q` (reflexively), because every block is a complete biclique. Block ids
are a topological order of `Q`, which has about 2.8 times fewer vertices than the PG has nodes.

**Exact index "ChronoChain"** ([`reach`](crate::reach)). On `Q` (PG) or on `G` (any DAG), per
vertex labels give fast negative and positive answers: chronology (`u > v` means no), longest-path
levels from both ends, the extreme reachable and reaching ids, and the positions on `k` chains
(the longest path first, then paths of maximum new coverage, in the spirit of the chain
compression of [\[ABJ89\]](crate::references#abj89) and [\[Jag90\]](crate::references#jag90)).
Pruned search guided by labels follows GRAIL and O'Reach ([\[YCZ10\]](crate::references#ycz10),
[\[HST22\]](crate::references#hst22); survey [\[ZBO23\]](crate::references#zbo23)). Each filter is
sound, because a missing chain position is encoded as ±∞. Undecided queries run a depth-first
search pruned by the filters with a visit budget, and then an exact sequential scan of the id
range `[X, Y]` over a [`DirtyBitset`](crate::bitset::DirtyBitset). Every answer is exact, and
memory per query thread is bounded by `n/8` bytes.

# Family 4: transitive reduction, path covers and width

**Transitive reduction** ([`reduce`](crate::reduce); [\[AGU72\]](crate::references#agu72),
[\[GK79\]](crate::references#gk79), [\[Sim88\]](crate::references#sim88),
[\[HMR93\]](crate::references#hmr93)). On the PG and its windows the lemma above settles the
question. For any other topologically numbered DAG the order-based certificate H3 is checked
first: if `minsucc(w) > maxsucc(u)` for every arc `(u, w)`, then no path of length at least 2 can
end inside `succ(u)`, so the graph is reduced. Otherwise only the arcs `(u, v)` with
`v ≥ t_u = min{minsucc(w) : w ∈ succ(u), minsucc(w) ≤ maxsucc(u)}` can be redundant, and a
bounded search decides each of them exactly. Small windows use a bitset closure.

**Minimum vertex-disjoint path cover**
([\[BG77\]](crate::references#bg77), [\[HK73\]](crate::references#hk73)). `MPC_d = n − ν`, where
`ν` is a maximum matching of the split bipartite graph (out-copies against in-copies). On the PG
the split graph is the disjoint union of the bicliques `K_{i_B, o_B}`, whose maximum matching is
`min(i_B, o_B)`, so for a window `W`

```text
MPC_d(W) = |W| − Σ_B min(i_B^W, o_B^W)
```

where `i_B^W` and `o_B^W` count the inputs and outputs of `B` inside `W`. On the 2022 PG this is
`2,181,021,971 − 1,016,828,057 = 1,164,193,914`. On other graphs Hopcroft–Karp computes `ν`, with a
Kőnig vertex cover ([\[Kon31\]](crate::references#kon31)) as a certificate.

**Width and antichains** ([`width`](crate::width)). By Dilworth's theorem
([\[Dil50\]](crate::references#dil50)) the width (maximum antichain) equals the minimum number of
chains covering the order. Fulkerson ([\[Ful56\]](crate::references#ful56)) reduces the minimum
chain cover to a matching on the transitive closure, and Ntafos and Hakimi
([\[NH79\]](crate::references#nh79)) to a minimum flow with lower bounds on the split network,
which avoids the closure. The generic exact method is that minimum flow
([\[CC04\]](crate::references#cc04), Dinic's algorithm [\[Din70\]](crate::references#din70),
max-flow min-cut [\[FF56\]](crate::references#ff56)); the antichain is read from the minimum cut.
Two PG-specific results bracket and then compute the width:

- **Frontier antichains.** For a cut `c`, `A_c = {u < c : succ(u) ⊆ [c, n)}` is an antichain,
  because a path from `u` to `v` with `u < v < c` would need an arc that ends below `c`.
  `max_c |A_c|` is a lower bound computed in one pass. On the PG `A_c` is the set of outputs created
  before `c` and not yet spent, and the bound is 136,188,373 at `c = 2,180,649,454`.
- **Width as a maximum closure.** For `PG = L(TG)` with TG acyclic, the width is the maximum of
  `Σ_{T ∈ S} (o_T − i_T)` over the down-sets `S` of transactions, and `δ⁺(S*)` (outputs created
  inside `S*` and spent outside `S*` or never spent) is a maximum antichain. A maximum-weight closure is a minimum cut
  ([\[Pic76\]](crate::references#pic76)), so the width of a window is one max-flow computation on
  its transaction graph. The antichain is then checked independently.

The full-PG width is reported as the bracket `[frontier, MPC_d]`. The closure on the full graph
(about 7.8·10⁸ vertices) is experimental. Parameterized algorithms
([\[MTKPGC19\]](crate::references#mtkpgc19), [\[CCMRT21\]](crate::references#ccmrt21),
[\[CCMRT22\]](crate::references#ccmrt22), [\[KP23\]](crate::references#kp23)) are cited only:
their parameter, the width, is at least 1.36·10⁸ on the PG.

# Conventions

- **Outputs.** Per-node and per-block results are raw little-endian arrays
  `<out>.<field>.<dtype>` (for example `run.E.u32le`) plus one JSON sidecar `<out>.json`
  ([`RunInfo`](crate::io::RunInfo): command line, graph, window, threads, wall time, peak RSS
  and the heap/file split at exit, the memory-budget reservations, the time of each phase,
  output files, parameters and results). "No node" is [`NONE`](crate::NONE) = `u32::MAX`. The
  format matches the repository's other tools, so outputs can be compared byte for byte.
- **Ids.** Node ids are `u32`, so `n ≤ u32::MAX − 1` is enforced at load time; arc counts and
  offsets are `u64`.
- **Memory.** Arrays above 1 GiB come from [`huge_vec`](crate::io::huge_vec) (transparent huge
  pages and a parallel first touch spread over both NUMA nodes). Every subcommand checks its
  estimated footprint against `--max-mem` with [`MemBudget`](crate::io::MemBudget) before
  allocating.
- **Errors.** Library functions return [`anyhow::Result`]; structural failures are
  [`DagError`](crate::DagError) values. The binary exits with 0 (success), 1 (a verification
  failed) or 2 (an error).
- **Aggregate measures.** Every per-node or per-block output is also summarised in the sidecar
  by [`summary::distribution`](crate::summary::distribution): count, sum, mean, extremes, a log2
  histogram, the top-k entries and, with `--buckets FILE`, the same per node range. PG ids are
  chronological, so the half-year chunks of the time mapping are id ranges and every measure
  comes with its evolution over time.
- **Determinism.** Results do not depend on the number of threads. Violation reports name the
  smallest offending node. Every randomized component (hashes, benchmarks) takes an explicit
  seed.

# Module map

| Module | Contents |
|---|---|
| [`graph`](crate::graph), [`sweep`](crate::sweep) | BV loading, arc-balanced chunks, the ordered parallel-decode engine |
| [`topo`](crate::topo), [`window`](crate::window) | identity-order check, relabelling, time windows |
| [`blocks`](crate::blocks) | `BlockIndex`, the biclique decomposition (PG fast path) |
| [`io`](crate::io), [`csr`](crate::csr), [`bitset`](crate::bitset), [`error`](crate::error) | arrays and files, memory budget, in-memory windows, dirty bitsets, errors |
| [`summary`](crate::summary) | distributions, histograms, top-k and time buckets of per-node arrays |
| [`semiring`](crate::semiring), [`dp`](crate::dp), [`cpm`](crate::cpm), [`paths`](crate::paths) | family 1 |
| [`dom`](crate::dom) | family 2 |
| [`reach`](crate::reach) | family 3 |
| [`reduce`](crate::reduce), [`width`](crate::width) | family 4 |
| [`cli`](crate::cli) | the `dag` subcommands |
| [`references`](crate::references) | bibliography, with an anchor per key |
