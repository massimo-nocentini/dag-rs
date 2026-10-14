#!/usr/bin/env bash
# Full-scale validation of dag on the 2022 Payment Graph.
#
# Runs the release binary one subcommand at a time (never two PG passes at once) and checks
# every JSON report, and the per-node arrays where §10.4 asks for byte identity, against the
# expected values of §10.4. Each check prints one TSV row
#     check <TAB> expected <TAB> observed <TAB> PASS|FAIL <TAB> wall_s <TAB> peak_rss_GiB
# to stdout and to <out>/validation.tsv; the exit status is 1 if any check fails.
#
# Usage:
#   scripts/validate_pg.sh [STEP...]           # default: all steps, then `check`
#   scripts/validate_pg.sh check               # only re-check existing outputs
#   scripts/validate_pg.sh --list              # list the steps
#
# Environment:
#   PG_BASENAME      forward BV basename (default /data/bitcoin/2022/pg/pg; needs -t too)
#   PG_ANALYSIS_DIR  prior exact results (default /data/bitcoin/2022/pg_analysis/results)
#   DAG_BIN           the dag binary (default target/release/dag of this crate)
#   VALIDATE_OUT     output directory (default ./validate_pg.out)
#   MEASURE          optional wrapper prefix; the script appends `-n <step> -- <cmd>`, e.g.
#                    MEASURE="measure.sh -o runs.jsonl -t validation"
#   DAG_BIN_OPTS      extra global options for dag (e.g. "--threads 56")
#   PG_BUCKETS       optional node-range bucket file (the time mapping's chunk_boundaries.tsv:
#                    one bucket per half-year); passed as --buckets to every producing step, so
#                    the JSON reports carry the per-half-year distributions
#   PG_CUTS          optional sorted u64 node ids (the time mapping's block_first_node.u64):
#                    `frontier --cuts`, the UTXO-set size at every blockchain block
#
# Wall time and peak RSS come from the `elapsed_s` and `peak_rss_bytes` fields that every
# subcommand writes in its JSON report (RunInfo).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PG=${PG_BASENAME:-/data/bitcoin/2022/pg/pg}
AN=${PG_ANALYSIS_DIR:-/data/bitcoin/2022/pg_analysis/results}
BIN=${DAG_BIN:-$here/target/release/dag}
OUT=${VALIDATE_OUT:-$PWD/validate_pg.out}
read -r -a MEAS <<<"${MEASURE:-}"
read -r -a GOPTS <<<"${DAG_BIN_OPTS:-}"
BLK=$OUT/blocks # BlockIndex prefix written by the `blocks` step
BK=()           # --buckets FILE, when PG_BUCKETS is set
[[ -n ${PG_BUCKETS:-} ]] && BK=(--buckets "$PG_BUCKETS")
CUTS=()         # --cuts FILE, when PG_CUTS is set
[[ -n ${PG_CUTS:-} ]] && CUTS=(--cuts "$PG_CUTS")

STEPS=(topo blocks dmax certify-dmax dmin height cpm count tr tr-generic tr-reduce-window
    path-cover frontier dom dom-stats dom-verify pdom-verify reach-counts reach-index
    reach-bench reach-query width-window pc-window-0 pc-window-1 pc-window-2)

# Windows of the `windows` rows (each ≤ 2^20 nodes, so `path-cover` cross-checks HK).
PCW=(0..1000000 1000000000..1001000000 2180000000..2181000000)

run() { # run STEP ARGS...: one dag invocation, optionally under the measurement wrapper
    local step=$1
    shift
    echo "== $step: dag $*" >&2
    local rc=0
    if ((${#MEAS[@]})); then
        "${MEAS[@]}" -n "$step" -- "$BIN" "${GOPTS[@]}" "$@" || rc=$?
    else
        "$BIN" "${GOPTS[@]}" "$@" || rc=$?
    fi
    echo "$rc" >"$OUT/$step.rc"
    echo "== $step: exit $rc" >&2
}

step() {
    local s=$1
    case $s in
    topo) run topo topo-check --graph "$PG" --out "$OUT/topo" ;;
    blocks) run blocks blocks --graph "$PG" --out "$BLK" --compare-bicliques "$AN/bicliques" "${BK[@]}" ;;
    dmax) run dmax dp --graph "$PG" --blocks "$BLK" --semiring maxplus-u32 --dir forward \
        --init sources --weights unit --pred --certify --compare "$AN/chains/longest.u32le" \
        "${BK[@]}" --out "$OUT/dmax" ;;
    certify-dmax) run certify-dmax certify --graph "$PG" --semiring maxplus-u32 --dir forward \
        --init sources --weights unit --values "$OUT/dmax.val.u32le" \
        --pred "$OUT/dmax.pred.u32le" --out "$OUT/certify-dmax" ;;
    dmin) run dmin dp --graph "$PG" --blocks "$BLK" --semiring minplus-u32 --dir forward \
        --init sources --weights unit --certify "${BK[@]}" --out "$OUT/dmin" ;;
    height) run height dp --graph "$PG" --blocks "$BLK" --semiring maxplus-u32 --dir backward \
        --init sources --weights unit --certify --compare "$AN/layers/height.u32le" \
        "${BK[@]}" --out "$OUT/height" ;;
    cpm) run cpm cpm --graph "$PG" --blocks "$BLK" --critical-arcs --slack --path \
        --compare-e "$AN/chains/longest.u32le" --compare-h "$AN/layers/height.u32le" \
        "${BK[@]}" --out "$OUT/cpm" ;;
    count) run count count-paths --graph "$PG" --blocks "$BLK" --dir both \
        --kinds extfloat,mod61 --summary-only "${BK[@]}" --out "$OUT/count" ;;
    tr) run tr tr verify --graph "$PG" --blocks "$BLK" --out "$OUT/tr" ;;
    tr-generic) run tr-generic tr verify --graph "$PG" --blocks none --out "$OUT/tr-generic" ;;
    tr-reduce-window) run tr-reduce-window tr reduce --graph "$PG" --blocks "$BLK" \
        --window 0..100000000 --out "$OUT/tr-reduce-window" ;;
    path-cover) run path-cover path-cover --graph "$PG" --blocks "$BLK" --out "$OUT/path-cover" ;;
    frontier) run frontier frontier --graph "$PG" --blocks "$BLK" "${BK[@]}" "${CUTS[@]}" \
        --profile-step 10000000 --out "$OUT/frontier" ;;
    dom) run dom dom build --graph "$PG" --blocks "$BLK" --sizes --depth-bound "${BK[@]}" \
        --out "$OUT/dom" ;;
    dom-stats) run dom-stats dom stats --prefix "$OUT/dom" --top 100 "${BK[@]}" ;;
    dom-verify) run dom-verify dom verify --graph "$PG" --idom "$OUT/dom.idom.u32le" \
        --out "$OUT/dom-verify" ;;
    pdom-verify) run pdom-verify dom verify --graph "$PG" --post --idom "$OUT/dom.ipdom.u32le" \
        --out "$OUT/pdom-verify" ;;
    reach-counts) run reach-counts reach-counts --graph "$PG" --blocks "$BLK" --which both \
        --log2m 10 --seed 0 "${BK[@]}" --out "$OUT/reach-counts" ;;
    reach-index) run reach-index reach-index build --graph "$PG" --blocks "$BLK" --chains 4 \
        --out "$OUT/reach-index" ;;
    reach-bench) run reach-bench reach-index bench --graph "$PG" --blocks "$BLK" --index "$OUT/reach-index" \
        --sources 64 --queries 10000 --seed 1 --max-query-secs 60 --out "$OUT/reach-bench" ;;
    reach-query)
        # 5,000 arcs (reachable), the same pairs reversed (unreachable: ids are topological),
        # 5,000 random pairs u < v, and the spine (source 84,564 to 2,181,015,947)
        python3 - "$BLK" "$OUT/reach-query.pairs.u32x2le" <<'PY'
import mmap, random, struct, sys
blk, out = sys.argv[1], sys.argv[2]
N = 2_181_021_971
def arr(name):
    f = open(f"{blk}.blk.{name}.u32le", "rb")
    return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
spend, first = arr("spend"), arr("first")
rng = random.Random(1)
arcs = []
while len(arcs) < 5000:
    u = rng.randrange(N)
    b = struct.unpack_from("<I", spend, 4 * u)[0]
    if b != 0xFFFFFFFF:
        arcs.append((u, struct.unpack_from("<I", first, 4 * b)[0]))
rand = []
while len(rand) < 5000:
    u, v = sorted((rng.randrange(N), rng.randrange(N)))
    if u < v:
        rand.append((u, v))
pairs = arcs + [(v, u) for u, v in arcs] + rand + [(84_564, 2_181_015_947)]
with open(out, "wb") as f:
    for u, v in pairs:
        f.write(struct.pack("<II", u, v))
PY
        run reach-query reach-index query --graph "$PG" --blocks "$BLK" --index "$OUT/reach-index" \
            --pairs "$OUT/reach-query.pairs.u32x2le" --out "$OUT/reach-query"
        ;;
    width-window) run width-window width --graph "$PG" --blocks "$BLK" --window 0..1000000 \
        --method auto --out "$OUT/width-window" ;;
    pc-window-[0-2])
        local i=${s#pc-window-}
        run "$s" path-cover --graph "$PG" --blocks "$BLK" --window "${PCW[$i]}" --explicit \
            --out "$OUT/$s"
        ;;
    *)
        echo "unknown step $s (see --list)" >&2
        exit 2
        ;;
    esac
}

check() {
    python3 - "$OUT" "$AN" "${PCW[@]}" <<'PY'
import json, mmap, os, struct, sys

out, an, pcw = sys.argv[1], sys.argv[2], sys.argv[3:]
N, M = 2_181_021_971, 8_639_773_499
rows, fails = [], 0

def load(name):
    p = os.path.join(out, name + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def get(d, path):
    for k in path.split("."):
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d

def row(check, expected, observed, ok, run):
    global fails
    j = load(run) if run else None
    wall = f"{j['elapsed_s']:.1f}" if j and j.get("elapsed_s") is not None else "-"
    rss = f"{j['peak_rss_bytes'] / 2**30:.1f}" if j and j.get("peak_rss_bytes") else "-"
    rc_file = os.path.join(out, f"{run}.rc") if run else None
    if rc_file and os.path.exists(rc_file):
        rc = open(rc_file).read().strip()
        if rc != "0":
            ok = False
            observed = f"{observed} (exit {rc})"
    status = "PASS" if ok else "FAIL"
    fails += not ok
    rows.append((check, str(expected), str(observed), status, wall, rss))

def eq(check, run, path, expected):
    obs = get(load(run), path)
    if isinstance(obs, float) and obs.is_integer():
        obs = int(obs)
    row(check, f"{expected:,}" if isinstance(expected, int) and not isinstance(expected, bool) else expected,
        f"{obs:,}" if isinstance(obs, int) and not isinstance(obs, bool) else obs, obs == expected, run)
    return obs

def u32_at(path, i):
    with open(path, "rb") as f:
        f.seek(4 * i)
        return struct.unpack("<I", f.read(4))[0]

def pred_chain_end(path, v):
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        steps = 0
        while True:
            p = struct.unpack_from("<I", m, 4 * v)[0]
            if p == 0xFFFFFFFF:
                return v, steps
            v, steps = p, steps + 1

# topo-check -----------------------------------------------------------------------------------
eq("topo: n", "topo", "results.n", N)
eq("topo: m", "topo", "results.m", M)
eq("topo: identity topological", "topo", "results.identity_topological", True)
eq("topo: sinks", "topo", "results.sinks", 136_124_643)
eq("topo: sources", "topo", "results.sources", 2_883_482)

# blocks ---------------------------------------------------------------------------------------
for k, v in [("nb", 777_851_179), ("sum_len", 2_178_138_489), ("sum_nin", 2_044_897_328),
             ("sum_len_times_nin", M), ("max_len", 13_107), ("max_nin", 20_000)]:
    eq(f"blocks: {k}", "blocks", f"results.report.{k}", v)
eq("blocks: first == bicliques/blocks.u32x3[.0]", "blocks", "results.comparison.block_mismatches", 0)
eq("blocks: create <=> bicliques/blk_start.u32", "blocks", "results.comparison.node_mismatches", 0)

# dp -------------------------------------------------------------------------------------------
def summary(run):
    return get(load(run), "results.summary") or {}

s = summary("dmax")
eq("dp maxplus fwd: max", "dmax", "results.summary.max", 3_559_224)
eq("dp maxplus fwd: argmax", "dmax", "results.summary.argmax", 2_181_015_947)
mean = s.get("mean")
row("dp maxplus fwd: mean", "2,397,482.094", f"{mean:,.3f}" if mean is not None else None,
    mean is not None and abs(mean - 2_397_482.094) < 5e-4, "dmax")
eq("dp maxplus fwd: == chains/longest.u32le", "dmax", "results.comparison.mismatches", 0)
pred = os.path.join(out, "dmax.pred.u32le")
if os.path.exists(pred):
    end, steps = pred_chain_end(pred, 2_181_015_947)
    row("dp maxplus fwd: pred chain of 2,181,015,947", "source 84,564 after 3,559,224 arcs",
        f"source {end:,} after {steps:,} arcs", (end, steps) == (84_564, 3_559_224), "dmax")
else:
    row("dp maxplus fwd: pred chain of 2,181,015,947", "source 84,564", "missing", False, "dmax")
eq("certify maxplus fwd (in dp)", "dmax", "results.certify.violations", 0)
eq("certify maxplus fwd: values", "certify-dmax", "results.violations", 0)
eq("certify maxplus fwd: preds", "certify-dmax", "results.pred_violations", 0)

eq("dp minplus fwd: max", "dmin", "results.summary.max", 25_348)
eq("dp minplus fwd: argmax", "dmin", "results.summary.argmax", 479_412_379)
mean = summary("dmin").get("mean")
row("dp minplus fwd: mean", "76.34", f"{mean:.4f}" if mean is not None else None,
    mean is not None and abs(mean - 76.34) < 5e-3, "dmin")
eq("certify minplus fwd", "dmin", "results.certify.violations", 0)

eq("dp maxplus bwd: == layers/height.u32le", "height", "results.comparison.mismatches", 0)
eq("dp maxplus bwd: max", "height", "results.summary.max", 3_559_224)
eq("dp maxplus bwd: argmax", "height", "results.summary.argmax", 84_564)
eq("certify maxplus bwd", "height", "results.certify.violations", 0)

# cpm ------------------------------------------------------------------------------------------
eq("cpm: T", "cpm", "results.summary.t", 3_559_224)
eq("cpm: critical nodes", "cpm", "results.summary.critical_nodes", 5_459_477)
cmp = {c["array"]: c["result"]["mismatches"] for c in get(load("cpm"), "results.comparisons") or []}
row("cpm: E == longest.u32le, H == height.u32le", "0, 0 mismatches",
    f"{cmp.get('E')}, {cmp.get('H')}", cmp.get("E") == 0 and cmp.get("H") == 0, "cpm")

# count-paths ----------------------------------------------------------------------------------
eq("count-paths: duality ok", "count", "results.summary.duality.ok", True)
eq("count-paths: M61 fwd total == bwd total", "count", "results.summary.duality.mod61_equal", True)
rel = get(load("count"), "results.summary.duality.extfloat_rel_diff")
row("count-paths: ExtFloat totals rel. diff", "<= 1e-6", f"{rel:.3e}" if rel is not None else None,
    rel is not None and rel <= 1e-6, "count")

# tr -------------------------------------------------------------------------------------------
eq("tr verify: reduced", "tr", "results.report.is_transitively_reduced", True)
eq("tr verify: certificate", "tr", "results.report.certificate", "blocks")
eq("tr verify --blocks none: H3 failing nodes", "tr-generic", "results.report.h3_failing_nodes", 0)
eq("tr verify --blocks none: reduced", "tr-generic", "results.report.is_transitively_reduced", True)

# path cover, frontier -------------------------------------------------------------------------
eq("path-cover: matching", "path-cover", "results.matching", 1_016_828_057)
eq("path-cover: MPC_d", "path-cover", "results.mpc_d", 1_164_193_914)
eq("frontier: size", "frontier", "results.size", 136_188_373)
eq("frontier: cut", "frontier", "results.cut", 2_180_649_454)

# dominators -----------------------------------------------------------------------------------
eq("dom verify: ok", "dom-verify", "results.report.ok", True)
eq("postdom verify: ok", "pdom-verify", "results.report.ok", True)
for run, tree in [("dom-verify", "dom"), ("pdom-verify", "postdom")]:
    r = get(load(run), "results.report") or {}
    fails_v = [r.get(k) for k in ("v0_fail", "v1_fail", "v2_fail")]
    row(f"{tree} verify: V0, V1, V2 failures", "0, 0, 0", ", ".join(map(str, fails_v)),
        fails_v == [0, 0, 0], run)
d = load("dom")
for tree, roots_min in [("dom", 2_883_482), ("pdom", 136_124_643)]:
    st = get(d, f"results.{tree}.stats") or {}
    rc = st.get("root_children")
    row(f"{tree}: root children >= {'sources' if tree == 'dom' else 'sinks'}", f">= {roots_min:,}",
        f"{rc:,}" if rc is not None else None, rc is not None and rc >= roots_min, "dom")
md = get(d, "results.dom.stats.max_depth")
row("dom: max depth <= 25,349", "<= 25,349", f"{md:,}" if md is not None else None,
    md is not None and md <= 25_349, "dom")

# reach-counts ---------------------------------------------------------------------------------
rcj = load("reach-counts")
for p in ("anc", "desc"):
    pl = None
    for rep in get(rcj, "results.passes") or []:
        if rep.get("name") == p:
            pl = rep["stats"]["peak_live"]
    row(f"reach-counts: peak live ({p})", "<= 5,139,210", f"{pl:,}" if pl is not None else None,
        pl is not None and pl <= 5_139_210, "reach-counts")
eq("reach-counts: anc == 0 on every source", "reach-counts", "results.sources_check.nonzero_anc", 0)
eq("reach-counts: sources checked", "reach-counts", "results.sources_check.sources", 2_883_482)
md = get(rcj, "results.desc.mean")
rsd = 1.04 / 2 ** (10 / 2)  # HyperLogLog relative standard deviation for 2^10 registers
if md is not None and rsd is not None:
    frac = md / N
    row("reach-counts: mean desc / N", f"0.4195 ± 3σ (σ = {rsd:.4f} rel.)", f"{frac:.4f}",
        abs(frac - 0.4195) <= 3 * rsd * 0.4195, "reach-counts")
else:
    row("reach-counts: mean desc / N", "0.4195 ± 3σ", None, False, "reach-counts")

# reach-index ----------------------------------------------------------------------------------
# a G path of L arcs from a coinbase output visits L blocks, i.e. L - 1 arcs of the quotient Q
ri = load("reach-index")
sp = get(ri, "results.stats.max_lvl")
row("reach-index: longest path of Q (lvl)", "3,559,223 = 3,559,224 - 1", f"{sp:,}" if sp is not None else None,
    sp == 3_559_223, "reach-index")
c0 = (get(ri, "results.stats.chain_lens") or [None])[0]
row("reach-index: spine (chain 0) blocks", "3,559,224", f"{c0:,}" if c0 is not None else None,
    c0 == 3_559_224, "reach-index")
wls = get(load("reach-bench"), "results.workloads") or []
dis = sum(w["disagreements"] for w in wls) if wls else None
q = sum(w["queries"] for w in wls)
row("reach-index bench: agreement with cone labels", "100 % of >= 10^4 queries",
    f"{q:,} queries ({', '.join(w['name'] for w in wls)}), {dis} disagreements",
    dis == 0 and q >= 10_000, "reach-bench")
mx = max([w["latency_ns"]["max"] / 1e9 for w in wls] + [w["par_max_query_s"] for w in wls]) if wls else None
row("reach-index bench: slowest query", "< 60 s", f"{mx:.3f} s" if mx is not None else None,
    mx is not None and mx < 60, "reach-bench")

# windows --------------------------------------------------------------------------------------
w = load("width-window")
c, mf = get(w, "results.width"), get(w, "results.cross_check.minflow_width")
ok = (c is not None and c == mf and get(w, "results.certificate_ok") is True
      and get(w, "results.cross_check.minflow_certificate_ok") is True)
row("width 0..1000000: closure == minflow (both certified)", "equal",
    f"{c:,} vs {mf:,}" if c is not None and mf is not None else None, ok, "width-window")
for i, win in enumerate(pcw):
    run = f"pc-window-{i}"
    j = load(run)
    hk, cf = get(j, "results.cross_check.hk_matching"), get(j, "results.matching")
    ok = (hk is not None and hk == cf and get(j, "results.cross_check.konig_ok") is True
          and get(j, "results.cross_check.explicit_ok") is True)
    row(f"path-cover {win}: HK == closed form (König, explicit cover)", "equal matchings",
        f"{hk:,} vs {cf:,} (MPC_d {get(j, 'results.mpc_d'):,})" if hk is not None and cf is not None else None,
        ok, run)

# aggregate measures: distributions, top-k, half-year buckets, curves --------------------------
BUCKETED = bool(os.environ.get("PG_BUCKETS"))

def hist_total(d):
    h = d.get("log2_hist") or {}
    return sum(h.get("bins") or []) + h.get("below_one", 0) + h.get("negative", 0)

def bucket_row(check, run, d, total, count_keys=("count", "absent")):
    """Every entry lies in exactly one of the 28 half-year buckets."""
    if not BUCKETED:
        return
    b = (d or {}).get("by_bucket")
    if b is None:
        row(check, "28 buckets, sum = total", None, False, run)
        return
    tot = sum(sum(x.get(k, 0) for k in count_keys) for x in b) + (d.get("unbucketed") or 0)
    labels = f"{b[0]['label']}..{b[-1]['label']}"
    row(check, f"28 buckets 2009H1..2022H2, {total:,} entries",
        f"{len(b)} buckets {labels}, {tot:,} entries, {d.get('unbucketed'):,} unbucketed",
        len(b) == 28 and labels == "2009H1..2022H2" and tot == total and d.get("unbucketed") == 0, run)

bd = get(load("blocks"), "results.distributions") or {}
ln, ni, ar = bd.get("len") or {}, bd.get("nin") or {}, bd.get("arcs") or {}
row("blocks: distributions (count, max len, max nin, Σ arcs)", "777,851,179; 13,107; 20,000; m",
    f"{ln.get('count')}; {ln.get('max')}; {ni.get('max')}; {ar.get('sum')}",
    ln.get("count") == 777_851_179 and ln.get("max") == 13_107 and ni.get("max") == 20_000
    and ar.get("sum") == M and hist_total(ln) == 777_851_179, "blocks")
bucket_row("blocks: per half-year", "blocks", ln, 777_851_179)

for run, nm in [("dmax", "dp maxplus fwd"), ("dmin", "dp minplus fwd"), ("height", "dp maxplus bwd")]:
    s = summary(run)
    ok = (s.get("nodes") == N and hist_total(s) == N - (s.get("zero_nodes") or 0)
          and (s.get("top") or [{}])[0].get("id") == s.get("argmax"))
    row(f"{nm}: log2 histogram + top-k", "hist total = n − 0̄, top[0] = argmax",
        f"hist {hist_total(s):,}, top[0] {(s.get('top') or [{}])[0].get('id')}", ok, run)
    bucket_row(f"{nm}: per half-year", run, s, N, ("count", "absent"))

c = get(load("cpm"), "results") or {}
cs, cd = c.get("summary") or {}, c.get("distributions") or {}
crit = (cd.get("critical_e") or {}).get("count")
slack0 = ((cd.get("slack") or {}).get("log2_hist") or {}).get("bins", [None])[0]
row("cpm: critical_e count == slack-0 nodes == critical nodes", "5,459,477",
    f"{crit}, {slack0}, {cs.get('critical_nodes')}",
    crit == slack0 == cs.get("critical_nodes") == 5_459_477, "cpm")
lp = c.get("longest_path") or {}
row("cpm --path: one longest path", "3,559,225 nodes, 84,564 -> 2,181,015,947",
    f"{lp.get('nodes')} nodes, {lp.get('start')} -> {lp.get('end')}",
    (lp.get("nodes"), lp.get("start"), lp.get("end")) == (3_559_225, 84_564, 2_181_015_947), "cpm")
row("cpm: widest critical layer", "width = max_critical_width, node range in [0, n)",
    f"{cs.get('max_critical_width')} at E = {cs.get('max_critical_layer')}, nodes {cs.get('max_critical_layer_nodes')}",
    cs.get("max_critical_layer_nodes") is not None and sum(cs.get("critical_width_log2_hist") or []) == 3_559_225,
    "cpm")
bucket_row("cpm: critical nodes per half-year", "cpm", cd.get("critical_e"), N)

cnt = get(load("count"), "results") or {}
fd = get(cnt, "distributions.forward_log2") or {}
row("count-paths: top-k forward", "max = summary max_log2", f"{fd.get('max')}",
    fd.get("max") is not None and fd.get("max") == get(cnt, "summary.forward.max_log2"), "count")
bucket_row("count-paths: through counts per half-year", "count", get(cnt, "distributions.through_log2"), N)

fr = get(load("frontier"), "results") or {}
prof = get(fr, "curve.profile") or []
pmax = max((r["max"] for r in prof), default=None)
row("frontier curve: max over the 10^7-node profile == frontier", "136,188,373",
    f"{pmax:,} over {len(prof)} ranges" if pmax is not None else None, pmax == 136_188_373, "frontier")
if BUCKETED:
    fb = get(fr, "curve.by_bucket") or []
    last = fb[-1]["at_hi"] if fb else None
    row("frontier curve: |A_n| at the end of 2022H2 == sinks", "136,124,643",
        f"{last:,} ({len(fb)} buckets)" if last is not None else None,
        last == 136_124_643 and len(fb) == 28, "frontier")

if os.environ.get("PG_CUTS"):
    cu = get(fr, "cuts") or {}
    row("frontier at every blockchain block: last = sinks, max <= frontier",
        "762,262 cuts, last 136,124,643, max <= 136,188,373",
        f"{cu.get('cuts')} cuts, last {cu.get('last')}, max {cu.get('max')} at {cu.get('argmax_cut')}",
        cu.get("cuts") == 762_262 and cu.get("last") == 136_124_643
        and (cu.get("max") or 1 << 62) <= 136_188_373, "frontier")

d = load("dom")
for tree in ("dom", "pdom"):
    rcd = get(d, f"results.{tree}.distributions.root_children") or {}
    rcs = get(d, f"results.{tree}.stats.root_children")
    row(f"{tree}: root_children distribution == stats", f"{rcs:,}" if rcs else None,
        f"{rcd.get('count'):,}, Σ size {rcd.get('sum'):,.0f}" if rcd.get("count") is not None else None,
        rcd.get("count") == rcs and rcd.get("sum") == N, "dom")
    bucket_row(f"{tree}: depth per half-year", "dom", get(d, f"results.{tree}.distributions.depth"), N)
st = load("dom.stats")
for tree in ("dom", "pdom"):
    a, b = get(st, f"results.{tree}.root_children"), get(d, f"results.{tree}.stats.root_children")
    tsv = os.path.join(out, f"dom.{tree}.top.tsv")
    lines = sum(1 for _ in open(tsv)) if os.path.exists(tsv) else None
    row(f"dom stats ({tree}): == dom build, top-100 TSV", f"{b:,}, 101 lines" if b else None,
        f"{a:,}, {lines} lines" if a is not None else None, a == b and lines == 101, "dom-stats")

rcj = load("reach-counts")
anc = get(rcj, "results.anc") or {}
bd_ = get(rcj, "results.blk_desc") or {}
row("reach-counts: absolute argmax, block argmax_node", "anc argmax in [0, n), blk argmax_node set",
    f"{anc.get('argmax')}, {bd_.get('argmax')} -> node {bd_.get('argmax_node')}",
    anc.get("argmax") is not None and 0 <= anc["argmax"] < N and bd_.get("argmax_node") is not None,
    "reach-counts")
bucket_row("reach-counts: desc per half-year", "reach-counts", get(rcj, "results.desc"), N)

rq = load("reach-query")
ans_p = os.path.join(out, "reach-query.answers.u8")
if rq and os.path.exists(ans_p):
    a = open(ans_p, "rb").read()
    ok = len(a) == 15_001 and all(a[:5000]) and not any(a[5000:10000]) and a[15_000] == 1
    row("reach-index query: arcs 1, reversed arcs 0, spine 1", "15,001 answers",
        f"{len(a):,} answers, {sum(a[:5000])} / {sum(a[5000:10000])} / {a[15_000]}, "
        f"{get(rq, 'results.qps'):,.0f} q/s", ok, "reach-query")
else:
    row("reach-index query: arcs 1, reversed arcs 0, spine 1", "15,001 answers", "missing", False, "reach-query")

eq("tr reduce 0..10^8: redundant arcs", "tr-reduce-window", "results.redundant", 0)

w = load("width-window")
row("width 0..1000000: antichain positions", "count = width",
    f"{get(w, 'results.antichain_ids.count')}, ids {get(w, 'results.antichain_ids.min')}..{get(w, 'results.antichain_ids.max')}",
    get(w, "results.antichain_ids.count") == get(w, "results.width"), "width-window")
for i, win in enumerate(pcw):
    j = load(f"pc-window-{i}")
    ch = get(j, "results.chains") or {}
    row(f"path-cover {win}: chains", "count = MPC_d, Σ length = window",
        f"{ch.get('count')} chains, longest {ch.get('max')}", ch.get("count") == get(j, "results.mpc_d")
        and ch.get("sum") == 1_000_000, f"pc-window-{i}")

with open(os.path.join(out, "validation.tsv"), "w") as f:
    f.write("check\texpected\tobserved\tstatus\twall_s\tpeak_rss_GiB\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
for r in rows:
    print("\t".join(r))
print(f"{len(rows) - fails} of {len(rows)} checks passed", file=sys.stderr)
sys.exit(1 if fails else 0)
PY
}

if [[ ${1:-} == --list ]]; then
    printf '%s\n' "${STEPS[@]}" check
    exit 0
fi
mkdir -p "$OUT"
[[ -x $BIN ]] || {
    echo "binary $BIN not found: cargo build --release" >&2
    exit 2
}
todo=("$@")
((${#todo[@]})) || todo=("${STEPS[@]}" check)
for s in "${todo[@]}"; do
    if [[ $s == check ]]; then
        check
    else
        step "$s"
    fi
done
