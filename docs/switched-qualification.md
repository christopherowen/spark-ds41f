# Running Release Tests on Switched DGX Sparks

How to build, verify, and benchmark a release candidate on two to eight DGX Sparks connected through a RoCE switch (`tp2-switched`..`tp8-switched` and `pp2-tp2-switched`..`pp2-tp4-switched`), and how to submit receipts back to `christopherowen/spark-ds41f`.

> **Hold unreleased benchmark numbers:** Candidate builds are not released. Keep measurements in a pull request or issue on `christopherowen/spark-ds41f` until the release that contains them is tagged (see [AGENTS.md](../AGENTS.md#hold-benchmarks-of-unreleased-work)).

## Prerequisites

- Two to eight DGX Sparks on a RoCE switch, configured with your site's switched node map under `config/nodes-switched*.local.json` from [config/examples/](../config/examples/) (see [switched-fabric.md](switched-fabric.md) for `roce_subnets`, `NCCL_IB_TC`, and port mapping).
- Passwordless SSH between the head node (rank 0) and every rank in the shape.
- At least `64 GiB` of free RAM on the build node (`spark-ds41f` serving container stopped during the build).

## 1. Check out the candidate and select a shape

Run on the head node (rank 0):

```bash
export CANDIDATE=candidate-73
export SHAPE=tp8-switched   # tp2-switched..tp8-switched, or pp2-tp2-switched / pp2-tp3-switched / pp2-tp4-switched

git clone https://github.com/christopherowen/spark-ds41f.git
cd spark-ds41f
git fetch --tags origin
git checkout "$CANDIDATE"

export CFG="experiments/2026-10-26-spark-ds41f-candidate/$CANDIDATE/$SHAPE.json"
./bin/spark --cluster-config "$CFG" doctor
```

## 2. Build and distribute the candidate image

Build once on a node with no serving container running, then sync the checkout and distribute the exact image digest to every rank in the shape:

```bash
./bin/spark --cluster-config "$CFG" build prepare
./bin/spark --cluster-config "$CFG" build check
./bin/spark --cluster-config "$CFG" build image --apply

./bin/spark --cluster-config "$CFG" cluster sync --revision "$CANDIDATE" --apply
./bin/spark --cluster-config "$CFG" build distribute --apply
```

## 3. Boot and run correctness & determinism gates (Boot 1)

```bash
export OUT="results/$CANDIDATE-$SHAPE"
mkdir -p "$OUT"

./bin/spark --cluster-config "$CFG" cluster start --replace --apply
./bin/spark --cluster-config "$CFG" doctor --live | tee "$OUT/doctor-live.txt"

# Strict temperature-0 batch invariance (--wide adds the 15-stream mixed scenario).
# Expected: token_diffs=0, logprob_diffs=0, identical=true.
python3 scripts/sonar_determinism.py --tokens 192 --long-lines 700 --wide \
  http://127.0.0.1:8000 | tee "$OUT/determinism-wide.jsonl"
python3 scripts/sonar_determinism.py --tokens 192 --long-lines 700 --wide --no-logprobs \
  http://127.0.0.1:8000 | tee "$OUT/determinism-wide-nologprobs.jsonl"

# Auto tool-calling gate (stock + deepseek-v4.1-flash-weightless under GLP-27 alpha=2.0).
# Expected: 80/80 PASS, 0 derailments.
python3 scripts/tool_gate.py --temperatures 0.0,0.3 \
  --out "$OUT/tool-gate.json" | tee "$OUT/tool-gate.txt"
```

## 4. Two-boot benchmark and Sonar release suites

Every qualified shape needs **two separate boots** (`b1` and `b2`) so comparisons keep a confidence interval. Run the block below on Boot 1, restart the cluster, and repeat with `BOOT=b2`:

```bash
export BOOT=b1   # after b1 finishes: cluster stop --apply, cluster start --replace --apply, then BOOT=b2

# Standard bench matrix (quality LRU 5/5, decode, sampled, prefill, prefix, admission).
# For new PP=2 shapes without a prior release reference, add: --compare none
./bin/spark --cluster-config "$CFG" bench --full \
  --output "$OUT/bench-$BOOT"

# Sonar release suites (for tp4-switched, tp8-switched, pp2-tp4-switched;
# use --suite release-tp3 for tp3-switched and pp2-tp3-switched):
./bin/spark --cluster-config "$CFG" bench --suite release \
  --label "$CANDIDATE-$BOOT" --output "$OUT/suite-release-$BOOT"
./bin/spark --cluster-config "$CFG" bench --suite single-stream \
  --label "$CANDIDATE-$BOOT" --output "$OUT/suite-single-$BOOT"
./bin/spark --cluster-config "$CFG" bench --suite sampled \
  --label "$CANDIDATE-$BOOT" --output "$OUT/suite-sampled-$BOOT"

# Optional on 8-node shapes (tp8-switched, pp2-tp4-switched): 32-stream concurrency
./bin/spark --cluster-config "$CFG" bench --suite c32 \
  --label "$CANDIDATE-$BOOT" --output "$OUT/suite-c32-$BOOT"
```

When switching to another shape on the same nodes, stop the current cluster first and pass `--change-shape` to `cluster start`:

```bash
./bin/spark --cluster-config "$CFG" cluster stop --apply
export SHAPE=pp2-tp4-switched
export CFG="experiments/2026-10-26-spark-ds41f-candidate/$CANDIDATE/$SHAPE.json"
./bin/spark --cluster-config "$CFG" cluster start --replace --change-shape --apply
```

## Priority shapes

1. **`tp8-switched`** — live verification of empty-index-slice decode RoPE batch invariance (`0868` / PR `#1323`), plus 2-boot `bench --full` and `release` / `single-stream` / `sampled` / `c32`.
2. **`pp2-tp4-switched` & `pp2-tp3-switched`** — switched qualification of `PP=2` (`36 GiB` / `24 GiB` KV, determinism, `tool_gate`, 2-boot `bench --full --compare none`, `c32`).
3. **`tp4-switched`, `tp3-switched`, `tp6-switched`, `tp7-switched`, `tp5-switched`, `tp2-switched`** — 2-boot `bench --full` and determinism (plus `release` / `single-stream` / `sampled` on `tp4-switched` and `tp3-switched`).

## What to submit

Open a pull request or issue on `christopherowen/spark-ds41f` (leaving out site IPs, hostnames you do not wish to share, and credentials) with:

- Commit SHA (`git rev-parse HEAD`), image ID (`docker image inspect "spark-ds41f:$CANDIDATE" --format '{{.Id}}'`), kernel version (`uname -r`), page size (`getconf PAGESIZE`), and exact `CFG`.
- `determinism-wide.jsonl`, `determinism-wide-nologprobs.jsonl`, and `tool-gate.json`.
- Both boots' `bench-b1/bench.json`, `bench-b2/bench.json`, and any `suite-*/` outputs — including any unfavourable or failed runs.
