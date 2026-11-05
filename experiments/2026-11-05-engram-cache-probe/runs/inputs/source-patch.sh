#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAKEFLAGS=-j1 MAX_JOBS=1
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=pack.threads GIT_CONFIG_VALUE_0=1 GIT_CONFIG_KEY_1=gc.auto GIT_CONFIG_VALUE_1=0
root=/home/chris/projects/engram-cache-probe
base="$root/deployment/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-05dd5db6/src/vllm"
cache=/home/chris/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-9b160482/src/vllm/.git/objects
export GIT_ALTERNATE_OBJECT_DIRECTORIES="$cache"
git clone --quiet --shared "$base" "$root/vllm"
printf '%s\n' "$cache" >> "$root/vllm/.git/objects/info/alternates"
cd "$root/vllm"
git remote rename origin prepared-cache
git remote add upstream https://github.com/local-inference-lab/vllm.git
git remote add origin https://github.com/christopherowen/vllm.git
git switch --quiet -c feat/engram-cache-probe
git apply "$root/candidate.diff"
git diff --check
python3 -m py_compile tests/kernels/test_deepseek_v41_engram_rows.py
cmp vllm/models/deepseek_v4_1/engram_rows/_row_plan.h "$root/bounded/_row_plan.h"
git add vllm/models/deepseek_v4_1/engram_rows/_row_plan.h tests/kernels/test_deepseek_v41_engram_rows.py
export GIT_AUTHOR_DATE='2026-11-05T22:55:00Z' GIT_COMMITTER_DATE='2026-11-05T22:55:00Z'
git -c user.name='Christopher Owen' -c user.email='3221756+christopherowen@users.noreply.github.com' commit --quiet -F - <<'MESSAGE'
Reuse warm Engram rows in bounded larger batches

Upstream base: local-inference-lab/vllm
19f2c20ed4d64ae22bd009ceacadd85e2a81e354; applied after candidate-38
1f33102547e4da5c64712390604695d1abd577c4.

Rationale: direct-to-mapped steps above 64 rows bypass populated cache
entries, including the 96 rows of a full TP4 decode batch. Probe through
128 rows while retaining the existing admission limit and cache size.
Longer direct-to-mapped prefills keep the current bypass.

Quality: copy the same immutable weight and scale bytes. No arithmetic,
quantization, eviction, packing, worker or interrupt-policy change.
Add GPU regression coverage for exact cached/uncached output, dead IDs,
rank masks and retaining hot rows after repeated larger misses.

Tests: offline native-planner byte oracle and cache-retention checks,
two bounded CPU fixture runs; GCC AddressSanitizer/UndefinedBehaviorSanitizer
over 26 paired fixtures. Python syntax and git diff --check pass.
GPU regression and serving boot/repeat qualification are pending.
Synthetic planned reads are not measurements of NVMe or serving speed.

Upstream status: experimental local patch; not submitted upstream.
MESSAGE
git show -s --format='commit %H%ntree %T%nparent %P'
git format-patch -1 --stdout > "$root/0001-engram-bounded-cache-probes.patch"
