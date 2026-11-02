#!/bin/bash
# Prepare candidate-37's vLLM tree as `bin/spark build prepare` does (base
# 19f2c20ed, the series applied with `git am --committer-date-is-author-date`
# under the build's committer), then branch for this experiment. Reproduces
# patch head 5b7cfa37 and tree af0fb152. REPO is a clean checkout of this
# repository; SRC any local vLLM clone holding the base revision.
set -euo pipefail
REPO=${REPO:-~/projects/spark-ds41f}
SRC=${SRC:-$REPO/.work/vllm-candidate-34}
DEST=${DEST:-~/src/vllm-markov-greedy-fusion}
SERIES=$REPO/experiments/2026-10-26-spark-ds41f-candidate/candidate-37/vllm/series
test ! -e "$DEST"
git clone --quiet --no-checkout "$SRC" "$DEST"
cd "$DEST"
git switch --quiet --create spark/build 19f2c20ed4d64ae22bd009ceacadd85e2a81e354
dir=$(dirname "$SERIES")
grep -v '^\s*#' "$SERIES" | sed '/^\s*$/d' | grep -v '^TODO-' | while read -r p; do
  git -c user.name="Christopher Owen" \
    -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am --quiet --committer-date-is-author-date "$dir/$p"
done
test "$(git rev-parse HEAD)" = 5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575
test "$(git rev-parse 'HEAD^{tree}')" = af0fb1525fc6aa4149b18dee7d8c3ae5120b6f33
git switch --quiet --create perf/dspark-fused-markov-greedy
