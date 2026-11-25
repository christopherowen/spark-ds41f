#!/usr/bin/env bash
# make_overlay.sh WORK [OUT]: rebuild the eight patched vLLM files of overlay/ from the series.
#
# WORK  an empty folder for the vLLM tree (about 1 GB).
# OUT   where to write the files (default: WORK/overlay). The script compares OUT/SHA256SUMS with
#       overlay/SHA256SUMS of this experiment, and each file with the committed copy.
#
# Steps: fetch local-inference-lab/vllm at the revision of the spark-ds41f-1120 lock, apply the
# first 702 entries of vllm/series (the release's series) with the recipe's own git am settings
# (spark_ds41f/build.py), check the tree 8fc99ca7 of the lock, apply 9030 to 9046, check the tree
# 47e196a2, then copy the eight files. Reads the recipe from ROOT (default: the checkout that
# holds this script). No GPU is needed.
set -euo pipefail
WORK=${1:?usage: make_overlay.sh WORK [OUT]}
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${ROOT:-$(cd "$HERE/../.." && pwd)}
REMOTE=${VLLM_REMOTE:-https://github.com/local-inference-lab/vllm}
REV=19f2c20ed4d64ae22bd009ceacadd85e2a81e354
BASE_TREE=8fc99ca7546c5c538f0fe64e7c82954f1d2d7ae0
TIER_TREE=47e196a297a31e2c4a7dd3a11f83b6b6ee02c638
FILES=(
  vllm/v1/simple_kv_offload/disk_backend.py
  vllm/v1/simple_kv_offload/worker.py
  vllm/v1/simple_kv_offload/manager.py
  vllm/v1/simple_kv_offload/metadata.py
  vllm/v1/simple_kv_offload/metrics.py
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py
  vllm/v1/sample/sink.py
  vllm/v1/worker/gpu/model_runner.py
)
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
OUT=${2:-$WORK/overlay}
mkdir -p "$WORK/vllm" "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$WORK/vllm"
[ -d .git ] && { echo "make_overlay: $WORK/vllm already holds a repository; use an empty WORK" >&2; exit 1; }
git -c gc.auto=0 init --quiet
git remote add upstream "$REMOTE"
git -c gc.auto=0 fetch --quiet --filter=blob:none --no-tags upstream "$REV"
git switch --quiet --create spark/build FETCH_HEAD

apply() {  # apply entries FROM..TO of the series with the recipe's git am settings
  python3 -I - "$ROOT" "$HERE/vllm/series" "$PWD" "$1" "$2" <<'PY'
import sys
from pathlib import Path
root, series, tree, start, end = sys.argv[1:6]
sys.path.insert(0, root)
from spark_ds41f.build import apply_patch_series, read_series
entries = read_series(Path(series))
if len(entries) != 719:
    raise SystemExit(f"make_overlay: the series has {len(entries)} entries, want 719")
apply_patch_series(Path(tree), Path(series).parent, entries[int(start):int(end)])
PY
}

apply 0 702
[ "$(git rev-parse 'HEAD^{tree}')" = "$BASE_TREE" ] || { echo "make_overlay: base tree differs" >&2; exit 1; }
apply 702 719
[ "$(git rev-parse 'HEAD^{tree}')" = "$TIER_TREE" ] || { echo "make_overlay: tier tree differs" >&2; exit 1; }
for f in "${FILES[@]}"; do
  mkdir -p "$OUT/$(dirname "$f")"
  cp "$f" "$OUT/$f"
done
(cd "$OUT" && sha256sum "${FILES[@]}" > SHA256SUMS && cat SHA256SUMS)
cmp -s "$OUT/SHA256SUMS" "$HERE/overlay/SHA256SUMS" \
  || { echo "make_overlay: the sums differ from overlay/SHA256SUMS" >&2; exit 1; }
for f in "${FILES[@]}"; do
  cmp -s "$OUT/$f" "$HERE/overlay/$f" || { echo "make_overlay: $f differs from overlay/$f" >&2; exit 1; }
done
echo "make_overlay: tree $TIER_TREE, ${#FILES[@]} files in $OUT, equal to overlay/"
