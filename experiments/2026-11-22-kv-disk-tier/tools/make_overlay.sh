#!/usr/bin/env bash
# make_overlay.sh WORK [OUT]: rebuild the eight patched vLLM files of overlay/ from the series.
#
# WORK     an empty folder for the vLLM tree (about 1 GB).
# OUT      where to write the files (default: WORK/overlay). The script compares each file with the
#          committed copy, and for the 1120 release it also compares OUT/SHA256SUMS with
#          overlay/SHA256SUMS of this experiment.
# RELEASE  the release whose series is the base: 1120 (default) or 1126.
#
# Steps: fetch local-inference-lab/vllm at the revision of the release's lock, apply the release's
# own entries of the series (702 of vllm/series for 1120, 742 of vllm/series-1126 for 1126) with the
# recipe's own git am settings (spark_ds41f/build.py), check the release's tree, apply 9030 to 9046,
# check the tree of the tier, then copy the eight files. For 1126, model_runner.py is compared with
# overlay-1126/ and the other seven files with overlay/, because only model_runner.py differs between
# the two releases. Reads the recipe from ROOT (default: the checkout that holds this script). No GPU
# is needed.
set -euo pipefail
WORK=${1:?usage: make_overlay.sh WORK [OUT]}
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${ROOT:-$(cd "$HERE/../.." && pwd)}
REMOTE=${VLLM_REMOTE:-https://github.com/local-inference-lab/vllm}
REV=19f2c20ed4d64ae22bd009ceacadd85e2a81e354
RELEASE=${RELEASE:-1120}
case "$RELEASE" in
  1120) SERIES=$HERE/vllm/series; ENTRIES=719; SPLIT=702
        BASE_TREE=8fc99ca7546c5c538f0fe64e7c82954f1d2d7ae0
        TIER_TREE=47e196a297a31e2c4a7dd3a11f83b6b6ee02c638 ;;
  1126) SERIES=$HERE/vllm/series-1126; ENTRIES=759; SPLIT=742
        BASE_TREE=f3783b0ec61851ba481f6a3a316d0a76d0e171d8
        TIER_TREE=e45d318c694a36121dda4d37a38f0a17cba58afb ;;
  *) echo "make_overlay: RELEASE must be 1120 or 1126" >&2; exit 1 ;;
esac
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
  python3 -I - "$ROOT" "$SERIES" "$PWD" "$1" "$2" "$ENTRIES" <<'PY'
import sys
from pathlib import Path
root, series, tree, start, end, want = sys.argv[1:7]
sys.path.insert(0, root)
from spark_ds41f.build import apply_patch_series, read_series
entries = read_series(Path(series))
if len(entries) != int(want):
    raise SystemExit(f"make_overlay: the series has {len(entries)} entries, want {want}")
apply_patch_series(Path(tree), Path(series).parent, entries[int(start):int(end)])
PY
}

apply 0 "$SPLIT"
[ "$(git rev-parse 'HEAD^{tree}')" = "$BASE_TREE" ] || { echo "make_overlay: base tree differs" >&2; exit 1; }
apply "$SPLIT" "$ENTRIES"
[ "$(git rev-parse 'HEAD^{tree}')" = "$TIER_TREE" ] || { echo "make_overlay: tier tree differs" >&2; exit 1; }
for f in "${FILES[@]}"; do
  mkdir -p "$OUT/$(dirname "$f")"
  cp "$f" "$OUT/$f"
done
(cd "$OUT" && sha256sum "${FILES[@]}" > SHA256SUMS && cat SHA256SUMS)
if [ "$RELEASE" = 1120 ]; then
  cmp -s "$OUT/SHA256SUMS" "$HERE/overlay/SHA256SUMS" \
    || { echo "make_overlay: the sums differ from overlay/SHA256SUMS" >&2; exit 1; }
fi
for f in "${FILES[@]}"; do
  ref=$HERE/overlay/$f
  [ "$RELEASE" = 1126 ] && [ -f "$HERE/overlay-1126/$f" ] && ref=$HERE/overlay-1126/$f
  cmp -s "$OUT/$f" "$ref" || { echo "make_overlay: $f differs from ${ref#$HERE/}" >&2; exit 1; }
done
echo "make_overlay: release $RELEASE, tree $TIER_TREE, ${#FILES[@]} files in $OUT, equal to the committed copies"
