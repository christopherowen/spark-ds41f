#!/bin/bash
# The release gate (gate-manifest.json) in the candidate-37 image, with the
# branch's changed files (tree/, listed in changed.txt) over the image's tree,
# or none of them with CONTROL=1. Outcomes under /tmp/842-final/c/release-tests/LABEL.
# io_uring (the disk Engram reader, the direct loader) needs seccomp unconfined and an
# unlimited memlock, as the lab's io_uring bundles have.
set -euo pipefail
W=/tmp/842-final
LABEL=${1:?label}
DST=/opt/spark-ds41f/candidate/vllm
MOUNTS=()
if [ "${CONTROL:-0}" != 1 ]; then
  while read -r f; do MOUNTS+=(-v "$W/tree/$f:$DST/$f:ro"); done < "$W/changed.txt"
fi
MANIFEST=${MANIFEST:-$W/gate-manifest.json}
mkdir -p "$W/c/engram-rows-test" "$W/c/direct-io-test" "$W/c/tilelang-release-tests"
exec docker run --rm --gpus all --ipc=host --network none \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/b -e TILELANG_CACHE_DIR=/c/tilelang-release-tests \
  -e ENGRAM_ROWS_TEST_DIR=/c/engram-rows-test -e DIRECT_IO_TEST_DIR=/c/direct-io-test \
  -v "$W/c:/c" -v "$MANIFEST:/b/manifest.json:ro" \
  -v "$W/sd/scripts/release_tests/release_test_runner.py:/b/release_test_runner.py:ro" \
  -v "$W/sd/scripts/release_tests/tblib_stub.py:/b/tblib_stub.py:ro" \
  -v "$W/sd/scripts/release_tests/offline_skip.py:/b/offline_skip.py:ro" \
  "${MOUNTS[@]}" -w "$DST" --entrypoint python3 spark-ds41f:candidate-37 \
  /b/release_test_runner.py /b/manifest.json "/c/release-tests/$LABEL"
