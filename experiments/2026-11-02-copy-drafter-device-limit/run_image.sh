#!/bin/bash
# Run a shell command in spark-ds41f:candidate-37 with the files of TREE changed since
# candidate-37's patch head (5b7cfa37f) mounted read-only over the image's vLLM tree,
# and the tblib stand-in (docs/testing.md) on PYTHONPATH.
#
# usage: TREE=/tmp/vllm-device-limit PLUGINS=/tmp/loop-test-plugins ./run_image.sh "COMMAND"
set -euo pipefail
TREE=${TREE:?the patched vLLM tree}
PLUGINS=${PLUGINS:?a directory holding tblib_stub.py}
V=/opt/spark-ds41f/candidate/vllm
mounts=()
for f in $(cd "$TREE" && { git diff --name-only 5b7cfa37f; git ls-files --others --exclude-standard; } | sort -u); do
  [ -f "$TREE/$f" ] && mounts+=(-v "$TREE/$f:$V/$f:ro")
done
exec docker run --rm --gpus all --ipc=host -v "$PLUGINS:/plugins:ro" "${mounts[@]}" \
  -e PYTHONPATH=/plugins -e HF_HUB_OFFLINE=1 -w $V --entrypoint bash \
  spark-ds41f:candidate-37 -c "$*"
