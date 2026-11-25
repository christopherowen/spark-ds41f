#!/usr/bin/env bash
# cpu_suite.sh TREE: run the CPU tests of the disk tier and of sink rejection in a vLLM tree.
#
# TREE  a vLLM tree with vllm/series applied: WORK/vllm after tools/make_overlay.sh WORK (tree
#       47e196a2), or the candidate-73 series with 9030 to 9046 applied on top.
#
# Runs three pytest commands, one after the other, on the CPU path:
#   tests/v1/simple_kv_offload/test_ds41_tier_sink.py (the tests of 9046),
#   tests/v1/core/test_sink_draft_zero.py (the sink scheduler suite of the fork),
#   tests/v1/simple_kv_offload and tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py
#   (the offload suite; test_worker.py and test_integration.py skip without CUDA).
#
# PYTHON  a Python 3.12 with a CPU build of torch, the packages of TREE/requirements/common.txt
#         and pytest (default: python3).
# DS41_ODIRECT_DIR  a folder on a filesystem with O_DIRECT, such as ext4 or xfs (default:
#         TREE/../odirect). tmpfs may refuse O_DIRECT.
# The tests run with HF_HUB_OFFLINE=1, and many of them build a ModelConfig for
# facebook/opt-125m, so the config of that model must be in the Hugging Face cache first:
#   hf download facebook/opt-125m config.json
set -euo pipefail
TREE=$(cd "${1:?usage: cpu_suite.sh TREE}" && pwd)
PYTHON=${PYTHON:-python3}
ODIRECT=${DS41_ODIRECT_DIR:-$TREE/../odirect}
mkdir -p "$ODIRECT"
cd "$TREE"
[ -z "$(git status --porcelain)" ] || { echo "cpu_suite: $TREE has changes" >&2; exit 1; }
echo "cpu_suite: tree $(git rev-parse 'HEAD^{tree}')"
run() {
  PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 DS41_ODIRECT_DIR="$ODIRECT" \
    "$PYTHON" -B -m pytest -q -p no:cacheprovider -rs "$@"
}
run tests/v1/simple_kv_offload/test_ds41_tier_sink.py
run tests/v1/core/test_sink_draft_zero.py
run tests/v1/simple_kv_offload tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py
