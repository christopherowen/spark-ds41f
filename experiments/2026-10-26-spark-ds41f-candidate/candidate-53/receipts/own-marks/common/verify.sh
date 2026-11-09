#!/bin/bash
# verify.sh ARM: no-GPU container with the arm's mounts; prints each overlaid file's SHA-256 as the
# container sees it and where Python imports the changed modules from.
B=$HOME/spark-lab/agents/candidate-53-own-marks; IMG=$(cat $B/image-id); arm=$1
mounts=()
if [ "$arm" = c53 ]; then list=$B/changed-files.txt; src=$B/ovl/c53; else list=$B/no1009-files.txt; src=$B/ovl/no1009; fi
while read -r f; do mounts+=(-v "$src/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $list
docker run --rm --network none "${mounts[@]}" -v $B/changed-files.txt:/list:ro -w /opt/spark-ds41f/candidate/vllm --entrypoint bash $IMG -c '
while read -r f; do echo "$(sha256sum < $f | cut -c1-12) $f"; done < /list
echo "pycache for changed modules in image:"; for f in $(grep "^vllm/.*\.py$" /list); do d=$(dirname $f)/__pycache__; b=$(basename $f .py); ls $d/$b.*.pyc 2>/dev/null; done | wc -l
python3 -c "
import importlib
for m in [\"vllm.v1.worker.gpu.model_runner\",\"vllm.v1.worker.gpu.spec_decode.tilelang_copy\",\"vllm.models.deepseek_v4_1.ced\",\"vllm.utils.mem_utils\",\"vllm.v1.worker.display_carveout\",\"vllm.models.deepseek_v4_1.tilelang.front\"]:
    try:
        mod = importlib.import_module(m); print(m, mod.__file__)
    except Exception as e: print(m, \"import error\", type(e).__name__, e)
from vllm.models.deepseek_v4_1.ced import CEDState; print(\"CEDState.stage_floors\", hasattr(CEDState, \"stage_floors\"))
from vllm.v1.worker import display_carveout as dc; print(\"display_carveout.compact\", hasattr(dc, \"compact\"))
" 2>&1 | grep -v Warning'
