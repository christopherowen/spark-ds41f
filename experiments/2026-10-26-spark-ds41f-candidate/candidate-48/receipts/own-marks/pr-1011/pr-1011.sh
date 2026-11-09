. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
EXP=$B/src/pr-1011/experiments/2026-11-08-vllm-mainline-patch-audit
W=$B/work/pr-1011; mkdir -p $W; chmod 777 $W
for img in C R; do
  [ $img = C ] && I=$C && tag=candidate48 || { I=$R; tag=1107; }
  for BOOT in 1 2; do
    $B/run.sh 1011 pytest-and-audit-$tag-boot$BOOT $I -- flock $L docker run --rm --name c48om-1011-$tag-b$BOOT --gpus all --ipc=host --entrypoint bash \
      -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/opt/release_tests \
      -v $RT:/opt/release_tests:ro -v $EXP:/opt/audit:ro -v $W:/opt/audit-out \
      -w /opt/spark-ds41f/candidate/vllm $I \
      -lc "python3 -m pytest -v -p no:cacheprovider -p tblib_stub -p offline_skip tests/v1/worker/test_gpu_worker.py tests/v1/worker/test_workspace.py -k \"llm_repr or batch_invariant or kpool\" | tee /opt/audit-out/pytest-$tag-boot${BOOT}.txt; python3 /opt/audit/screen_vllm_mainline_patch_audit.py --boot ${BOOT} --output /opt/audit-out/audit-$tag-boot${BOOT}.json"
  done
done
