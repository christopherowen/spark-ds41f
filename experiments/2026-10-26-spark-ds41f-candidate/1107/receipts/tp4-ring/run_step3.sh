#!/usr/bin/env bash
set -uo pipefail
cd /home/swank/projects/spark-ds41f
GATE_DIR=/home/swank/spark-lab/1107-tp4-ring/gate
mkdir -p "$GATE_DIR"

# Background heartbeat refresher every 5 minutes
(
  while true; do
    python3 -c "import json, datetime; p=\"/home/swank/spark-hold.json\"; d=json.load(open(p)); d[\"heartbeat\"]=datetime.datetime.now(datetime.timezone.utc).strftime(\"%Y-%m-%dT%H:%M:%SZ\"); json.dump(d, open(p,\"w\"), indent=2)" || true
    sleep 300
  done
) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

echo "=== [1] Booting production 1107/tp4.json with telemetry ==="
python3 /home/swank/spark-lab/1107-tp4-ring/telemetry/boot_with_telemetry.py \
  experiments/2026-10-26-spark-ds41f-candidate/1107/tp4.json \
  boot1-tp4-prod \
  /home/swank/spark-lab/1107-tp4-ring/telemetry/boot1-tp4-prod.json

for n in dgx1 dgx2 dgx3 dgx4; do
  ssh "$n" "sudo find /dev/shm -maxdepth 1 -type f -name \"psm_*\" -mmin +10 -delete || true"
done

bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4.json doctor --live | tee /home/swank/spark-lab/1107-tp4-ring/telemetry/doctor-live-tp4-prod.txt

run_gate() {
  local name="$1"
  shift
  echo "=== [GATE] $name ==="
  if "$@" > "$GATE_DIR/$name.json"; then
    echo "=== [GATE] $name -> EXIT 0 ==="
  else
    local rc=$?
    echo "=== [GATE] $name -> EXIT $rc ==="
  fi
}

run_gate token_determinism_wide_logprobs python3 experiments/2026-10-12-r6g/token_determinism_text.py http://10.0.1.71:8000 --tokens 192 --long-lines 700 --wide
run_gate token_determinism_wide python3 experiments/2026-10-12-r6g/token_determinism_text.py http://10.0.1.71:8000 --tokens 192 --long-lines 700 --wide --no-logprobs
run_gate token_determinism_1100_logprobs python3 experiments/2026-10-12-r6g/token_determinism_text.py http://10.0.1.71:8000 --tokens 192 --long-lines 1100
run_gate token_determinism_1100 python3 experiments/2026-10-12-r6g/token_determinism_text.py http://10.0.1.71:8000 --tokens 192 --long-lines 1100 --no-logprobs
run_gate distinct_streams_c8 python3 scripts/distinct_streams.py http://10.0.1.71:8000 --streams 8 --samples 4 --tokens 256
run_gate distinct_streams_c4 python3 scripts/distinct_streams.py http://10.0.1.71:8000 --streams 4 --samples 4 --tokens 256
run_gate prefix_extension python3 experiments/2026-10-25-kv-zeroing/prefix_extension.py http://10.0.1.71:8000 --threshold 16336
run_gate controls_digest python3 experiments/2026-10-24-tilelang-completion/controls_digest.py http://10.0.1.71:8000 --tokens 96
run_gate controls_digest_prompt_logprobs python3 experiments/2026-10-24-tilelang-completion/controls_digest.py http://10.0.1.71:8000 --tokens 96 --scenarios prompt-logprobs
run_gate sampled_digest python3 experiments/2026-10-24-tilelang-completion/sampled_digest.py http://10.0.1.71:8000 --tokens 192
run_gate image_digest python3 experiments/2026-10-24-tilelang-completion/image_digest.py http://10.0.1.71:8000 --tokens 96
run_gate vision_check python3 experiments/2026-09-25-vision/vision_check.py http://10.0.1.71:8000
run_gate acceptance_sweep python3 scripts/acceptance_sweep.py http://10.0.1.71:8000 --tokens 256
run_gate determinism python3 experiments/2026-09-29-determinism/determinism.py http://10.0.1.71:8000 --repeats 5
run_gate determinism_concurrent python3 experiments/2026-09-29-determinism/determinism_concurrent.py http://10.0.1.71:8000
run_gate prompt_logprobs_check python3 experiments/2026-10-24-prompt-logprobs-sink/prompt_logprobs_check.py http://10.0.1.71:8000 --label 1107-tp4
run_gate logprob_ids_check python3 experiments/2026-10-25-logprob-token-ids/logprob_ids_check.py http://10.0.1.71:8000 --label 1107-tp4
run_gate sink_rejection_batch python3 experiments/2026-10-19-sink-rejection-v2/batch.py http://10.0.1.71:8000
run_gate sink_rejection_longctx python3 experiments/2026-10-19-sink-rejection-v2/longctx.py http://10.0.1.71:8000
run_gate sink_rejection_probes python3 experiments/2026-10-19-sink-rejection-v2/probes.py http://10.0.1.71:8000 --nodes dgx1,dgx2,dgx3,dgx4 --container spark-ds41f
run_gate long_context_960k python3 experiments/2026-10-04-tp4-memory-tuning/long_context.py http://10.0.1.71:8000 --tokens 960000 --depths 0.1,0.9
run_gate kv_counters_prod python3 experiments/2026-10-25-kv-zeroing/kv_counters.py http://10.0.1.71:8000 --nodes dgx1,dgx2,dgx3,dgx4 --container spark-ds41f

echo "=== [2] Running 5 Quality Evals + AgentX c8 from dgx4 ==="
ssh dgx4 "bash /home/swank/spark-lab/1107-tp4-ring/run_evals_dgx4.sh"

run_gate kv_counters_after_evals python3 experiments/2026-10-25-kv-zeroing/kv_counters.py http://10.0.1.71:8000 --nodes dgx1,dgx2,dgx3,dgx4 --container spark-ds41f

echo "=== STEP 3 COMPLETE ==="
