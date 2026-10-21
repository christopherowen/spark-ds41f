#!/usr/bin/env bash
# End-to-end smoke test of vLLM 0222 on one idle node (dgx4): a dummy-weight Llama-shaped
# model (about 1.6B parameters, random weights, DS4.1's tokenizer) served from the r6x image
# with the candidate overlay, V2 runner, async scheduling and the multiprocess executor (the
# worker process and its output thread carry the late-admission reply). Each arm boots a
# server on 127.0.0.1:8199 and runs smoke_client.py: greedy outputs of eight prompts, then
# four closed-loop streams decomposed by analyze_trace.py.
#
#   smoke.sh            arms: late admission off (-1), then on at layer 18 of 24
#
# Outputs must be identical; with late admission the requests stop waiting a step that does
# not carry them. Not a performance measurement: the steps are a few milliseconds long.
set -u
LAB=${LAB:-$HOME/spark-lab/late-admission}
V=/opt/spark3/candidate/vllm
IMG=vllm-ds41f-kkref:19f2c20ed4d6-r6x
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/* | head -1)
D=$LAB/smoke
mkdir -p "$D/tiny"
cp "$SNAP/tokenizer.json" "$SNAP/tokenizer_config.json" "$D/tiny/"
cat > "$D/tiny/config.json" <<'EOF'
{"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 2048,
 "intermediate_size": 8192, "num_hidden_layers": 24, "num_attention_heads": 16,
 "num_key_value_heads": 16, "vocab_size": 129280, "max_position_embeddings": 4096,
 "rms_norm_eps": 1e-6, "torch_dtype": "bfloat16", "bos_token_id": 0, "eos_token_id": 1}
EOF
mounts=()
for f in $(cd "$LAB/candidate" && find vllm -type f -name '*.py'); do
  mounts+=(-v "$LAB/candidate/$f:$V/$f:ro")
done

for layer in -1 18; do
  docker rm -f la-smoke > /dev/null 2>&1
  docker run -d --name la-smoke --gpus all --network host --shm-size 16g --memory 48g -w "$V" "${mounts[@]}" \
    -v "$D/tiny:/w/tiny:ro" -v "$LAB/smoke_client.py:/w/smoke_client.py:ro" \
    -v "$LAB/analyze_trace.py:/w/analyze_trace.py:ro" \
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_LATE_ADMISSION_LAYER="$layer" \
    --entrypoint python3 "$IMG" -m vllm.entrypoints.openai.api_server \
    --model /w/tiny --served-model-name tiny --load-format dummy --seed 0 \
    --async-scheduling --distributed-executor-backend mp --max-model-len 2048 \
    --max-num-seqs 8 --gpu-memory-utilization 0.25 --host 127.0.0.1 --port 8199 > /dev/null
  for _ in $(seq 600); do
    curl -sf -m 2 http://127.0.0.1:8199/v1/models > /dev/null && break
    docker inspect -f '{{.State.Running}}' la-smoke 2>/dev/null | grep -q true || break
    sleep 1
  done
  echo "== late admission layer $layer"
  docker exec la-smoke python3 /w/smoke_client.py http://127.0.0.1:8199 /tmp/out.json \
    || docker logs --tail 60 la-smoke
  docker logs la-smoke 2>&1 | grep -i -E "error|traceback" | head -5
  docker rm -f la-smoke > /dev/null
done
