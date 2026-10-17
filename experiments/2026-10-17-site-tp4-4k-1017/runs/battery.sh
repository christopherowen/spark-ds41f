#!/usr/bin/env bash
# Test battery used for both engines (run on the head against :8000). Usage: battery.sh <label>.
# idletest, the smoke request and vision_tools_demo send chat_template_kwargs thinking=false explicitly
# (the SGLang line's default; this recipe's vLLM defaults reasoning on). Tools: v41bench.py, idletest.py,
# needle_salted.py, qeval.py (MiaAI-Lab DeepSeek-v4.1-Flash-DGX-Sparks scripts/qeval.py), vision_tools_demo.py.
set -u
L="$1"; D="$HOME/dsv41-1017/$L"; mkdir -p "$D"; cd "$D" || exit 1
BASE=http://127.0.0.1:8000/v1; MODEL=deepseek-v4.1-flash; T="$HOME/dsv41-tools"
stamp() { echo "$1 $(date -u +%FT%TZ)"; }

stamp start
curl -s -m 120 $BASE/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":50,"temperature":0,"chat_template_kwargs":{"thinking":false}}' > smoke.json
stamp smoke-done
curl -fsS --max-time 15 http://127.0.0.1:8000/metrics > metrics-before-idle.txt
sed 's#"temperature": 0, "stream": True,#"temperature": 0, "chat_template_kwargs": {"thinking": False}, "stream": True,#' "$T/idletest.py" > idletest.py; grep -c '"thinking": False' idletest.py > idletest-thinking-flag.txt
IDLE_S=45 python3 idletest.py > idletest.txt 2>&1; stamp "idletest rc=$?"
curl -fsS --max-time 15 http://127.0.0.1:8000/metrics > metrics-after-idle.txt
mkdir -p "$D/bench"; python3 "$T/v41bench.py" --base $BASE --model $MODEL --label "$L" --out "$D/bench" --levels 1,4,8,16 > bench.txt 2>&1; stamp "bench rc=$?"
python3 "$HOME/dsv41-1017/needle_salted.py" $BASE 175000 > needle.json 2>&1; stamp "needle rc=$?"
grep MemAvailable /proc/meminfo > mem-after-needle.txt
( cd "$HOME/dsv41-4x-prod" && python3 scripts/qeval.py run "$L" --url $BASE/chat/completions --concurrency 1 ) > qeval.txt 2>&1; stamp "qeval rc=$?"
cp -p "$HOME/dsv41-4x-prod"/qeval-"$L"*.json . 2>/dev/null; ls "$HOME/dsv41-4x-prod" | grep -i qeval > qeval-files.txt
sed 's#"temperature": 0,#"temperature": 0, "chat_template_kwargs": {"thinking": False},#g' "$T/vision_tools_demo.py" > vision_tools_demo.py; python3 vision_tools_demo.py $BASE > vision.txt 2>&1; stamp "vision rc=$?"
stamp DONE > DONE
