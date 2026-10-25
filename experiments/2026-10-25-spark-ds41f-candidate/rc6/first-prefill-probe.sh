#!/usr/bin/env bash
# The first long prefills after a fresh boot: the deploy's step 5b. Run it on dgx1 from the
# repository root right after `cluster start --apply` reports ready, before any other request
# (doctor --live included).
#
# It sends three cold 32K prefills of the same document, then one cold 128K, one at a time, each
# with a fresh cache salt. Around each request it records every node's MemAvailable and NVMe
# sectors read, plus the server's prompt-token counter: another client's traffic shows up as a
# larger step in that counter. Afterwards it lists any line the containers logged during the probe
# about compiling, preparing or tuning kernels.
#
# Interpretation:
# - If the first 32K is slower than the next two while each node's MemAvailable drops once and
#   stays down, the first long prefill allocated memory.
# - If it is slower with flat memory, something set itself up on first use, for example
#   collectives connecting lazily.
# - If it is level, the warm-up covers the path.
set -euo pipefail

out=${1:?usage: first-prefill-probe.sh OUTPUT_DIR}
url=${PROBE_URL:-http://10.0.1.71:8000}
nodes=(dgx1 dgx2 dgx3 dgx4)
mkdir -p "$out"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)

snapshot() {
  local label=$1 tokens host
  tokens=$(curl -s "$url/metrics" | awk '/^vllm:prompt_tokens_total/ {s += $2} END {printf "%d", s}')
  for host in "${nodes[@]}"; do
    ssh "$host" 'cat /proc/meminfo /proc/diskstats' |
      awk -v label="$label" -v host="$host" -v tokens="$tokens" '
        /^MemAvailable:/ {available = $2}
        $3 == "nvme0n1" {sectors = $6}
        END {printf "%s %s mem_available_kib=%d nvme_sectors_read=%d prompt_tokens_total=%d\n",
             label, host, available, sectors, tokens}'
  done | tee -a "$out/nodes.txt"
}

request() {
  local label=$1 size=$2
  python3 -m servebench prefill "$url" --sizes "$size" --repeats 1 --label "$label" \
    --output "$out/$label.json"
}

snapshot before
request first-32k-1 32768
snapshot after-32k-1
request first-32k-2 32768
snapshot after-32k-2
request first-32k-3 32768
snapshot after-32k-3
request first-128k 131072
snapshot after-128k

for host in "${nodes[@]}"; do
  echo "== $host: kernel lines logged since $started"
  ssh "$host" "docker logs --since $started spark-ds41f 2>&1 |
    grep -iE 'compil|prepar|jit|tuning|autotun|capturing|not prepared' | head -20 || true"
done | tee "$out/kernel-lines.txt"

python3 - "$out" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
for label in ("first-32k-1", "first-32k-2", "first-32k-3", "first-128k"):
    sample = json.loads((out / f"{label}.json").read_text())["samples"][0]
    print(f"{label:12} {sample['prompt_tokens']:>7} tokens  ttft {sample['ttft_s']:.3f} s  {sample['tps']:.0f} tok/s")
EOF
