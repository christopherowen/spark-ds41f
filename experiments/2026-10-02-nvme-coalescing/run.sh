#!/usr/bin/env bash
# NVMe interrupt-coalescing A/B on the live production profile (no restart).
# Arms: on (NVIDIA's default 0x107: 8 completions or 100 us), off (0), on again.
# Per arm on every node: set and verify feature 8; NVMe interrupt and read counters
# before/after; latency.py on the model blob; then one lean bench from dgx1.
set -u
NODES="dgx1 dgx2 dgx3"
OUT=$HOME/spark3-lab/nvme/results-$(date -u +%Y%m%dT%H%M)
mkdir -p "$OUT"
REPO=$HOME/projects/spark3-vllm-ds41f
BLOB=$(ls -S $HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/blobs/* | head -1)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }

set_all() {  # $1 = hex value
  for n in $NODES; do
    ssh -n "$n" "sudo -n nvme set-feature /dev/nvme0 -f 8 --value $1 >/dev/null && sudo -n nvme get-feature /dev/nvme0 -f 8 | grep -o 'Current value:0x[0-9a-f]*'" | sed "s/^/$n /" | tee -a "$OUT/run.log"
  done
}
counters() {  # $1 = label
  for n in $NODES; do
    ssh -n "$n" "echo \$(grep nvme0q /proc/interrupts | awk '{s=0; for(i=2;i<=NF;i++) if (\$i ~ /^[0-9]+\$/) s+=\$i; t+=s} END {print t}') \$(awk '{print \$1, \$3, \$5, \$7}' /sys/block/nvme0n1/stat) \$(date +%s.%N)" | sed "s/^/$1 $n /" >> "$OUT/counters.txt"
  done
}
restore() { log "restore: coalescing on (0x107)"; set_all 0x107; }
trap restore EXIT

i=0
for arm in on off on; do
  i=$((i + 1)); label="$i-$arm"
  log "arm $label"
  if [ "$arm" = on ]; then set_all 0x107; else set_all 0x0; fi
  sleep 2
  for n in $NODES; do
    ssh -n "$n" "python3 $HOME/spark3-lab/nvme/latency.py $BLOB --seconds 5 --burst 32 --bursts 400" | sed "s/^/$label /" >> "$OUT/latency.jsonl" &
  done
  wait
  counters "$label-start"
  (cd "$REPO" && bin/spark bench --suites decode,prefill --decode-cases prose,code --concurrency 1,8 \
     --min-samples 3 --max-samples 3 --prefill-text source --prefill-sizes 4096,32768 --prefill-repeats 2 \
     --compare none --output "results/private/bench/nvme-$label" > "$OUT/bench-$label.txt" 2>&1)
  log "bench $label exit $?"
  (cd "$REPO" && bin/spark bench --suites prefill --prefill-text novel --prefill-sizes 4096,32768 \
     --prefill-repeats 2 --compare none --output "results/private/bench/nvme-$label-novel" > "$OUT/bench-$label-novel.txt" 2>&1)
  log "novel $label exit $?"
  counters "$label-end"
done
log "done; results in $OUT"
