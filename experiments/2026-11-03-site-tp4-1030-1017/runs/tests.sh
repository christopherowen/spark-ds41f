#!/usr/bin/env bash
# One labelled run on the head against :8000: the battery of ../../2026-10-17-site-tp4-4k-1017/runs/battery.sh
# (idletest, v41bench C1/C4/C8/C16, the ~545K salted needle, qeval 75, vision/tools 7), then a salted cold-prefill
# sweep (2 x ~3K/13K/54K/109K tokens) and a ~1M-token salted needle. Usage: tests.sh <label>.
# DIR holds battery.sh, needle_salted.py and the battery's client tools.
set -u
L="$1"; DIR=${DIR:-$HOME/bench}; D=$DIR/$L; mkdir -p "$D"; B=http://127.0.0.1:8000/v1
echo "start $(date -u +%FT%TZ)"
[ -f "$D/DONE" ] && { echo "REFUSE: $D/DONE exists"; exit 1; }
curl -fsS -m 10 http://127.0.0.1:8000/metrics | grep -E '^vllm:num_requests_(running|waiting)' > "$D/load-at-start.txt"
bash "$DIR/battery.sh" "$L" > "$DIR/battery-$L.log" 2>&1; echo "battery rc=$? $(date -u +%T)"
cd "$D"
for rep in 1 2; do for nw in 1100 4300 17500 35000; do python3 "$DIR/needle_salted.py" $B $nw; done; done > prefill-sweep.jsonl 2>&1; echo "sweep rc=$? $(date -u +%T)"
grep MemAvailable /proc/meminfo > mem-before-1m.txt
python3 "$DIR/needle_salted.py" $B 318000 > needle-1m.json 2>&1; echo "needle-1m rc=$? $(date -u +%T)"
grep MemAvailable /proc/meminfo > mem-after-1m.txt
echo "TESTS_DONE $(date -u +%FT%TZ)"
