#!/usr/bin/env bash
# One battery for every arm. Runs on the head against :8000, as a system
# unit. Read-only against the engine: nothing is restarted or killed (memwatch warns only).
#   bash trial_battery.sh <label>        results in $T/<label>/ (T: OUT, default ~/receipt)
# 1. RoCE counters on every rail of all four nodes (before); 2. the stage 2 battery (tests-label.sh: idletest, v41bench
# C1-C16, 175K needle, qeval, vision/tools, salted prefill sweep, 1M needle) with the 4-node MemAvailable sampler;
# 3. the stall probe (short request beside a 545K prefill, batch 8192); 4. the loop probe (legitimate repetitive output);
# 5. DSML client and agent checks; 6. DSML fixture capture, quoted and thinking; 7. RoCE counters (after).
set -u
# tests.sh, memwatch.sh and stall_probe.py: ../../2026-11-03-site-tp4-1030-1017/runs/ (tests.sh writes under $T/client/).
# dsml_fixtures*.json: earlier captures of capture_fixtures.py on the same model (the 2026-11-06 reference), not shipped.
L=${1:?label}; T=${OUT:-$HOME/receipt}; D=$T/$L; B=http://127.0.0.1:8000/v1
mkdir -p "$D"; cd "$D" || exit 1
[ -f "$D/BATTERY_DONE" ] && { echo "REFUSE: $D/BATTERY_DONE exists"; exit 1; }
stamp() { echo "$1 $(date -u +%FT%TZ)"; }
counters() {
  for h in localhost ${WORKERS:?the other ranks fabric names}; do
    cmd='for d in rocep1s0f1 roceP2p1s0f1; do c=/sys/class/infiniband/$d/ports/1/hw_counters; p=/sys/class/infiniband/$d/ports/1/counters; echo "$(hostname -s) $d retrans=$(cat $c/roce_adp_retrans) oos=$(cat $c/out_of_sequence) rnr=$(cat $c/rnr_nak_retry_err 2>/dev/null) xmit_words=$(cat $p/port_xmit_data) rcv_words=$(cat $p/port_rcv_data)"; done'
    if [ $h = localhost ]; then bash -c "$cmd"; else ssh -o BatchMode=yes -o ConnectTimeout=5 $h "$cmd"; fi
  done
}
stamp start
curl -fsS -m 10 http://127.0.0.1:8000/v1/models > models.json
curl -fsS -m 10 http://127.0.0.1:8000/metrics | grep -E '^vllm:num_requests_(running|waiting)\{' > load-at-start.txt
counters > roce-before.txt
stop=$D/memwatch.stop; rm -f "$stop"
bash $T/memwatch.sh "$D/memwatch.tsv" "$stop" & mw=$!
bash $T/tests.sh "$L" > "$D/tests.log" 2>&1; stamp "tests rc=$?"
python3 $T/stall_probe.py $B "$L" > stall.jsonl 2>&1; stamp "stall rc=$?"
python3 $T/loop_probe.py loop.jsonl > loop.log 2>&1; stamp "loop rc=$?"
python3 $T/overlay_check.py clients > dsml-clients.jsonl 2>&1; stamp "dsml-clients rc=$?"
python3 $T/overlay_check.py agents > dsml-agents.jsonl 2>&1; stamp "dsml-agents rc=$?"
python3 $T/capture_fixtures.py fixtures-quoted.json quoted > fixtures-quoted.log 2>&1; stamp "fixtures-quoted rc=$?"
python3 $T/capture_fixtures.py fixtures-thinking.json thinking > fixtures-thinking.log 2>&1; stamp "fixtures-thinking rc=$?"
python3 $T/overlay_check.py compare $T/dsml_fixtures.json fixtures-quoted.json > fixtures-quoted-cmp.jsonl 2>&1
python3 $T/overlay_check.py compare $T/dsml_fixtures_thinking.json fixtures-thinking.json > fixtures-thinking-cmp.jsonl 2>&1
counters > roce-after.txt
touch "$stop"; wait $mw
cp -p $T/client/battery-"$L".log $T/client/"$L"/*.txt $T/client/"$L"/*.json $T/client/"$L"/*.jsonl "$D/" 2>/dev/null
cp -rp $T/client/"$L"/bench "$D/" 2>/dev/null
stamp BATTERY_DONE | tee "$D/BATTERY_DONE"
