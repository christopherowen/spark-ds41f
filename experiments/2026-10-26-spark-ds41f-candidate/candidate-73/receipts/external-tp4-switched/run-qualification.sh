#!/bin/bash
set -u
cd /home/operator/spark-ds41f || exit 2
CFG=config/qualification-stock.json
OUT=/home/operator/c73-qualification
mkdir -p "$OUT"
run() {
 name="$1"; shift
 printf '%s START %s\n' "$(date -Is)" "$name" >> "$OUT/phases.log"
 "$@" > "$OUT/$name.log" 2>&1
 rc=$?
 printf '%s END %s rc=%s\n' "$(date -Is)" "$name" "$rc" >> "$OUT/phases.log"
 printf '%s\t%s\n' "$name" "$rc" >> "$OUT/status.tsv"
 return "$rc"
}
for boot in b1 b2; do
 run "start-$boot" ./bin/spark --cluster-config "$CFG" cluster start --replace --change-shape --apply || exit 10
 run "doctor-$boot" ./bin/spark --cluster-config "$CFG" doctor --live
 if [ "$boot" = b1 ]; then
  run determinism-wide python3 scripts/sonar_determinism.py --tokens 192 --long-lines 700 --wide http://REDACTED_IP:8889
  run determinism-wide-nologprobs python3 scripts/sonar_determinism.py --tokens 192 --long-lines 700 --wide --no-logprobs http://REDACTED_IP:8889
  run tool-gate-stock python3 scripts/tool_gate.py --base http://REDACTED_IP:8889/v1 --models deepseek-v4.1-flash --temperatures 0.0,0.3 --out "$OUT/tool-gate-stock.json"
 fi
 run "bench-$boot" ./bin/spark --cluster-config "$CFG" bench --full --url http://REDACTED_IP:8889 --compare manifests/benchmarks/2026-11-20-spark-ds41f-1120-switched.json --output "$OUT/bench-$boot"
 for suite in release single-stream sampled; do
  run "suite-$suite-$boot" ./bin/spark --cluster-config "$CFG" bench --suite "$suite" --url http://REDACTED_IP:8889 --label "candidate-73-stock-$boot" --output "$OUT/suite-$suite-$boot"
 done
 run "stop-$boot" ./bin/spark --cluster-config "$CFG" cluster stop --apply || exit 11
done
