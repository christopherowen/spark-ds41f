#!/usr/bin/env bash
# The dynamic check: exp/gate_replay.py on candidate-54 (none), round 2 (r2f) and round 3
# (r3), each twice, in the order none r2f r3 r3 r2f none (each run one hold of the GPU
# lock, taken by run_exp.sh after checking ~/spark-hold.json). The recipe's
# VLLM_LATE_ADMISSION_LAYER=30. Outputs: raw/replay/<label>.json and .txt.
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r3
R2=$HOME/spark-lab/agents/candidate-55-rework-978-r2
cd "$WD"; mkdir -p raw/replay
export EXTRA_ENV="-e VLLM_LATE_ADMISSION_LAYER=30"
for spec in c54-1:none r2-1:$R2/arms/r2f r3-1:r3 r3-2:r3 r2-2:$R2/arms/r2f c54-2:none; do
  label=${spec%%:*}; arm=${spec#*:}
  scripts/run_exp.sh "$arm" raw/replay/$label.txt gate_replay.py $label /c/replay-$label.json
  cp cache/replay-$label.json raw/replay/$label.json
done
python3 - <<'PY'
import json
labels = ["c54-1", "c54-2", "r2-1", "r2-2", "r3-1", "r3-2"]
runs = {l: json.load(open(f"raw/replay/{l}.json")) for l in labels}
keys = ["frontend_requests_sha256", "trace_sha256", "outputs_sha256", "steps", "requests"]
summary = {l: {k: r[k] for k in keys} | {"mm_prefix_fastpath": r["mm_prefix_fastpath"]} for l, r in runs.items()}
ref = runs["c54-1"]
summary["identical_to_candidate_54"] = {l: all(r[k] == ref[k] for k in keys) and r["engine"] == ref["engine"]
                                        and r["by_scenario"] == ref["by_scenario"] for l, r in runs.items()}
summary["engine"] = ref["engine"]
summary["by_scenario"] = ref["by_scenario"]
json.dump(summary, open("raw/replay/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
PY
