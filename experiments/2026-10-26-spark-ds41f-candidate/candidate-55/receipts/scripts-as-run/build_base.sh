#!/bin/bash
# Run on sp8 in ~/spark-lab/agents/round-55-57-select (CPU only). The candidate-54 base tree, built from its
# series: vLLM 19f2c20ed + the 671 active patches of candidate-54/vllm/series (resolved to repository paths
# in c54-series.txt; the patch files were copied from the laptop's scratch clone of origin/main 920e1d50),
# applied with git am, the fixed committer and --committer-date-is-author-date, as spark_ds41f/build.py does.
# The object store is a local clone of round 2's base-vllm (which holds 19f2c20ed); nothing is fetched.
set -e
W=~/spark-lab/agents/round-55-57-select; cd $W
git clone -q --no-hardlinks --no-checkout ~/spark-lab/agents/round-51-53-select/base-vllm c54-vllm
cd c54-vllm
git checkout -q -B spark/build 19f2c20ed4d64ae22bd009ceacadd85e2a81e354
git config gc.auto 0
n=0
while read f; do
  git -c user.name="Christopher Owen" -c user.email=3221756+christopherowen@users.noreply.github.com \
    am --quiet --committer-date-is-author-date "$W/repo/$f"
  n=$((n+1))
done < $W/c54-series.txt
echo "applied $n"
git rev-parse HEAD HEAD^{tree}      # 4e8133ff10a1..., 5f021d9a7f01...: candidate-54's patch_head and expected_tree
git diff --check 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 HEAD && echo diff-check-clean
git branch -f round-55-57-base HEAD
git checkout -q round-55-57-base
# Later, after apply_alone.py and show_conflicts.py had run (two-way markers) and before split_hunks.py and
# rebase_context.py ran: diff3 markers, so a resolution can see the merge base.
git config merge.conflictstyle diff3
