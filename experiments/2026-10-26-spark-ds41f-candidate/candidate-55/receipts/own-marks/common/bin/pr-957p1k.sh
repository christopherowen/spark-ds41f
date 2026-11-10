#!/bin/bash
# #957 part 1 (0795): the compressor projection cells (h957k.py), ABBA, then the PR's test.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh; mkdir -p $B/out/pr-957p1/json
for spec in "base 1" "c56p1 1" "c56p1 2" "base 2"; do set -- $spec
  $OM 957p1 proj-$1-boot$2 $1 "${RECIPE_ENV[@]}" -- python3 /h/h957k.py --boot $2 --out /out/json/proj-$1-boot$2.json
done
