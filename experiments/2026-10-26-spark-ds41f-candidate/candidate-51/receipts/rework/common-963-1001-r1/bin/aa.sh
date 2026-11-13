#!/bin/bash
# After final.sh: an A/A rerun of #963-r1 sections D (certified screen, both screen-config
# sets) and E (split-K), whose kernels have byte-identical SASS in both arms
# (out/pr-common/sass-*.txt): ABBA in one session, two boots per arm.
W=$HOME/spark-lab/agents/candidate-51-rework-963-1001; RW=$W/bin/rw.sh; mkdir -p $W/out/pr-963/json
while kill -0 2839360 2>/dev/null; do sleep 20; done
for spec in "base 1" "c51r1 1" "c51r1 2" "base 2"; do set -- $spec
  $RW 963 aa-DE-$1-boot$2 $1 -e H963_SECTIONS=DE -- python3 /h/h963aa.py /out/json/aa-DE-$1-boot$2.json
done
