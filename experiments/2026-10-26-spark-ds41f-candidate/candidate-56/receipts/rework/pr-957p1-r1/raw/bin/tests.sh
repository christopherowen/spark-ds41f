#!/bin/bash
# 0795-r1 on candidate-54's image: the compressor test (TP1 identity at TP2..TP8) and the
# L2 schedule test (COMP's own budget); then the same two files on candidate-54 itself.
W=$HOME/spark-lab/agents/candidate-56-rework-957p1; OM=$W/bin/om.sh
for spec in "r1 1" "base 1"; do set -- $spec
  $OM pytest-$1-boot$2 $1 -- pytest --noconftest -o cache_dir=/tmp/pc -q \
    tests/models/test_deepseek_v4_1_compressor.py tests/models/test_deepseek_v4_1_l2_schedule.py
done
