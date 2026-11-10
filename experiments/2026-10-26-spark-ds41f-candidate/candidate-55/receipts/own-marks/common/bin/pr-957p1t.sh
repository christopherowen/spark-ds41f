#!/bin/bash
# #957 part 1 (0795): tests/models/test_deepseek_v4_1_compressor.py, as candidate-56's README runs it.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh
for spec in "c56p1 1" "c56p1 2" "base 1"; do set -- $spec
  $OM 957p1 pytest-$1-boot$2 $1 -- pytest --noconftest -o cache_dir=/tmp/pc -q tests/models/test_deepseek_v4_1_compressor.py
done
