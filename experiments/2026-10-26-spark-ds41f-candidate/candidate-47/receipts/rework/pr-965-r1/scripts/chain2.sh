#!/bin/bash
# After chain1: attribute the one other failure of the worker/spec_decode subset, and rerun the
# c47 diagnosis with short kernel names.
cd ~/spark-lab/agents/candidate-47-rework-965
until grep -q chain1-done chain1.log; do sleep 10; done
T=tests/v1/worker/test_workspace.py::test_v41_tilelang_attention_reserves_scratch_in_its_model_lane
./run.sh attr-workspace-c47 c47 pytest -p tblib_stub -q -rf $T
./run.sh attr-workspace-fix-to-0766 fix-to-0766 pytest -p tblib_stub -q -rf $T
./run.sh attr-workspace-fix fix pytest -p tblib_stub -q -rf $T
./run.sh diag-c47-named c47 python3 /harness/diag_declared.py
echo chain2-done
