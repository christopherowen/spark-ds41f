#!/bin/bash
# The replay queue: each job takes the lock on its own (run.sh), so other agents interleave.
W=$HOME/spark-lab/agents/round3-957p5
$W/scripts/replay.sh replay-986fit-seed0 0 20000; echo "replay-986fit-seed0 rc=$?"
$W/scripts/replay.sh replay-986fit-seed1 1 20000; echo "replay-986fit-seed1 rc=$?"
$W/scripts/replay.sh replay-tp3fit-seed0 0 20000 /w/out/fit-tp3-seed0.json; echo "replay-tp3fit-seed0 rc=$?"
$W/scripts/replay.sh replay-tp3fit-seed1 1 20000 /w/out/fit-tp3-seed1.json; echo "replay-tp3fit-seed1 rc=$?"
$W/scripts/pytest.sh pytest-c57p5-run2 c57p5; echo "pytest-c57p5-run2 rc=$?"
$W/scripts/pytest.sh pytest-c54 c54; echo "pytest-c54 rc=$?"
echo queue-done
