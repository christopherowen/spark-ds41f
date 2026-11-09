#!/bin/bash
# extract.sh N... : export each PR head's experiment directory into src/pr-N
set -e
B=$HOME/spark-lab/agents/candidate-47-own-marks; cd $B/src/repo
for n in "$@"; do
  d=$(git diff --name-only $(git merge-base origin/main pr-$n) pr-$n | grep '^experiments/.*/README.md$' | xargs dirname)
  rm -rf $B/src/pr-$n; mkdir -p $B/src/pr-$n
  git archive pr-$n $d | tar xf - -C $B/src/pr-$n
  echo "pr-$n $(git rev-parse pr-$n) $d"; ls $B/src/pr-$n/$d
done
