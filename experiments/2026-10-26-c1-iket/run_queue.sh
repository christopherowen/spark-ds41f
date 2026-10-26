#!/usr/bin/env bash
# The c1 IKET study's kernel-local runs on its assigned node, one after another;
# lab.py's GPU lock orders them against other sessions' runs. Stops if a release
# request appears or the smoke run fails.
#
# usage: run_queue.sh SMOKE_OUT [BUNDLE:OUT ...]   (outputs under ~/spark-lab/agents/c1-iket/out)
set -u
cd ~/spark-lab/agents/c1-iket/spark-ds41f
out=~/spark-lab/agents/c1-iket/out
run() {  # bundle, output name
  for request in ~/spark-request.json ~/spark3-request.json; do
    if [ -e "$request" ]; then echo "=== $(date "+%F %T %Z") $request present: stopping"; exit 3; fi
  done
  if [ -e "$out/$2" ]; then echo "=== $(date "+%F %T %Z") $out/$2 exists: stopping"; exit 4; fi
  echo "=== $(date "+%F %T %Z") start $1 -> $out/$2"
  python3 scripts/lab.py kernel-local "experiments/2026-10-26-c1-iket/bundles/$1" --out "$out/$2"
  local code=$?
  echo "=== $(date "+%F %T %Z") end $1: exit $code"
  return $code
}
run smoke "$1" || exit 1
shift
for pair in "$@"; do run "${pair%%:*}" "${pair##*:}"; done
