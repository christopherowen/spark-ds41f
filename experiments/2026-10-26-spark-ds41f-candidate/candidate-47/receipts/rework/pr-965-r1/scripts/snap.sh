#!/bin/bash
# usage: snap.sh <repo-dir> <base-rev> <head-rev> <overlay-name>
# Copies every file that differs between base and head (as of head) into ovl/<overlay-name>/.
set -e
W=$HOME/spark-lab/agents/candidate-47-rework-965
REPO=$1; BASE=$2; HEAD=$3; NAME=$4
rm -rf $W/ovl/$NAME; mkdir -p $W/ovl/$NAME
cd $REPO
git diff --name-only --diff-filter=AM $BASE $HEAD | while IFS= read -r f; do
  mkdir -p "$W/ovl/$NAME/$(dirname "$f")"
  git show "$HEAD:$f" > "$W/ovl/$NAME/$f"
done
echo "$NAME: $(git rev-parse $HEAD) vs $(git rev-parse $BASE): $(cd $W/ovl/$NAME && find . -type f | wc -l) files"
