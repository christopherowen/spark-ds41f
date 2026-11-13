#!/bin/bash
# apply_series.sh <vllm-dir> <series-file> [stop-after-pattern]
# Applies active patches (non-comment lines) with git am, fixed committer, as spark_ds41f/build.py does.
set -euo pipefail
dir=$1; series=$2; stop=${3:-}
sdir=$(cd "$(dirname "$series")" && pwd)
n=0
while IFS= read -r line; do
  case "$line" in ""|\#*) continue;; esac
  git -C "$dir" -c user.name="Christopher Owen" -c user.email="3221756+christopherowen@users.noreply.github.com" am --quiet --committer-date-is-author-date "$sdir/$line"
  n=$((n+1))
  if [ -n "$stop" ] && [[ "$line" == *"$stop"* ]]; then break; fi
done < "$series"
echo "applied $n"
