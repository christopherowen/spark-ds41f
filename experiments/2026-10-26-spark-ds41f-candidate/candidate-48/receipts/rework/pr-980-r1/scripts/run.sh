#!/bin/bash
# run.sh <label> -- <command...>: runs the command, tees output to out/<label>.txt,
# appends the command, start/end UTC and exit code to commands.txt.
W=$HOME/spark-lab/agents/candidate-48-rework-980
label=$1; shift; [ "$1" = "--" ] && shift
start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$@" > "$W/out/$label.txt" 2>&1
rc=$?
end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{ printf "## %s\nstart_utc %s end_utc %s exit %s\n" "$label" "$start" "$end" "$rc"; printf "%q " "$@"; printf "\n\n"; } >> "$W/commands.txt"
exit $rc
