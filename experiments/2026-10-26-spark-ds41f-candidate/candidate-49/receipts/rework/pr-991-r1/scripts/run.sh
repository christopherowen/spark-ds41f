#!/usr/bin/env bash
# usage: run.sh <label> <image-id> -- <command...>
# Records the exact command, host, image ID, UTC start/end, load and exit code
# to out/commands.txt and the combined stdout/stderr to out/<label>.txt.
set -u
W=$HOME/spark-lab/agents/candidate-49-rework-991
label=$1; image=$2; shift 2; [ "${1:-}" = "--" ] && shift
d=$W/out; mkdir -p "$d"
out=$d/$label.txt
if [ -e "$HOME/spark-hold.json" ]; then echo "HOLD present, not running" >&2; cat "$HOME/spark-hold.json" >&2; exit 99; fi
running=$(docker ps --format "{{.Image}} {{.Names}}" | tr "\n" ";")
img_id=$(docker image inspect --format "{{.Id}}" "$image")
tags=$(docker image inspect --format "{{join .RepoTags \",\"}}" "$image")
{
echo "### $label"
echo "host: $(hostname) (sp8)"
echo "image: $img_id ($tags)"
echo "start_utc: $(date -u +%FT%TZ)"
echo "loadavg_before: $(cut -d" " -f1-3 /proc/loadavg)"
echo "docker_ps_before: ${running:-none}"
printf "command:"; printf " %q" "$@"; echo
} >> "$d/commands.txt"
"$@" > "$out" 2>&1
rc=$?
{ echo "end_utc: $(date -u +%FT%TZ)"; echo "loadavg_after: $(cut -d" " -f1-3 /proc/loadavg)"; echo "exit_code: $rc"; echo "output: $label.txt"; echo; } >> "$d/commands.txt"
echo "rc=$rc"; tail -n 30 "$out"
