#!/usr/bin/env bash
# usage: run.sh <label> <image-id> -- <command...>
# Records the exact command, host, image ID, UTC start/end, load and exit code to
# out/commands.txt and the combined stdout/stderr to out/<label>.txt.
# Refuses to run while ~/spark-hold.json exists or a serving container runs.
set -u
B=$HOME/spark-lab/agents/round3-956
label=$1; image=$2; shift 2; [ "${1:-}" = "--" ] && shift
d=$B/out; mkdir -p "$d" "$d/json"
out=$d/$label.txt
if [ -e "$HOME/spark-hold.json" ]; then echo "HOLD present, not running" >&2; cat "$HOME/spark-hold.json" >&2; exit 99; fi
running=$(docker ps --format "{{.Image}} {{.Names}}" | tr "\n" ";")
if docker ps --format "{{.Names}} {{.Command}}" | grep -qi "vllm serve\|spark-ds41f-serve\|api_server"; then echo "serving container present, not running" >&2; docker ps >&2; exit 98; fi
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
{ echo "end_of: $label"; echo "end_utc: $(date -u +%FT%TZ)"; echo "loadavg_after: $(cut -d" " -f1-3 /proc/loadavg)"; echo "exit_code: $rc"; echo "output: $label.txt"; echo; } >> "$d/commands.txt"
echo "rc=$rc"; tail -n 12 "$out"
