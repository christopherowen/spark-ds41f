#!/bin/bash
# Every container of this rework runs through here: take /tmp/spark-lab-gpu.lock, then refuse to run
# if ~/spark-hold.json exists or any container is running; log the block to out/commands.log.
# usage: run.sh <name> <docker run args...>
set -uo pipefail
W=$HOME/spark-lab/agents/round3-957p5
name=$1; shift
log=$W/out/commands.log
exec 9>/tmp/spark-lab-gpu.lock
issued=$(date -u +%FT%TZ)
flock 9
{
  echo "### $name"
  echo "host: $(hostname) (sp8)"
  echo "issued_utc: $issued"
  echo "start_utc: $(date -u +%FT%TZ)"
  echo "loadavg_before: $(cut -d' ' -f1-3 /proc/loadavg)"
} >> "$log"
if [ -e "$HOME/spark-hold.json" ]; then echo "hold_file: present; skipped" >> "$log"; echo "hold file present" >&2; exit 3; fi
ps_now=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}')
if [ -n "$ps_now" ]; then echo "docker_ps_before: $ps_now; skipped" >> "$log"; echo "a container is running" >&2; exit 4; fi
{ echo "hold_file: absent"; echo "docker_ps_before: none"; echo "command: docker run $*"; } >> "$log"
docker run "$@" > "$W/out/$name.txt" 2>&1
rc=$?
{ echo "end_utc: $(date -u +%FT%TZ)"; echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"; echo "exit_code: $rc"; echo "output: out/$name.txt"; echo; } >> "$log"
exit $rc
