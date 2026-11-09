#!/bin/bash
# usage: run.sh <label> -- <command...>
# Appends a block to commands.txt and writes the command output to out/<label>.txt.
cd ~/spark-lab/agents/candidate-48-rework-1010-r2
LABEL=$1; shift; [ "$1" = "--" ] && shift
mkdir -p out
IMG=$(docker image inspect spark-ds41f:candidate-48 --format "{{.Id}}")
{
  echo "### $LABEL"
  echo "host: $(hostname) (sp8)"
  echo "image: $IMG (spark-ds41f:candidate-48)"
  echo "start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "loadavg_before: $(cut -d" " -f1-3 /proc/loadavg)"
  echo "docker_ps_before: $(docker ps --format "{{.Image}} {{.Names}};" | tr "\n" " " | sed "s/ $//")"
  echo "hold_file: $(cat ~/spark-hold.json 2>/dev/null | tr -d "\n")"
  echo "command: $*"
} >> commands.txt
"$@" > out/$LABEL.txt 2>&1
RC=$?
{
  echo "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "loadavg_after: $(cut -d" " -f1-3 /proc/loadavg)"
  echo "exit_code: $RC"
  echo "output: $LABEL.txt"
  echo
} >> commands.txt
exit $RC
