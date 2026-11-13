#!/usr/bin/env bash
# usage: run4.sh <label> <arm> <topology> <port>
# One four-rank run of bench_979_integrated.py on sp1..sp4 (rank = node order).
# Each node's rank runs under that node's /tmp/spark-lab-gpu.lock: the locks are taken one
# node at a time in the order sp1, sp2, sp3, sp4 (a blocking flock that queues behind other
# agents' jobs), each holder idling until all four are held. With every lock held, each node
# is checked for ~/spark-hold.json (abort) and a running container (abort), and only then are
# the four containers released together. Appends the exact commands, image IDs, UTC times,
# load, docker ps and exit codes to out/commands.txt; each rank's stdout+stderr goes to
# out/<label>-<node>.txt and rank 0's JSON to out/<label>.json.
set -u
W=$HOME/spark-lab/agents/candidate-49-own-marks-979
REPO=$W/repo
label=$1; arm=$2; topo=$3; port=$4
OUT=$W/out; mkdir -p "$OUT"
NODES="sp1 sp2 sp3 sp4"
L=/tmp/spark-lab-gpu.lock
IMG=spark-ds41f:1107
S="ssh -o BatchMode=yes"
for n in $NODES; do
  cmd=$(python3 "$W/render_cmds.py" "$REPO" "$W" "$n" "$arm" "$topo" "$label" "$port")
  printf '%s\n' "$cmd" > "$W/cmds/$label-$n.sh" 2>/dev/null || { mkdir -p "$W/cmds"; printf '%s\n' "$cmd" > "$W/cmds/$label-$n.sh"; }
  $S "$n" "mkdir -p $W/cmds $W/locks && rm -f $W/locks/$label.*" && scp -q "$W/cmds/$label-$n.sh" "$n:$W/cmds/"
done
echo "### $label" >> "$OUT/commands.txt"
echo "arm: $arm  topology: $topo  master_port: $port" >> "$OUT/commands.txt"
echo "lock_request_utc: $(date -u +%FT%TZ)" >> "$OUT/commands.txt"
pids=()
for n in $NODES; do
  # A holder gives its lock back after 60 min without a go (an orphan never keeps a GPU).
  $S "$n" "exec flock $L bash -c 'touch $W/locks/$label.held; t=0; while [ ! -e $W/locks/$label.go ] && [ ! -e $W/locks/$label.abort ]; do sleep 0.2; t=\$((t+1)); [ \$t -gt 18000 ] && exit 96; done; [ -e $W/locks/$label.abort ] && exit 97; exec bash $W/cmds/$label-$n.sh'" > "$OUT/$label-$n.txt" 2>&1 &
  pids+=($!)
  until $S "$n" "[ -e $W/locks/$label.held ]"; do
    kill -0 "${pids[-1]}" 2>/dev/null || break
    sleep 5
  done
  echo "lock_held $n: $(date -u +%FT%TZ)" >> "$OUT/commands.txt"
done
abort=""
for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || abort="$abort holder-exited"; done
{
  for n in $NODES; do
    st=$($S "$n" "hostname; docker image inspect --format '{{.Id}}' $IMG; cut -d' ' -f1-3 /proc/loadavg; echo docker_ps=[\$(docker ps --format '{{.Names}}' | tr '\n' ' ')]; [ -e ~/spark-hold.json ] && echo HOLD || echo no-hold" | tr '\n' '|')
    echo "node $n: $st"
    case "$st" in *HOLD*) abort="$abort $n:hold";; esac
    case "$st" in *"docker_ps=[]"*) ;; *) abort="$abort $n:docker";; esac
  done
} >> "$OUT/commands.txt"
for n in $NODES; do echo "command $n: flock $L $(cat "$W/cmds/$label-$n.sh")" >> "$OUT/commands.txt"; done
if [ -n "$abort" ]; then
  for n in $NODES; do $S "$n" "touch $W/locks/$label.abort"; done
  echo "ABORTED before start:$abort" >> "$OUT/commands.txt"
fi
echo "start_utc: $(date -u +%FT%TZ)" >> "$OUT/commands.txt"
[ -z "$abort" ] && for n in $NODES; do $S "$n" "touch $W/locks/$label.go" & done; wait_go=$!
rcs=""
for p in "${pids[@]}"; do wait "$p"; rcs="$rcs $?"; done
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "exit_codes (sp1 sp2 sp3 sp4):$rcs"
  for n in $NODES; do echo "loadavg_after $n: $($S "$n" "cut -d' ' -f1-3 /proc/loadavg; rm -f $W/locks/$label.*")"; done
  echo "outputs: $label-sp1.txt .. $label-sp4.txt, $label.json"
  echo
} >> "$OUT/commands.txt"
sed -n '/=====RESULT-JSON-BEGIN=====/,/=====RESULT-JSON-END=====/p' "$OUT/$label-sp1.txt" | sed '1d;$d' > "$OUT/$label.json"
echo "rc:$rcs"; grep -E '^\[' "$OUT/$label-sp1.txt" | cut -c1-400
