#!/bin/bash
# Catch a run-iket dry run that stops launching (runs 30 and 42). When the newest dry-run
# injection log has not grown for 120 s while the traced harness lives, record GPU power and
# clocks, each harness thread's state, wait channel and kernel stack, py-spy stacks and gdb
# backtraces into hang-N/, keep a copy of the log, then attach the host's cuda-gdb for the
# kernels, blocks and warps still on the GPU. The copy comes first: cuda-gdb's attach
# re-initializes the injection, which truncates its log (run 43's capture 2). Once a stall:
# again only after that log grows. Usage: hangcatch.sh N; ends when run N is done.
here=~/spark-lab/agents/iket-debug
out=$here/hang-$1
mkdir -p "$out"
seen=""
k=0
until grep -q "all done" "$here/run-$1.log" 2>/dev/null; do
  sleep 20
  newest=$(find ~/.cache/spark-lab/compile/iket -path '*/dry/tracker/pid_*/smodel.injection.log' \
    -mmin -30 -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -n 1)
  [ -n "$newest" ] || continue
  read -r mtime size path <<< "$newest"
  age=$(( $(date +%s) - ${mtime%.*} ))
  [ "$age" -ge 120 ] || continue
  [ "$seen" != "$path:$size" ] || continue
  # A log that ends in the tracker's Finalize lines belongs to a dry run that finished; the
  # live harness is then the measured run, decoding its records at exit (run 43's capture 1).
  ! tail -n 1 "$path" | grep -q "Finalize" || continue
  # The harness itself: a python whose script is /b/iket_step.py (not run-iket or iket_profile.py).
  harness=""
  for p in $(pgrep -f "/b/iket_step.py on"); do
    if tr '\0' ' ' < "/proc/$p/cmdline" | grep -q '^[^ ]*python[^ ]* /b/iket_step.py on'; then
      harness="$harness $p"
    fi
  done
  [ -n "$harness" ] || continue
  seen="$path:$size"
  k=$((k + 1))
  echo "$(date '+%F %T %Z') capture $k: $path ($size bytes) quiet ${age} s; harness$harness"
  {
    echo "$(date '+%F %T %Z') $path: $size bytes, quiet ${age} s"
    tail -n 5 "$path"
    echo "### GPU power, SM clock, utilization, 5 samples a second apart"
    for i in 1 2 3 4 5; do
      nvidia-smi --query-gpu=power.draw,clocks.sm,utilization.gpu --format=csv,noheader
      sleep 1
    done
    for p in $harness; do
      echo "### pid $p: children $(pgrep -P "$p" | tr '\n' ' ')"
      for t in /proc/"$p"/task/*; do
        echo "$(basename "$t") $(cat "$t/comm") $(awk '{print $3}' "$t/stat") $(sudo -n cat "$t/wchan")"
      done
    done
  } > "$out/capture-$k.txt" 2>&1
  cp "$path" "$out/injection-$k.log"
  for p in $harness; do
    sudo -n sh -c "cat /proc/$p/task/*/stack" > "$out/kstack-$k-$p.txt" 2>&1
    sudo -n "$here/venv/bin/py-spy" dump --pid "$p" > "$out/pyspy-$k-$p.txt" 2>&1
    sudo -n timeout 300 gdb -p "$p" -batch -nx -iex "set sysroot /proc/$p/root" \
      -ex "set pagination off" -ex "thread apply all bt 30" > "$out/gdb-$k-$p.txt" 2>&1
    sudo -n timeout 400 /usr/local/cuda/bin/cuda-gdb -p "$p" -batch -nx \
      -iex "set sysroot /proc/$p/root" -ex "set pagination off" -ex "info cuda kernels" \
      -ex "info cuda blocks" -ex "info cuda warps" -ex "cuda block (0,0,0) thread (0,0,0)" \
      -ex "x/24i \$pc" > "$out/cudagdb-$k-$p.txt" 2>&1
  done
  echo "$(date '+%F %T %Z') capture $k written"
done
