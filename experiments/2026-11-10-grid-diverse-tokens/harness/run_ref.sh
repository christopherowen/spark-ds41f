#!/bin/bash
# The reference forward's routes (CPU, no GPU), under /tmp/spark-lab-gpu.lock so its CPU and
# memory traffic does not overlap another job's GPU timing.
B=$HOME/spark-lab/agents/grid-fidelity; cd $B
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
if [ -e "$HOME/spark-hold.json" ]; then echo "$(date -u +%FT%TZ) ref skipped: hold file" >> out/progress.log; exit 3; fi
{ echo "### ref-routes"; echo "start_utc: $(date -u +%FT%TZ)"; echo "loadavg_before: $(cut -d' ' -f1-3 /proc/loadavg)";
  echo "docker_ps_before: $(docker ps --format '{{.Names}} {{.Image}}' | tr '\n' ';')"; echo "hold_file: absent"; } >> out/commands.txt
docker run --rm --name gf-ref-routes --memory 24g --memory-swap 24g --cpus 12 --network none -e HF_HUB_OFFLINE=1 \
  -e REF_THREADS=12 -v $HOME/.cache/huggingface/hub:/hub:ro -v $B/h/ref:/work:ro -v $B/out:/out \
  --entrypoint python3 spark-ds41f:candidate-57 -I /work/ref_routes.py $SNAP /out/ref-sequences.json /out/ref-routes.pt \
  > out/ref-routes.txt 2>&1
rc=$?
{ echo "command: docker run --rm --name gf-ref-routes --memory 24g --cpus 12 ... spark-ds41f:candidate-57 -I /work/ref_routes.py \$SNAP /out/ref-sequences.json /out/ref-routes.pt";
  echo "end_utc: $(date -u +%FT%TZ)"; echo "exit_code: $rc"; echo; } >> out/commands.txt
echo "$(date -u +%FT%TZ) ref-routes rc=$rc" >> out/progress.log
