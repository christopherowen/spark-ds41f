#!/usr/bin/env bash
# PR #991 rework r1, bench boot 3: bench-r1-boot2 overlapped an unlocked GPU container
# (silly_spence, 08:21:10-08:23:31Z, another session). This repeats the bench once more;
# inside the lock it first waits (up to 10 min) until no other container runs, and logs
# the containers seen during the run.
. $HOME/spark-lab/agents/candidate-49-rework-991/env.sh
G=$W/variants/r1/gemm.py
TL=$W/variants/r1/test_deepseek_v41_tilelang_linear.py
MG=$V/vllm/models/deepseek_v4_1/tilelang/gemm.py
MT=$V/tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py
J=$W/out/json
$W/run.sh bench-r1-boot3 $C -- flock $L bash -c "for i in \$(seq 120); do [ -z \"\$(docker ps -q)\" ] && break; sleep 5; done; echo containers_at_start: \$(docker ps --format {{.Names}}); ( while sleep 3; do echo \"\$(date -u +%FT%TZ) \$(docker ps --format {{.Names}} | tr \"\\n\" \" \")\"; done ) > $W/out/bench-r1-boot3-neighbors.log & m=\$!; docker run --rm --gpus all --ipc=host --name rw991-bench-b3 --entrypoint \"\" -v /usr/local/cuda-13.0/bin/nvdisasm:/usr/local/bin/nvdisasm:ro -v $G:$MG:ro -v $TL:$MT:ro -v $EXP/bench_hw_e2m1x2_quant.py:/tmp/bench_hw_e2m1x2_quant.py:ro -v $J:/out -w $V $C bash -lc \"python3 /tmp/bench_hw_e2m1x2_quant.py --out /out/bench-r1-boot3.json\"; rc=\$?; kill \$m; exit \$rc"
