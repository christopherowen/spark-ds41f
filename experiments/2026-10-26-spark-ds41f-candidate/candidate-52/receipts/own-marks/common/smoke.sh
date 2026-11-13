B=/home/chris/spark-lab/agents/candidate-52-own-marks
M968="-e PYTHONPATH=/opt/spark-ds41f/candidate/vllm -v $B/h:/h:ro"
M969="--cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --security-opt=seccomp=unconfined --security-opt=label=disable -v $B/nvme:/mnt/nvme -v $B/bench969:/bench969:ro -e ENGRAM_ROWS_TEST_DIR=/mnt/nvme/engram-rows-test"
bash $B/om.sh smoke h968-c52 c52 $M968 -- python3 /h/h968.py smoke-c52; echo rc=$?
bash $B/om.sh smoke rows-c52 c52 $M969 -- python3 /bench969/rows969.py smoke-c52; echo rc=$?
bash $B/om.sh smoke bench-cache-c52 c52 $M969 -- python3 /bench969/run969.py smoke-c52 1; echo rc=$?
