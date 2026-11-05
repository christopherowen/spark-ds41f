#!/bin/bash
set -eu
export PATH=/home/swank/spark-lab/collective-pdl/venv/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cd /home/swank/spark-lab/collective-pdl/sparknet
make check >../runs/source-checks.log 2>&1
printf '0\n' >../runs/source-checks.exit
cd ..
git clone --shared --no-checkout /home/swank/projects/spark-ds41f deployment >runs/deployment-prepare.log 2>&1
cd deployment
git remote set-url origin https://github.com/christopherowen/spark-ds41f.git
git switch -c feat/collective-overlap 2c4b23afcd834ba2adf6a15ca434e59fa4f49f74 >>../runs/deployment-prepare.log 2>&1
cp /home/swank/projects/spark-ds41f/config/nodes-ring4.local.json config/nodes-ring4.local.json
cp /home/swank/projects/spark-ds41f/config/nodes.json config/nodes.json
PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet bin/spark doctor >../runs/doctor.log 2>&1 || echo $? >../runs/doctor.exit
PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet bin/spark build check >../runs/build-check.log 2>&1 || echo $? >../runs/build-check.exit
