. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
for b in 1 2; do
  $B/run.sh 972 supplementary-default-cadence-candidate48-boot$b $C -- flock $L docker run --rm --name c48om-972-cad-b$b --entrypoint python3 -v $B/shim/check_972_default_cadence.py:/tmp/check_972_default_cadence.py:ro $C /tmp/check_972_default_cadence.py
done
$B/run.sh 972 supplementary-default-cadence-1107-boot1 $R -- flock $L docker run --rm --name c48om-972-cad-1107 --entrypoint python3 -v $B/shim/check_972_default_cadence.py:/tmp/check_972_default_cadence.py:ro $R /tmp/check_972_default_cadence.py
