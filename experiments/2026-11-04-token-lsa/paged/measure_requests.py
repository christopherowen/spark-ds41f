"""Measure requests under the repository's cooling, fan and memory controls."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))
from spark_ds41f import bench
from spark_ds41f.config import read_cluster,read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    parser.add_argument('--private',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists()
    cluster = read_cluster(args.profile)
    nodes = read_json(cluster['nodes_config'])
    enabled = bench.fan_control_enabled(cluster)
    conditions = {'profile':args.profile,'fan_control_enabled':enabled,'fan_curve':[],
                  'scope':'diagnostic request timing; token acceptance recorded separately'}
    halted,child = [],{}
    def halt(reason):
        if not halted:
            halted.append(reason)
            if 'process' in child and child['process'].poll() is None:
                child['process'].send_signal(signal.SIGINT)
    memory = bench.MemoryWatch(nodes,float(cluster['host']['memguard_gib'])+1,halt)
    try:
        conditions['cooling'] = bench.cool_nodes(nodes,nodes['nodes'],55,600,fan_control=enabled)
        conditions['fan_curve'] = bench.start_fan_curve(nodes,nodes['nodes'],fan_control=enabled)
        problems = bench.fan_curve_problems(conditions['fan_curve'])
        if problems and not all(r['service'] == 'absent' for r in conditions['fan_curve']):
            raise RuntimeError('; '.join(problems))
        conditions['fan_warnings'] = problems
        memory.wait_ready()
        child['process'] = subprocess.Popen([sys.executable,str(HERE/'request_screen.py'),
            'bench','--private',args.private,'--out',args.out],cwd=ROOT)
        code = child['process'].wait()
        if code or halted:
            raise RuntimeError('request measurement failed: '+repr((code,halted)))
    finally:
        conditions['fan_stop_failures'] = bench.stop_fan_curve(nodes,nodes['nodes'],conditions['fan_curve'])
        conditions['thermal'] = memory.thermal_summary()
        conditions['memory'] = memory.close()
        conditions['halted'] = halted
        out.with_name(out.stem+'-conditions.json').write_text(json.dumps(conditions,indent=2)+'\n')
    assert not conditions['fan_stop_failures']
    assert not any(v.get('thermal_slowdown_ms') for v in conditions['thermal'].values()),'thermal slowdown invalidates comparison'


if __name__ == '__main__':
    main()
