"""Cooling and memory controls around a separate final-answer benchmark client."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from timing_conditions import fan_curve_ready
from spark_ds41f import bench
from spark_ds41f.config import read_cluster,read_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',required=True)
    p.add_argument('--benchmark',choices=['code','ruler','longbench'],required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--private',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['pilot','heldout'],default='pilot')
    p.add_argument('--ruler-stage',choices=['full','screen'],default='full')
    args=p.parse_args()
    assert not args.out.exists()
    cluster=read_cluster(args.profile);nodes=read_json(cluster['nodes_config'])
    conditions={'profile':args.profile,'benchmark':args.benchmark,'fan_curve':[]}
    halted=[];child={}
    def halt(reason):
        if not halted:
            halted.append(reason)
            if 'process' in child and child['process'].poll() is None:
                child['process'].send_signal(signal.SIGINT)
    memory=bench.MemoryWatch(nodes,float(cluster['host']['memguard_gib'])+1,halt)
    try:
        enabled=bench.fan_control_enabled(cluster)
        conditions['cooling']=bench.cool_nodes(nodes,nodes['nodes'],55,600,fan_control=enabled)
        conditions['fan_curve']=bench.start_fan_curve(nodes,nodes['nodes'],fan_control=enabled)
        problems=bench.fan_curve_problems(conditions['fan_curve'])
        conditions['fan_warnings']=problems
        if problems or not fan_curve_ready(conditions['fan_curve'],[n['name'] for n in nodes['nodes']]):
            raise RuntimeError('; '.join(problems) or 'required performance fan curve not verified on every node')
        conditions['performance_fan_curve_verified']=True
        memory.wait_ready()
        folder='2026-11-05-context-quality' if args.benchmark=='ruler' else '2026-11-05-answer-quality'
        client='generate_code.py' if args.benchmark=='code' else 'evaluate.py'
        command=[sys.executable,str(HERE.parent/folder/client),'--cache',args.cache,
                 '--private',args.private,'--out',str(args.out)]
        command+=['--concurrency','4'] if args.benchmark=='code' else ['--split',args.split]
        if args.benchmark=='ruler':command+=['--stage',args.ruler_stage]
        child['process']=subprocess.Popen(command,cwd=ROOT)
        code=child['process'].wait()
        if code or halted:raise RuntimeError('benchmark failed: '+repr((code,halted)))
    finally:
        conditions['fan_stop_failures']=bench.stop_fan_curve(nodes,nodes['nodes'],conditions['fan_curve'])
        conditions['thermal']=memory.thermal_summary();conditions['memory']=memory.close()
        conditions['halted']=halted
        args.out.with_name(args.out.stem+'-conditions.json').write_text(json.dumps(conditions,indent=2)+'\n')
    assert not conditions['fan_stop_failures']
    assert not any(t.get('thermal_slowdown_ms') for t in conditions['thermal'].values())


if __name__=='__main__':main()
