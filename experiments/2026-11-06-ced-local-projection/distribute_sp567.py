"""Copy the verified image with the normal CLI inside an owned stopped cohort."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT));import spark_ds41f as spark
spec=importlib.util.spec_from_file_location('cohort',EXP/'sp567_cohort.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
PROFILE=str(EXP.relative_to(ROOT)/'sp567-image.json')
NODES=['10.30.0.13','10.30.0.14','10.30.0.15']
READ="""from pathlib import Path
import json
p=Path.home()
print(json.dumps({'hold':json.loads((p/'spark-hold.json').read_text()),
'legacy':(p/'spark3-hold.json').exists(),
'memory':int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024}))
"""

def guard(token):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        states=list(pool.map(lambda node:c.remote(node,READ),NODES))
    for state in states:
        h=state['hold']
        assert h.get('holder')==c.HOLDER and h.get('token')==token and h.get('nodes')==c.NAMES
        assert h.get('phase')=='copying' and not state['legacy']
        assert state['memory']>=48*2**30,'host memory below the copy reserve'
    observed=c.remote(NODES[0],"from pathlib import Path; print((Path.home()/'spark-lab/ced-sp567/global-observation.json').read_text())")
    c.GLOBAL_OBSERVATION=types.SimpleNamespace(read_text=lambda:json.dumps(observed))
    c.global_scope()
    assert spark.available_memory_gib()>=48,'build host memory below the copy reserve'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token',required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==args.commit
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    cluster,nodes,_=spark.configuration(argparse.Namespace(cluster_config=PROFILE))
    assert [n['name'] for n in nodes['nodes']]==c.NAMES and not cluster['deployment']['launch_enabled']
    metadata=json.loads((EXP/'image-native-reuse.json').read_text())
    identity,problem=spark.local_image(cluster,cluster['container']['image'],PROFILE)
    assert not problem and identity==metadata['content_identity'],problem
    guard(args.token)
    pending=[]
    for node in nodes['nodes']:
        found=spark.image_identity(nodes,node,cluster['container']['image'])
        assert found in (None,identity),'existing immutable tag has different content; refuse overwrite'
        if found is None:pending.append(node['name'])
    print(json.dumps({'actual_transfer_targets':pending,'all_content_verification_targets':c.NAMES,
                      'source_content_identity':identity}),flush=True)
    command=[str(ROOT/'bin/spark'),'--cluster-config',PROFILE,'build','distribute']
    subprocess.run(command,cwd=ROOT,check=True)
    if not args.apply:return
    guard(args.token)
    process=subprocess.Popen(command+['--apply'],cwd=ROOT,start_new_session=True)
    began=time.monotonic()
    try:
        while process.poll() is None:
            time.sleep(5);guard(args.token)
            assert time.monotonic()-began<780,'distribution time limit reached'
        assert process.returncode==0,'normal image distribution failed'
        _,problem,remaining=spark.image_copies(cluster,nodes,PROFILE)
        assert not problem and not remaining,(problem,remaining)
        print(json.dumps({'all_nodes_verified':c.NAMES,'scope':'image copy only; no GPU job on SP8'}),flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

if __name__=='__main__':main()
