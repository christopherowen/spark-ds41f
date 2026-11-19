"""CPU-only contract check: native payloads/timing survive identity recording."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

path=Path(__file__).with_name('bench_requests.py')
spec=importlib.util.spec_from_file_location('bench_requests',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original_run=module.bench.Bench.run
original_argv=sys.argv
payloads=[('/v1/completions',{'prompt':'public test','max_tokens':1,'cache_salt':'a'})]
before=copy.deepcopy(payloads)

def native(self,label,payloads,*args,**kwargs):
    return {'wall_s':0.625,'requests':[{'ok':True,'elapsed_s':0.5,'ttft_s':0.375,
                                      'output_sha256':'fixed','completion_tokens':1}]}

def main(argv):
    assert argv==['--help']
    bench=object.__new__(module.bench.Bench)
    bench.run('first',payloads)
    changed=copy.deepcopy(payloads);changed[0][1]['cache_salt']='b'
    bench.run('second',changed)
    return 0

try:
    module.bench.Bench.run=native;module.main=main
    with tempfile.TemporaryDirectory() as directory:
        manifest=Path(directory)/'requests.jsonl'
        sys.argv=[str(path),str(manifest),'--help']
        assert module.run()==0
        assert payloads==before,'recording changed a native request'
        first,second=[json.loads(line) for line in manifest.read_text().splitlines()]
        assert first['wall_s']==second['wall_s']==0.625
        a,b=first['requests'][0],second['requests'][0]
        assert a['elapsed_s']==b['elapsed_s']==0.5 and a['ttft_s']==0.375
        assert a['model_input_sha256']==b['model_input_sha256']
        assert a['payload_sha256']!=b['payload_sha256']
        assert a['output_sha256']==b['output_sha256']=='fixed'
    print('native payloads and elapsed time preserved; cache salts separated from model-input identity')
finally:
    module.bench.Bench.run=original_run;sys.argv=original_argv

# A lease writer must be joined before release so it cannot recreate a hold.
window_path=path.with_name('run_window.py')
spec=importlib.util.spec_from_file_location('request_window',window_path)
window=importlib.util.module_from_spec(spec);spec.loader.exec_module(window)
state={'writer_stopped':False,'released':False}
window.owns=lambda token:True
window.run=lambda *args,**kwargs:0

def each(program,data=None):
    if data and data.get('action')=='remove':
        assert state['writer_stopped'],'release raced a live heartbeat writer'
        state['released']=True
        return {}
    return {'node':{'image':window.IMAGE,'running':True,'experiment_mounts':[],'search_env':[]}}

window.each=each
with tempfile.TemporaryDirectory() as directory:
    window.restore('test',Path(directory),'restore',lambda:state.update(writer_stopped=True))
assert state['released']
print('lease writer stops before verified service restoration releases holds')
