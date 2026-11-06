import pathlib,os,json,sys,runpy,signal,hashlib
assert not list(pathlib.Path('/dev').glob('nvidia*'))
c=pathlib.Path('/sys/fs/cgroup');limits={n:(c/n).read_text().strip() for n in ['cpu.max','memory.max','memory.swap.max']};a,b=map(int,limits['cpu.max'].split());assert a<=b and limits['memory.max']==str(2*1024**3) and limits['memory.swap.max']=='0'
manifest=json.loads(pathlib.Path('/proof/hash-manifest.json').read_text());runtime={}
for name in ['vllm/v1/worker/gpu/sample/logprob.py','vllm/v1/engine/async_llm.py','vllm/v1/engine/core_client.py']:
 raw=(pathlib.Path('/opt/spark-ds41f/candidate/vllm')/name).read_bytes();actual=hashlib.sha256(raw).hexdigest();assert actual==manifest[name]['before_sha256'];runtime[name]=actual
print(json.dumps({'kind':'actual_CPU_container_resource_and_runtime_guard','limits':limits,'no_GPU_devices':True,'actual_frozen_runtime_files':runtime}),flush=True)
signal.alarm(30);sys.argv=['/proof/witness.py',*sys.argv[1:]];runpy.run_path('/proof/witness.py',run_name='__main__')
