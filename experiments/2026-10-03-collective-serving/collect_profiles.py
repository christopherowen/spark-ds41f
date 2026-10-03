import concurrent.futures, json, pathlib, shlex, subprocess, sys
label = sys.argv[1]
assert label.replace('-', '').isalnum()
out = pathlib.Path('.work/collective-serving/profiles') / label
out.mkdir(parents=True, exist_ok=False)
remote = r'''
import pathlib,json,hashlib,subprocess,sys
root=pathlib.Path('/home/swank/projects/spark3-vllm-ds41f/cache/kkref/profiles/ring4-collective-20261003')
out=root/sys.argv[1];out.mkdir(exist_ok=False)
traces=list(root.glob('*.pt.trace.json.gz'))
assert len(traces)==int(sys.argv[2]), [p.name for p in traces]
traces=sorted(traces,key=lambda p:p.stat().st_mtime)[:1]
records=[]
for path in traces:
 target=out/path.name;path.rename(target)
 summary=out/'kernels.json'
 subprocess.run(['python3','/home/swank/projects/spark3-ring4-qualification/experiments/2026-09-29-determinism/summarize_kernels.py',str(summary),str(target)],check=True,capture_output=True)
 records.append({'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'summary':json.loads(summary.read_text())})
if int(sys.argv[2])==1:
 for path in root.glob('profiler_out_*.txt'):path.rename(out/path.name)
print(json.dumps(records))
'''
def collect(n):
 result=subprocess.check_output(['ssh',f'swank@dgx{n}',shlex.join(['sudo','python3','-c',remote,label,sys.argv[2] if len(sys.argv)>2 else '1'])],text=True)
 records=json.loads(result)
 (out/f'dgx{n}.json').write_text(json.dumps(records,indent=2)+'\n')
 print(n,[(round(r['summary']['span_ms']),round(r['summary']['kernel_ms']),len(r['summary']['kernels'])) for r in records],flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(collect,range(1,5)))
