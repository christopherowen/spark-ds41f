"""Summarize native serving receipts without pooling boots or excluding failures."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]);out={'benchmarks':{},'profiles':{},'agent_workflows':{}}
for p in sorted(root.glob('bench-*/bench.json')):
 d=json.loads(p.read_text());r={k:d.get(k) for k in ('started_utc','finished_utc','aborted','workload','cooling','memory','thermal','discarded_samples')}
 r['source_report']=str(p.relative_to(root));r['identity']=d['identity'];r['suites']={}
 r['command_attempts']=[]
 for receipt in sorted(root.glob('*.txt.json')):
  attempt=json.loads(receipt.read_text());command=attempt.get('command',[])
  if '--output' in command and Path(command[command.index('--output')+1]).name==p.parent.name:
   r['command_attempts'].append({'receipt':receipt.name, **attempt})
 for suite,value in d['suites'].items():
  r['suites'][suite]=value.get('points',value)
 out['benchmarks'][p.parent.name]=r
for p in sorted(root.glob('profiles/*/dgx?.json')):
 for r in json.loads(p.read_text()):
  s=r['summary'];kernels=s['kernels'];nccl=[k for k in kernels if 'nccl' in k[0].lower()]
  out['profiles'].setdefault(p.parent.name,{})[p.stem]={k:r[k] for k in ('path','bytes','sha256')}|{'span_ms':s['span_ms'],'summed_kernel_ms':s['kernel_ms'],'nccl_calls':sum(k[3] for k in nccl),'summed_nccl_ms':sum(k[4] for k in nccl),'nccl_kernels':nccl}
for p in sorted(root.glob('agent-*.json')):
 d=json.loads(p.read_text())
 if 'runs' not in d:continue
 out['agent_workflows'][p.stem]={'source_report':p.name,'workload_sha256':d['workload_sha256'],'passed':sum(bool(r['ok']) for r in d['runs']),'total':len(d['runs']),'seconds':[r['seconds'] for r in d['runs']],'error':d.get('error')}
print(json.dumps(out,indent=2))
