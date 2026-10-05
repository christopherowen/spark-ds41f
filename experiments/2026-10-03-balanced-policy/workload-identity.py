"""Fingerprint the exact deterministic source prompts used by the serving screen."""
import hashlib,importlib.machinery,importlib.util,json,random,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
l=importlib.machinery.SourceFileLoader('screen_workload',str(root/'bin/spark'));s=importlib.util.spec_from_loader(l.name,l);m=importlib.util.module_from_spec(s);l.exec_module(m)
out={'python':sys.version,'seed':0,'bench_sha256':hashlib.sha256((root/'bin/spark').read_bytes()).hexdigest(),'source_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in m.standard_library_sources()},'screens':{}}
for repeats in (2,4):
 rng=random.Random(0);rows=[]
 for _ in range(repeats):
  for size in rng.sample([4096,32768,65536],3):
   seed=rng.randrange(1<<30);prompt=m.source_text(int(size*0.97),seed)+'\n\nReply with the word ok.'
   rows.append({'nominal_tokens':size,'source_seed':seed,'characters':len(prompt),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
 out['screens'][str(repeats)]=rows
print(json.dumps(out,indent=2))
