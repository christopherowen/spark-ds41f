#!/usr/bin/env python3
"""Cache the exact simulated R580->R610 transaction, without installing it."""
import hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'.work/driver-cuda-refresh/packages'
OUT.mkdir(parents=True,exist_ok=True)
args=['nvidia-driver-610-open=610.57.04-0ubuntu0.24.04.3',
      'linux-modules-nvidia-610-open-nvidia-hwe-24.04=7.0.0-1019.19~24.04.2+1',
      'linux-modules-nvidia-610-open-nvidia-64k-hwe-24.04=7.0.0-1019.19~24.04.2+1']
sim=subprocess.check_output(['apt-get','-s','install',*args],text=True)
(OUT/'simulation.txt').write_text(sim)
assert 'Inst nvidia-dkms-' not in sim, 'Unexpected NVIDIA DKMS registration'
old=[];new=[]
for line in sim.splitlines():
 if line.startswith('Remv '):
  name=line.split()[1]
  version=subprocess.check_output(['dpkg-query','-W','-f=${Version}',name],text=True)
  old.append(name+'='+version)
 if line.startswith('Inst '):
  m=re.match(r'Inst (\S+)(?: \[[^]]+\])? \((\S+)',line)
  assert m,line
  new.append(m[1]+'='+m[2])
assert old and new,'Run this cache preparation on R580 before changing hosts'
for label,items in [('old',old),('new',new)]:
 d=OUT/label;d.mkdir(exist_ok=True)
 subprocess.run(['apt-get','download',*items],cwd=d,check=True)
(OUT/'transaction.json').write_text(json.dumps({'requested':args,'old':old,'new':new,'sha256':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*/*.deb')}},indent=2)+'\n')
print(OUT/'transaction.json')
