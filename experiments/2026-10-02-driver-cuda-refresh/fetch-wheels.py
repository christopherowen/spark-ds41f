#!/usr/bin/env python3
import hashlib,json,pathlib,urllib.request,zipfile
p=pathlib.Path(__file__).resolve().parent
out=p.parents[1]/'.work/driver-cuda-refresh/cuda-context';out.mkdir(parents=True,exist_ok=True)
for f in json.loads((p/'cuda-wheels.json').read_text()):
 dest=out/f['filename']
 if not dest.exists(): urllib.request.urlretrieve(f['url'],dest)
 assert hashlib.sha256(dest.read_bytes()).hexdigest()==f['sha256'],dest
 if f['name']=='nvidia-cuda-nvcc':
  with zipfile.ZipFile(dest) as z:
   names=[n for n in z.namelist() if n.endswith('/ptxas')];assert len(names)==1
   (out/'ptxas').write_bytes(z.read(names[0]));(out/'ptxas').chmod(0o755)
for name in ('cuda-requirements.txt','Dockerfile'):
 (out/name).write_bytes((p/name).read_bytes())
print(out)
