import base64,hashlib,json,pathlib,shlex,subprocess,zlib
root=pathlib.Path.cwd();private=pathlib.Path(__file__).resolve().parent.parent;out=private/'previous-candidates';assert not out.exists()
code='''import base64,hashlib,json,pathlib,zlib
root=pathlib.Path('/home/swank/projects/spark-ds41f/results/private/suites');files={}
for n in [36,37]:
 for suffix in ['', '-single-stream', '-sampled']:
  d=root/f'candidate-{n}-screen-tp3{suffix}'/f'candidate-{n}'
  for p in d.glob('*.json'):
   before=p.stat();b=p.read_bytes();after=p.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
   files[str(p.relative_to(root))]={'source':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size':len(b),'mtime_ns':after.st_mtime_ns,'base64':base64.b64encode(b).decode()}
print(base64.b64encode(zlib.compress(json.dumps(files).encode())).decode())'''
r=subprocess.run(['ssh','dgx1','python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True)
files=json.loads(zlib.decompress(base64.b64decode(r.stdout)));assert files;out.mkdir();receipt={}
for name,a in files.items():
 p=out/name;p.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(a.pop('base64'));assert hashlib.sha256(b).hexdigest()==a['sha256'];p.write_bytes(b);receipt[name]=a
(out/'receipt.json').write_text(json.dumps({'recorded_runs_only':True,'no_inference_requests':True,'files':receipt},indent=2)+'\n');print('Downloaded',len(files),'native files,',sum(a['size'] for a in files.values()),'bytes')
