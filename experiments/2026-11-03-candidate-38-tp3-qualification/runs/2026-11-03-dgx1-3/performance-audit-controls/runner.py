import datetime,gzip,hashlib,importlib.util,io,json,pathlib,shutil,sys,tarfile
repo=pathlib.Path.cwd();phase=sys.argv[1]
base=pathlib.Path(__file__).resolve().parent.parent/'snapshots'/phase
script=repo/'experiments/2026-11-03-candidate-38-tp3-qualification/analyze_performance.py'
sys.path.insert(0,str(script.parent))
module=importlib.util.spec_from_file_location('performance_audit',script);m=importlib.util.module_from_spec(module);module.loader.exec_module(m)
assert m.audit(base)['evidence_passed']
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=base.parent.parent/'performance-controls'/stamp;out.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
 return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())

def write(p,d):
 data=(json.dumps(d,indent=1)+'\n').encode();p.write_bytes(gzip.compress(data,mtime=0) if p.suffix=='.gz' else data)

def change(p,f):
 d=read(p);f(d);write(p,d)

def refresh(p):
 receipt=read(p/'receipt.json');raw=p/'raw';files={}
 with tarfile.open(p/'raw.tar.gz','w:gz') as tar:
  for path in sorted(raw.rglob('*')):
   if not path.is_file():continue
   name=str(path.relative_to(raw));data=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data));files[name]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source':str(path)}
 receipt.update(files=files,archive_sha256=sha(p/'raw.tar.gz'));write(p/'receipt.json',receipt)

label='candidate-38'+('-copy' if phase=='release-copy' else '')
run='candidate-38-tp3-qualification-'+phase.removeprefix('release-')
suite=lambda p,arm=label,suffix='':p/'raw/results/private/suites'/(run+suffix)/arm/'suite.json'

def failed_step(p):change(suite(p),lambda d:d['steps'][0].update(exit=1))
def wrong_image(p):change(suite(p),lambda d:[v.update(image_id='sha256:'+'0'*64) for v in d['metadata']['identity']['nodes'].values()])
def wrong_source(p):change(suite(p),lambda d:d['metadata']['identity'].update(client_commit='0'*40))
def same_boot(p):
 first=read(suite(p))['metadata']['identity']['nodes']
 for suffix in ['', '-single-stream','-sampled']:
  change(suite(p,label+'-repeat',suffix),lambda d:[v.update(container_started_at=first[n]['container_started_at']) for n,v in d['metadata']['identity']['nodes'].items()])
def missing_telemetry(p):
 path=p/'raw/results/private/determinism'/run/('suite-'+label+'.telemetry.json');change(path,lambda d:d['nodes'].pop('dgx3'))
def nonfinite_score(p):
 path=suite(p).with_name('model.json.gz');change(path,lambda d:d['scores'][0]['logprob'].__setitem__(0,float('nan')))
def missing_memory(p):change(suite(p).with_name('conditions.json'),lambda d:d['memory'].pop('dgx3'))
def raw_request_failure(p):
 path=suite(p).with_name('c8.json')
 def mutate(d):d['request_table']['rows'][0][d['request_table']['columns'].index('completion_tokens')]=0
 change(path,mutate)

cases=[('failed-step',failed_step,'failed or different ordered steps'),('wrong-common-image',wrong_image,'image/source differs'),('wrong-client-source',wrong_source,'live/client mismatch'),('reused-boot',same_boot,'repeat is not another boot'),('missing-rank-telemetry',missing_telemetry,'telemetry incomplete'),('nonfinite-model-score',nonfinite_score,'non-finite model log probabilities'),('missing-rank-memory',missing_memory,'conditions omit a rank'),('raw-request-failure',raw_request_failure,'raw request table differs')]
results=[]
for name,mutator,expected in cases:
 path=out/name;shutil.copytree(base,path,ignore=shutil.ignore_patterns('analysis*.json'));mutator(path);refresh(path)
 error=None
 try:m.audit(path)
 except Exception as e:error=f'{type(e).__name__}: {e}'
 result={'case':name,'rejected_for_expected_reason':error is not None and expected in error,'error':error,'mutated_archive_sha256':sha(path/'raw.tar.gz')};results.append(result);print(json.dumps(result),flush=True)
receipt={'phase':phase,'base_archive_sha256':read(base/'receipt.json')['archive_sha256'],'analyzer_sha256':sha(script),'control_runner_sha256':sha(pathlib.Path(__file__)),'positive_control_passed':True,'results':results,'passed':all(r['rejected_for_expected_reason'] for r in results)}
write(out/'results.json',receipt);shutil.copy2(__file__,out/'runner.py');shutil.copy2(script,out/'analyzer.py')
print(str(out));raise SystemExit(0 if receipt['passed'] else 1)
