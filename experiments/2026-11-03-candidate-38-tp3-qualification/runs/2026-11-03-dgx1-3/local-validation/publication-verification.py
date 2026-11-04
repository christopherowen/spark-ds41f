import hashlib,json,math,pathlib,subprocess
repo=pathlib.Path.cwd();root=repo/'experiments/2026-11-03-candidate-38-tp3-qualification';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def reject(value):raise ValueError('nonfinite JSON constant: '+value)
json_files=list(root.rglob('*.json'))
for p in json_files:json.loads(p.read_text(),parse_constant=reject)
verified=0
for p in root.rglob('publication-receipt.json'):
 d=json.loads(p.read_text());files=d.get('published') or d.get('files') or d.get('recorded_files_sha256');assert files,p
 for name,v in files.items():
  expected=v['sha256'] if isinstance(v,dict) else v;assert sha(p.parent/name)==expected,(p,name);verified+=1
result=json.loads((root/'results.json').read_text())
for name,digest in result['source_files'].items():assert sha(root/name)==digest,name
restore=root/'runs/2026-11-03-dgx1-3/restoration';r=json.loads((restore/'summary.json').read_text());assert r['passed'] and not r['issues'];assert sha(restore/'wrapper.py')==r['wrapper_sha256'];assert sha(restore/'collector-function.py')==r['collector_function_sha256']
for n in ['dgx1','dgx2','dgx3']:
 d=json.loads((restore/n/'runtime.json').read_text());assert d['checkout']=={'head':result['deployment_commit'],'status':'','branch':''};assert not d['issues'];assert d['container']['image_id']==result['restoration']['image_id'];assert d['container']['status']=='running';assert d['container']['restart_count']==0
 if n=='dgx1':assert not any(d['hold_files'].values()) and not d['lab_clock_lock_marker']
changes=subprocess.check_output(['git','diff','--name-only','cb657ab1f5dba4e7cca48a48b0c2fe4e8d4a711f','--','config','manifests','patches','upstreams.lock.json','requirements','releases'],text=True);assert not changes,changes
subprocess.run(['git','diff','--check'],check=True)
out={'passed':True,'json_files_with_only_finite_numbers':len(json_files),'published_files_hash_verified':verified,'normalized_source_files_hash_verified':len(result['source_files']),'restoration_source_image_guard_checks':True,'promoted_inputs_and_immutable_evidence_unchanged':True,'verification_script_sha256':sha(pathlib.Path(__file__))}
p=pathlib.Path(__file__).with_name('final-publication-verification.json');assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
