import datetime,gzip,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path.home()/'spark-lab/agents/prefix-endpoint'; repo=root/'deployment'; exp=repo/'experiments/2026-11-05-prefix-endpoint'; (exp/'vllm').mkdir(parents=True,exist_ok=True); (exp/'runs').mkdir(exist_ok=True)
base=repo/'experiments/2026-10-26-spark-ds41f-candidate/candidate-38'
patch=exp/'vllm/0001-retain-encoder-prompt-endpoints.patch'; shutil.copyfile(root/'incoming/endpoint.patch',patch)
old=base/'vllm/series'; lines=[os.path.relpath((old.parent/p).resolve(),patch.parent) for p in old.read_text().splitlines() if p.strip() and not p.startswith('#')]; lines.append(patch.name); (patch.parent/'series').write_text('\n'.join(lines)+'\n')
lock=json.loads((base/'upstreams.lock.json').read_text()); manifest=json.loads((base/'source.json').read_text())
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'vllm',text=True).strip(); tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root/'vllm',text=True).strip()
lock['source_manifest']=str((exp/'source.json').relative_to(repo)); lock['sources']['vllm']['patch_series']=str((patch.parent/'series').relative_to(repo))
sys.path.insert(0,str(repo)); from spark_ds41f.build import patchset_sha256
os.chdir(repo)
record={'patch_head':head,'expected_tree':tree,'patch_series':lock['sources']['vllm']['patch_series'],'patchset_sha256':patchset_sha256(lock['sources']['vllm'])}
lock['sources']['vllm'].update(record);manifest['vllm'].update(record)
for name,d in [('upstreams.lock.json',lock),('source.json',manifest)]: (exp/name).write_text(json.dumps(d,indent=2)+'\n')
for shape in ('tp3','tp4'):
 d={'notes':'Unpromoted bounded-replay encoder endpoint retention; CPU validated, serving gates pending.','extends':{'config':str((base/f'{shape}.json').relative_to(repo))},'set':{'upstreams_config':str((exp/'upstreams.lock.json').relative_to(repo)),'container':{'image':'spark-ds41f:prefix-endpoint-'+head[:8],'expected_labels':{'local.spark-ds41f.vllm.tree':tree}},'deployment':{'launch_enabled':False}}};(exp/f'{shape}.json').write_text(json.dumps(d,indent=2)+'\n')
base_record={'deployment_commit':'2c4b23afcd834ba2adf6a15ca434e59fa4f49f74','promoted_baseline':'2026-10-30-spark-ds41f-1030','candidate_lock':str((base/'upstreams.lock.json').relative_to(repo)),'vllm_upstream_base':lock['sources']['vllm']['revision'],'candidate_vllm_head':'1f33102547e4da5c64712390604695d1abd577c4','candidate_vllm_tree':'c061e5f6931306fa4655ba531428f639c365bf10','test_image':'sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b','candidate_head':head,'candidate_tree':tree,'source_pr_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'source-pr',text=True).strip()};(exp/'base.json').write_text(json.dumps(base_record,indent=2)+'\n')
for p in (root/'runs').iterdir():
 if p.suffix in ('.txt','.stderr','.exit','.json','.log') and p.name!='prepare.log':
  if p.stat().st_size>12000:
   with gzip.open(exp/'runs'/f'{p.name}.gz','wb') as f: f.write(p.read_bytes())
  else: shutil.copyfile(p,exp/'runs'/p.name)
for name in ('prepare.py','test.sh','final-test.sh','wide-check.sh'):
 shutil.copyfile(root/name,exp/name)
shutil.copyfile(root/'incoming/cpu_runner.py',exp/'cpu_runner.py')
print(json.dumps(base_record,indent=2))
