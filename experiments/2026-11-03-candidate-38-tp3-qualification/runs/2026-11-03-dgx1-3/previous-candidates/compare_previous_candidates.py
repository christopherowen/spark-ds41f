import hashlib,json,pathlib,sys
repo=pathlib.Path.cwd();sys.path.insert(0,str(repo))
from servebench import suite
from servebench.__main__ import compare_results,read_json
from scripts.servebench_suite_table import AA_FLOOR_PCT
private=pathlib.Path(__file__).resolve().parent.parent;prior=private/'previous-candidates';receipt=json.loads((prior/'receipt.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,a in receipt['files'].items():assert sha(prior/name)==a['sha256']
raw=private/'snapshots/release-default/raw/results/private/suites';judge=lambda a,b:compare_results(a,b,3.0)
result={'recorded_runs_only':True,'prior_boots_per_candidate':1,'caveat':'Previous candidates have one recorded standard boot, not a repeat-boot qualification. CopyDrafter fault status is separate. No old model corpus was transferred.','comparisons':{}}
for n in [36,37]:
 for suffix in ['','-single-stream','-sampled']:
  base=prior/f'candidate-{n}-screen-tp3{suffix}'/f'candidate-{n}';a=raw/f'candidate-38-tp3-qualification-default{suffix}'/'candidate-38';b=a.parent/'candidate-38-repeat'
  docs=[json.loads((p/'suite.json').read_text()) for p in [base,a,b]];assert len({d['suite_sha256'] for d in docs})==1
  assert all(step['exit']==0 for d in docs for step in d['steps'])
  ident=docs[0]['metadata']['identity'];assert not ident['client_dirty'] and not ident['live_problems'] and len(set(x['image_id'] for x in ident['nodes'].values()))==1
  conditions=json.loads((base/'conditions.json').read_text());assert not conditions['halted'] and not conditions['fan_stop_failures'];assert all(not x['swap_growth_gib'] for x in conditions['memory'].values());assert all(not x['thermal_slowdown_ms'] for x in conditions['thermal'].values())
  x=suite.compare(read_json,str(base),str(a),judge);y=suite.compare(read_json,str(base),str(b),judge);aa=suite.compare(read_json,str(a),str(b),judge)
  result['comparisons'][str(n)+suffix]={'prior_identity':ident,'prior_conditions':conditions,'reference_to_boot1':x,'reference_to_boot2':y,'headlines':suite.with_floor(x,aa,floors=AA_FLOOR_PCT,reference_to_repeat=y)}
p=prior/'comparison.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
for name,c in result['comparisons'].items():
 print(name)
 for row in c['headlines']:
  if row['metric']!='output tok/s':print(row['step'],row['metric'],round(row['change_pct'],3),'margin',round(row['margin_pct'],3),row['verdict'])
 for step,x in c['reference_to_boot1']['steps'].items():
  if step=='c4' and 'skipped' not in x:print('C4 p99',next(r for r in x[1]['rows'] if r['metric']=='TTFT p99 s'))
print('saved',p,'sha256',sha(p))
