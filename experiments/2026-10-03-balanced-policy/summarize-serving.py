"""Compare matched serving screens without pooling different boots into a CI."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
load=lambda name:json.loads((a.root/name/'bench.json').read_text())
result={}
for label,control,candidate in [('first','control1','selected'),('repeat','control2','selected2')]:
 dc=load('serving-'+control+'-decode'+('-r1' if label=='first' else ''))
 dn=load('serving-'+candidate+'-decode')
 pc=load('serving-'+control+'-prefill');pn=load('serving-'+candidate+'-prefill')
 for x,y in [(dc,dn),(pc,pn)]:
  for key in ('seed','suites','concurrency','decode_cases','prefill_sizes','prefill_repeats','prompts_sha256'):
   assert x['workload'][key]==y['workload'][key],(label,key)
  assert x['identity']['topology']==y['identity']['topology']
  assert not x['identity']['client_dirty'] and not y['identity']['client_dirty']
  assert not x['identity']['live_problems'] and not y['identity']['live_problems']
  assert {n['image_id'] for n in x['identity']['nodes'].values()}=={n['image_id'] for n in y['identity']['nodes'].values()}
 out={'decode':{},'prefill':{},'quality':{'control':dc['suites']['quality']['passed'],'candidate':dn['suites']['quality']['passed']},'prefix':{'control':pc['suites']['prefix'],'candidate':pn['suites']['prefix']}}
 for key,b in dc['suites']['decode']['points'].items():
  n=dn['suites']['decode']['points'][key]
  row={'control_tps':b['tps'],'candidate_tps':n['tps'],'change_pct':100*(n['tps']['mean']/b['tps']['mean']-1),'control_verified':b['verified_per_draft'],'candidate_verified':n['verified_per_draft'],'control_accepted':b['accepted_per_draft'],'candidate_accepted':n['accepted_per_draft']}
  if b['step_ms']['n']:row.update(control_step_ms=b['step_ms'],candidate_step_ms=n['step_ms'])
  assert b['failed_requests']==n['failed_requests']==0
  out['decode'][key]=row
 for key,b in pc['suites']['prefill']['points'].items():
  n=pn['suites']['prefill']['points'][key];bs=pc['suites']['prefill']['samples'][key];ns=pn['suites']['prefill']['samples'][key]
  assert [s['prompt_tokens'] for s in bs]==[s['prompt_tokens'] for s in ns]
  assert b['failed_requests']==n['failed_requests']==0
  out['prefill'][key]={'actual_tokens':[s['prompt_tokens'] for s in bs],'control_tps':b['prefill_tps'],'candidate_tps':n['prefill_tps'],'change_pct':100*(n['prefill_tps']['mean']/b['prefill_tps']['mean']-1),'paired_ttft_change_pct':[100*(y['ttft_s']/x['ttft_s']-1) for x,y in zip(bs,ns)]}
 result[label]=out
print(json.dumps(result,indent=2))
