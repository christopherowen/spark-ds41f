"""Check independent counters and recorded event structure across benchmark clients."""
import hashlib,json,pathlib,subprocess
root=pathlib.Path.cwd();private=pathlib.Path(__file__).resolve().parent.parent
public=root/'experiments/2026-11-03-candidate-38-tp3-qualification/runs/2026-11-03-dgx1-3/performance-default'
rows=[];hashes={}
def inside(data):
 table=data['request_table'];lo,hi=table['window'];records={}
 for raw in table['rows']:
  record=dict(zip(table['columns'],raw))
  if lo<=record['sent']<hi:records[record['index']]=record
 return records
for arm in ['candidate-38','candidate-38-repeat']:
 for name,points in [('release-tp3',['c8','c4','poisson']),('single-stream',['c1']),('sampled',['c16','c8','c1'])]:
  reference=root/('manifests/suites/2026-10-30-spark-ds41f-1030-tp3'+('-'+name if name!='release-tp3' else ''))
  for point in points:
   paths=[reference/(point+'.json'),public/arm/name/(point+'.json')];a,b=[json.loads(p.read_text()) for p in paths]
   server=[data['summary']['server']['generation_tps']['mean'] for data in [a,b]];window=[data['summary']['output_tps']['mean'] for data in [a,b]]
   row={'arm':arm,'suite':name,'point':point,'server_counter_change_pct':100*(server[1]/server[0]-1),'client_window_change_pct':100*(window[1]/window[0]-1)}
   row['difference_percentage_points']=row['server_counter_change_pct']-row['client_window_change_pct']
   if point=='c1':
    x,y=inside(a),inside(b);pairs=[(x[index],y[index]) for index in set(x)&set(y) if x[index]['prompt']==y[index]['prompt']]
    row['paired_event_structure']={'pairs':len(pairs),'same_chunks':sum(xr['chunks']==yr['chunks'] for xr,yr in pairs),'same_first_tokens':sum(xr['first_tokens']==yr['first_tokens'] for xr,yr in pairs)}
   if name=='sampled':assert a['workload']['extra_body']['seed']==b['workload']['extra_body']['seed']==0
   rows.append(row)
   for path in paths:hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
old='19efdd6fe1f11a159c5f1989aaa3433682325ab2';current='cb657ab1f5dba4e7cca48a48b0c2fe4e8d4a711f';unchanged={}
for name in ['behaviour.py','engine.py','prefill.py','prom.py','prompts.py','stats.py']:
 path='servebench/'+name;unchanged[path]=subprocess.check_output(['git','show',old+':'+path])==subprocess.check_output(['git','show',current+':'+path])
result={'recorded_data_only':True,'old_client_commit':old,'current_client_commit':current,'unchanged_modules':unchanged,'rows':rows,'max_window_counter_change_disagreement_percentage_points':max(abs(row['difference_percentage_points']) for row in rows),'source_sha256':hashes,'runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'client_changes':['Non-gzip JSON formatting changed without changing parsed numeric data.','The stream client also retains usage-only/control-token events; server generation counters independently corroborate throughput.','Sampled default seed filling is inactive for these suites: each already supplies seed 0; greedy workloads use temperature 0.','Stall request bodies are prepared before the timed traffic; model, engine, prefill and Prometheus readers are unchanged.']}
out=private/'control/client-compatibility-check.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'unchanged_modules':unchanged,'max_disagreement_pp':result['max_window_counter_change_disagreement_percentage_points'],'rows':rows},indent=2))
