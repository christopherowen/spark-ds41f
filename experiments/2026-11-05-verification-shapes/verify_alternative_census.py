from pathlib import Path
import json
exp=Path(__file__).resolve().parent; proposed=json.loads((exp/'runs/alternatives-final.json').read_text())['results']; result=[]
for file,name,policy in [('exact-native-final-shapes.json','exact','selected'),('native-row-ladder-final-shapes.json','row-ladder','native_rows_only'),('native-shape-union-final-shapes.json','shape-union','native_rows_plus_exact')]:
 actual=json.loads((exp/'runs'/file).read_text())['results']
 for rec in actual:
  if rec['verification_width']==6:
   expected={8:(34,23),16:(48,35)}[rec['request_capacity']];assert (rec['full_graph_count'],rec['context_graph_count'])==expected;continue
  target=next(r for r in proposed if r['strategy']==policy and r['request_capacity']==rec['request_capacity'])
  assert rec['full_graph_count']==target['target_graphs']
  assert rec['context_graph_count']==target['context_graphs']
  assert sum(r['target']-r['live'] for r in rec['all_shapes'])==target['sum_padded_target_rows']
  assert sum(r['target_reqs']-r['reqs'] for r in rec['all_shapes'])==target['sum_padded_requests']
  assert rec['piecewise_graph_count']=={8:16,16:22}[rec['request_capacity']]
  result.append({'variant':name,'requests':rec['request_capacity'],'full':rec['full_graph_count'],'context':rec['context_graph_count'],'piecewise':rec['piecewise_graph_count'],'matches_policy_census':True})
(exp/'runs/implemented-alternative-census.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
