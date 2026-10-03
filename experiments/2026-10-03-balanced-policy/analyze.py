"""Summarize per-call interface coverage and slowest-rank timings."""
import argparse,json,statistics,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('runs',nargs='+');a=p.parse_args()
result={}
for run in a.runs:
 reports=[]
 for rank in range(4):
  path=a.root/'probes'/run/f'dgx{rank+1}.txt'
  reports.append(next(json.loads(l) for l in path.read_text().splitlines() if l.startswith('{"rank":') and '"proxy":' in l))
 assert all(r['passed'] for r in reports)
 cases=[]
 for index,t in enumerate(reports[0]['timings']):
  rows=[r['timings'][index] for r in reports]
  key=tuple(t[k] for k in ('dtype','operation','elements_per_rank'))
  assert all(tuple(r[k] for k in ('dtype','operation','elements_per_rank'))==key for r in rows)
  samples=[max(r['microseconds_per_call'][j] for r in rows) for j in range(len(t['microseconds_per_call']))]
  nodes={}
  for rank,row in enumerate(rows):
   v={dev:values['tx_vport_rdma_unicast_bytes'] for dev,values in row['port_deltas'].items()}
   total=sum(v.values())
   nodes[f'dgx{rank+1}']={'tx_rdma_bytes':v,'shares':{d:x/total if total else None for d,x in v.items()},'cw_share':sum(x for d,x in v.items() if d.endswith('f0np0'))/total if total else None,'proxy_payload_bytes':row.get('proxy_payload_bytes',{}),'nonzero_errors':{d:{k:v for k,v in vals.items() if v} for d,vals in row['rdma_error_deltas'].items() if any(vals.values())}}
  cases.append({'dtype':key[0],'operation':key[1],'elements_per_rank':key[2],'bytes_per_rank':key[2]*(2 if 'bfloat16' in key[0] else 4),'slowest_rank_us':samples,'median_us':statistics.median(samples),'nodes':nodes})
 result[run]=cases
print(json.dumps(result,indent=2))
