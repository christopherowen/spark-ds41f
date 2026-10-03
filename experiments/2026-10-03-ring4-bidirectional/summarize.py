"""Summarize saved per-rank collective receipts; timings are screens, not CIs."""
from pathlib import Path
import argparse,json,statistics
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('runs',nargs='+');args=parser.parse_args()
result={}
for run in args.runs:
 entry={'ranks':{}}
 for rank in range(4):
  node=f'dgx{rank+1}'
  path=args.root/'probes'/run/(node+'.txt')
  reports=[json.loads(l) for l in path.read_text().splitlines() if l.startswith('{"rank":')]
  report=next(x for x in reports if 'proxy' in x)
  p=report['proxy'];dirs={}
  for direction,peer in [('clockwise',(rank+1)%4),('counterclockwise',(rank+3)%4)]:
   dirs[direction]=sum(p['bytes_posted_per_hca'][p['hcas'].index(h)] for h in p['peer_hca_routes'][str(peer)])
  r={'passed':report['passed'],'proxy_bytes':dirs,'ops_posted':p['ops_posted'],'numerical_checks':report['numerical_checks'],'timings':[]}
  for t in report['timings']:
   ports=t.get('port_deltas',{})
   tx={direction:sum(v['tx_bytes_phy'] for k,v in ports.items() if suffix in k) for direction,suffix in [('clockwise','f0np0'),('counterclockwise','f1np1')]}
   errors={}
   for h,counters in t.get('rdma_error_deltas',{}).items():
    for name,count in counters.items():
     if count:errors[h+':'+name]=count
   r['timings'].append({'dtype':t['dtype'],'operation':t['operation'],'elements_per_rank':t['elements_per_rank'],'samples_us':t['microseconds_per_call'],'median_us':statistics.median(t['microseconds_per_call']),'physical_tx_bytes':tx,'rdma_errors':errors})
  tele=args.root/run/(node+'-telemetry.jsonl')
  if tele.exists():
   samples=[json.loads(l) for l in tele.read_text().splitlines() if l.startswith('{')]
   gpu=[x['gpu'].split(',') for x in samples if x['gpu'] and len(x['gpu'].split(','))==5]
   r['telemetry']={'samples':len(samples),'gpu_max_c':max(float(x[0]) for x in gpu),'gpu_peak_w':max(float(x[1]) for x in gpu),'thread_samples':sum(bool(x['threads']) for x in samples)}
  entry['ranks'][node]=r
 cases = list(entry['ranks']['dgx1']['timings'])
 entry['collectives'] = []
 for i, case in enumerate(cases):
  samples = [max(r['timings'][i]['samples_us'][j] for r in entry['ranks'].values()) for j in range(len(case['samples_us']))]
  entry['collectives'].append({k:case[k] for k in ('dtype','operation','elements_per_rank')} | {'slowest_rank_samples_us':samples, 'median_us':statistics.median(samples)})
 result[run]=entry
print(json.dumps(result,indent=2))
