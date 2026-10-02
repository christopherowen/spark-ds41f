"""Normalize all completed launches, preserving rank timings and error evidence.

Usage: python3 summarize.py path/to/extracted/raw-directory
"""
import json,pathlib,statistics,sys,re
root=pathlib.Path(sys.argv[1])
def objects(text):
 dec=json.JSONDecoder();offset=0
 while (i:=text.find('{"rank":',offset))>=0:
  try:value,end=dec.raw_decode(text[i:]);offset=i+end;yield value
  except ValueError:offset=i+1
summary={}
for run in sorted(root.glob('run*')):
 if not run.is_dir() or not (run/'results.json').exists():continue
 rows=[];forwards=[];graphs={}
 for p in sorted(run.glob('dgx?.txt')):
  log=p.read_text();rank=int(p.stem[-1])-1
  rings=[tuple(map(int, values)) for values in re.findall(r'Ring (\d+) : (\d+) -> (\d+) -> (\d+)',log)]
  commands=json.loads((run/(p.stem+'-command.json')).read_text())
  env=dict(value.split('=',1) for i,value in enumerate(commands) if i and commands[i-1]=='--env')
  expected=int(env.get('NCCL_MAX_NCHANNELS','1'))
  legal=bool(rings) and all(center==rank and {previous,following}=={(rank-1)%4,(rank+1)%4} for channel,previous,center,following in rings)
  channels=sorted({ring[0] for ring in rings})
  choices=sorted(set(re.findall(r'(AllReduce|AllGather|ReduceScatter): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}',log)))
  graphs[p.stem]={'legal_neighbor_ring':legal,'channels':channels,'expected_channels':expected,
                 'channels_match':channels==list(range(expected)),
                 'gdr_disabled':bool(re.search(r'Connected all rings, use ring PXN \d+ GDR 0',log)),
                 'environment':env, 'selected_plans':[{'operation':o,'logged_bytes':int(b),'algorithm':a,'protocol':p,'channel_low':int(lo),'channel_high':int(hi)} for o,b,a,p,lo,hi in choices]}
  for obj in objects(log):
   if obj.get('passed'):rows.append(obj)
   if 'forwarding_counters' in obj:forwards.append(obj)
 errors={};buffers={};memory={}
 for n in range(1,5):
  pre=json.loads((run/f'dgx{n}-hardware-before.json').read_text());post=json.loads((run/f'dgx{n}-hardware-after.json').read_text())
  errors[f'dgx{n}']={dev:{k:v-pre[dev][k] for k,v in vals.items()} for dev,vals in post.items()}
  pre=json.loads((run/f'dgx{n}-ports-before.json').read_text());post=json.loads((run/f'dgx{n}-ports-after.json').read_text())
  buffers[f'dgx{n}']={dev:vals['rx_out_of_buffer']-pre['ports'][dev]['rx_out_of_buffer'] for dev,vals in post['ports'].items()}
  memory[f'dgx{n}']={label:{k.strip():int(v.strip().split()[0]) for line in snap['memory'].splitlines() for k,v in [line.split(':',1)] if k in ('MemTotal','MemAvailable','Slab','SUnreclaim')} for label,snap in [('before_kib',pre),('after_kib',post)]}
 result={'results':json.loads((run/'results.json').read_text()),'ranks_passed':len(rows),'proxy':[r['proxy'] for r in rows],'error_deltas':errors,'rx_out_of_buffer_deltas':buffers,'memory':memory,'forwarding':forwards,'latency':[]}
 result['totals']={key:sum(dev[key] for node in errors.values() for dev in node.values()) for key in ('roce_adp_retrans','packet_seq_err','out_of_sequence')}
 result['totals']['rx_out_of_buffer']=sum(sum(node.values()) for node in buffers.values())
 if len(rows)==4 and all(r.get('timings') for r in rows):
  for idx,t in enumerate(rows[0]['timings']):
   samples=[max(r['timings'][idx]['microseconds_per_call'][j] for r in rows) for j in range(5)]
   result['latency'].append({k:t[k] for k in ('dtype','elements_per_rank','operation')}|{'slowest_rank_samples_us':samples,'median_us':statistics.median(samples),'rdma_error_deltas':{k:sum(v.get(k,0) for r in rows for v in r['timings'][idx].get('rdma_error_deltas',{}).values()) for k in ('roce_adp_retrans','packet_seq_err','out_of_sequence')} if all('rdma_error_deltas' in r['timings'][idx] for r in rows) else None})
 for idx, row in enumerate(result['latency']):
  ports={f"rank{r['rank']}/{dev}":d for r in rows for dev,d in r['timings'][idx].get('port_deltas',{}).items()}
  row['port_deltas']=ports
  tx=[d['tx_bytes_phy'] for d in ports.values()]
  row['physical_tx_max_min_ratio']=max(tx)/min(tx) if tx and min(tx)>0 else None
  row['rx_out_of_buffer']=sum(d['rx_out_of_buffer'] for d in ports.values())
 result['graph_checks']=graphs
 result['numerical_checks']={str(r['rank']):r.get('numerical_checks',[]) for r in rows}
 summary[run.name]=result
 print(run.name,result['totals'],'passed',len(rows))
 for row in result['latency']:
  if row['dtype']=='torch.bfloat16':print(row['operation'],row['elements_per_rank'],round(row['median_us'],2),row['rdma_error_deltas'])
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
