"""Analytical fixture only; no GPU timing evidence is created."""
import importlib.util,json,pathlib,hashlib,subprocess,tempfile
path=pathlib.Path('/home/swank/spark-lab/collective-pdl/deployment/experiments/2026-11-05-collective-pdl/analyze_launch_chain.py')
spec=importlib.util.spec_from_file_location('analysis',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
value=mod.interval([1.0,3.0]);assert value['mean']==2 and abs(value['ci95'][1]-14.706204736432095)<1e-10
with tempfile.TemporaryDirectory() as temp:
 root=pathlib.Path(temp);paths=[]
 for phase,mean in [('entry',20),('staged',19),('arrived',21)]:
  for run in (1,2):
   directory=root/f'{phase}-{run}';directory.mkdir();cases=[]
   for rows in (1,6,9,16,24,48,96):
    for operation in ('all_reduce','all_gather','mixed'):
     for background in (0,4,32):
      cases.append({'rows':rows,'operation':operation,'background_mib':background,'rank_samples_us':[[mean+run/10]*40 for _ in range(3)],'exact':True})
   outcomes=[]
   for rank in range(3):
    log=directory/f'{rank}.log';log.write_text('ANALYTICAL TEST FIXTURE ONLY')
    outcomes.append({'exit':0,'problems':[],'log':log.name,'sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'benchmark_result':{'passed':True,'source_commit':phase,'rank':rank,'cases':cases}})
   receipt={'passed':True,'mode':'benchmark','phase':phase,'process_run':run,'source_commit':phase,'source_tree':phase,'gates_receipt_sha256':'fixture','jobs':[{'kind':'benchmark','outcomes':outcomes}],'image':'fixture','map_sha256':'fixture','harness_commit':'fixture','benchmark_sha256':'fixture','memory':{},'thermal':{}}
   target=directory/'receipt.json';target.write_text(json.dumps(receipt));paths.append(target)
 result=root/'analysis.json'
 subprocess.run(['python3',str(path),*map(str,paths),'--out',str(result)],check=True)
 data=json.loads(result.read_text());assert len(data['cases'])==63
 assert all(abs(row['versus_entry']['staged']['delta_us']['mean']+1)<1e-12 for row in data['cases'])
 assert all(row['versus_entry']['staged']['both_rounds_faster'] for row in data['cases'])
 log=paths[0].parent/'0.log';log.write_text('changed')
 try: mod.read_run(paths[0])
 except ValueError: pass
 else: raise AssertionError('changed log was accepted')
print('PASS: df=1 interval, full six-run fixture and corrupted-log refusal. Analytical test only; no GPU execution.')
