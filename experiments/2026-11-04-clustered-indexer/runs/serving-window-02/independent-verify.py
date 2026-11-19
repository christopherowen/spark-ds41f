from pathlib import Path
import subprocess,json
home=Path.home();p=home/'spark-lab/checkouts/clustered-serving';original=home/'projects/spark-ds41f'
x=json.loads(subprocess.check_output(['docker','inspect','spark-ds41f'],text=True))[0]
print(json.dumps({'image_id':x['Image'],'running':x['State']['Running'],
 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
 'source_dirty':subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True).strip(),
 'original_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip(),
 'original_dirty':subprocess.check_output(['git','status','--porcelain'],cwd=original,text=True).strip(),
 'reservations':{name:(home/name).exists() for name in ['spark-hold.json','spark3-hold.json','spark-request.json','spark3-request.json']},
 'search_env':[e for e in x['Config']['Env'] if e.startswith('VLLM_DS41_CLUSTERED_SEARCH=')],
 'search_mounts':[m['Destination'] for m in x['Mounts'] if 'clustered' in m['Source']]}))
