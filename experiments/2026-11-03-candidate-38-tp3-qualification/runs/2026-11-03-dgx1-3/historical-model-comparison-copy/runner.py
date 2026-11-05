"""Compare recovered model records on the client; never contact an inference endpoint."""
import datetime,gzip,hashlib,json,math,pathlib,shutil,subprocess,sys
repo=pathlib.Path.cwd();sys.path.insert(0,str(repo))
from servebench import behaviour
private=pathlib.Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def read_model(path):
 data=json.loads(gzip.decompress(path.read_bytes()));assert data['schema']=='servebench-model/1'
 generation=data['generation']['sequences'];scores=data['scores']
 assert len(generation)==len(scores)==256
 assert {r['id'] for r in generation}=={r['id'] for r in scores}==set(range(256))
 assert all(not r.get('error') and r['prompt_ids'] and r['output_ids'] for r in generation)
 assert all(not r.get('error') and r['tokens'] and len(r['tokens'])==len(r['logprob'])==len(r['rank']) for r in scores)
 assert all(math.isfinite(lp) for r in scores for lp in r['logprob'])
 assert all(math.isfinite(lp) for r in scores for position in r['top_logprobs'] for lp in position)
 return data

def comparison(a,b,corpora):
 x={r['id']:r for r in a['generation']['sequences']};y={r['id']:r for r in b['generation']['sequences']}
 out={'generation':behaviour.compare_generation(a['generation'],b['generation']),'equal_prompt_contexts':sum(x[i]['prompt_ids']==y[i]['prompt_ids'] for i in x),'finish_reason_changes':sum(x[i].get('finish_reason')!=y[i].get('finish_reason') for i in x),'same_served_model':a['server']['model']==b['server']['model'],'warnings':[]}
 if not out['same_served_model']:out['warnings'].append('served model IDs differ')
 context_a=corpora.get(a['corpus']['sha256']);context_b=corpora.get(b['corpus']['sha256'])
 if context_a is None or context_b is None:
  out['warnings'].append('scored corpus context unavailable; probability comparison omitted');return out
 allowed={i for i in context_a if context_a[i]['prompt_ids']==context_b[i]['prompt_ids'] and context_a[i]['output_ids']==context_b[i]['output_ids']}
 out['comparable_scored_contexts']=len(allowed)
 if len(allowed)!=256:out['warnings'].append('only exactly equal scored prompt/output contexts compare')
 if allowed:out['scores']=behaviour.compare_scores([r for r in a['scores'] if r['id'] in allowed],[r for r in b['scores'] if r['id'] in allowed])
 return out

def main():
 candidates=[pathlib.Path(p).resolve() for p in sys.argv[1:]];assert candidates
 historical=private/'preflight/historical-1030-models'
 paths=[historical/name/'model.json.gz' for name in ['spark-ds41f-1030','spark-ds41f-1030-again']]+candidates
 docs={str(p.relative_to(repo)):read_model(p) for p in paths};names=list(docs)
 corpora={behaviour.corpus_digest(d['generation']['sequences']):{r['id']:r for r in d['generation']['sequences']} for d in docs.values()}
 for name,data in docs.items():
  context=corpora.get(data['corpus']['sha256'])
  if context is not None:assert all(r['tokens']==context[r['id']]['output_ids'] for r in data['scores']),name
 pairs=[(names[0],names[1])]+[(base,candidate) for base in names[:2] for candidate in names[2:]]
 comparisons=[{'base':a,'candidate':b,**comparison(docs[a],docs[b],corpora)} for a,b in pairs]
 # A self-pair verifies full-context score alignment without manufacturing a new run.
 sanity=comparison(docs[names[0]],docs[names[0]],corpora)
 assert sanity['generation']['identical']==256 and sanity['scores']['identical_fraction']==1 and sanity['scores']['kl']['mean']==0
 dependencies={str(p.relative_to(repo)):sha(p) for p in [repo/'servebench/behaviour.py',repo/'servebench/stats.py']}
 for name,digest in dependencies.items():assert hashlib.sha256(subprocess.check_output(['git','show','cb657ab1f5dba4e7cca48a48b0c2fe4e8d4a711f:'+name])).hexdigest()==digest
 result={'recorded_models_only':True,'no_inference_requests':True,'python':sys.version,'analysis_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dependencies':dependencies,'runner_sha256':sha(pathlib.Path(__file__)),'self_pair_check_passed':True,'files':{name:{'sha256':sha(repo/name),'label':d['label'],'server':d['server'],'started_utc':d['started_utc'],'workload':d['workload'],'generation_corpus_sha256':behaviour.corpus_digest(d['generation']['sequences']),'scored_corpus_sha256':d['corpus']['sha256'],'scored_positions':sum(len(r['tokens']) for r in d['scores']),'all_selected_and_top_logprobs_finite':True} for name,d in docs.items()},'comparisons':comparisons}
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=private/'model-comparisons'/stamp;out.mkdir(parents=True)
 (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,out/'runner.py')
 (out/'receipt.json').write_text(json.dumps({'argv':sys.argv,'cwd':str(repo),'analysis_sha256':sha(out/'analysis.json'),'runner_sha256':sha(out/'runner.py')},indent=2)+'\n')
 print(str(out));print(json.dumps([{'base':r['base'],'candidate':r['candidate'],'generation_identical':r['generation']['identical'],'equal_prompt_contexts':r['equal_prompt_contexts'],'comparable_scored_contexts':r.get('comparable_scored_contexts'),'scores':{k:r.get('scores',{}).get(k) for k in ['positions','identical_fraction','top1_agreement']},'kl_mean':r.get('scores',{}).get('kl',{}).get('mean'),'warnings':r['warnings']} for r in comparisons]))

if __name__=='__main__':main()
