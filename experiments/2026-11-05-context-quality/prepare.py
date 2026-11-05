"""Pinned official RULER generators; CPU-only frozen inputs, no model requests."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

HERE=Path(__file__).resolve().parent
LOCK_PATH=HERE/'ruler.lock.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['pilot','heldout'],required=True)
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--lengths',nargs='+',type=int)
    args=parser.parse_args()
    lock=json.loads(LOCK_PATH.read_text())
    assert '.'.join(map(str,sys.version_info[:3]))==lock['python'],'use the pinned preparation interpreter'
    source=args.source.resolve();tokenizer=args.tokenizer.resolve();assets=args.assets.resolve();out=args.out.resolve()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==lock['source']['revision']
    subprocess.run(['git','diff','--exit-code'],cwd=source,check=True,stdout=subprocess.DEVNULL)
    assert not out.exists();out.mkdir(parents=True)
    for name,expected in lock['tokenizer']['files'].items():assert sha(tokenizer/name)==expected['sha256'],name
    for name,expected in lock['data_artifacts'].items():
        assert sha(assets/name)==expected['sha256'],name
    from transformers import AutoTokenizer
    token=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True,trust_remote_code=True)
    data=source/'scripts/data';custom=yaml.safe_load((source/'scripts/synthetic.yaml').read_text())
    defaults=module(data/'synthetic/constants.py','ruler_data_constants').TASKS
    metrics=module(source/'scripts/eval/synthetic/constants.py','ruler_eval_constants').TASKS
    names=args.tasks or lock['tasks'];lengths=args.lengths or lock['generator_lengths']
    assert set(names)<=set(lock['tasks']) and set(lengths)<=set(lock['generator_lengths'])
    receipt={'source_lock_sha256':sha(LOCK_PATH),'split':args.split,'complete':False,'cases':[],'errors':[],'model_requests_sent':0,'prepare_source_sha256':sha(Path(__file__)),'python':'.'.join(map(str,sys.version_info[:3]))}
    def save(): (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    env=dict(os.environ,NLTK_DATA=str(assets/'nltk-data'),TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    try:
        for name in names:
            config=custom[name];base=defaults[config['task']]
            for length in lengths:
                for repeat in range(lock['samples_per_split'][args.split]):
                    seed=lock['seed_base']+lock['split_seed_offsets'][args.split]+lock['tasks'].index(name)*1000+lock['generator_lengths'].index(length)*100+repeat
                    identity=f'{name}-l{length}-s{seed}'
                    folder=out/identity;folder.mkdir()
                    command=[sys.executable,str(data/'synthetic'/(config['task']+'.py')),
                             '--save_dir',str(folder),'--save_name',name,'--tokenizer_path',str(tokenizer),
                             '--tokenizer_type','hf','--max_seq_length',str(length),
                             '--tokens_to_generate',str(base['tokens_to_generate']),'--num_samples','1',
                             '--random_seed',str(seed),'--template',base['template']+base.get('answer_prefix','')]
                    for k,v in config['args'].items():command += ['--'+k,str(v)]
                    if config['task']=='qa':command += ['--pre_samples',str(lock['split_seed_offsets'][args.split]//10+lock['generator_lengths'].index(length)*100+repeat)]
                    (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                    started=time.monotonic()
                    with (folder/'output.txt').open('x') as stream:
                        result=subprocess.run(command,env=env,cwd=source,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
                    assert result.returncode==0,identity+' generator failed'
                    path=folder/name/'validation.jsonl'
                    records=[json.loads(line) for line in path.read_text().splitlines()];assert len(records)==1
                    row=records[0];prompt=row['input']+row.get('answer_prefix','');answers=row['outputs']
                    assert answers and all(isinstance(a,str) and a for a in answers)
                    count=len(token.tokenize(prompt));assert count+base['tokens_to_generate']<=length
                    metric=metrics[config['task']]['metric_fn']
                    assert metric([', '.join(answers)],[answers])==100
                    assert metric(['§⊘☃∅'],[answers])==0
                    case={'id':identity,'task':name,'family':config['task'],'generator_length':length,'seed':seed,
                          'prompt':prompt,'prompt_tokens_untemplated':count,'outputs':answers,
                          'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
                    (folder/'request.json').write_text(json.dumps(case)+'\n')
                    public={k:v for k,v in case.items() if k not in ['prompt','outputs']}
                    public.update(manifest_sha256=sha(path),request_sha256=sha(folder/'request.json'),generator_seconds=time.monotonic()-started)
                    if 'token_position_answer' in row:public['first_answer_depth_fraction']=row['token_position_answer']/count
                    with (out/'requests.jsonl').open('a') as stream:stream.write(json.dumps(case)+'\n')
                    receipt['cases'].append(public);save();print(json.dumps(public),flush=True)
        receipt['complete']=True;receipt['requests_sha256']=sha(out/'requests.jsonl');save()
    except BaseException as error:
        receipt['errors'].append({'type':type(error).__name__,'message':str(error)});save();raise


if __name__=='__main__':main()
