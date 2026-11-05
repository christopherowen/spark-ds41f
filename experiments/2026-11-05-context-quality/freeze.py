"""Freeze complete generated inputs and the pinned official grader before inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--split',choices=['pilot','heldout'],required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    lock=json.loads((HERE/'ruler.lock.json').read_text());r=json.loads((args.cache/'receipt.json').read_text())
    assert r['complete'] and not r['errors'] and r['split']==args.split
    assert r['source_lock_sha256']==sha(HERE/'ruler.lock.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()==lock['source']['revision']
    subprocess.run(['git','diff','--exit-code'],cwd=args.source,check=True,stdout=subprocess.DEVNULL)
    expected=len(lock['tasks'])*len(lock['generator_lengths'])*lock['samples_per_split'][args.split]
    assert len(r['cases'])==expected
    records=[json.loads(line) for line in (args.cache/'requests.jsonl').read_text().splitlines()]
    assert len(records)==expected and [x['id'] for x in records]==[x['id'] for x in r['cases']]
    for row,receipt in zip(records,r['cases']):
        assert row['prompt_sha256']==receipt['prompt_sha256']==hashlib.sha256(row['prompt'].encode()).hexdigest()
        assert row['prompt_tokens_untemplated']+65536+4096<=1048576
    assert sha(args.cache/'requests.jsonl')==r['requests_sha256']
    grader=args.source/'scripts/eval/synthetic/constants.py'
    target=args.cache/'grader.py'
    assert not target.exists();shutil.copyfile(grader,target)
    manifest={'lock':lock,'split':args.split,'requests_sha256':r['requests_sha256'],
              'grader_sha256':sha(target),'case_count':expected,'preparation_receipt_sha256':sha(args.cache/'receipt.json'),
              'protocol':dict(lock['generation_protocol'],concurrency=1,cache_reuse=False),
              'scope':'Project chat/high-thinking RULER task coverage; official string matching of completed final content; no standard leaderboard claim'}
    path=args.cache/'evaluation.json';assert not path.exists();path.write_text(json.dumps(manifest,indent=2)+'\n')
    assert not args.out.exists()
    args.out.write_text(json.dumps({'split':args.split,'evaluation_sha256':sha(path),'requests_sha256':r['requests_sha256'],
                                   'grader_sha256':sha(target),'case_count':expected,'model_outputs_seen':0},indent=2)+'\n')
    print(args.out.read_text())


if __name__=='__main__':main()
