"""Pinned HumanEvalPlus inputs and separate CPU-side executable grading.

Generation and grading are separate jobs. This module never opens the serving
API, and generated programs must run inside the restricted CPU container.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE/'evalplus.lock.json').read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def inputs(cache):
    for name,item in LOCK['artifacts'].items():
        path = cache/(name+'.bin')
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], name
    with gzip.open(cache/'dataset_gzip.bin','rt') as stream:
        cases = [json.loads(line) for line in stream]
    assert sorted(c['task_id'] for c in cases) == sorted(LOCK['protocol']['task_ids'])
    return {c['task_id']:c for c in cases}


def source_tree(cache):
    target = cache/('evalplus-'+LOCK['revision'])
    with tarfile.open(cache/'source_tar.bin') as archive:
        for member in archive.getmembers():
            if member.isfile() and '/evalplus/' in member.name and member.name.endswith('.py'):
                path = cache/member.name
                assert path.read_bytes() == archive.extractfile(member).read(), str(path)+' differs from pinned archive'
    assert (target/'evalplus/eval/__init__.py').is_file()
    return target


def prepare(cache):
    cache.mkdir(parents=True,exist_ok=True)
    for name,item in LOCK['artifacts'].items():
        path = cache/(name+'.bin')
        if not path.exists():
            partial = path.with_suffix('.part')
            with urllib.request.urlopen(item['url'],timeout=120) as source,partial.open('xb') as output:
                while chunk := source.read(1024*1024):
                    output.write(chunk)
            assert digest(partial) == item['sha256']
            partial.rename(path)
    cases = inputs(cache)
    target = cache/('evalplus-'+LOCK['revision'])
    if not target.exists():
        with tarfile.open(cache/'source_tar.bin') as archive:
            archive.extractall(cache,filter='data')
    source_tree(cache)
    records = []
    for task_id in LOCK['protocol']['task_ids']:
        case = cases[task_id]
        prompt = LOCK['protocol']['instruction_prefix']+'\n```python\n'+case['prompt'].strip()+'\n```'
        records.append({'id':task_id,'entry_point':case['entry_point'],'prompt':prompt,
                        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    path = cache/'requests.jsonl'
    raw = ''.join(json.dumps(r)+'\n' for r in records)
    if path.exists():
        assert path.read_text() == raw
    else:
        path.write_text(raw)
    print(json.dumps({'tasks':len(records),'requests_sha256':digest(path),
                      'source_revision':LOCK['revision'],'dataset_sha256':LOCK['artifacts']['dataset_gzip']['sha256']}))


def final_code(content):
    """Only final content is eligible; never search reasoning for a program.

    Accept one Python fence or a plain Python script. Ambiguous multiple code
    blocks remain unparsed rather than selecting whichever passes the tests.
    """
    if not isinstance(content,str) or not content.strip():
        return None
    if '```' not in content:
        return content.strip()
    if content.count('```') != 2:
        return None
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```',content,re.DOTALL|re.IGNORECASE)
    return blocks[0].strip() if len(blocks) == 1 else None


def official(cache):
    assert Path('/.dockerenv').is_file() and os.environ.get('SPARK_EVALPLUS_ISOLATED') == '1', 'restricted CPU container required'
    inputs(cache)  # Verify the archive before importing the pinned source.
    source = source_tree(cache)
    sys.path.insert(0,str(source))
    from evalplus.eval import PASS,untrusted_check
    from evalplus.gen.util import trusted_exec
    return PASS,untrusted_check,trusted_exec


def check(case,code,checker):
    passed,untrusted,trusted = checker
    result = {}
    reference = case['prompt']+case['canonical_solution']
    for name in ('base','plus'):
        values = case[name+'_input']
        expected,times = trusted(reference,values,case['entry_point'],record_time=True)
        status,details = untrusted('humaneval',code,values,case['entry_point'],expected,
                                   case['atol'],times,fast_check=False)
        result[name] = {'status':status,'passed':status == passed,'tests':len(values),
                        'executed':len(details),'passed_tests':sum(details)}
    result['correct'] = all(result[name]['passed'] for name in ('base','plus'))
    return result


def grade(cache,samples,out):
    assert not out.exists()
    cases = inputs(cache)
    checker = official(cache)
    rows = [json.loads(s) for s in samples.read_text().splitlines()]
    assert [r['id'] for r in rows] == LOCK['protocol']['task_ids'], 'one frozen-order attempt per task'
    results = []
    for row in rows:
        code = final_code(row.get('content'))
        result = {'id':row['id'],'correct':False,'status':'incomplete' if row.get('finish_reason') != 'stop' else 'unparsed-final'}
        if row.get('finish_reason') == 'stop' and code:
            result.update(check(cases[row['id']],code,checker),status='graded')
            result['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
        results.append(result)
        out.write_text(json.dumps({'source_lock':LOCK,'samples_sha256':digest(samples),'records':results,
                                   'complete':len(results) == len(rows)},indent=2)+'\n')
        print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','self-check','grade'])
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--samples',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.cache)
    elif args.action == 'self-check':
        case = inputs(args.cache)['HumanEval/0']
        checker = official(args.cache)
        good = check(case,case['prompt']+case['canonical_solution'],checker)
        bad = check(case,'def '+case['entry_point']+'(*args):\n    return None\n',checker)
        assert good['correct'] and not bad['correct']
        print(json.dumps({'canonical':good,'deliberately_wrong':bad}))
    else:
        assert args.samples and args.out
        grade(args.cache,args.samples,args.out)


if __name__ == '__main__':
    main()
