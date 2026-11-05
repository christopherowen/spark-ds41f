"""Pinned benchmark inputs and outcome-independent, disjoint task selection."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import urllib.request

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE/'sources.lock.json').read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def fetch(url,path,expected):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url,timeout=120) as source,temp.open('wb') as dest:
            while chunk := source.read(1024*1024):
                dest.write(chunk)
        assert digest(temp) == expected, 'download hash mismatch'
        temp.rename(path)
    assert digest(path) == expected, str(path)+' hash mismatch'


def download(cache):
    fetch(LOCK['dataset']['url'],cache/'data.json',LOCK['dataset']['sha256'])
    for name,sha in LOCK['files'].items():
        fetch('https://raw.githubusercontent.com/THUDM/LongBench/'+LOCK['revision']+'/'+name,
              cache/'upstream'/name,sha)


def template(cache):
    path = cache/'upstream/prompts/0shot.txt'
    assert digest(path) == LOCK['files']['prompts/0shot.txt']
    return path.read_text()


def prompt(item,text):
    for marker,key in [('DOC','context'),('Q','question'),('C_A','choice_A'),('C_B','choice_B'),('C_C','choice_C'),('C_D','choice_D')]:
        text = text.replace('$'+marker+'$',item[key].strip())
    return text


def grader(cache):
    path = cache/'upstream/pred.py'
    assert digest(path) == LOCK['files']['pred.py']
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'extract_answer')
    # Run the pinned official pure extractor, without importing its server,
    # Torch, dataset loader or inference-time defaults. MIT notice stays cached.
    env = {'re':re}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),env)
    return env['extract_answer']


def select(catalog):
    spec = LOCK['protocol']
    groups = {}
    for item in catalog:
        for name,(low,high) in spec['token_bins'].items():
            if low <= item['prompt_tokens_untemplated'] < high:
                groups.setdefault((item['domain'],name),[]).append(item)
    result = {'pilot':[],'heldout':[],'strata':[]}
    for (domain,name),rows in sorted(groups.items()):
        rows.sort(key=lambda x:hashlib.sha256((spec['split_seed']+domain+name+x['id']).encode()).hexdigest())
        a,b = spec['pilot_per_domain_bin'],spec['heldout_per_domain_bin']
        assert len(rows) >= a+b, (domain,name,len(rows))
        result['pilot'] += [x['id'] for x in rows[:a]]
        result['heldout'] += [x['id'] for x in rows[a:a+b]]
        result['strata'].append({'domain':domain,'bin':name,'eligible':len(rows),
                                'pilot':[x['id'] for x in rows[:a]],'heldout':[x['id'] for x in rows[a:a+b]]})
    assert not set(result['pilot']) & set(result['heldout'])
    assert len(result['strata']) == 12, 'all six domains and both token bins required'
    result.update(dataset_sha256=LOCK['dataset']['sha256'],source_revision=LOCK['revision'],
                  protocol=spec,catalog_sha256=hashlib.sha256(json.dumps(catalog,sort_keys=True).encode()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['download','catalog'])
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path)
    args = parser.parse_args()
    if args.command == 'download':
        download(args.cache)
        return
    from tokenizers import Tokenizer
    assert digest(args.cache/'data.json') == LOCK['dataset']['sha256']
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    text = template(args.cache)
    catalog = []
    for item in json.loads((args.cache/'data.json').read_text()):
        rendered = prompt(item,text)
        catalog.append({'id':item['_id'],'domain':item['domain'],'difficulty':item['difficulty'],
            'benchmark_length':item['length'],'prompt_tokens_untemplated':len(tokenizer.encode(rendered).ids),
            'prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest()})
    (args.cache/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    selection = select(catalog)
    (args.cache/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps({'pilot':len(selection['pilot']),'heldout':len(selection['heldout']),'strata':selection['strata']}),flush=True)


if __name__ == '__main__':
    main()
