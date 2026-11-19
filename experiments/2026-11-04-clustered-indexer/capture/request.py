"""Collect real operands with one public-source request; no performance benchmark."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from spark_ds41f.bench import source_text, standard_library_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8001')
    parser.add_argument('--estimated-tokens', type=int, default=131072)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--window-id')
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args()
    if not options.prepare_only:
        hold = json.loads((Path.home()/'spark-hold.json').read_text())
        if (socket.gethostname() != 'dgx1' or not options.window_id
                or hold.get('window') != options.window_id
                or hold.get('holder') != 'spark-lab: owner-authorized clustered-indexer-input-capture'):
            raise RuntimeError('request requires this capture window on dgx1')
    out = options.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'schema_version': 1, 'passed': False, 'prepared_only': options.prepare_only,
               'workload': 'one Python standard-library source prompt; one greedy completion token',
               'seed': options.seed, 'estimated_tokens': options.estimated_tokens,
               'python': platform.python_version(), 'window': options.window_id,
               'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               'started_at': dt.datetime.now(dt.timezone.utc).isoformat()}

    def request(route, body=None, timeout=30):
        payload = json.dumps(body).encode() if body is not None else None
        message = urllib.request.Request(options.base_url.rstrip('/') + route, data=payload,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(message, timeout=timeout) as response:
            return json.load(response)

    try:
        prompt = source_text(options.estimated_tokens, options.seed)
        (out/'prompt.txt').write_text(prompt)
        receipt['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
        receipt['prompt_characters'] = len(prompt)
        receipt['corpus_manifest'] = [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                     for p in standard_library_sources()]
        if options.prepare_only:
            receipt['prepared'] = True
            return
        model = request('/v1/models')['data'][0]['id']
        tokenized = request('/tokenize', {'model': model, 'prompt': prompt,
                                         'add_special_tokens': True}, timeout=120)
        count = int(tokenized['count'])
        receipt.update(model=model, actual_prompt_tokens=count)
        # Native index compression ratios are 1 and 2. Leave a full query chunk
        # beyond the 32769-key diagnostic threshold in both geometries.
        if not 2 * (32769 + 256) <= count < int(tokenized['max_model_len']) - 1:
            raise RuntimeError(f'prompt token count {count} misses the capture domain or model limit')
        response = request('/v1/completions', {'model': model, 'prompt': prompt,
            'temperature': 0, 'max_tokens': 1, 'ignore_eos': True, 'stream': False}, timeout=1800)
        (out/'response.json').write_text(json.dumps(response, indent=2)+'\n')
        receipt.update(passed=True, usage=response.get('usage'),
                       response_sha256=hashlib.sha256((out/'response.json').read_bytes()).hexdigest())
    except Exception:
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        receipt['ended_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        # Keep prompt contents and token IDs node-local. Only identity is printed.
        print(json.dumps({k:v for k,v in receipt.items() if k!='corpus_manifest'}), flush=True)


if __name__ == '__main__':
    main()
