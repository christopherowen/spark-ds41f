"""Check an existing servebench model result against frozen needle fixtures, on CPU."""
import argparse
import array
import gzip
import hashlib
import json
from pathlib import Path
import sys

from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--fixtures', type=Path, required=True)
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.fixtures/'manifest.json').read_text())
assert manifest['complete']
for name, expected in manifest['tokenizer_sha256'].items():
    assert hashlib.sha256((args.snapshot/name).read_bytes()).hexdigest() == expected
opener = gzip.open if args.result.suffix == '.gz' else open
with opener(args.result, 'rt') as stream:
    result = json.load(stream)
assert result['workload']['extra_body'] == {'chat_template_kwargs': {'thinking': False}}
assert result['workload']['max_tokens'] == manifest['max_tokens']
tokenizer = DeepseekV41Tokenizer.from_pretrained(str(args.snapshot), local_files_only=True)
sequences = result['generation']['sequences']
assert len(sequences) == len(manifest['cases'])
checks = []
for index, (case, sequence) in enumerate(zip(manifest['cases'], sequences)):
    check = {'id': case['id'], 'error': sequence.get('error')}
    assert sequence['id'] == index
    if not check['error']:
        ids = array.array('I', sequence['prompt_ids'])
        if sys.byteorder != 'little':
            ids.byteswap()
        check['prompt_ids_equal'] = hashlib.sha256(ids.tobytes()).hexdigest() == case['chat_token_ids_sha256']
        text = tokenizer.decode(sequence['output_ids'], skip_special_tokens=True).strip()
        check['output_text'] = text
        try:
            answer = json.loads(text)
        except ValueError:
            answer = None
        check['strict_json_exact'] = answer == case['expected']
        check['values_present'] = {key: value in text for key, value in case['expected'].items()}
    check['passed'] = not check['error'] and check['prompt_ids_equal'] and check['strict_json_exact']
    checks.append(check)
report = {'schema': 'ced-needle-check/1', 'result': str(args.result),
          'fixture_manifest_sha256': hashlib.sha256((args.fixtures/'manifest.json').read_bytes()).hexdigest(),
          'passed': all(check['passed'] for check in checks), 'checks': checks}
args.out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'passed': report['passed'], 'cases': len(checks),
                  'failed': [case['id'] for case in checks if not case['passed']]}))
raise SystemExit(0 if report['passed'] else 1)
