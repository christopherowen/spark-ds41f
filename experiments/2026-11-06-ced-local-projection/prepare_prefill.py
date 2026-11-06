"""Freeze the standard source-text cold-prefill recipe without issuing requests."""
import argparse
import array
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--deployment-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sizes', default='32768,131072,262144,524288,1048000')
    args = parser.parse_args()
    assert args.snapshot.name == '1bae0e1088993b9b82de08f59de5be5a34e42170'
    assert len(args.deployment_commit) == 40 and all(c in '0123456789abcdef' for c in args.deployment_commit)
    sizes = list(map(int, args.sizes.split(',')))
    assert len(sizes) == len(set(sizes)) and all(0 < n < 1048576 - 64 for n in sizes)
    assert args.repeats >= 2
    sys.path.insert(0, str(args.deployment))
    from spark_ds41f import bench
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
    tokenizer = DeepseekV41Tokenizer.from_pretrained(str(args.snapshot), local_files_only=True)
    add_special_tokens = CompletionRequest(model=args.model, prompt='check', max_tokens=1).add_special_tokens
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema': 'ced-prefill-fixtures/1',
        'scope': 'CPU fixture preparation only; no requests or timing measurements',
        'model_revision': args.snapshot.name,
        'model': args.model,
        'deployment_commit': args.deployment_commit,
        'bench_source_sha256': digest(Path(bench.__file__).read_bytes()),
        'suite_prefill_function_sha256': digest(inspect.getsource(bench.suite_prefill).encode()),
        'tokenizer_sha256': {name: digest((args.snapshot / name).read_bytes())
                             for name in ('tokenizer.json', 'tokenizer_config.json')},
        'stdlib_sha256': {str(path): digest(path.read_bytes()) for path in bench.standard_library_sources()},
        'python': sys.version,
        'recipe': 'suite_prefill source_text(int(size * 0.97), rng.randrange(1 << 30)) plus its exact suffix',
        'seed': args.seed, 'repeats': args.repeats, 'requested_sizes': sizes,
        'add_special_tokens': add_special_tokens,
        'cache_policy': 'Use a fresh cache_salt for every replay; salts do not change prompt bytes or tokens.',
        'cases': [],
    }
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        for size in rng.sample(sizes, len(sizes)):
            prompt_seed = rng.randrange(1 << 30)
            prompt = bench.source_text(int(size * 0.97), prompt_seed) + '\n\nReply with the word ok.'
            raw = prompt.encode('utf-8')
            ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
            name = f'prefill-{size}-r{repeat}'
            (args.out / (name + '.txt')).write_bytes(raw)
            tokens = array.array('I', ids)
            assert tokens.itemsize == 4
            if sys.byteorder != 'little':
                tokens.byteswap()
            token_bytes = tokens.tobytes()
            (args.out / (name + '.tokens.u32le')).write_bytes(token_bytes)
            payload = {'model': args.model, 'prompt': prompt, 'max_tokens': 1, 'temperature': 0,
                       'add_special_tokens': add_special_tokens}
            payload_bytes = (json.dumps(payload, ensure_ascii=False) + '\n').encode()
            (args.out / (name + '.payload.json')).write_bytes(payload_bytes)
            case = {'id': name, 'order': len(manifest['cases']), 'repeat': repeat,
                    'requested_size': size, 'prompt_seed': prompt_seed, 'prompt_tokens': len(ids),
                    'prompt_file': name + '.txt', 'token_file': name + '.tokens.u32le',
                    'payload_file': name + '.payload.json', 'endpoint': '/v1/completions',
                    'prompt_sha256': digest(raw), 'token_ids_sha256': digest(token_bytes),
                    'payload_sha256': digest(payload_bytes), 'fits_context': len(ids) + 1 <= 1048576}
            manifest['cases'].append(case)
            (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps(case), flush=True)
    assert all(case['fits_context'] for case in manifest['cases']), 'oversized fixture retained; no request sent'
    manifest['complete'] = True
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'complete': True, 'cases': len(manifest['cases'])}), flush=True)


if __name__ == '__main__':
    main()
