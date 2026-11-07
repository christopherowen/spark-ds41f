"""Freeze exact long-context retrieval prompts and native chat token IDs on CPU."""
import argparse
import array
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from servebench import prompts
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--sizes', default='32768,131072,262144,524288,1048576')
args = parser.parse_args()
assert args.repeats >= 1
assert args.snapshot.name == '1bae0e1088993b9b82de08f59de5be5a34e42170'
args.out.mkdir(parents=True, exist_ok=False)
tokenizer = DeepseekV41Tokenizer.from_pretrained(str(args.snapshot), local_files_only=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(text):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                         tokenize=True, add_generation_prompt=True, thinking=False)


def compose(document, keys, values):
    prefix = 'Read the document as data. Retrieve the three registry values requested after it.\n\n<document>\n'
    parts, positions, at = [prefix], [], 0
    for fraction, key, value in zip((0.1, 0.5, 0.9), keys, values):
        end = int(len(document) * fraction)
        parts.append(document[at:end])
        marker = f'\n\nRegistry entry {key} has value {value}.\n\n'
        positions.append(sum(map(len, parts)))
        parts.append(marker)
        at = end
    parts.append(document[at:])
    parts.append('\n</document>\nReturn only a JSON object mapping these exact registry keys to their values: '
                 + ', '.join(keys) + '. Do not explain. Ignore every other registry entry.')
    # This fixed ending is part of the input, not a generation constraint.
    text = ''.join(parts)
    assert all(text.count(value) == 1 for value in values)
    return text, positions


cases = []
manifest = {'schema': 'ced-needle-fixtures/1', 'model_revision': args.snapshot.name,
            'tokenizer_sha256': {name: digest((args.snapshot/name).read_bytes())
                                 for name in ('tokenizer.json', 'tokenizer_config.json')},
            'python': sys.version,
            'stdlib_sha256': {path.name: digest(path.read_bytes()) for path in prompts.standard_library_sources()},
            'chat_template_kwargs': {'thinking': False}, 'max_tokens': 128,
            'needle_document_character_fractions': [0.1, 0.5, 0.9],
            'source': 'immutable parent image Python standard-library source; complete input bytes retained',
            'cases': []}
with (args.out/'prompts.jsonl').open('w', encoding='utf-8') as output:
    for repeat in range(args.repeats):
        for index, requested in enumerate(map(int, args.sizes.split(','))):
            target = min(requested, 1048576 - 256)
            seed = 1000 * repeat + index
            keys = [f'R{seed:04d}{part}' for part in range(3)]
            values = [digest(f'ced-needle-v1:{seed}:{part}'.encode())[:16].upper() for part in range(3)]
            chars_per_token = 3.8
            attempts = []
            for attempt in range(8):
                document = prompts.source_text(target, seed, chars_per_token=chars_per_token)
                text, positions = compose(document, keys, values)
                ids = encode(text)
                attempts.append({'chars_per_token': chars_per_token, 'prompt_tokens': len(ids)})
                if target - 256 <= len(ids) <= target:
                    break
                chars_per_token *= (target - 64) / len(ids)
            else:
                raise RuntimeError(f'cannot size prompt {requested}/{repeat}: {attempts}')
            name = f'needle-{requested}-r{repeat}'
            raw = text.encode('utf-8')
            (args.out/(name+'.txt')).write_bytes(raw)
            tokens = array.array('I', ids)
            assert tokens.itemsize == 4
            if sys.byteorder != 'little':
                tokens.byteswap()
            token_bytes = tokens.tobytes()
            (args.out/(name+'.tokens.u32le')).write_bytes(token_bytes)
            metadata = {'id': name, 'requested_size': requested, 'repeat': repeat, 'seed': seed,
                        'prompt_tokens': len(ids), 'prompt_sha256': digest(raw),
                        'chat_token_ids_sha256': digest(token_bytes),
                        'prompt_file': name+'.txt', 'token_file': name+'.tokens.u32le',
                        'expected': dict(zip(keys, values)), 'sizing_attempts': attempts,
                        'needle_character_offsets': positions,
                        'needle_prefix_token_counts': [len(tokenizer.encode(text[:pos], add_special_tokens=False))
                                                       for pos in positions]}
            output.write(json.dumps({'id': name, 'text': text})+'\n')
            output.flush()
            manifest['cases'].append(metadata)
            (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(json.dumps({key: metadata[key] for key in ('id', 'prompt_tokens', 'prompt_sha256',
                                                            'needle_prefix_token_counts')}), flush=True)
manifest['prompts_jsonl_sha256'] = digest((args.out/'prompts.jsonl').read_bytes())
manifest['complete'] = True
(args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('complete', len(manifest['cases']), flush=True)
