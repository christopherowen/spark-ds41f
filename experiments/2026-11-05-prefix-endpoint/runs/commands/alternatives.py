"""CPU-only alternatives and exhaustive native page-boundary audit."""
import hashlib
import json
import subprocess
from pathlib import Path

area = Path('/home/chris/spark-lab/agents/prefix-endpoint')
source = area / 'vllm'
out = area / 'alternative-runs'
out.mkdir(exist_ok=False)
coordinator_path = 'vllm/v1/core/kv_cache_coordinator.py'
test_path = 'tests/v1/core/test_prefix_caching.py'
selected = (source / coordinator_path).read_text()
original = subprocess.check_output(['git', '-C', str(source), 'show',
    '1f33102547e4da5c64712390604695d1abd577c4:' + coordinator_path], text=True)
tests = (source / test_path).read_text()
assert 'return (resend, request.num_prompt_tokens)' in selected
variants = [
    ('endpoint-only', selected.replace('return (resend, request.num_prompt_tokens)',
                                      'return (request.num_prompt_tokens,)'), tests),
    ('end-minus-one', selected.replace('return (resend, request.num_prompt_tokens)',
                                      'return (resend, max(request.num_prompt_tokens - 1, 0))'), tests),
    ('dense-256-original-policy', original, tests.replace(
        '@pytest.mark.parametrize("retention_interval", [0, 512])',
        '@pytest.mark.parametrize("retention_interval", [256])')),
]
sweep = tests.replace('@pytest.mark.parametrize("retention_interval", [0, 512])',
                     '@pytest.mark.parametrize("retention_interval", [0, 256, 512, 4096])')
sweep = sweep.replace('@pytest.mark.parametrize("prompt_length", [4351, 4352, 4353, 4479, 4480, 4608])',
                     '@pytest.mark.parametrize("prompt_length", range(4096, 4608))\n'
                     '@pytest.mark.parametrize("extension_length", [0, 1, 127, 128, 129, 256])')
sweep = sweep.replace('@pytest.mark.parametrize("chunked", [False, True])\n'
                      '@pytest.mark.parametrize("producer_live", [False, True])\n'
                      'def test_ced_retains_encoder_endpoint_for_extended_prompt(',
                      '@pytest.mark.parametrize("chunked", [False])\n'
                      '@pytest.mark.parametrize("producer_live", [False])\n'
                      'def test_ced_retains_encoder_endpoint_for_extended_prompt(')
sweep = sweep.replace('    retention_interval, prompt_length, chunked, producer_live\n',
                      '    retention_interval, prompt_length, chunked, producer_live, extension_length\n')
sweep = sweep.replace('consumer = make_request("extension", tokens + [9000] * 129, 128, sha256)',
                      'consumer = make_request("extension", tokens + [9000] * extension_length, 128, sha256)')
sweep = sweep.replace('assert hit_tokens == prompt_length // 256 * 256',
                      'assert hit_tokens == min(prompt_length, prompt_length + extension_length - 128) // 256 * 256')
variants.append(('selected-boundary-sweep', selected, sweep))

receipts = []
for label, implementation, test_code in variants:
    d = out / label
    d.mkdir()
    (d / 'coordinator.py').write_text(implementation)
    (d / 'test_prefix_caching.py').write_text(test_code)
    argv = ['docker', 'run', '--rm', '--name', 'prefix-endpoint-audit',
            '--runtime=runc', '--network=none', '--cpus=2', '--memory=4g', '--memory-swap=4g',
            '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
            '-e', 'HF_HUB_OFFLINE=1', '-e', 'VLLM_PLUGINS=', '-e', 'PYTHONPATH=/source',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
            '-v', f'{source}:/source:ro', '-v', f'{area / "incoming"}:/probe:ro',
            '-v', f'{d / "coordinator.py"}:/source/{coordinator_path}:ro',
            '-v', f'{d / "test_prefix_caching.py"}:/source/{test_path}:ro',
            '-w', '/source', '--entrypoint', 'python3', 'spark-ds41f:candidate-38',
            '/probe/cpu_runner.py', '--noconftest', '-p', 'no:cacheprovider', test_path,
            '-k', 'ced_retains_encoder_endpoint_for_extended_prompt or ced_endpoint_retention_respects_completion_and_eviction', '-q']
    receipt = {'label': label, 'argv': argv,
               'coordinator_sha256': hashlib.sha256(implementation.encode()).hexdigest(),
               'tests_sha256': hashlib.sha256(test_code.encode()).hexdigest()}
    with (d / 'output.txt').open('x') as stream:
        receipt['exit_code'] = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
    (d / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    receipts.append(receipt)
    (out / 'receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(label, receipt['exit_code'], flush=True)
