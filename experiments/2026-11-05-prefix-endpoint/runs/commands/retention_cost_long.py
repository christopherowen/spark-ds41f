import hashlib
import json
import subprocess
from pathlib import Path

area = Path('/home/chris/spark-lab/agents/prefix-endpoint')
source = area / 'vllm'
out = area / 'retention-cost-long'
out.mkdir(exist_ok=False)
coordinator_path = 'vllm/v1/core/kv_cache_coordinator.py'
test_path = 'tests/v1/core/test_prefix_caching.py'
selected = (source / coordinator_path).read_text()
original = subprocess.check_output(['git', '-C', str(source), 'show',
    '1f33102547e4da5c64712390604695d1abd577c4:' + coordinator_path], text=True)
tests = (source / test_path).read_text()
dense = tests.replace('@pytest.mark.parametrize("retention_interval", [0, 512])',
                      '@pytest.mark.parametrize("retention_interval", [256])')
start = dense.index('def make_ced_kv_cache_config(')
end = dense.index('\ndef test_ced_private_swa', start)
dense = dense[:start] + dense[start:end].replace('prefix_cache_retention_interval=0,',
                                               'prefix_cache_retention_interval=128,') + dense[end:]
stats = '''
@pytest.mark.parametrize("cost_length", [262464, 1000256])
def test_retention_cost_receipt(cost_length):
    import json
    config = make_ced_kv_cache_config()
    config.num_blocks = 32768
    config.prefix_cache_retention_interval = INTERVAL
    for group in config.kv_cache_groups:
        group.kv_cache_spec = replace(group.kv_cache_spec,
            block_size=256 if group.layer_names == ["global"] else 128)
    manager = make_kv_cache_manager(config, max_model_len=1048576, hash_block_size=128)
    tokens = list(range(cost_length))
    producer = make_request("cost", tokens, 128, sha256)
    assert manager.allocate_slots(producer, len(tokens)) is not None
    producer.num_computed_tokens = len(tokens)
    blocks = [list(m.req_to_blocks[producer.request_id])
              for m in manager.coordinator.single_type_managers]
    manager.free(producer)
    counts = [sum(b.block_hash is not None for b in group if not b.is_null) for group in blocks]
    consumer = make_request("cost-extension", tokens + [9000] * 129, 128, sha256)
    _, hit_tokens, _ = manager.get_computed_blocks(consumer)
    print("RETENTION_COST", json.dumps({"interval": INTERVAL, "prompt_tokens": len(tokens),
        "hit_tokens": hit_tokens, "hashed_blocks_by_group": counts,
        "groups": ["global", "encoder", "private_decoder", "private_drafter"]}))
'''
variants = []
for label, implementation, interval in [('original-sparse', original, 512),
                                       ('selected-sparse', selected, 512),
                                       ('original-dense', original, 256)]:
    variants.append((label, implementation, tests + stats.replace('INTERVAL', str(interval)),
                     'retention_cost_receipt'))
receipts = []
for label, implementation, test_code, selection in variants:
    d = out / label
    d.mkdir()
    (d / 'coordinator.py').write_text(implementation)
    (d / 'test_prefix_caching.py').write_text(test_code)
    argv = ['docker', 'run', '--rm', '--name', 'prefix-retention-cost', '--runtime=runc',
            '--network=none', '--cpus=2', '--memory=4g', '--memory-swap=4g',
            '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
            '-e', 'HF_HUB_OFFLINE=1', '-e', 'VLLM_PLUGINS=', '-e', 'PYTHONPATH=/source',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
            '-v', f'{source}:/source:ro', '-v', f'{area / "incoming"}:/probe:ro',
            '-v', f'{d / "coordinator.py"}:/source/{coordinator_path}:ro',
            '-v', f'{d / "test_prefix_caching.py"}:/source/{test_path}:ro',
            '-w', '/source', '--entrypoint', 'python3', 'spark-ds41f:candidate-38',
            '/probe/cpu_runner.py', '--noconftest', '-p', 'no:cacheprovider', test_path,
            '-k', selection, '-q', '-s']
    receipt = {'label': label, 'argv': argv,
               'coordinator_sha256': hashlib.sha256(implementation.encode()).hexdigest(),
               'tests_sha256': hashlib.sha256(test_code.encode()).hexdigest()}
    with (d / 'output.txt').open('x') as stream:
        receipt['exit_code'] = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
    (d / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    receipts.append(receipt)
    (out / 'receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(label, receipt['exit_code'], flush=True)
