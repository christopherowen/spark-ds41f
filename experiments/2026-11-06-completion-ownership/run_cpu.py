"""Run delayed-completion checks in a disposable CPU-only candidate image."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import resource

root = Path('/input')
output = Path('/output')
source = Path('/opt/spark-ds41f/candidate/vllm')
identity = json.loads((root / 'identity.json').read_text())
receipt = {'identity': identity, 'cpu_only': True, 'started': time.time()}
mutation = os.environ.get('COMPLETION_MUTATION', '')
receipt['mutation'] = mutation or None
output.mkdir(exist_ok=True)
try:
    for name, record in identity['files'].items():
        path = source / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['image_sha256'], name
        if record['overlay']:
            shutil.copyfile(root / 'overlay' / name, path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['tested_sha256'], name
    mutations = {
        'confidence': ('vllm/v1/worker/gpu/spec_decode/adaptive_verification.py',
                       '        self._pending_resets.append(req_idx)\n', '',
                       'test_old_confidence_dma_lands_after_slot_reuse'),
        'generation': ('vllm/v1/worker/gpu/pp_utils.py',
                       '        self.req_idx_gen_np[req_idx] += 1', '        pass',
                       'test_pp_completion_filters_reassigned_slots_and_releases_owners'),
        'snapshot': ('vllm/v1/worker/gpu/async_utils.py',
                     '            num_verified_draft_tokens.clone()',
                     '            num_verified_draft_tokens',
                     'test_delayed_verification_output_keeps_its_snapshot_and_releases_tensors'),
    }
    if mutation:
        name, before, after, test = mutations[mutation]
        path = source / name
        contents = path.read_text()
        assert contents.count(before) == 1
        path.write_text(contents.replace(before, after))
        receipt['fault_injection'] = {'file': name, 'before': before, 'after': after,
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    os.chdir(source)
    sys.path[:0] = [str(root), str(source)]
    os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', VLLM_TARGET_DEVICE='cpu',
                     CUDA_VISIBLE_DEVICES='-1', VLLM_PLUGINS='',
                     HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                     OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    import cpu_declarations
    from tests.v1.core.test_copy_drafts_scheduler import _OPT_CONFIG
    config = output / 'opt-config'
    config.mkdir(exist_ok=True)
    (config / 'config.json').write_text(json.dumps(_OPT_CONFIG))
    os.environ['VLLM_TEST_DEFER_FREE_MODEL'] = str(config)
    from tests.v1.core import utils
    # Supply an offline model config to the unmodified scheduler test factory.
    defaults = list(utils.create_scheduler.__defaults__)
    defaults[0] = str(config)
    utils.create_scheduler.__defaults__ = tuple(defaults)
    import pytest
    selected = [
        str(root / 'test_delayed_ownership.py'),
        'tests/v1/core/test_async_scheduler.py',
        'tests/v1/core/test_deferred_block_free.py',
        'tests/v1/spec_decode/test_adaptive_verification.py',
        'tests/v1/worker/test_pp_utils.py',
        'tests/v1/executor/test_multiproc_gather_responses.py',
    ]
    if mutation:
        selected = [str(root / 'test_delayed_ownership.py') + '::' + test]
    receipt['command'] = ['pytest', '--noconftest', '-c', '/dev/null', '-p',
                          'cpu_declarations', '-q', '--tb=short', *selected,
                          '--junitxml=/output/tests.xml']
    code = pytest.main(receipt['command'][1:])
    receipt['exit_code'] = int(code)
except BaseException as error:
    receipt['error'] = repr(error)
    raise
finally:
    receipt['seconds'] = time.time() - receipt['started']
    receipt['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
raise SystemExit(code)
