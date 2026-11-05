"""Lower grouped GEMM to CUDA without GPU access or device compilation."""
import argparse
import hashlib
import json

import tilelang
from tilelang import tvm
from vllm.models.deepseek_v4_1.tilelang.moe import grouped_fp8_fp4_gemm

parser = argparse.ArgumentParser()
parser.add_argument('--expect', choices=('before', 'after'), required=True)
args = parser.parse_args()
target = tvm.target.Target({'kind': 'cuda', 'arch': 'sm_121'})
results = []
for gather, k in ((True, 512), (False, 576), (False, 768)):
    sources = {}
    for panel in (0, 3):
        func = grouped_fp8_fp4_gemm.get_tir(
            32, 384, k, 6, gather, 64, 128,
            coded_scales=True, narrow_codes=True,
            trigger_after_loop=True, swizzle_panel=panel)
        with target, tvm.transform.PassContext(
            opt_level=3, config=grouped_fp8_fp4_gemm.pass_configs):
            artifact = tilelang.lower(func, target=target,
                                      enable_device_compile=False,
                                      enable_host_codegen=False)
        source = artifact.kernel_source
        expected = args.expect == 'after' and panel == 3
        found = 'rasterization2DRow<3>' in source
        assert found == expected, (gather, k, panel, expected,
                                   [l for l in source.splitlines() if 'raster' in l])
        sources[panel] = source
        results.append({'gather': gather, 'k': k, 'panel': panel,
                        'swizzle_emitted': found,
                        'source_sha256': hashlib.sha256(source.encode()).hexdigest()})
    assert (sources[0] == sources[3]) == (args.expect == 'before')
print(json.dumps({'expect': args.expect, 'target': 'sm_121',
                  'device_compilation': False, 'gpu_execution': False,
                  'cases': results, 'result': 'pass'}, indent=2))
