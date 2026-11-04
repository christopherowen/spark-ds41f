#!/usr/bin/env python3
"""Pinned sparknet kernels: compile coverage plus a local fake-peer replay.

The peer supplies pinned-memory payloads and flags; no NIC or serving rank is
used. This checks instrumentation compatibility, not a physical fabric.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=['network'])
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--stream', action='store_true')
args = parser.parse_args()
import numpy as np
import torch
from stream.collector import HERE, enable
collector = enable(args.out / 'stream', ring_bytes=4096, chunk_bytes=65536, patched=True) if args.stream else None
from sparknet.oneshot import _oneshot_tilelang as ar
from sparknet.oneshot import _allgather_tilelang as ag
from sparknet.oneshot import _exchange_tilelang as ex
from sparknet.oneshot import _reducescatter_tilelang as rs

compiled = []
for topology, world, hcas, ring in [('triangle', 3, 2, False), ('ring', 4, 2, True), ('switched', 4, 4, False)]:
    for operation, module in [('allreduce', ar), ('allgather', ag), ('exchange', ex), ('reducescatter', rs)]:
        prefix = () if operation == 'allgather' else ('float32',)
        module._build(*prefix, world, 0, 128, 2, 64, hcas, ring)()
        compiled.append([topology, operation])

if collector:
    library = collector.lib
else:
    so = args.out / 'emulator.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-pthread',
                    str(HERE / 'writer.cpp'),'-lz','-o',str(so)],check=True)
    library = ctypes.CDLL(str(so))
library.stream_store_u32_release.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
input_tensor = torch.arange(1024,device='cuda',dtype=torch.float32) / 16
region = torch.zeros(65536,dtype=torch.uint8,pin_memory=True)
region_array = region.numpy()
recv, flags, send, ctrl, slot_bytes = 0, 16384, 20480, 32768, 4096
ctrl_words = region_array[ctrl:].view(np.uint32)
ctrl_words[28 // 4] = 1_000_000
results = {}
for operation in ['allreduce','allgather']:
    region_array.fill(0)
    ctrl_words[28 // 4] = 1_000_000
    counters = torch.zeros(5,device='cuda',dtype=torch.int32)
    out = torch.empty(1024 if operation=='allreduce' else 2048,device='cuda',dtype=torch.float32)
    launcher = (ar._build('float32',2,0,128,2,64,2,False)() if operation=='allreduce'
                else ag._build(2,0,128,2,64,2,False)())
    for seq in range(1,4):
        errors = []
        def peer():
            try:
                deadline = time.monotonic() + 2
                while int(ctrl_words[0]) != seq:
                    if time.monotonic() > deadline:
                        raise TimeoutError('kernel did not ring its doorbell')
                    time.sleep(.0001)
                time.sleep(.002)
                slot = seq & 1
                remote = (2 + slot) * slot_bytes
                region_array[remote:remote+4096].view(np.float32).fill(7)
                for lane in range(2):
                    offset = flags + ((2 + slot) * 2 + lane) * 64
                    library.stream_store_u32_release(region.data_ptr()+offset, seq)
            except Exception as error:
                errors.append(str(error))
        thread = threading.Thread(target=peer)
        thread.start()
        common = [input_tensor.view(torch.int32),out.view(torch.int32),counters,region.data_ptr()]
        if operation=='allreduce':
            launcher(*common,256,4096,recv,flags,send,ctrl,slot_bytes,1,2,3,4,1,0)
            expected = input_tensor + 7
        else:
            launcher(*common,256,4096,256,256,recv,flags,send,ctrl,slot_bytes,1,2,3,4,1)
            expected = torch.cat([input_tensor,torch.full_like(input_tensor,7)])
        torch.cuda.synchronize()
        thread.join(3)
        assert not thread.is_alive() and not errors, errors
        assert torch.equal(out, expected)
    results[operation] = hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()
coverage = {}
if collector:
    collector.close()
    from stream.export import export
    results['trace'] = export(args.out / 'stream', args.out / 'trace.json')
    for file in (args.out / 'stream/sources').glob('*.json'):
        manifest=json.loads(file.read_text())
        phases={s['phase'] for s in manifest['sites'] if 'phase' in s}
        required={'store','signal','wait','compute' if manifest['kernel'] in ['oneshot_allreduce_kernel','oneshot_reducescatter_kernel'] else 'load'}
        assert required <= phases, (manifest['kernel'],phases,required)
        coverage[file.name] = {'kernel':manifest['kernel'],'phases':sorted(phases)}
results.update(operation='network',stream=args.stream,compiled=compiled,coverage=coverage,
               scope='local fake peer in pinned memory; no NIC traffic or fabric qualification')
(args.out / 'result.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results),flush=True)
