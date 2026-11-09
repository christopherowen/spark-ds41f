"""Where the TileLang packed head's screening-table view points after
PackedBf16LmHeadMethod packs (and, with #998, compacts) its carve-out buffer.

Runs the vocab test's sequence (n=16384, k=4096, carve-out required, TileLang
backend) up to just before the screen table is built, then reports the view's
address against the live carve-out buffers and the CUDA driver's mapping of it.
It does not write through the view."""
import ctypes
import json

import torch

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.online import packed_bf16
from vllm.models.deepseek_v4_1.tilelang import vocab
from vllm.v1.worker import display_carveout

display_carveout._REQUIRED = True
cuda = ctypes.CDLL("libcuda.so.1")
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12


def mapping(ptr):
    start = ctypes.c_ulonglong()
    size = ctypes.c_size_t()
    rc1 = cuda.cuPointerGetAttribute(ctypes.byref(start), CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, ctypes.c_ulonglong(ptr))
    rc2 = cuda.cuPointerGetAttribute(ctypes.byref(size), CU_POINTER_ATTRIBUTE_RANGE_SIZE, ctypes.c_ulonglong(ptr))
    return {"rc": [rc1, rc2], "range_start": hex(start.value) if not rc1 else None,
            "range_mib": (size.value >> 20) if not rc2 else None}


def buffers():
    return [{"start": hex(c.pointer.value), "mib": c.nbytes >> 20} for c in display_carveout._carveouts]


class _HeadLayer(torch.nn.Module):
    pass


report = {}
seen = {}
orig_ensure = vocab._ensure_screen_weights


def probe(layer):
    wq = layer._carveout_screen_wq
    w = layer.weight.data
    seen["after_pack_and_compact"] = {
        "carveout_buffers": buffers(),
        "weight_ptr": hex(w.data_ptr()),
        "weight_mib": round(w.numel() * w.element_size() / 2**20, 2),
        "screen_wq_ptr": hex(wq.data_ptr()),
        "screen_wq_mib": round(wq.numel() / 2**20, 2),
        "screen_wq_in_live_buffer": any(display_carveout._holds(c, wq) for c in display_carveout._carveouts),
        "screen_wq_driver_mapping": mapping(wq.data_ptr()),
        "screen_wq_end_driver_mapping": mapping(wq.data_ptr() + wq.numel() - 1),
    }
    raise SystemExit(0)


n, k = 16384, 4096
config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
with set_current_vllm_config(config):
    method = packed_bf16.PackedBf16LmHeadMethod()
    layer = _HeadLayer()
    with torch.device("cuda"):
        method.create_weights(layer, k, [n], k, n, torch.bfloat16)
    wq = layer._carveout_screen_wq
    report["after_create_weights"] = {
        "carveout_buffers": buffers(),
        "weight_ptr": hex(layer.weight.data_ptr()),
        "screen_wq_ptr": hex(wq.data_ptr()),
        "screen_wq_in_live_buffer": any(display_carveout._holds(c, wq) for c in display_carveout._carveouts),
        "screen_wq_driver_mapping": mapping(wq.data_ptr()),
    }
    generator = torch.Generator(device="cuda").manual_seed(42)
    layer.weight.data.copy_((torch.randn(n, k, generator=generator, device="cuda") * 0.02).bfloat16())
    vocab._ensure_screen_weights = probe
    try:
        method.process_weights_after_loading(layer)
    except SystemExit:
        pass
    report.update(seen)
print(json.dumps(report, indent=1))
