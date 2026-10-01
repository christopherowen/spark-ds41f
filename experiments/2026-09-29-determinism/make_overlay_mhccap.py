#!/usr/bin/env python3
"""Build overlay mhc-capture: ref2's b12x_layers.py plus a debug capture of mHC inputs.

usage: make_overlay_mhccap.py   (on dgx1; writes ~/spark3-overlay/mhc-capture/b12x_layers.py)

With SPARK3_DEBUG_MHC_CAPTURE=DIR:MIN_ROWS:MAX_ROWS set and DIR/arm present,
rank 0 saves, once per (layer, operation) for the layers in
SPARK3_DEBUG_MHC_CAPTURE_LAYERS (default 1,20,39), the first eager mHC call of
at least MIN_ROWS rows: its residual streams, previous output and mixes (up to
MAX_ROWS rows) and the layer's mHC weights, to DIR/mhc-<layer>-<operation>.pt.
The arm file keeps the startup profiling pass (dummy inputs) out.
"""
import os

HOME = os.path.expanduser("~")
src = open(f"{HOME}/spark3-overlay/ref2/vllm/models/deepseek_v4_1/b12x_layers.py").read()
hook = '''

# Debug (SPARK3_DEBUG_MHC_CAPTURE=DIR:MIN_ROWS:MAX_ROWS, DIR/arm present): rank 0
# saves one large eager step's mHC inputs and weights per layer and operation.
_MHC_CAPTURE = os.environ.get("SPARK3_DEBUG_MHC_CAPTURE")
_MHC_CAPTURED: set = set()


def _capture_mhc_inputs(layer_name, residual, fn, scale, base, norm, pre,
                        previous_output, previous_post, previous_comb):
    import re

    if torch.cuda.is_current_stream_capturing():
        return
    directory, min_rows, max_rows = _MHC_CAPTURE.split(":")
    rows = int(residual.shape[0])
    if rows < int(min_rows) or not os.path.exists(os.path.join(directory, "arm")):
        return
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return
    name = str(_resolve_layer_name(layer_name))
    found = re.findall(r"layers\\.(\\d+)", name)
    layers = os.environ.get("SPARK3_DEBUG_MHC_CAPTURE_LAYERS", "1,20,39").split(",")
    if not found or found[-1] not in layers:
        return
    operation = "pre" if previous_output is None else "post_pre"
    if (found[-1], operation) in _MHC_CAPTURED:
        return
    _MHC_CAPTURED.add((found[-1], operation))
    keep = min(rows, int(max_rows))

    def take(t):
        return None if t is None else t[:keep].detach().to("cpu", copy=True)

    torch.save(
        {"layer": name, "operation": operation, "rows": rows, "residual": take(residual),
         "pre": take(pre), "previous_output": take(previous_output),
         "previous_post": take(previous_post), "previous_comb": take(previous_comb),
         "fn": fn.detach().cpu(), "scale": scale.detach().cpu(), "base": base.detach().cpu(),
         "norm": norm.detach().cpu()},
        os.path.join(directory, f"mhc-{found[-1]}-{operation}.pt"),
    )
'''
anchor = '''@torch.library.custom_op(
    "vllm::dsv41_mhc_pre", mutates_args=("residual_out", "y", "post", "comb", "pre_out")
)'''
assert src.count(anchor) == 1
src = src.replace(anchor, hook.lstrip("\n") + "\n\n" + anchor)
body = '''    mhc_module = b12x_layer(_resolve_layer_name(layer_name))._b12x_mhc
    operation = "pre" if previous_output is None else "post_pre"'''
assert src.count(body) == 1
src = src.replace(body, '''    if _MHC_CAPTURE:
        _capture_mhc_inputs(layer_name, residual, fn, scale, base, norm, pre,
                            previous_output, previous_post, previous_comb)
''' + body)
out = f"{HOME}/spark3-overlay/mhc-capture"
os.makedirs(out, exist_ok=True)
compile(src, "b12x_layers.py", "exec")
open(f"{out}/b12x_layers.py", "w").write(src)
print("wrote", f"{out}/b12x_layers.py")
