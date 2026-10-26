# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch (PDL) along DeepSeek-V4.1's decode chain.

A decode step is one chain of about 1,850 small kernels in CUDA graphs; at one
stream the graph-internal launch gaps on its critical path (median 0.5 us) are
3.5% of the step. With PDL a kernel is launched while the kernel before it on
its stream still runs, so its launch and CTA start-up overlap that kernel's
tail instead of following its completion.

A kernel on the chain begins with ``chain()``:

- ``T.pdl_sync()`` (``griddepcontrol.wait``) before anything else: the kernel
  reads and writes no global memory until every kernel it depends on has
  completed and its writes are visible, so the arithmetic, the inputs it sees
  and the order of its stores are those of a serialized launch, to the bit.
  TileLang then launches the kernel with programmatic stream serialization,
  which CUDA graph capture keeps as a programmatic edge.
- ``T.pdl_trigger()`` (``griddepcontrol.launch_dependents``) once every CTA has
  passed the wait: the next kernel may launch now, and its CTAs wait in turn.
  At most two kernels of the chain are resident at once, the running one and
  the next, so waiting CTAs never pile up ahead of the work they wait for.

A kernel that does not begin with ``chain()`` launches as before, after its
predecessor completes; one that does, after a kernel that does not, starts as
that kernel's CTAs exit. Kernels compiled with warp specialization do not use
it: their producer warps' TMA loads would have to follow the wait too.

The routed experts' grouped GEMMs begin with ``wait()`` and call ``trigger()``
after their main loop instead, so the kernel after them launches near their
end. They fill every SM for most of a layer's MoE time, and the shared expert's
kernels on the aux stream beside them run in the SM slots their last CTAs free.
Triggered at the top, the routed SwiGLU's CTAs launched into those slots and
waited there; the shared expert then finished after the routed down GEMM, on
the critical path (GB10 traces at 16 and 48 rows), about what PDL saved
elsewhere in the layer.

TileKernels' kernels on the chain (``per_token_cast``, ``norm_forward``,
``moe_topk_gate_forward`` here) use DeepSeek's own PDL switch, set only for the
call: these place the wait first and the trigger after their stores. The switch
stays off for every other TileKernels kernel. Some read an input before their
wait (the mHC post its mixes, RoPE its positions), which is safe only where the
kernel launched just before them did not write it.
"""

import contextlib

import tilelang.language as T


@T.macro
def chain():
    T.pdl_sync()
    T.pdl_trigger()


@T.macro
def wait():
    T.pdl_sync()


@T.macro
def trigger():
    T.pdl_trigger()


@contextlib.contextmanager
def _tile_kernels():
    """TileKernels with its PDL switch on inside the block (each call compiles and
    launches with it; the kernel cache keys on the switch)."""
    import tile_kernels

    previous = tile_kernels.get_pdl()
    tile_kernels.set_pdl(True)
    try:
        yield tile_kernels
    finally:
        tile_kernels.set_pdl(previous)


def per_token_cast(*args, **kwargs):
    """TileKernels' ``quant.per_token_cast``, chained."""
    with _tile_kernels() as tk:
        return tk.quant.per_token_cast(*args, **kwargs)


def norm_forward(*args, **kwargs):
    """TileKernels' ``quant.norm_forward``, chained."""
    with _tile_kernels() as tk:
        return tk.quant.norm_forward(*args, **kwargs)


def moe_topk_gate_forward(*args, **kwargs):
    """TileKernels' ``moe.moe_topk_gate_forward``, chained."""
    with _tile_kernels() as tk:
        return tk.moe.moe_topk_gate_forward(*args, **kwargs)
