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

TileKernels' casts and norms (``tile_kernels()``) use DeepSeek's own PDL
switch, which places the wait first and the trigger after the stores.
"""

import functools

import tilelang.language as T


@T.macro
def chain():
    T.pdl_sync()
    T.pdl_trigger()


@functools.cache
def tile_kernels():
    """TileKernels, with PDL on for every kernel it compiles from now on."""
    import tile_kernels

    tile_kernels.config.set_pdl(True)
    return tile_kernels
