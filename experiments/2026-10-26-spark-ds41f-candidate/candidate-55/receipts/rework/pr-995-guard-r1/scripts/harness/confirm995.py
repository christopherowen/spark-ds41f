#!/usr/bin/env python3
"""#995 guard confirmation of the gate's copy cells (round 3), as candidate-53's confirm977.py did for
#977: benchmarks/kernels/benchmark_copy_drafts.py imported unmodified and run as --gate --cases copy,
its timer's replays raised from 21 to 201 (same 64 MiB flush, same median), because the 21-replay
medians of this ~15-55 us step are bimodal (they switch between levels about 2 us apart).
Usage: confirm995.py (from the vLLM tree's root)"""
import functools, importlib.util, sys
path = "benchmarks/kernels/benchmark_copy_drafts.py"
sys.argv = [path, "--gate", "--cases", "copy"]
spec = importlib.util.spec_from_file_location("bcd", path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.timed = functools.partial(mod.timed, replays=201)
mod.main()
