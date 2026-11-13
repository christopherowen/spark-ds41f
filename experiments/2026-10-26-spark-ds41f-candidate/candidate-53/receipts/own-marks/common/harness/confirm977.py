#!/usr/bin/env python3
"""#977 confirmation of the <=32K cells (candidate-53 own marks): the PR's
bench_copy_match_long_context.py imported unmodified, with CONTEXTS narrowed to 4K/16K/32K
and its timer's replays raised from 21 to 201 (same flush, same median). Usage:
confirm977.py BENCH_PATH --label L --out OUT"""
import functools, importlib.util, sys
path = sys.argv[1]; sys.argv = [path] + sys.argv[2:]
spec = importlib.util.spec_from_file_location("bench977", path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.CONTEXTS = (4096, 16384, 32768)
mod.timed = functools.partial(mod.timed, replays=201)
mod.main()
