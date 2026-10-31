# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pytest plugin (``-p tests.kernels.deepseek_v41_pdl_sources``): every DeepSeek-V4.1
kernel a test session launches is checked for PDL in its CUDA source.

Run with the DS4.1 TileLang kernel tests, it records each TileLang kernel at its
first launch and, when the session ends, checks every ``__global__`` function:

- a kernel of the PDL chain (``CHAIN``; ``gemm_mxfp8`` when compiled without warp
  specialization) calls ``cudaGridDependencySynchronize()`` outside any block
  (every thread waits) and before any statement that names one of its pointer
  parameters, then ``cudaTriggerProgrammaticLaunchCompletion()``;
- a warp-specialized GEMM (``WARP_SPECIALIZED``, ``gemm_mxfp8``'s prefill tiles)
  has no PDL: its producer warps' loads would have to follow the wait too;
- any other kernel that waits (TileKernels' casts, norms and top-k gate, which the
  deployment launches through ``pdl``'s wrappers with DeepSeek's switch on) waits
  before it names a pointer parameter.

A violation fails the session. ``EXEMPT`` names the PDL test's deliberate early
reader.
"""

import re

CHAIN = {
    "activation_swiglu_clamp",
    "attention_front",
    "cache_writer",
    "compress",
    "compress_gather_partial_state",
    "gemm_bf16_splitk",
    "gemm_fp8w_bf16",
    "gemm_fp8w_bf16_cast",
    "gemm_fp8w_bf16_partials",
    "gemm_mxfp8_partials",
    "gemm_splitk_reduce",
    "gemm_splitk_reduce_cast",
    "gemm_splitk_reduce_two",
    "glp_steer",
    "indexer_mxfp4_quantize",
    "indexer_score",
    "indexer_topk_merge",
    "indexer_topk_segment",
    "metadata_builder",
    "metadata_chunk",
    "metadata_decode",
    "metadata_pages",
    "mhc_collapse",
    "mhc_collapse_column",
    "mhc_finalize",
    "mhc_post",
    "mhc_post_collapse",
    "mhc_project_streams",
    "moe_combine",
    "moe_dispatch",
    "moe_grouped_fp8_fp4_gemm",
    "moe_swiglu_cast_kernel",
    "rope",
    "router_partials",
    "routing_masks",
    "sparse_mla",
}
WARP_SPECIALIZED = {"gemm_bf16", "gemm_bf16_partials", "gemm_bf16_two"}
# test_deepseek_v41_tilelang_pdl's reader that loads before its wait on purpose
EXEMPT = {"pdl_early_reader"}
WAIT = "cudaGridDependencySynchronize()"
TRIGGER = "cudaTriggerProgrammaticLaunchCompletion()"
_GLOBAL = re.compile(
    r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?(\w+)\s*\("
)

_kernels: dict[int, object] = {}


def _matching(text: str, start: int, open_: str, close: str) -> int:
    """The index just past the bracket closing the one before ``start``."""
    depth = 1
    while depth:
        depth += {open_: 1, close: -1}.get(text[start], 0)
        start += 1
    return start


def functions(source: str):
    """``(name, pointer parameter names, body)`` of every ``__global__`` function."""
    for match in _GLOBAL.finditer(source):
        end = _matching(source, match.end(), "(", ")")
        params = source[match.end() : end - 1]
        open_body = source.index("{", end)
        body = source[open_body + 1 : _matching(source, open_body + 1, "{", "}") - 1]
        pointers = [
            re.findall(r"\w+", param)[-1] for param in params.split(",") if "*" in param
        ]
        yield match.group(1), pointers, body


def violations(source: str, warp_specialized: bool) -> list[str]:
    """What is wrong with one compiled module's kernels, if anything."""
    out = []
    for name, pointers, body in functions(source):
        base = re.sub(r"_kernel(_\d+)?$", "", name)
        if base in EXEMPT:
            continue
        wait = body.find(WAIT)
        if base in WARP_SPECIALIZED or (base == "gemm_mxfp8" and warp_specialized):
            if wait >= 0 or TRIGGER in body:
                out.append(f"{name}: warp-specialized, but uses PDL")
            continue
        chain = base in CHAIN or base == "gemm_mxfp8"
        if wait < 0:
            if chain:
                out.append(f"{name}: on the PDL chain, but never waits")
            continue
        before = body[:wait]
        if before.count("{") != before.count("}"):
            out.append(f"{name}: waits inside a block, so some threads may not")
        # A parameter, not a member such as threadIdx.x.
        early = [
            p for p in pointers if re.search(rf"(?<![\w.]){re.escape(p)}\b", before)
        ]
        if early:
            out.append(f"{name}: names {', '.join(early)} before the wait")
        if chain and body.find(TRIGGER) < wait:
            out.append(f"{name}: does not trigger its dependents after the wait")
    return out


def pytest_configure(config):
    import importlib

    # ``tilelang.jit`` is also the decorator tilelang re-exports, so the module is
    # imported by name.
    kernel_module = importlib.import_module("tilelang.jit.kernel")
    launch = kernel_module.JITKernel.__call__

    def recorded(self, *args, **kwargs):
        _kernels.setdefault(id(self), self)
        return launch(self, *args, **kwargs)

    kernel_module.JITKernel.__call__ = recorded


def pytest_sessionfinish(session, exitstatus):
    problems, checked, waiting = [], 0, 0
    for kernel in _kernels.values():
        source = kernel.get_kernel_source()
        configs = {
            str(key): value for key, value in (kernel.pass_configs or {}).items()
        }
        warp_specialized = not any(
            "disable_warp_specialized" in key and value
            for key, value in configs.items()
        )
        checked += 1
        waiting += WAIT in source
        problems.extend(violations(source, warp_specialized))
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    lines = [
        f"PDL sources: {checked} kernels launched, {waiting} with PDL, "
        f"{len(problems)} violations"
    ] + [f"  {problem}" for problem in sorted(set(problems))]
    for line in lines:
        if reporter is not None:
            reporter.write_line(line)
        else:
            print(line)
    if problems:
        session.exitstatus = 1
