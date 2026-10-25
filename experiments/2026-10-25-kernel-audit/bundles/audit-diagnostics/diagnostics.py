#!/usr/bin/env python3
"""The kernel audit's diagnostics on one node (dgx4), against vLLM 26382f5cf's kernels
(the image's own): why fix 1's live-page tables were slower at a full 1M context, and
why loading mHC's fn tiles ahead (fix 3, reverted) was slower everywhere.

For each variant of make_diagnostics.py: its outputs against the 26382f5cf form (the
page tables on every live column; the projection's every output bit for bit), then its
time a call (CUDA graphs of 100 calls, the forms interleaved, the median of five
samples), and each form's generated CUDA source (written under the node's compile
cache, /c/kernel-audit-diagnostics) with counts of its vector accesses and loops. Exits 1 if a variant's
outputs differ.
"""
import json
import statistics
import sys
import traceback

import torch

sys.path.insert(0, "/b")
import metadata_variants  # noqa: E402
import mhc_variants  # noqa: E402

DEVICE = "cuda"
CALLS = 100
SAMPLES = 5
problems = []


def graph(function):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            function()
    return capture


def elapsed(capture) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def time_forms(calls: dict) -> dict:
    """Median microseconds a call of each form, the forms interleaved."""
    captures = {name: graph(call) for name, call in calls.items()}
    times = {name: [] for name in calls}
    names = list(calls)
    for sample in range(SAMPLES):
        for name in names[sample % len(names):] + names[: sample % len(names)]:
            times[name].append(elapsed(captures[name]))
    return {name: round(statistics.median(t), 3) for name, t in times.items()}


SOURCES = "/c/kernel-audit-diagnostics"  # the node's compile-cache mount


def dump(kernel, name: str) -> str:
    """Write a kernel's CUDA source under SOURCES; a line of counts that tell scalar
    from vector memory accesses."""
    import os

    try:
        text = kernel.get_kernel_source()
    except Exception as error:  # noqa: BLE001
        return f"{name}: no source ({error})"
    os.makedirs(SOURCES, exist_ok=True)
    with open(f"{SOURCES}/{name}.cu", "w") as handle:
        handle.write(text)
    counts = {key: text.count(key) for key in ("float4", "int4", "uint4", "#pragma unroll",
                                               "for (", "__syncthreads", "if (")}
    return f"{name}: {len(text.splitlines())} lines, {counts}, {SOURCES}/{name}.cu"


# --- the page tables (fix 1) ------------------------------------------------------------

BLOCK_TOKENS, MAIN_WIDTH, WIDTH, PAGE = 256, 4096, 128, 256
PAGE_SHAPES = {"c1 4K": (1, 6, 4096), "c1 1M": (1, 6, 1 << 20),
               "c16 64K": (16, 6, 65536), "c16 1M": (16, 6, 1 << 20),
               "prefill 16K": (1, 16384, 16384), "prefill 64K": (1, 16384, 65536)}


def page_step(requests, rows_each, context, main_page):
    gen = torch.Generator(device=DEVICE).manual_seed(context + rows_each)
    rows = requests * rows_each
    row = torch.arange(rows, device=DEVICE)
    req = (row // rows_each).to(torch.int32)
    pos = (context - rows_each + row % rows_each).to(torch.int64)
    visible = ((pos + 1) // (BLOCK_TOKENS // main_page)).to(torch.int32)
    starts = torch.arange(0, rows + 1, rows_each, dtype=torch.int32, device=DEVICE)
    table = torch.randint(1, 1 << 20, (16, MAIN_WIDTH), generator=gen, device=DEVICE,
                          dtype=torch.int32)
    return rows, pos, req, visible, starts, pos[::rows_each].contiguous(), table


def pages() -> dict:
    results = {}
    main_page = 256
    names = ("old", "dyn", "static_var", "static_expr", "select", "two_loop", "grid8",
             "skip1", "skip4", "skip8")
    kernels = {}
    for name in names:
        try:
            kernels[name] = getattr(metadata_variants, f"decode_metadata_{name}")(
                PAGE, 128, WIDTH, False, False, MAIN_WIDTH, main_page)
        except Exception:  # noqa: BLE001
            problems.append(f"decode_metadata_{name} did not compile")
            print(f"decode_metadata_{name}: did not compile\n{traceback.format_exc()}")
    for shape, (requests, rows_each, context) in PAGE_SHAPES.items():
        rows, pos, req, visible, starts, request_positions, table = page_step(
            requests, rows_each, context, main_page)
        outs, calls = {}, {}
        for name, kernel in kernels.items():
            out = (torch.full((rows, WIDTH), 7, dtype=torch.int32, device=DEVICE),
                   torch.full((rows,), 7, dtype=torch.int32, device=DEVICE),
                   torch.full((rows,), 7, dtype=torch.int32, device=DEVICE),
                   torch.full((rows, MAIN_WIDTH), 7, dtype=torch.int32, device=DEVICE))
            outs[name] = out

            def call(kernel=kernel, out=out):
                kernel(pos, req, table, out[0], out[1], out[2], visible, starts,
                       request_positions, request_positions, req, table, out[3], 0,
                       MAIN_WIDTH)

            call()
            calls[name] = call
        torch.cuda.synchronize()
        live = (visible.long() + main_page - 1) // main_page
        mask = torch.arange(MAIN_WIDTH, device=DEVICE)[None, :] < live[:, None]
        for name in calls:
            same = all(torch.equal(a, b) for a, b in zip(outs[name][:3], outs["old"][:3]))
            same &= torch.equal(outs[name][3][mask], outs["old"][3][mask])
            if not same:
                problems.append(f"decode_metadata_{name} {shape}: live pages differ")
        results[shape] = time_forms(calls)
        print(json.dumps({"page tables": shape, "us": results[shape]}), flush=True)
    for name, kernel in kernels.items():
        print(dump(kernel, f"decode_metadata_{name}"))
    return results


# --- the mHC entry projection (fix 3) -----------------------------------------------------

HIDDEN, MIXES = 5120, 24
PROJECT_SHAPES = {"decode c1": 6, "decode c8": 48, "decode c16": 96, "prefill 8K": 8192}


def project() -> dict:
    results = {}
    for mode in ("post_pre", "pre"):
        kernels = {}
        for name in "ABCDE":
            try:
                kernels[name] = getattr(mhc_variants, f"project_streams_{name}")(
                    HIDDEN, mode, block_M=16, world=1)
            except Exception:  # noqa: BLE001
                problems.append(f"project_streams_{name} {mode} did not compile")
                print(f"project_streams_{name} {mode}: did not compile\n"
                      f"{traceback.format_exc()}")
        for shape, tokens in PROJECT_SHAPES.items():
            g = torch.Generator(device=DEVICE).manual_seed(tokens)

            def randn(*shape, scale=1.0):
                return torch.randn(*shape, generator=g, device=DEVICE) * scale

            x = randn(1, tokens, HIDDEN).bfloat16()
            residual = randn(tokens, 4 * HIDDEN).bfloat16()
            post = 2 * torch.sigmoid(randn(tokens, 4))
            comb = torch.softmax(randn(tokens, 4, 4), dim=-1)
            incoming = torch.softmax(randn(tokens, 4), dim=-1) + 1e-6
            fn = randn(MIXES, 4 * HIDDEN, scale=0.02)
            splits = HIDDEN // 128
            outs, calls = {}, {}
            for name, kernel in kernels.items():
                out = (torch.empty(tokens, 4 * HIDDEN, dtype=torch.bfloat16, device=DEVICE),
                       torch.empty(splits, tokens, 1, MIXES, device=DEVICE),
                       torch.empty(splits, tokens, 1, device=DEVICE),
                       torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE),
                       torch.empty(splits, tokens, device=DEVICE))
                outs[name] = out

                def call(kernel=kernel, out=out):
                    kernel(x, residual, post, comb, incoming, fn, *out)

                call()
                calls[name] = call
            torch.cuda.synchronize()
            for name in calls:
                written = outs[name][1:] if mode == "pre" else outs[name]
                base = outs["A"][1:] if mode == "pre" else outs["A"]
                for a, b in zip(written, base):
                    bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
                    if not torch.equal(a.view(bits), b.view(bits)):
                        problems.append(f"project_streams_{name} {mode} {shape}: bits differ")
                        break
            results[f"{mode} {shape}"] = time_forms(calls)
            print(json.dumps({"projection": f"{mode} {shape}",
                              "us": results[f"{mode} {shape}"]}), flush=True)
        for name, kernel in kernels.items():
            print(dump(kernel, f"project_streams_{name}_{mode}"))
    return results


def main() -> int:
    summary = {"page tables": pages(), "projection": project(), "problems": problems}
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
