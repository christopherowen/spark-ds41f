"""What GLP steering costs a captured decode step when no row is marked, and what a
conditional graph node would save.

A step is a CUDA graph of 40 stand-in layers (a few elementwise launches each on a
[T, 4, 5120] BF16 residual), with layers 1-39 followed by GLP steering as one of:

- ``none``: no steering (an engine without a GLP file);
- ``launch``: the glp-3 kernel, one launch a layer that marked rows alone use;
- ``torch_if``: that launch in a PyTorch conditional node (``begin_capture_to_if_node``),
  which sets its own condition with a kernel of its own, every layer;
- ``shared_if``: that launch in a conditional node whose condition one kernel sets
  once a step (any row marked), shared by every layer.

Each arm is timed unmarked and with every row marked, at several row counts, by
replaying its graph in interleaved rounds; the median microseconds a step are
reported, with the change from ``none``. ``shared_if`` and ``torch_if`` must leave
the residual, post and comb bit-identical to ``launch``, marked and unmarked, or the
run fails. One JSON line a measurement, then a ``summary`` line.
"""

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from cuda.bindings import driver, runtime

from vllm.models.deepseek_v4_1.tilelang import glp as tilelang_glp

HC, HIDDEN, LAYERS = 4, 5120, 40
ARMS = ("none", "launch", "torch_if", "shared_if")

ANY_MARKED = r"""
#include <cuda_runtime.h>
extern "C" __global__ void glp_any_marked(cudaGraphConditionalHandle handle,
                                          const int* rows, int n) {
  int mine = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) mine |= rows[i];
  int any = __syncthreads_or(mine);
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, any != 0);
}
"""


def ok(result):
    error, *rest = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA error {error!r}")
    return rest[0] if len(rest) == 1 else rest


def any_marked_kernel():
    major, minor = torch.cuda.get_device_capability()
    with tempfile.TemporaryDirectory() as work:
        source, cubin = Path(work) / "any.cu", Path(work) / "any.cubin"
        source.write_text(ANY_MARKED)
        subprocess.run(
            ["nvcc", "-cubin", f"-arch=sm_{major}{minor}", "-o", str(cubin), str(source)],
            check=True,
        )
        module = ok(driver.cuModuleLoadData(cubin.read_bytes()))
    return module, ok(driver.cuModuleGetFunction(module, b"glp_any_marked"))


def capture_info(stream: int):
    info = ok(runtime.cudaStreamGetCaptureInfo(stream))
    # (status, id, graph, dependencies, edge data, count) since CUDA 13.
    return info[2], info[3], info[-1]


class SharedCondition:
    """One condition a captured graph, set by one kernel a step; any number of
    conditional nodes read it."""

    def __init__(self, function):
        self.function = function
        self.handle = None

    def set(self, rows: torch.Tensor) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        graph, _, _ = capture_info(stream)
        # Every launch of the graph starts the condition at 0 (cudaGraphCondAssignDefault).
        self.handle = ok(runtime.cudaGraphConditionalHandleCreate(graph, 0, 1))
        ok(
            driver.cuLaunchKernel(
                self.function, 1, 1, 1, 256, 1, 1, 0, driver.CUstream(stream),
                ((int(self.handle), rows.data_ptr(), rows.numel()),
                 (ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_int)),
                0,
            )
        )

    def run_if(self, body) -> None:
        stream = torch.cuda.current_stream()
        graph, dependencies, count = capture_info(stream.cuda_stream)
        params = runtime.cudaGraphNodeParams()
        params.type = runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle = self.handle
        params.conditional.type = runtime.cudaGraphConditionalNodeType.cudaGraphCondTypeIf
        params.conditional.size = 1
        node = ok(runtime.cudaGraphAddNode(graph, dependencies, None, count, params))
        child_graph = params.conditional.phGraph_out[0]
        ok(
            runtime.cudaStreamUpdateCaptureDependencies(
                stream.cuda_stream, [node], None, 1,
                runtime.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
            )
        )
        child = torch.cuda.Stream()
        ok(
            runtime.cudaStreamBeginCaptureToGraph(
                child.cuda_stream, child_graph, None, None, 0,
                runtime.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed,
            )
        )
        with torch.cuda.stream(child):
            body()
        ok(runtime.cudaStreamEndCapture(child.cuda_stream))


class Step:
    def __init__(self, arm: str, tokens: int, marked: bool, condition: SharedCondition):
        self.arm, self.condition = arm, condition
        generator = torch.Generator(device="cuda").manual_seed(tokens)
        self.initial = (
            torch.randn((tokens, HC, HIDDEN), device="cuda", generator=generator).bfloat16(),
            torch.rand((tokens, HC), device="cuda", generator=generator),
            torch.rand((tokens, HC, HC), device="cuda", generator=generator),
        )
        self.residual, self.post, self.comb = (t.clone() for t in self.initial)
        self.parts = torch.randn(
            (1, tokens, HIDDEN), device="cuda", generator=generator
        ).bfloat16()
        self.scratch = torch.zeros((tokens, HC * HIDDEN), device="cuda", dtype=torch.bfloat16)
        directions = torch.randn((LAYERS, HIDDEN), device="cuda", generator=generator)
        self.directions = directions / directions.norm(dim=1, keepdim=True)
        self.alpha = torch.tensor([0.75], device="cuda")
        self.rows = torch.full((tokens,), int(marked), device="cuda", dtype=torch.int32)
        self.kernel = tilelang_glp._kernel(HC, HIDDEN, 1)
        self.graph = torch.cuda.CUDAGraph()

    def steer(self, layer: int) -> None:
        self.kernel(
            self.parts, self.residual, self.post, self.comb, self.rows,
            self.directions[layer], self.alpha,
        )

    def forward(self) -> None:
        pred = None
        if self.arm == "shared_if":
            self.condition.set(self.rows)
        if self.arm == "torch_if":
            pred = self.rows.bool().any()
        for layer in range(LAYERS):
            for _ in range(4):
                self.scratch.add_(1)
            if layer == 0 or self.arm == "none":
                continue
            if self.arm == "launch":
                self.steer(layer)
            elif self.arm == "torch_if":
                graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
                graph.begin_capture_to_if_node(pred)
                try:
                    self.steer(layer)
                finally:
                    graph.end_capture_to_conditional_node()
            else:
                self.condition.run_if(lambda layer=layer: self.steer(layer))

    def capture(self) -> None:
        self.steer(1)  # compile outside the capture
        self.reset()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(self.graph, stream=stream):
            self.forward()
        torch.cuda.synchronize()

    def reset(self) -> None:
        for live, initial in zip((self.residual, self.post, self.comb), self.initial):
            live.copy_(initial)
        torch.cuda.synchronize()

    def once(self) -> tuple[torch.Tensor, ...]:
        self.reset()
        self.graph.replay()
        torch.cuda.synchronize()
        return tuple(t.clone() for t in (self.residual, self.post, self.comb))

    def time_us(self, replays: int) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            self.graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / replays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", default="2,16,96")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--replays", type=int, default=300)
    args = parser.parse_args()
    torch.zeros(1, device="cuda")  # the primary context, current for the driver calls
    _, function = any_marked_kernel()
    failed = False
    summary = {}
    for tokens in (int(t) for t in args.tokens.split(",")):
        for marked in (False, True):
            steps, errors = {}, {}
            for arm in ARMS:
                try:
                    step = Step(arm, tokens, marked, SharedCondition(function))
                    step.capture()
                    steps[arm] = step
                except Exception as error:  # an arm that cannot capture is reported
                    errors[arm] = f"{type(error).__name__}: {error}"
                    torch.cuda.synchronize()
            reference = steps["launch"].once() if "launch" in steps else None
            identical = {}
            for arm in ("torch_if", "shared_if"):
                if arm in steps and reference is not None:
                    identical[arm] = all(
                        torch.equal(a, b) for a, b in zip(steps[arm].once(), reference)
                    )
                    failed |= not identical[arm]
            if not marked and reference is not None:
                untouched = all(
                    torch.equal(a, b) for a, b in zip(reference, steps["launch"].initial)
                )
                identical["launch_leaves_unmarked"] = untouched
                failed |= not untouched
            samples = {arm: [] for arm in steps}
            for step in steps.values():
                step.time_us(50)
            for _ in range(args.rounds):
                for arm, step in steps.items():
                    samples[arm].append(step.time_us(args.replays))
            medians = {arm: statistics.median(values) for arm, values in samples.items()}
            base = medians.get("none")
            record = {
                "tokens": tokens,
                "marked": marked,
                "step_us": {arm: round(value, 2) for arm, value in medians.items()},
                "over_none_us": {
                    arm: round(value - base, 2) for arm, value in medians.items()
                } if base is not None else {},
                "spread_us": {
                    arm: round(max(values) - min(values), 2) for arm, values in samples.items()
                },
                "identical_to_launch": identical,
                "errors": errors,
            }
            failed |= "none" in errors or "launch" in errors
            summary[f"{tokens}{'-marked' if marked else ''}"] = record["over_none_us"]
            print(json.dumps(record), flush=True)
    print(json.dumps({"summary": summary, "failed": failed}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
