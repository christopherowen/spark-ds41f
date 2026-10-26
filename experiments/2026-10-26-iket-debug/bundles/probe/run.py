"""Can IKET run on GB10? Where its profiler comes from, and one profiled run.

1. The device, its per-block shared-memory reservation, and the IKET files the
   image's CuTe DSL wheels install (console scripts, the ``iket`` package, native
   libraries and their ELF machine).
2. ``run-iket`` help, and the architecture checks in the IKET sources.
3. probe_kernel.py run directly, then under ``run-iket profile``; the traces it
   writes are summarized (schema, launches, ranges, markers, payloads).

Traces and sources go under /c/iket-probe/<stamp> (the node's compile cache, the
one writable mount). Every step prints a ``=== step`` line; a failed step is
reported and the next one runs.
"""

import ctypes
import glob
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
ROOT = Path("/c/iket-probe") / STAMP
FAILED = []


def step(name):
    print(f"\n=== {name}", flush=True)


def run(argv, timeout=900, env=None):
    print("$ " + " ".join(argv), flush=True)
    try:
        process = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"!! {error}", flush=True)
        return None
    output = (process.stdout + process.stderr).splitlines()
    for line in output[-120:]:
        print("  " + line)
    print(f"  (exit {process.returncode}, {len(output)} lines)", flush=True)
    return process


def device():
    import torch

    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"name": props.name, "capability": f"{props.major}.{props.minor}",
                      "sms": props.multi_processor_count, "torch_cuda": torch.version.cuda}))
    libs = sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so*")) + sorted(
        glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*")))
    if libs:
        cudart = ctypes.CDLL(libs[0])
        value = ctypes.c_int()
        # cudaDevAttrReservedSharedMemoryPerBlock = 111
        status = cudart.cudaDeviceGetAttribute(ctypes.byref(value), 111, 0)
        print(json.dumps({"cudart": libs[0], "status": status, "reserved_shared_per_block": value.value}))
    driver = Path("/proc/driver/nvidia/version")
    if driver.exists():
        print(driver.read_text().splitlines()[0])


def elf_machine(path):
    with open(path, "rb") as handle:
        header = handle.read(20)
    if header[:4] != b"\x7fELF":
        return "not ELF"
    machine = int.from_bytes(header[18:20], "little")
    return {183: "aarch64", 62: "x86_64"}.get(machine, str(machine))


def packages():
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"] or ""):
        name = dist.metadata["Name"] or ""
        if not name.lower().startswith(("nvidia-cutlass-dsl", "iket")):
            continue
        files = [f for f in (dist.files or []) if "iket" in str(f).lower()]
        print(f"{name} {dist.version}: {len(dist.files or [])} files, {len(files)} named iket")
        for entry in dist.entry_points:
            if "iket" in entry.name.lower() or "iket" in entry.value.lower():
                print(f"  entry point {entry.group}: {entry.name} = {entry.value}")
        for f in files[:150]:
            path = Path(dist.locate_file(f))
            size = path.stat().st_size if path.exists() else -1
            kind = elf_machine(path) if path.suffix == ".so" or ".so." in path.name else ""
            print(f"  {f} {size} {kind}")
    for module in ("iket", "cutlass.iket", "cutlass.base_dsl.iket", "cutlass.cute.iket"):
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None
        print(f"find_spec({module}) -> {spec.origin if spec else None}")
    print("which run-iket:", shutil.which("run-iket"))


def arch_checks():
    roots = set()
    for module in ("iket",):
        spec = importlib.util.find_spec(module)
        if spec and spec.submodule_search_locations:
            roots.update(spec.submodule_search_locations)
    script = shutil.which("run-iket")
    if script:
        print(Path(script).read_text()[:2000])
    pattern = re.compile(r"sm_?1[0-2][0-9]|SM_?1[0-2][0-9]|compute_capability|capability|cc_major|arch_|get_arch|supported", re.I)
    for root in sorted(roots):
        for path in sorted(Path(root).rglob("*.py")):
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if pattern.search(line):
                    print(f"{path.relative_to(root)}:{number}: {line.strip()[:160]}")


def summarize(directory):
    files = sorted(Path(directory).rglob("*"))
    for path in files:
        if path.is_file():
            print(f"  {path.relative_to(directory)} {path.stat().st_size}")
    traces = sorted(Path(directory).rglob("*.trace.json"), key=lambda p: p.stat().st_size, reverse=True)
    if not traces:
        print("  no *.trace.json")
        return False
    data = json.loads(traces[0].read_text())
    names = data.get("stringTable", [])
    print("  top-level keys:", sorted(data))
    launches = data.get("launches", [])
    print(f"  launches: {len(launches)}; strings: {len(names)}")
    for launch in launches[:6]:
        print("  launch keys:", sorted(launch))
        print("  launch scalars:", json.dumps({k: v for k, v in launch.items() if not isinstance(v, (list, dict))})[:600])
        for key in ("ranges", "markers", "warpLifetimes"):
            items = launch.get(key, [])
            print(f"   {key}: {len(items)}; first: {json.dumps(items[:2])[:600]}")
    return True


def main():
    step("device")
    try:
        device()
    except Exception as error:  # noqa: BLE001
        FAILED.append("device")
        print(f"!! {error!r}")

    step("packages")
    packages()

    step("run-iket help")
    help_ok = run(["run-iket", "--help"], timeout=120)
    run(["run-iket", "profile", "--help"], timeout=120)
    run([sys.executable, "-m", "iket.cli.main", "--help"], timeout=120)
    if help_ok is None or help_ok.returncode:
        FAILED.append("run-iket --help")

    step("architecture checks in the IKET sources")
    try:
        arch_checks()
    except Exception as error:  # noqa: BLE001
        print(f"!! {error!r}")

    step("probe kernels, run directly")
    direct = run([sys.executable, "/b/probe_kernel.py", str(ROOT / "direct")])
    if direct is None or direct.returncode:
        FAILED.append("direct")

    step("probe kernels under run-iket profile")
    profiled = ROOT / "profile"
    process = run(["run-iket", "--output-dir", str(profiled), "profile", "--postprocess", "all", "--",
                   sys.executable, "/b/probe_kernel.py", str(ROOT / "profile-src")], timeout=1800)
    if process is None or process.returncode:
        FAILED.append("profile")
    if profiled.exists() and not summarize(profiled):
        FAILED.append("profile trace")

    step("verdict")
    print(json.dumps({"root": str(ROOT), "failed": FAILED}))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
