#!/usr/bin/env python3
"""Offline, isolated ARM64 NVBit tool build; never launches a GPU program.

Inputs are reviewed upstream archives with pinned hashes. No upstream installer
or Makefile is executed. Extracted sources retain upstream license files. Build
flags follow NVBit 1.8.1: C++17 and no maxrregcount on CUDA >=12.3. Tools remain
experimental; a successful build does not establish instrumentation coverage.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile


INPUTS = {
    "nvbit-arm64.tar.bz2": "d1701e1fdcb1a9661b561accf8da817d911f6123977eacfd903f704f29b0ab0f",
    "nixnan-main.tar.gz": "7f67ecab09550a9b94229f2b1dfdd4f16d37e38ab88751167a5cc22736b1282d",
    "json-3.12.0.hpp": "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63",
    "gpu-fpx.tar.gz": "7b18be51eb7b3d362de44f0e76c12b889cf12c3318bbbb0575c80a3bb12c18cd",
}


def run(command: list[str], cwd: Path) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, timeout=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool", choices=("nixnan", "detector", "analyzer", "instr_count", "instr_count_cuda_graph"), required=True)
    args = parser.parse_args()
    inputs, output = args.inputs.resolve(), args.output.resolve()
    for name, expected in INPUTS.items():
        actual = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"input checksum mismatch: {name}: {actual}")
    # Refuse reuse: failed and successful builds keep separate immutable receipts.
    output.mkdir(parents=True, exist_ok=False)
    # Retain original GPU-FPX LICENSE and NOTICE beside its maintained copy.
    for name in ("nvbit-arm64.tar.bz2", "nixnan-main.tar.gz", "gpu-fpx.tar.gz"):
        with tarfile.open(inputs / name) as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise SystemExit(f"unsupported archive member: {member.name}")
                if not (output / member.name).resolve().is_relative_to(output):
                    raise SystemExit(f"unsafe archive path: {member.name}")
            archive.extractall(output)
    nvbit = output / "nvbit_release_aarch64"
    nixnan = output / "parfloat-nixnan-55a6587"
    core = nvbit / "core"
    if args.tool == "nixnan":
        source = nixnan / "src"
        (source / "nlohmann").mkdir(exist_ok=True)
        shutil.copyfile(inputs / "json-3.12.0.hpp", source / "nlohmann/json.hpp")
        data = (source / "instructions.json").read_bytes()
        (source / "instructions.h").write_text(
            "unsigned char instructions_json[] = {" +
            ",".join(str(byte) for byte in data) +
            "};\nunsigned int instructions_json_len = " + str(len(data)) + ";\n"
        )
        injected = {"inject_funs.cu", "memcheck.cu", "fp-fun.cu"}
    elif args.tool in ("instr_count", "instr_count_cuda_graph"):
        source = nvbit / "tools" / args.tool
        injected = {"inject_funcs.cu"}
    else:
        # GPU-FPX as maintained in the exact Nixnan main snapshot, not the old
        # LLNL Makefile that downloads an x86-only NVBit 1.5.5 build.
        source = nixnan / "nvbit_release/tools/GPU-FPX" / args.tool
        injected = {"inject_funcs.cu"}
    nvcc = shutil.which("nvcc")
    cxx = shutil.which("g++")
    if not nvcc or not cxx:
        raise SystemExit("nvcc and g++ are required")
    run([nvcc, "--version"], output)
    run([cxx, "--version"], output)
    includes = ["-I" + str(core), "-I" + str(source), "-I" + str(source.parent)]
    objects = []
    for file in sorted([*source.glob("*.cu"), *source.glob("*.cc")]):
        obj = output / (file.name + ".o")
        if file.suffix == ".cu":
            command = [nvcc, "-ccbin", cxx, "-std=c++17", "-D_FORCE_INLINES",
                       "-arch=sm_121", "-O3", "-Xcompiler=-Wall,-fPIC", *includes]
            if file.name in injected:
                command += ["-Xptxas", "-astoolspatch", "--keep-device-functions"]
            else:
                command += ["-dc", "-Xptxas", "-cloning=no"]
        else:
            command = [cxx, "-std=c++17", "-O3", "-Wall", "-fPIC", *includes]
        run([*command, "-c", str(file), "-o", str(obj)], output)
        objects.append(str(obj))
    run([nvcc, "-ccbin", cxx, "-std=c++17", "-arch=sm_121", "-O3",
         *objects, "-L" + str(core), "-lnvbit", "-lcuda", "-lcudart_static",
         "-shared", "-o", str(output / (args.tool + ".so"))], output)
    print("Built " + str(output / (args.tool + ".so")), flush=True)


if __name__ == "__main__":
    main()
