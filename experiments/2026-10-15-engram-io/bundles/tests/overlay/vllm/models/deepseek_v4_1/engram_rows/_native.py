# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and load the Engram disk reader (``_reader.c``) with the system C compiler.

The extension needs only the Python headers and liburing. It is built on first use
into ``$XDG_CACHE_HOME/vllm/engram_rows/<digest>``, where the digest covers the C
sources and the build identity, so a changed source rebuilds.
"""

import functools
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path

_LOCK = threading.Lock()
_SOURCE = Path(__file__).with_name("_reader.c")
_NAME = "_engram_rows_reader"
ABI_VERSION = 1


def _pkg_config(*args: str) -> str:
    return subprocess.check_output(["pkg-config", *args, "liburing"], text=True).strip()


def _build() -> Path:
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError("the Engram disk reader needs a C compiler (CC or cc)")
    compiler[0] = shutil.which(compiler[0])
    if not shutil.which("pkg-config") or subprocess.run(
        ["pkg-config", "--exists", "liburing"], capture_output=True
    ).returncode:
        raise RuntimeError("the Engram disk reader needs liburing and pkg-config")
    include = Path(sysconfig.get_path("include"))
    if not (include / "Python.h").is_file():
        raise RuntimeError(f"the Engram disk reader needs the Python headers in {include}")
    cflags = shlex.split(_pkg_config("--cflags"))
    ldflags = shlex.split(_pkg_config("--libs"))
    flags = ["-O2", "-std=c99", "-shared", "-fPIC", "-pthread", "-D_FILE_OFFSET_BITS=64",
             "-Wall", "-Wextra", "-Werror", f"-I{include}", *cflags]
    identity = {
        "abi": ABI_VERSION,
        "python": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "compiler": compiler,
        "version": subprocess.check_output([*compiler, "--version"], text=True),
        "flags": flags,
        "ldflags": ldflags,
        "liburing": _pkg_config("--modversion"),
    }
    sources = sorted(_SOURCE.parent.glob("*.[ch]"))
    digest = hashlib.sha256(
        b"".join(p.read_bytes() for p in sources) + json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:24]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = root / "vllm" / "engram_rows" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (_NAME + sysconfig.get_config_var("EXT_SUFFIX"))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / target.name
            result = subprocess.run([*compiler, *flags, str(_SOURCE), *ldflags, "-o", str(output)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("the Engram disk reader did not build:\n" + result.stderr)
            os.replace(output, target)
    return target


@functools.cache
def load():
    with _LOCK:
        spec = importlib.util.spec_from_file_location(_NAME, _build())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.ABI_VERSION != ABI_VERSION:
            raise RuntimeError("the Engram disk reader's ABI does not match its loader")
        return module
