# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the b12x project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and load the direct loader's C99 helper (``_storage.c``) with the system C
compiler, without linking the PyTorch C++ ABI.

It needs the Python and CUDA headers (``CUDA_HOME``, default /usr/local/cuda), the CUDA
runtime and driver libraries, and liburing with pkg-config. It is built on first use
into ``$XDG_CACHE_HOME/vllm/direct_io/<digest>``, where the digest covers the C sources
and the build identity, so a changed source rebuilds.

Ported from local-inference-lab/b12x b12x/loader/_native.py at 236ddff04a5f
(Apache-2.0). Changed: liburing is required, the cache directory and module name are
vLLM's, and the build identity no longer names the PyTorch version (the helper does
not link it).
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
_SOURCE = Path(__file__).with_name("_storage.c")
_NAME = "_direct_io_storage"
_DEFAULT_PREBUILT_ROOT = Path("/opt/spark-ds41f/native")
ABI_VERSION = 1


def _pkg_config(*args: str) -> str:
    return subprocess.check_output(["pkg-config", *args, "liburing"], text=True).strip()


def _source_digest() -> str:
    sources = sorted(_SOURCE.parent.glob("*.[ch]"))
    return hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()[:24]


def _prebuilt_target() -> Path | None:
    if "CC" in os.environ:
        return None
    default_cuda = Path("/usr/local/cuda")
    cuda_env = os.environ.get("CUDA_HOME")
    if cuda_env and Path(cuda_env).resolve() != default_cuda.resolve():
        return None
    cc_path = shutil.which("cc")
    if cc_path is not None and Path(cc_path).resolve() not in {
        Path(p).resolve() for p in ("/usr/bin/cc", "/bin/cc") if Path(p).exists()
    }:
        return None
    root = Path(os.environ.get("VLLM_NATIVE_CACHE_DIR", _DEFAULT_PREBUILT_ROOT))
    target = (
        root
        / "direct_io"
        / _source_digest()
        / (_NAME + sysconfig.get_config_var("EXT_SUFFIX"))
    )
    return target if target.is_file() else None


def prebuild(root: Path | str | None = None) -> Path:
    """Compile the helper and install it under ``root / 'direct_io' / <source_digest>``."""
    built = _build(ignore_prebuilt=True)
    dest_dir = Path(
        root or os.environ.get("VLLM_NATIVE_CACHE_DIR", _DEFAULT_PREBUILT_ROOT)
    ) / "direct_io" / _source_digest()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / built.name
    if built.resolve() != dest.resolve():
        shutil.copy2(built, dest)
    return dest


def _build(*, ignore_prebuilt: bool = False) -> Path:
    if not ignore_prebuilt:
        prebuilt = _prebuilt_target()
        if prebuilt is not None:
            return prebuilt
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError("the direct loader needs a C99 compiler (CC or cc)")
    compiler[0] = shutil.which(compiler[0])
    if (
        not shutil.which("pkg-config")
        or subprocess.run(
            ["pkg-config", "--exists", "liburing"], capture_output=True
        ).returncode
    ):
        raise RuntimeError("the direct loader needs liburing and pkg-config")
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    includes = [Path(sysconfig.get_path("include")), cuda / "include"]
    headers = [
        includes[0] / "Python.h",
        includes[1] / "cuda_runtime_api.h",
        includes[1] / "driver_types.h",
    ]
    for path in headers:
        if not path.is_file():
            raise RuntimeError(f"the direct loader's build needs {path}")
    flags = [
        "-O2",
        "-std=c99",
        "-shared",
        "-fPIC",
        "-pthread",
        "-D_FILE_OFFSET_BITS=64",
        "-Wall",
        "-Wextra",
        "-Werror",
        *shlex.split(_pkg_config("--cflags")),
    ]
    ldflags = shlex.split(_pkg_config("--libs"))
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
        "cuda": str(cuda),
        "headers": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in headers
        },
    }
    sources = sorted(_SOURCE.parent.glob("*.[ch]"))
    digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in sources)
        + json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:24]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = root / "vllm" / "direct_io" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (_NAME + sysconfig.get_config_var("EXT_SUFFIX"))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / target.name
            command = [
                *compiler,
                *flags,
                *(f"-I{path}" for path in includes),
                str(_SOURCE),
                f"-L{cuda / 'lib64'}",
                f"-Wl,-rpath,{cuda / 'lib64'}",
                "-lcudart",
                "-lcuda",
                *ldflags,
                "-o",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(
                    "the direct loader's helper did not build:\n" + result.stderr
                )
            os.replace(output, target)
    return target


@functools.cache
def load():
    with _LOCK:
        spec = importlib.util.spec_from_file_location(_NAME, _build())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.ABI_VERSION != ABI_VERSION:
            raise RuntimeError("the direct loader helper's ABI does not match")
        return module
