# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and load the routed MoE expert slot reader (``_reader.c``).

The extension requires only the Python headers and liburing. It compiles on
first use into ``$XDG_CACHE_HOME/vllm/expert_slots/<digest>`` so source or
toolchain updates rebuild automatically.
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
_NAME = "_expert_slots_reader"
_DEFAULT_PREBUILT_ROOT = Path("/opt/spark-ds41f/native")
ABI_VERSION = 1


def _pkg_config(*args: str) -> str:
    return subprocess.check_output(
        ["pkg-config", *args, "liburing"], text=True
    ).strip()


def _source_digest() -> str:
    sources = sorted(_SOURCE.parent.glob("*.[ch]"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:24]


def _prebuilt_target() -> Path | None:
    if "CC" in os.environ:
        return None
    cc_path = shutil.which("cc")
    if cc_path is not None and Path(cc_path).resolve() not in {
        Path(p).resolve() for p in ("/usr/bin/cc", "/bin/cc") if Path(p).exists()
    }:
        return None
    root = Path(os.environ.get("VLLM_NATIVE_CACHE_DIR", _DEFAULT_PREBUILT_ROOT))
    target = (
        root
        / "expert_slots"
        / _source_digest()
        / (_NAME + sysconfig.get_config_var("EXT_SUFFIX"))
    )
    return target if target.is_file() else None


def prebuild(root: Path | str | None = None) -> Path:
    """Compile the reader and install it under ``root / 'expert_slots' / <source_digest>``."""
    built = _build(ignore_prebuilt=True)
    dest_dir = (
        Path(root or os.environ.get("VLLM_NATIVE_CACHE_DIR", _DEFAULT_PREBUILT_ROOT))
        / "expert_slots"
        / _source_digest()
    )
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
        raise RuntimeError("the expert slot reader needs a C compiler (CC or cc)")
    compiler[0] = shutil.which(compiler[0])
    if (
        not shutil.which("pkg-config")
        or subprocess.run(
            ["pkg-config", "--exists", "liburing"], capture_output=True
        ).returncode
    ):
        raise RuntimeError("the expert slot reader needs liburing and pkg-config")
    include = Path(sysconfig.get_path("include"))
    if not (include / "Python.h").is_file():
        raise RuntimeError(
            f"the expert slot reader needs the Python headers in {include}"
        )
    cflags = shlex.split(_pkg_config("--cflags"))
    ldflags = shlex.split(_pkg_config("--libs"))
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
        f"-I{include}",
        *cflags,
    ]
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
        b"".join(p.read_bytes() for p in sources)
        + json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:24]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = root / "vllm" / "expert_slots" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (_NAME + sysconfig.get_config_var("EXT_SUFFIX"))
    if target.is_file():
        return target
    with _LOCK:
        if target.is_file():
            return target
        fd, tmp = tempfile.mkstemp(
            dir=cache, prefix=_NAME + ".", suffix=target.suffix
        )
        os.close(fd)
        try:
            subprocess.run(
                [*compiler, *flags, str(_SOURCE), "-o", tmp, *ldflags],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(tmp, target)
        except subprocess.CalledProcessError as exc:
            Path(tmp).unlink(missing_ok=True)
            raise RuntimeError(
                f"failed to build {_SOURCE.name}:\n{exc.stderr}"
            ) from exc
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
    return target


@functools.cache
def load():
    path = _build()
    spec = importlib.util.spec_from_file_location(_NAME, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "ABI_VERSION", None) != ABI_VERSION:
        raise RuntimeError(
            f"{path} has ABI_VERSION={getattr(module, 'ABI_VERSION', None)}, "
            f"expected {ABI_VERSION}"
        )
    return module
