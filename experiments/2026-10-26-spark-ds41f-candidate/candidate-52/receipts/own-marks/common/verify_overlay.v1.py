#!/usr/bin/env python3
"""verify_overlay.py ARM: inside a container of one arm, show what the container sees and runs.
- SHA-256 prefix of each of the 13 candidate files at its container path;
- for the overlaid Python modules the PRs change, that the code Python executes is the mounted file's
  (names only the candidate's code has, found in the loaded code objects, not just in the source text);
- the Engram reader: its source digest, which extension _native.load() returns (the image's prebuilt for
  candidate-50's sources, or a runtime build of the mounted sources), and that extension's ABI.
"""

import hashlib
import importlib
import json
import sys
import types

ARM = sys.argv[1]
ROOT = "/opt/spark-ds41f/candidate/vllm/"
FILES = [l.strip() for l in open("/om/changed-files.txt") if l.strip()]


def names(code):
    out = set(code.co_names) | set(code.co_varnames)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out |= names(c)
    return out


def module_names(mod):
    src = open(mod.__file__).read()
    return names(compile(src, mod.__file__, "exec")), mod


res = {"arm": ARM, "files": {}}
for f in FILES:
    res["files"][f] = hashlib.sha256(open(ROOT + f, "rb").read()).hexdigest()[:12]

checks = {}
sched = importlib.import_module("vllm.v1.core.sched.scheduler")
checks["scheduler_loaded_from"] = sched.__file__
checks["scheduler_runs_max_decode_new_tokens"] = hasattr(sched.Scheduler, "_max_decode_new_tokens")
moe = importlib.import_module("vllm.models.deepseek_v4_1.tilelang.moe")
checks["moe_loaded_from"] = moe.__file__
# The code object the module's loader runs (a valid __pycache__ entry, or the mounted source compiled).
code_names = names(moe.__spec__.loader.get_code(moe.__name__))
checks["moe_module_code_has_x_bf16"] = "x_bf16" in code_names
checks["moe_module_code_has_cert_e"] = "cert_e" in code_names
checks["moe_cached_pyc"] = getattr(moe, "__cached__", None)
sched_names = names(sched.__spec__.loader.get_code(sched.__name__))
checks["scheduler_module_code_has_max_decode_new_tokens"] = "_max_decode_new_tokens" in sched_names
for name in ("attention", "sp_prefill", "tilelang.cache_writer", "tilelang.front"):
    m = importlib.import_module("vllm.models.deepseek_v4_1." + name)
    checks[f"{name}_loaded_from"] = m.__file__
from vllm.models.deepseek_v4_1.engram_rows import _native  # noqa: E402

checks["engram_source_digest"] = _native._source_digest()
checks["engram_prebuilt_target"] = str(_native._prebuilt_target())
ext = _native.load()
checks["engram_extension_loaded"] = ext.__file__
checks["engram_extension_sha256"] = hashlib.sha256(open(ext.__file__, "rb").read()).hexdigest()[:16]
checks["engram_extension_abi"] = ext.ABI_VERSION
res["checks"] = checks
print(json.dumps(res, indent=1))
