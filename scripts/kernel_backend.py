"""Kernel backend policy: which library runs the DS4.1 model compute kernels.

TileLang is the promoted backend (r6): it selects the TileLang compiler,
DeepSeek's TileKernels and the DS4.1 TileLang kernels in the vLLM patch stack,
with sparknet's one-shot collectives. The promoted configurations name it
explicitly. A configuration without ``kernel_backend`` means B12X, as every
configuration recorded before r6 does. NCCL and the B12X checkpoint loader are
outside this policy: both backends keep them.

Until the vLLM series retired its kernel-family switch, TileLang also needed
``VLLM_DS41_KERNEL_BACKEND=tilelang`` (vLLM defaulted to B12X). A series without
the switch records the ``tilelang-only`` capability; its configurations must not
set the variable, which that vLLM no longer reads.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath

import source_labels


# The backend of a configuration that does not name one (every record before r6).
DEFAULT = "b12x"
# Serve arguments and the environment value each backend requires.
BACKENDS = {
    "b12x": {
        "--attention-backend": "B12X",
        "--linear-backend": "b12x",
        "--moe-backend": "b12x",
    },
    "tilelang": {
        "--attention-backend": "TILELANG",
        "--linear-backend": "tilelang",
        "--moe-backend": "tilelang",
    },
}
ENVIRONMENT = "VLLM_DS41_KERNEL_BACKEND"
CACHE_ENVIRONMENT = "TILELANG_CACHE_DIR"
# The TileLang backend needs these optional build sources, this capability in
# the vLLM source manifest record, and an image built from them.
SOURCES = ("tilelang", "tile_kernels")
CAPABILITY = "tilelang-kernels"
# A vLLM series whose DS4.1 model has no kernel-family switch: TileLang is its only
# family and VLLM_DS41_KERNEL_BACKEND is gone.
TILELANG_ONLY = "tilelang-only"
# New DS4.1 series keep one serving contract; older frozen sources may still
# implement these settings and retain their original configuration behavior.
SERVING_CONTRACT = "ds41-serving-contract"
RETIRED_SERVING_ENVIRONMENT = (
    "VLLM_DS41_ENGRAM_OVERLAP",
    "VLLM_DS41_MARKOV_NVFP4",
    "VLLM_DS41_DRAFT_NVFP4_HEAD",
)


def backend(cluster: dict) -> str:
    return cluster.get("kernel_backend", DEFAULT)


def argument(cluster: dict, flag: str) -> str | None:
    args = cluster.get("serve_args", [])
    if flag not in args:
        return None
    i = args.index(flag) + 1
    return args[i] if i < len(args) else None


def problems(cluster: dict) -> list[str]:
    """Serve arguments and environment that contradict the selected backend."""
    selected = backend(cluster)
    if selected not in BACKENDS:
        return [f"unknown kernel_backend {selected!r}; choose {' or '.join(BACKENDS)}"]
    errors = []
    for flag, wanted in BACKENDS[selected].items():
        value = argument(cluster, flag)
        if value != wanted:
            errors.append(f"kernel_backend {selected} requires {flag} {wanted}, got {value}")
    try:
        spec = json.loads(argument(cluster, "--speculative-config") or "{}")
    except ValueError:
        spec = None
    if isinstance(spec, dict) and spec:
        wanted = BACKENDS[selected]["--attention-backend"]
        if spec.get("attention_backend") != wanted:
            errors.append(
                f"kernel_backend {selected} requires speculative attention_backend {wanted}, "
                f"got {spec.get('attention_backend')}"
            )
    env = cluster.get("environment", {})
    # A value that is present must name the selected backend; whether TileLang
    # needs the variable at all is its vLLM series' (source_problems).
    value = env.get(ENVIRONMENT)
    if value is not None and value != selected:
        errors.append(f"kernel_backend {selected} requires {ENVIRONMENT}={selected}")
    if selected == "tilelang":
        cache = PurePosixPath(str(env.get(CACHE_ENVIRONMENT, "")))
        if not cache.is_absolute() or not cache.is_relative_to("/cache") or cache == PurePosixPath("/cache"):
            errors.append(f"kernel_backend tilelang requires {CACHE_ENVIRONMENT} under /cache/")
    return errors


def source_problems(cluster: dict, upstreams: dict, manifest: dict) -> list[str]:
    """The TileLang backend's build sources, vLLM capability and image trees."""
    if backend(cluster) != "tilelang":
        return []
    errors = []
    for name in SOURCES:
        if name not in upstreams.get("sources", {}):
            errors.append(f"kernel_backend tilelang requires source {name} in its upstreams lock")
    vllm = manifest.get("vllm", {})
    capabilities = vllm.get("capabilities", [])
    if SERVING_CONTRACT in capabilities:
        for name in RETIRED_SERVING_ENVIRONMENT:
            if name in cluster.get("environment", {}):
                errors.append(
                    f"{name} is retired in this vLLM series "
                    f"(capability {SERVING_CONTRACT}); drop it from the environment"
                )
    if CAPABILITY not in capabilities:
        errors.append(f"kernel_backend tilelang requires a vLLM source manifest with capability {CAPABILITY}")
    value = cluster.get("environment", {}).get(ENVIRONMENT)
    if TILELANG_ONLY in capabilities:
        if value is not None:
            errors.append(
                f"{ENVIRONMENT} is gone from this vLLM series (capability {TILELANG_ONLY}): "
                "drop it from the environment"
            )
    elif value != "tilelang":
        # The series still has the switch, which defaults to B12X.
        errors.append(f"kernel_backend tilelang requires {ENVIRONMENT}=tilelang")
    labels = cluster.get("container", {}).get("expected_labels", {})
    for name in ("vllm", *SOURCES):
        tree = manifest.get(name, {}).get("expected_tree")
        if tree is None or source_labels.get(labels, f"{name}.tree") != tree:
            errors.append(
                f"kernel_backend tilelang requires container.expected_labels "
                f"{source_labels.key(labels, f'{name}.tree')} to match its source manifest tree"
            )
    if (
        "b12x" not in upstreams.get("sources", {})
        and source_labels.get(labels, "b12x.tree") is not None
    ):
        errors.append(
            f"kernel_backend tilelang without b12x requires omitting "
            f"{source_labels.key(labels, 'b12x.tree')} from container.expected_labels"
        )
    return errors
