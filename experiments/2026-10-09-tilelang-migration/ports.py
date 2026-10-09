"""The ports of the TileLang migration: for each, the vLLM files it changes on the
migration branch, the environment its arm sets, and any package it mounts.

An arm is the r6c TP4 recipe plus exactly one port. Ports are measured against the
control, never against each other.
"""

# vLLM migration branch (~/projects/vllm-ds41-tilelang-migration, from r6c's 125c404e4).
VLLM_BRANCH = "tilelang-migration"

PORTS = {
    # 1. The CuTe DSL L2 weight prefetch -> TileLang (same work split and PTX).
    "prefetch": {
        "files": (
            "models/glm5next/nvidia/l2_prefetch.py",
            "models/glm5next/nvidia/l2_prefetch_tilelang.py",
        ),
        "environment": {"VLLM_L2_PREFETCH_KERNELS": "tilelang"},
    },
}

# sparknet's one-shot collectives: the image pins 0.2.0, which predates the TileLang
# family, so both arms mount sparknet main (b61660f) and differ only in the family.
SPARKNET_MAIN = {
    "revision": "b61660f",
    "mount": ["{home}/sparknet-main/sparknet", "/usr/local/lib/python3.12/dist-packages/sparknet", "ro"],
}
