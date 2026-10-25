#!/usr/bin/env python3
"""Write the direct loader's single-node kernel-lab bundles (scripts/lab.py kernel-local).

usage: make_kernel_bundles.py VLLM_REPO [--image spark-ds41f:19f2c20ed4d6-next-6]

Both bundles mount bundles/direct (make_arms.py, the branch's package files) over the
image's vLLM tree, so they run the branch's loader in candidate 6's image without a build:

- kernel/gpu-tests: pytest over the branch's three test files, copied here at the bundle's
  revision (test_direct_loader_cuda.py needs the GPU; the other two run anywhere). Its
  O_DIRECT checkpoints go under /c, the node's compile cache, a real filesystem.
- kernel/rank-load: rank-load/rank_compare.py, which loads one TP4 rank's share of the
  served checkpoint (DeepSeek-V4.1-Flash-1017, the hub mounted at /hub as serving mounts
  it) with the direct loader, then with B12X's, and compares their weight digests.
- kernel/plugins: plugins/plugins.py, what each VLLM_PLUGINS setting loads in the image
  (the recipe's setting for the direct loader).

Both build the native helper into /c/xdg-direct-loader (XDG_CACHE_HOME) and run with
io_uring allowed (the serving container's seccomp and memlock).
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENT = HERE.relative_to(ROOT).as_posix()
SOURCE_ROOT = "/opt/spark-ds41f/candidate/vllm"
TESTS = (
    "tests/model_executor/model_loader/test_direct_loader_cuda.py",
    "tests/model_executor/model_loader/test_direct_loader.py",
    "tests/model_executor/model_loader/test_weight_digest.py",
)
SNAPSHOT = "models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
ENV = {"HF_HUB_OFFLINE": "1", "XDG_CACHE_HOME": "/c/xdg-direct-loader"}


def package_mounts() -> list[list[str]]:
    source = json.loads((HERE / "bundles/direct/source.json").read_text())
    return [[f"{EXPERIMENT}/bundles/direct/overlay/{path}", f"{SOURCE_ROOT}/{path}"] for path in source["files"]]


def write(bundle: str, candidate: dict) -> None:
    path = HERE / "kernel" / bundle / "candidate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm", type=Path)
    parser.add_argument("--image", default="spark-ds41f:19f2c20ed4d6-next-6")
    args = parser.parse_args()
    revision = json.loads((HERE / "bundles/direct/source.json").read_text())["revision"]
    tests = HERE / "kernel" / "gpu-tests"
    tests.mkdir(parents=True, exist_ok=True)
    test_mounts = []
    for path in TESTS:
        name = path.replace("/", "_")
        (tests / name).write_bytes(subprocess.run(["git", "-C", str(args.vllm), "show", f"{revision}:{path}"],
                                                  check=True, capture_output=True).stdout)
        test_mounts.append([name, f"{SOURCE_ROOT}/{path}"])
    write("gpu-tests", {
        "description": f"The direct loader's tests (vLLM {revision[:9]}: 0498-0499) in candidate 6's image "
                       "with the branch's loader files mounted. DIRECT_IO_TEST_DIR is on /c, where O_DIRECT works.",
        "image": args.image,
        "env": {**ENV, "DIRECT_IO_TEST_DIR": "/c/direct-io-test"},
        "workdir": "vllm",
        "argv": ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
                 "--basetemp", "/c/pytest-direct-loader", *TESTS],
        "verdict": "exit",
        "io_uring": True,
        "mounts": [*test_mounts, *package_mounts()],
    })
    write("rank-load", {
        "description": f"One TP4 rank's share of DeepSeek-V4.1-Flash-1017 loaded with the direct loader "
                       f"(vLLM {revision[:9]}) and with B12X's, weight digests compared (rank_compare.py).",
        "image": args.image,
        "env": {**ENV, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6"},
        "workdir": "vllm",
        "argv": ["/b/rank_compare.py", f"/hub/{SNAPSHOT}", "/c/direct-loader-rank-load", "--tp", "4", "--rank", "0"],
        "verdict": "status",
        "io_uring": True,
        "mounts": [["rank_load.py", "/b/rank_load.py"], ["rank_compare.py", "/b/rank_compare.py"],
                   ["~/.cache/huggingface/hub", "/hub"], *package_mounts()],
    })
    write("plugins", {
        "description": "The image's general plugins: what an unset, empty or b12x_loader VLLM_PLUGINS loads, "
                       f"and the loader --load-format direct gets (vLLM {revision[:9]}); plugins.py.",
        "image": args.image,
        "env": ENV,
        "workdir": "vllm",
        "argv": ["/b/plugins.py"],
        "verdict": "status",
        "mounts": [["plugins.py", "/b/plugins.py"], *package_mounts()],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
