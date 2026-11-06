"""Bounded own-window metadata audit; never issues a model or benchmark request."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("functional", "performance"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--model-contract-sha256")
    args = parser.parse_args()
    if sys.platform != "linux" or args.output.exists():
        raise SystemExit("remote Linux only; preserve prior audit evidence")
    cgroup = Path("/sys/fs/cgroup") / next(line.split("::", 1)[1].lstrip("/")
        for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    limits = {key: (cgroup / key).read_text().strip() for key in ("cpu.max", "memory.max", "memory.swap.max")}
    quota, period = map(int, limits["cpu.max"].split())
    if quota > period or limits["memory.max"] != str(4 * 1024**3) or limits["memory.swap.max"] != "0":
        raise SystemExit("metadata audit requires actual CPU1/memory4GiB/no-swap cgroup bounds")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 95))
    signal.alarm(120)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    image, spec = json.loads(args.image_receipt.read_text()), json.loads(args.spec.read_text())
    try:
        if args.kind == "functional":
            import c45_functional
            plan = json.loads(Path(__file__).with_name("orchestration-readiness.json").read_text())
            result = c45_functional.validate(args.output.parent, spec, image, plan["semantic_validator"]["contract"])
        else:
            import c45_performance
            result = c45_performance.validate(spec, image, args.commit, args.model_contract_sha256)
    except Exception as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    result.update(audit_commit=args.commit, effective_cgroup_limits=limits, kind=args.kind)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({key: result.get(key) for key in ("kind", "passed", "error", "effective_cgroup_limits")}))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
