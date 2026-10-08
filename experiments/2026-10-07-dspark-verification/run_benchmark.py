#!/usr/bin/env python3
"""r6a acceptance benchmark, r6's protocol (experiments/2026-10-05-tilelang-r6/run_benchmark.sh).

usage: experiments/2026-10-07-dspark-verification/run_benchmark.py tp4|tp3

Run on dgx1 from the deployment checkout, detached. Opens a lab window that restores
the recipe's promoted profile, boots r6a-<recipe>.json once and runs quality, the
single-stream decode profile, decode on prose and code with reasoning (three samples)
and real-text prefill (two repeats) at the recipe's limits: TP4 at 1-16 streams and
32K-512K plus a 1,000,000-token prefill on the same boot, TP3 at 1-8 streams and
32K-500K. Copies the reports to runs-<recipe>/ and closes the window.
"""
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent.relative_to(ROOT).as_posix()
sys.path.insert(0, str(ROOT / "scripts"))
import lab  # noqa: E402

RECIPES = {
    "tp4": ("config/cluster-tp4.json", "1,2,4,8,16", "32768,262144,500000,1048576"),
    "tp3": ("config/cluster.json", "1,2,4,8", "32768,262144,500000"),
}
URL = "http://10.0.1.71:8000"


def main() -> int:
    recipe = sys.argv[1] if len(sys.argv) > 1 else "tp4"
    production, streams, sizes = RECIPES[recipe]
    lab.PRODUCTION_CONFIG = production
    config = f"{EXP}/r6a-{recipe}.json"
    work = f".work/r6a/{recipe}"
    runs = ROOT / EXP / f"runs-{recipe}"
    runs.mkdir(parents=True, exist_ok=True)
    note = {"text": f"r6a {recipe} benchmark"}
    stop = threading.Event()

    def beats() -> None:
        while not stop.wait(60):
            lab.beat(note["text"])

    def spark(name: str, *argv: str) -> int:
        note["text"] = f"r6a {recipe}: {name}"
        lab.log(f"{name}: bin/spark --cluster-config {config} {' '.join(argv)}")
        with (runs / f"{name}.txt").open("w") as handle:
            code = subprocess.run([str(ROOT / "bin" / "spark"), "--cluster-config", config, *argv],
                                  cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
        lab.log(f"{name}: exit {code}")
        return code

    lab.window_open(90, f"r6a {recipe} acceptance benchmark")
    threading.Thread(target=beats, daemon=True).start()
    try:
        if spark("start", "cluster", "start", "--replace", "--apply"):
            return 1
        spark("quality", "bench", "--url", URL, "--suites", "quality", "--output", f"{work}-quality")
        with (runs / "profile.txt").open("w") as handle:
            subprocess.run([sys.executable, "experiments/2026-09-29-determinism/profile_decode.py", URL,
                            "--tokens", "160"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if spark("bench", "bench", "--url", URL, "--suites", "decode,prefill", "--decode-cases", "prose,code",
                 "--concurrency", streams, "--min-samples", "3", "--max-samples", "3",
                 "--prefill-text", "source", "--prefill-sizes", sizes, "--prefill-repeats", "2",
                 "--output", f"{work}-bench") == 0:
            shutil.copy(ROOT / f"{work}-bench" / "bench.json", runs / "bench.json")
        if recipe == "tp4" and spark("bench-1m", "bench", "--url", URL, "--suites", "prefill",
                                     "--prefill-text", "source", "--prefill-sizes", "1000000",
                                     "--prefill-repeats", "2", "--output", f"{work}-bench-1m") == 0:
            shutil.copy(ROOT / f"{work}-bench-1m" / "bench.json", runs / "bench-1m.json")
        return 0
    finally:
        stop.set()
        lab.window_close(f"r6a {recipe} benchmark done")


if __name__ == "__main__":
    sys.exit(main())
