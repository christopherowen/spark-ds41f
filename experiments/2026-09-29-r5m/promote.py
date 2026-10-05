"""Promote r5m: configuration, baseline manifest and docs.

usage: promote.py   (in the r5 worktree, after build.sh and run.sh)

r5m ships the exact tiled_topk.py measured as an overlay on r5l, so r5l's
reference bench remains the benchmark record.
"""
import json
import re
import subprocess
from pathlib import Path

BASE = "2026-09-29-karmic-kraken-r5m"
PREV = "2026-09-29-karmic-kraken-r5l"
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5m"
PREV_TAG = "vllm-ds41f-kkref:04c30fa98e79-r5l"
RESULTS = Path.home() / "projects/spark3-vllm-ds41f/results/private"
REF = f"manifests/benchmarks/{PREV}.json"


def sub(path, old, new, count=1):
    p = Path(path)
    text = p.read_text()
    assert old in text, (path, old[:80])
    p.write_text(text.replace(old, new, count))


run_log = (RESULTS / "r5m/run.log").read_text()
built_line = re.search(
    r"built \S+ (sha256:[0-9a-f]{64}) in (\d+)s; lowest MemAvailable ([0-9.]+) GiB", run_log
)
IMAGE_ID = built_line[1]
assert "tiled_topk.py matches the measured overlay on every node" in run_log
quality = json.loads((RESULTS / "bench/r5m-quality/bench.json").read_text())["suites"]["quality"]["passed"]
needle_lines = (RESULTS / "r5m/needle.txt").read_text().splitlines()
needle_pass = sum("PASS" in line for line in needle_lines)
tokens = re.search(r"prompt (\d+) tokens", needle_lines[0])[1]
NEEDLE = f"{needle_pass}/{len(needle_lines)} at {int(tokens):,} tokens"
created = subprocess.run(
    ["docker", "image", "inspect", TAG, "--format", "{{.Created}}"],
    capture_output=True, text=True, check=True,
).stdout.strip()
built = subprocess.run(
    ["date", "-u", "-d", created, "+%Y-%m-%dT%H:%M:%SZ"],
    capture_output=True, text=True, check=True,
).stdout.strip()

# 1. Configuration.
arm = json.loads(Path("experiments/2026-09-29-r5m/cluster-candidate.json").read_text())
arm["promoted_baseline"] = BASE
Path("config/cluster.json").write_text(json.dumps(arm, indent=2) + "\n")
sub("upstreams.lock.json", f'"generated_from_baseline": "{PREV}"', f'"generated_from_baseline": "{BASE}"')

# 2. Baseline manifest.
manifest = json.loads(Path(f"manifests/baselines/{PREV}.json").read_text())
source = json.loads(Path("manifests/sources/2026-09-29-r5m-candidate-source.json").read_text())
manifest.update(
    captured_at=subprocess.run(["date", "-u", "+%Y%m%dT%H%M%SZ"], capture_output=True,
                               text=True, check=True).stdout.strip(),
    image_tag=TAG,
    image_id=IMAGE_ID,
    change_from_previous=(
        "B12X 0003 breaks exact top-k score ties by lowest logical position in the DSA "
        "radix top-k (buffered arm and exact overflow fallback), so indexer selections "
        "repeat exactly; before, identical runs chose different positions for 78% of "
        "layer 2's prefill rows. DeepSeek's reference torch.topk repeats for the same "
        "scores and dgpp pins the same rule. No other change from r5l."
    ),
    evidence={
        "experiments": ["experiments/2026-09-29-topk-ties/", "experiments/2026-09-29-r5m/"],
        "summary": (
            "Measured as an overlay of the byte-identical tiled_topk.py (SHA-256 f1bdacb5) on "
            "r5l: zero selection differences between full-row runs and against the split "
            "(473,493 rows per encoder indexer layer), prefill chunks at 8K-200K within 0.7% "
            "of r5l over two rounds, decode matched r5k and r5l in an alternating screen "
            "(ties, r5k, r5l twice each), acceptance unchanged; B12X unit tests 6/6 new and "
            f"103/103 existing. Built image: tiled_topk.py identical on every node, LRU "
            f"{quality}/5, needle retrieval {NEEDLE}, doctor --live clean. The benchmark "
            "reference remains r5l's, since the measured file is the one shipped."
        ),
    },
    benchmark_reference=REF,
    capacity_benchmark_reference=REF,
    previous_baseline=PREV,
)
manifest["nodes"] = {n: dict(v, image_id=IMAGE_ID) for n, v in manifest["nodes"].items()}
manifest["source_identity"].update(
    b12x_patch_head=source["b12x"]["patch_head"],
    b12x_tree=source["b12x"]["expected_tree"],
)
manifest["runtime_choices"]["topk_ties"] = (
    "lowest logical position (B12X 0003); indexer selections repeat exactly"
)
manifest["build"].update(
    image=TAG, image_id=IMAGE_ID, elapsed_seconds=int(built_line[2]),
    min_available_gib=float(built_line[3]), built_utc=built,
)
Path(f"manifests/baselines/{BASE}.json").write_text(json.dumps(manifest, indent=2) + "\n")

# 3. Baseline and image references; the benchmark reference stays r5l's.
for path in ("docker/README.md", "docs/replicate.md", "docs/upstreams.md", "README.md",
             "docs/current-state.md"):
    p = Path(path)
    text = p.read_text().replace(PREV_TAG, TAG)
    text = text.replace(f"manifests/baselines/{PREV}", f"manifests/baselines/{BASE}")
    text = text.replace(f"`{PREV}`", f"`{BASE}`")
    p.write_text(text)
sub("docs/upstreams.md", "(sha256:57fa6379…)", f"(sha256:{IMAGE_ID[7:15]}…)")
sub("docs/replicate.md", f"This reproduces `manifests/baselines/{BASE}.json`",
    f"This reproduces `manifests/baselines/{BASE}.json`")

# 4. Current state and README.
cs = "docs/current-state.md"
sub(cs, "B12X `f8069b2c` + switchless RoCEnante and CuTe DSL 4.7.1 pin patches",
    "B12X `f8069b2c` + switchless RoCEnante, CuTe DSL 4.7.1 pin and top-k position-tie patches")
sub(cs, "| Indexer under sequence parallelism | each rank scores and selects its own rows and "
    "all-gathers the top-k positions (patch 0025) |",
    "| Indexer under sequence parallelism | each rank scores and selects its own rows and "
    "all-gathers the top-k positions (patch 0025) |\n"
    "| Indexer top-k ties | lowest logical position (B12X 0003): selections repeat exactly |")
sub(cs, "Measured on this configuration:",
    "Measured on r5l, which differs only by the top-k tie rule (measured neutral):")
rd = "README.md"
sub(rd, "  CuTe DSL pin moved to the 4.7.1 that vLLM requires; its FP4 KV writer\n"
    "  rounds like DeepSeek's reference quantizer), with B12X attention,",
    "  CuTe DSL pin moved to the 4.7.1 that vLLM requires; its FP4 KV writer\n"
    "  rounds like DeepSeek's reference quantizer, and its indexer top-k breaks\n"
    "  score ties by position, so selections repeat), with B12X attention,")
sub(rd, "Current baseline, measured with `bin/spark bench` from dgx1:",
    "Measured on r5l (r5m differs only by the top-k tie rule, measured neutral) with\n"
    "`bin/spark bench` from dgx1:")

# 5. TODO: the tie item is done; record what still varies.
todo = Path("TODO.md")
text = todo.read_text()
start = text.index("- **Indexer ties make runs irreproducible.**")
end = text.index("\n- ", start + 10)
text = text[:start] + (
    "- **Temperature-0 outputs still vary between identical requests** (6 of 6\n"
    "  distinct at one stream). r5m's B12X 0003 made indexer selections exact\n"
    "  (`experiments/2026-09-29-topk-ties`), so the rest comes from elsewhere:\n"
    "  verification batch shapes that change with acceptance, and kernels whose\n"
    "  summation order varies. A deterministic profile would need batch-invariant\n"
    "  verify kernels; measure its cost before proposing it.\n"
) + text[end:]
todo.write_text(text)
print("promoted", BASE)
