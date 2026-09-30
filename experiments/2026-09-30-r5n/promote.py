"""Promote r5n: configuration, baseline manifest and docs.

usage: promote.py   (in the r5 worktree, after screen.sh, build.sh and run.sh)

r5n ships the exact dense_gemm.py measured as an overlay on r5m by screen.sh,
so r5l's reference bench remains the benchmark record.
"""
import json
import re
import subprocess
from pathlib import Path

BASE = "2026-09-30-karmic-kraken-r5n"
PREV = "2026-09-29-karmic-kraken-r5m"
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5n"
PREV_TAG = "vllm-ds41f-kkref:04c30fa98e79-r5m"
RESULTS = Path.home() / "projects/spark3-vllm-ds41f/results/private"
SCREEN = (
    "Overlay screen of the shipped file against r5m: prefill chunks at 8K-200K within 0.73% "
    "(two rounds each); decode alternating overlay, r5m twice each: step times +0.3-0.5%, "
    "JSON at eight streams -0.1%, prose at eight streams -1.8% (r5m's own boot-to-boot "
    "spread 1.7%; recorded as a possible small cost)."
)


def sub(path, old, new, count=1):
    p = Path(path)
    text = p.read_text()
    assert old in text, (path, old[:80])
    p.write_text(text.replace(old, new, count))


assert SCREEN != "SCREEN_SUMMARY", "fill in the screen summary first"
run_log = (RESULTS / "r5n/run.log").read_text()
built_line = re.search(
    r"built \S+ (sha256:[0-9a-f]{64}) in (\d+)s; lowest MemAvailable ([0-9.]+) GiB", run_log
)
IMAGE_ID = built_line[1]
assert "dense_gemm.py matches the measured overlay on every node" in run_log
assert "doctor exit 0" in run_log
quality = json.loads((RESULTS / "bench/r5n-quality/bench.json").read_text())["suites"]["quality"]["passed"]
needle_lines = (RESULTS / "r5n/needle.txt").read_text().splitlines()
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
arm = json.loads(Path("experiments/2026-09-30-r5n/cluster-candidate.json").read_text())
arm["promoted_baseline"] = BASE
Path("config/cluster.json").write_text(json.dumps(arm, indent=2) + "\n")
sub("upstreams.lock.json", f'"generated_from_baseline": "{PREV}"', f'"generated_from_baseline": "{BASE}"')

# 2. Baseline manifest.
manifest = json.loads(Path(f"manifests/baselines/{PREV}.json").read_text())
source = json.loads(Path("manifests/sources/2026-09-30-r5n-candidate-source.json").read_text())
manifest.update(
    captured_at=subprocess.run(["date", "-u", "+%Y%m%dT%H%M%SZ"], capture_output=True,
                               text=True, check=True).stdout.strip(),
    image_tag=TAG,
    image_id=IMAGE_ID,
    change_from_previous=(
        "B12X 0004 fences the async proxy before each dense GEMM mainloop stage release on "
        "the TMA load path. The MMA warps' last generic-proxy reads of a stage could observe "
        "its TMA refill when another kernel's CTAs shared the SM, so DS4.1's shared-expert down "
        "projection returned wrong columns in about 5% of decode calls beside the routed MoE. "
        "No other change from r5m."
    ),
    evidence={
        "experiments": ["experiments/2026-09-29-determinism/", "experiments/2026-09-30-r5n/"],
        "summary": (
            "Found by exact-bits captures of the mismatching calls (the last k sub-block of k "
            "tiles 0 and 1 read from their stages' refills). Standalone stress of the "
            "serving-shaped linear beside the routed MoE: 20/12000 (6 rows) and 1/12000 (48 rows) "
            "wrong before, 0/12000 each after; with the experimental deterministic MoE, serving "
            f"repeats exactly with the side-stream overlap on. {SCREEN} Built image: dense_gemm.py "
            f"identical to the measured overlay on every node, LRU {quality}/5, needle retrieval "
            f"{NEEDLE}, doctor --live clean. The benchmark reference remains r5l's, since the "
            "measured file is the one shipped and measured neutral."
        ),
    },
    previous_baseline=PREV,
)
manifest["nodes"] = {n: dict(v, image_id=IMAGE_ID) for n, v in manifest["nodes"].items()}
manifest["source_identity"].update(
    b12x_patch_head=source["b12x"]["patch_head"],
    b12x_tree=source["b12x"]["expected_tree"],
)
manifest["runtime_choices"]["dense_gemm_stage_release"] = (
    "async-proxy fence before each TMA-refilled stage release (B12X 0004)"
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
sub("docs/upstreams.md", "(sha256:f5d3d192…)", f"(sha256:{IMAGE_ID[7:15]}…)")
sub("docs/upstreams.md", "+ patches/b12x/series (switchless RoCEnante routing, CuTe DSL 4.7.1)",
    "+ patches/b12x/series (switchless RoCEnante routing, CuTe DSL 4.7.1,\n"
    "          top-k position ties, dense GEMM stage fence)")

# 4. Current state and README.
cs = "docs/current-state.md"
sub(cs, "Promoted 2026-09-29 as", "Promoted 2026-09-30 as")
sub(cs, "B12X `f8069b2c` + switchless RoCEnante, CuTe DSL 4.7.1 pin and top-k position-tie patches",
    "B12X `f8069b2c` + switchless RoCEnante, CuTe DSL 4.7.1 pin, top-k position-tie and dense "
    "GEMM stage-fence patches")
sub(cs, "| Indexer top-k ties | lowest logical position (B12X 0003): selections repeat exactly |",
    "| Indexer top-k ties | lowest logical position (B12X 0003): selections repeat exactly |\n"
    "| Dense GEMM stage release | async-proxy fence before the TMA refill (B12X 0004): the "
    "shared expert no longer returns wrong columns beside the routed MoE |")
sub(cs, "Measured on r5l, which differs only by the top-k tie rule (measured neutral):",
    "Measured on r5l, which differs only by the top-k tie rule and the dense GEMM stage fence "
    "(both measured neutral):")
rd = "README.md"
sub(rd, "  score ties by position, so selections repeat), with B12X attention,",
    "  score ties by position, so selections repeat, and its dense GEMM fences\n"
    "  shared-memory stage reads before the TMA refill), with B12X attention,")
sub(rd, "Measured on r5l (r5m differs only by the top-k tie rule, measured neutral) with",
    "Measured on r5l (r5n differs only by the top-k tie rule and the dense GEMM stage\n"
    "fence, both measured neutral) with")

# 5. TODO: record what the determinism experiment found.
todo = Path("TODO.md")
text = todo.read_text()
start = text.index("- **Determinism.** Temperature-0 outputs differ within one boot")
end = text.index("\n## ", start)
text = text[:start] + (
    "- **Determinism.** `experiments/2026-09-29-determinism` found three sources: the\n"
    "  atomic routed-MoE combine, four-way split-K turbo, and a dense GEMM race that\n"
    "  also gave wrong shared-expert outputs (fixed in r5n). An experimental\n"
    "  deterministic mode (its 0004-0006, split-K through the FP32 reducer) repeats\n"
    "  exactly at one stream with no single-stream cost; JSON at eight streams is\n"
    "  about 4% slower. Before proposing it: profile that cost against r5n with\n"
    "  verification work held fixed, try a masked top-k sum without dead-route\n"
    "  clearing, and check repeatability across batch compositions.\n"
) + text[end:]
start = text.index("- **Temperature-0 outputs still vary between identical requests**")
end = text.index("\n- ", start + 10)
text = text[:start] + (
    "- **Temperature-0 outputs still vary between identical requests** on r5n: the\n"
    "  atomic MoE combine and split-K turbo change summation order run to run\n"
    "  (see the determinism item under decode).\n"
) + text[end:]
todo.write_text(text)
print("promoted", BASE)
