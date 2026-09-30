"""Promote r5o: configuration, baseline manifest and docs.

usage: promote.py   (in the r5 worktree, after screen.sh, build.sh and run.sh)

r5o ships the exact three files measured as an overlay on r5n by screen.sh, so
r5l's reference bench remains the benchmark record.
"""
import json
import re
import subprocess
from pathlib import Path

BASE = "2026-09-30-karmic-kraken-r5o"
PREV = "2026-09-30-karmic-kraken-r5n"
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5o"
PREV_TAG = "vllm-ds41f-kkref:04c30fa98e79-r5n"
RESULTS = Path.home() / "projects/spark3-vllm-ds41f/results/private"
SCREEN = (
    "Overlay screen of the shipped files against r5n on one pinned cost table: prefill chunks "
    "at 8K-200K within 0.42% (three rounds each); decode alternating twice: step times within "
    "0.3%, eight-stream throughput level or higher at matched verified and accepted drafts per "
    "draft; long-prompt repeatability of the deterministic MoE unchanged (1/5 with and without)."
)


def sub(path, old, new, count=1):
    p = Path(path)
    text = p.read_text()
    assert old in text, (path, old[:80])
    p.write_text(text.replace(old, new, count))


assert SCREEN != "SCREEN_SUMMARY", "fill in the screen summary first"
run_log = (RESULTS / "r5o/run.log").read_text()
built_line = re.search(
    r"built \S+ (sha256:[0-9a-f]{64}) in (\d+)s; lowest MemAvailable ([0-9.]+) GiB", run_log
)
IMAGE_ID = built_line[1]
for needed in ("shipped files match the measured ones on every node", "sass gate exit 0",
               "doctor exit 0"):
    assert needed in run_log, needed
gate = (RESULTS / "r5o/sass-gate.txt").read_text().strip().splitlines()[-1]
quality = json.loads((RESULTS / "bench/r5o-quality/bench.json").read_text())["suites"]["quality"]["passed"]
needle_lines = (RESULTS / "r5o/needle.txt").read_text().splitlines()
needle_pass = sum("PASS" in line for line in needle_lines)
tokens = re.search(r"prompt (\d+) tokens", needle_lines[0])[1]
NEEDLE = f"{needle_pass}/{len(needle_lines)} at {int(tokens):,} tokens"
created = subprocess.run(["docker", "image", "inspect", TAG, "--format", "{{.Created}}"],
                         capture_output=True, text=True, check=True).stdout.strip()
built = subprocess.run(["date", "-u", "-d", created, "+%Y-%m-%dT%H:%M:%SZ"],
                       capture_output=True, text=True, check=True).stdout.strip()

# 1. Configuration.
arm = json.loads(Path("experiments/2026-09-30-r5o/cluster-candidate.json").read_text())
arm["promoted_baseline"] = BASE
Path("config/cluster.json").write_text(json.dumps(arm, indent=2) + "\n")
sub("upstreams.lock.json", f'"generated_from_baseline": "{PREV}"', f'"generated_from_baseline": "{BASE}"')

# 2. Baseline manifest.
manifest = json.loads(Path(f"manifests/baselines/{PREV}.json").read_text())
source = json.loads(Path("manifests/sources/2026-09-30-r5o-candidate-source.json").read_text())
manifest.update(
    captured_at=subprocess.run(["date", "-u", "+%Y%m%dT%H%M%SZ"], capture_output=True,
                               text=True, check=True).stdout.strip(),
    image_tag=TAG,
    image_id=IMAGE_ID,
    change_from_previous=(
        "B12X 0005 fences the async proxy before the TMA-refilled stage releases of the BF16 "
        "prefill projection, the mHC TF32 and BF16 TMA prefill projections and the contiguous "
        "attention forward. Their compiled code released each stage with shared loads still "
        "pending, the dense GEMM's bug before r5n; beside co-resident kernels the mHC TF32 "
        "projection returned wrong outputs in up to 5% of calls and the BF16 prefill projection "
        "in up to 0.4%. No other change from r5n."
    ),
    evidence={
        "experiments": ["experiments/2026-09-30-proxy-fence-audit/", "experiments/2026-09-30-r5o/"],
        "summary": (
            "Found by a scoreboard dataflow over the compiled r5m kernels' SASS (loads pending at "
            "every stage arrive of the three kernels) and reproduced standalone beside the routed "
            "MoE or a copy: mHC TF32 609/12000 and BF16 prefill up to 45/12000 wrong shipped, "
            f"0/12000 fenced; the varlen attention fence is preventive. {SCREEN} Built image: the "
            f"three files identical to the measured overlay on every node; SASS gate ({gate}); "
            f"LRU {quality}/5, needle retrieval {NEEDLE}, doctor --live clean. The benchmark "
            "reference remains r5l's, since the measured files are the ones shipped."
        ),
    },
    previous_baseline=PREV,
)
manifest["nodes"] = {n: dict(v, image_id=IMAGE_ID) for n, v in manifest["nodes"].items()}
manifest["source_identity"].update(
    b12x_patch_head=source["b12x"]["patch_head"],
    b12x_tree=source["b12x"]["expected_tree"],
)
manifest["runtime_choices"]["prefill_stage_release"] = (
    "async-proxy fence before each TMA-refilled stage release in the BF16 and mHC prefill "
    "projections and contiguous attention (B12X 0005)"
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
prev_short = re.search(r"\(sha256:([0-9a-f]{8})…\)", Path("docs/upstreams.md").read_text())[1]
sub("docs/upstreams.md", f"(sha256:{prev_short}…)", f"(sha256:{IMAGE_ID[7:15]}…)")
sub("docs/upstreams.md", "top-k position ties, dense GEMM stage fence)",
    "top-k position ties, TMA stage-release fences)")

# 4. Current state and README.
cs = "docs/current-state.md"
sub(cs, "top-k position-tie and dense GEMM stage-fence patches",
    "top-k position-tie, dense GEMM and prefill stage-fence patches")
sub(cs, "the shared expert no longer returns wrong columns beside the routed MoE |",
    "the shared expert no longer returns wrong columns beside the routed MoE |\n"
    "| Prefill stage releases | async-proxy fence before the TMA refill in the BF16 and mHC "
    "prefill projections and contiguous attention (B12X 0005) |")
sub(cs, "the top-k tie rule and the dense GEMM stage fence (neither with a measured cost):",
    "the top-k tie rule and the TMA stage-release fences (none with a measured cost):")
rd = "README.md"
sub(rd, "  score ties by position, so selections repeat, and its dense GEMM fences\n"
    "  shared-memory stage reads before the TMA refill), with B12X attention,",
    "  score ties by position, so selections repeat, and its dense GEMM and prefill\n"
    "  kernels fence shared-memory stage reads before the TMA refill), with B12X attention,")
sub(rd, "Measured on r5l (r5n differs only by the top-k tie rule and the dense GEMM stage\n"
    "fence, neither with a measured cost) with",
    "Measured on r5l (r5o differs only by the top-k tie rule and the TMA stage-release\n"
    "fences, none with a measured cost) with")

# 5. TODO: the prefill races are fixed; record what the audit left open.
todo = Path("TODO.md")
text = todo.read_text()
start = text.index("- **Two more TMA stage-release races in r5n**")
end = text.index("\n- ", start + 10)
text = text[:start] + (
    "- **TMA stage releases outside DS4.1's serving set** (`experiments/2026-09-30-proxy-fence-audit`):\n"
    "  r5o fences every serving kernel the audit found; still open upstream are the\n"
    "  NVFP4/W6A8 MoE releases, a SASS pass over kernels DS4.1 does not run, two\n"
    "  single-stage write-after-read races in raw paged kernels, and mbarrier init fences.\n"
) + text[end:]
todo.write_text(text)
print("promoted", BASE)
