"""Promote r5l: configuration, baseline manifest, benchmark reference and docs.

usage: promote.py NEEDLE   (in the r5 worktree; NEEDLE like "3/3 at 181,203 tokens")
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

NEEDLE = sys.argv[1]
BASE = "2026-09-29-karmic-kraken-r5l"
PREV = "2026-09-29-karmic-kraken-r5k"
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5l"
PREV_TAG = "vllm-ds41f-kkref:04c30fa98e79-r5k"
RESULTS = Path.home() / "projects/spark3-vllm-ds41f/results/private"
BENCH = RESULTS / "bench/r5l-reference/bench.json"
SOURCE = RESULTS / "bench/r5l-prefill-source/bench.json"
REF = Path(f"manifests/benchmarks/{BASE}.json")
RUN_LOG = RESULTS / "r5l/run.log"
built_line = re.search(
    r"built \S+ (sha256:[0-9a-f]{64}) in (\d+)s; lowest MemAvailable ([0-9.]+) GiB",
    RUN_LOG.read_text(),
)
IMAGE_ID = built_line[1]
BUILD_SECONDS = int(built_line[2])
BUILD_MIN_GIB = float(built_line[3])


def sub(path, old, new, count=1):
    p = Path(path)
    text = p.read_text()
    assert old in text, (path, old[:80])
    p.write_text(text.replace(old, new, count))


def val(x):
    return x["mean"] if isinstance(x, dict) else x


bench = json.loads(BENCH.read_text())
suites = bench["suites"]
points = suites["decode"]["points"]
source = json.loads(SOURCE.read_text())["suites"]["prefill"]["points"]
filler = suites["prefill"]["points"]
prefix = suites["prefix"]
admission = suites["admission"]
memory = bench["memory"]
quality = suites["quality"]["passed"]
dgx1_min = memory["dgx1"]["min_available_gib"]


def mean(case, streams, field):
    return val(points[f"{case}-c{streams}"][field])


def k(tps):
    return f"{tps / 1000:.1f}k"


src = {int(size): val(p["prefill_tps"]) for size, p in source.items()}
fil = {int(size): val(p["prefill_tps"]) for size, p in filler.items()}
source_row = ", ".join(
    f"{label} {k(src[size])}"
    for label, size in (("4K", 4096), ("16K", 16384), ("32K", 32768), ("64K", 65536),
                        ("131K", 131072), ("200K", 200000))
)
filler_row = ", ".join(
    f"{label} {k(fil[size])}" for label, size in (("2K", 2048), ("32K", 32768), ("64K", 65536),
                                                   ("131K", 131072))
)
built = subprocess.run(
    ["docker", "image", "inspect", TAG, "--format", "{{.Created}}"],
    capture_output=True, text=True, check=True,
).stdout.strip()[:19] + "Z"

# 1. Configuration: the candidate arm, as run for the reference.
arm = json.loads(Path("experiments/2026-09-29-r5l/cluster-candidate.json").read_text())
arm["promoted_baseline"] = BASE
Path("config/cluster.json").write_text(json.dumps(arm, indent=2) + "\n")
sub("upstreams.lock.json", f'"generated_from_baseline": "{PREV}"', f'"generated_from_baseline": "{BASE}"')

# 2. Benchmark reference.
shutil.copyfile(BENCH, REF)

# 3. Baseline manifest.
manifest = json.loads(Path(f"manifests/baselines/{PREV}.json").read_text())
manifest.update(
    captured_at=bench["finished_utc"].replace("-", "").replace(":", "")[:15] + "Z",
    image_tag=TAG,
    image_id=IMAGE_ID,
    change_from_previous=(
        "vLLM 0025 splits the replicated indexer's rows across the TP ranks during "
        "sequence-parallel prefill: each rank scores and selects top-k positions for its own "
        "rows and all-gathers them, which removes two thirds of the indexer on every rank "
        "(one 4,096-token chunk: -5% at 64K of context, -10% at 131K, -15% at 200K). "
        "vLLM 0026 adds a debug-mode integrity check of the display carve-out weights "
        "(SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS), off in configuration and free when off."
    ),
    evidence={
        "experiments": ["experiments/2026-09-29-indexer-split/", "experiments/2026-09-29-r5l/"],
        "summary": (
            f"LRU {quality}/5; needle retrieval {NEEDLE}; split check on 473,461 rows per "
            "encoder indexer layer: split-versus-full differences within 2-7% of a second "
            "full-row run's (radix top-k ties), none in deterministic layers, no unwritten rows "
            "or candidate changes, and zero differences of either kind with the B12X "
            "lowest-position tie-break (experiments/2026-09-29-topk-ties); carve-out unit tests "
            "10/10 in the image and two debug-mode carve-out checks per node in the check boot; "
            f"real-text prefill {source_row} tok/s; decode matched r5k in an alternating "
            "screen (r5k, r5l, ties twice each); lowest dgx1 MemAvailable "
            f"{dgx1_min:.2f} GiB in the reference run."
        ),
    },
    benchmark_reference=str(REF),
    previous_baseline=PREV,
    capacity_benchmark_reference=str(REF),
)
manifest["nodes"] = {n: dict(v, image_id=IMAGE_ID) for n, v in manifest["nodes"].items()}
source_manifest = json.loads(Path("manifests/sources/2026-09-29-r5l-candidate-source.json").read_text())
manifest["source_identity"].update(
    vllm_patch_head=source_manifest["vllm"]["patch_head"],
    vllm_tree=source_manifest["vllm"]["expected_tree"],
)
manifest["runtime_choices"].update(
    indexer_split="always under sequence-parallel prefill (patch 0025; "
    "SPARK3_DS41_INDEXER_SPLIT_CHECK=1 validates it against the full-row indexer)",
    display_carveout_check="off (SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS unset; patch 0026 debug mode)",
)
manifest["build"].update(
    image=TAG, image_id=IMAGE_ID, elapsed_seconds=BUILD_SECONDS, min_available_gib=BUILD_MIN_GIB,
    built_utc=built,
    build_directory=".work/build/vllm-04c30fa98e79-b12x-f8069b2c0be1-<input hash>",
)
Path(f"manifests/baselines/{BASE}.json").write_text(json.dumps(manifest, indent=2) + "\n")

# 4. Image and baseline references.
for path in ("docker/README.md", "docs/replicate.md", "docs/upstreams.md", "README.md",
             "docs/current-state.md", "TODO.md"):
    p = Path(path)
    p.write_text(p.read_text().replace(PREV, BASE).replace(PREV_TAG, TAG))
sub("docs/upstreams.md", "(sha256:5416f8ff…)", f"(sha256:{IMAGE_ID[7:15]}…)")

# 5. Current state.
cs = "docs/current-state.md"
sub(cs, "Promoted 2026-09-28 as", "Promoted 2026-09-29 as")
sub(cs, "patches 0001-0024 (0005-0009 and 0011 off by default; 0023 off in configuration)",
    "patches 0001-0026 (0005-0009, 0011 and 0026 off by default; 0023 off in configuration)")
sub(cs, "| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the "
    "one-shot RoCE all-reduce; CED encoder layers (patches 0014-0018) |",
    "| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the "
    "one-shot RoCE all-reduce; CED encoder layers (patches 0014-0018) |\n"
    "| Indexer under sequence parallelism | each rank scores and selects its own rows and "
    "all-gathers the top-k positions (patch 0025) |")
text = Path(cs).read_text()
start = text.index("Measured on this configuration:")
end = text.index("The previous state")
text = text[:start] + (
    f"Measured on this configuration: LRU coherence gate {quality}/5; needle retrieval\n"
    f"{NEEDLE}; single-stream prose/code about {mean('prose', 1, 'tps'):.0f}/"
    f"{mean('code', 1, 'tps'):.0f} tok/s with reasoning,\n"
    f"code answers {mean('code-nothink', 1, 'tps'):.0f} tok/s; code at eight streams "
    f"{mean('code', 8, 'tps'):.0f} tok/s ({mean('code-nothink', 8, 'tps'):.0f} tok/s for code\n"
    f"answers); cold prefill {filler_row} tok/s\n"
    f"({source_row} on real text); dgx1 minimum\n"
    f"MemAvailable {dgx1_min:.2f} GiB under load.\n"
) + text[end:]
Path(cs).write_text(text)

# 6. README baseline bullets and performance.
rd = "README.md"
sub(rd, "- sequence-parallel prefill once a prompt chunk's reduce-scatter outgrows\n"
    "  the one-shot RoCE all-reduce (205 tokens): the encoder layers' row-wise\n"
    "  work runs on a third of the rows per rank;\n",
    "- sequence-parallel prefill once a prompt chunk's reduce-scatter outgrows\n"
    "  the one-shot RoCE all-reduce (205 tokens): the encoder layers' row-wise\n"
    "  work runs on a third of the rows per rank, and so does the sparse-attention\n"
    "  indexer, whose cost grows with context depth (a 4,096-token chunk at 200K\n"
    "  of context runs 15% faster);\n")
rows = []
for case in ("prose", "code"):
    for streams in (1, 2, 4, 8):
        rows.append(
            f"| {case} | {streams} | {mean(case, streams, 'tps'):.1f} | "
            f"{mean(case, streams, 'per_stream_decode_tps'):.1f} | {mean(case, streams, 'ttft_s'):.2f} s |"
        )
answers = []
for case, label in (("prose-nothink", "prose"), ("code-nothink", "code"), ("json-nothink", "JSON")):
    answers.append(f"| {label} | " + " | ".join(f"{mean(case, s, 'tps'):.1f}" for s in (1, 2, 4, 8)) + " |")
acc = {c: val(points[f"{c}-c1"]["accepted_per_draft"]) for c in ("prose", "code", "code-nothink")}
step = {c: val(points[f"{c}-c1"]["step_ms"]) for c in ("prose", "code")}
text = Path(rd).read_text()
start = text.index("| Prompt | Streams | Aggregate tok/s | Per-stream decode tok/s | First token |")
end = text.index("The quick default takes three or four samples")
text = text[:start] + (
    "| Prompt | Streams | Aggregate tok/s | Per-stream decode tok/s | First token |\n"
    "|---|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
    "With reasoning off (the answer itself), aggregate tok/s at 1/2/4/8 streams:\n\n"
    "| Prompt | 1 | 2 | 4 | 8 |\n|---|---:|---:|---:|---:|\n" + "\n".join(answers) + "\n\n"
    "| Other measurements | |\n|---|---|\n"
    f"| Quality gate (fixed LRU task, 5 repeats) | {quality}/5 |\n"
    f"| Long-context retrieval (phrase at 10%, 50%, 90% depth) | {NEEDLE} |\n"
    f"| Single-stream decode step | about {step['prose']:.0f} ms on prose and {step['code']:.0f} ms on "
    f"code; accepted drafts per step {acc['prose']:.1f} (prose), {acc['code']:.1f} (code), "
    f"{acc['code-nothink']:.1f} (code answers) |\n"
    f"| Cold prefill, repeated filler | {filler_row} tok/s |\n"
    f"| Cold prefill, real text (Python source) | {source_row} tok/s |\n"
    f"| Prefix-cache replay, 32K prompt | {val(prefix['cold_ttft_s']):.2f} s cold, "
    f"{val(prefix['warm_ttft_s']):.2f} s warm |\n"
    f"| Four concurrent 64K contexts | all admitted without preemption, peak KV use "
    f"{admission['peak_kv_cache_usage'] * 100:.0f}%, "
    f"{val(admission['per_stream_decode_tps']):.1f} tok/s per stream |\n"
    "| Four concurrent 180K contexts (r5k) | all admitted without preemption, peak KV use 36%, "
    "11.9 tok/s per stream |\n"
    "| KV capacity | 1,348,708 tokens in 2.2 GiB per rank (5.1 full 256K contexts) |\n"
    f"| Host memory headroom | dgx1 at least {dgx1_min:.2f} GiB MemAvailable under load "
    "(3 GiB guard); startup passes the 5 GiB guard |\n\n"
) + text[end:]
old = ("at 95% confidence; temperature-0 outputs differ between identical requests,\n"
       "which moves acceptance from sample to sample.")
if old in text:
    text = text.replace(old, (
        "at 95% confidence; temperature-0 outputs differ between identical requests\n"
        "(the indexer's radix top-k keeps an arbitrary subset of positions tied at its\n"
        "threshold), which moves acceptance from sample to sample."), 1)
Path(rd).write_text(text)

# 7. TODO: current reference and the top-k tie item.
todo = Path("TODO.md")
text = todo.read_text()
start = text.index("| Workload | 1 | 2 | 4 | 8 |")
end = text.index("- **Time to first token (short prompts):**")
table = "| Workload | 1 | 2 | 4 | 8 |\n|---|---:|---:|---:|---:|\n"
for case, label in (("json-nothink", "JSON, answer only"), ("code-nothink", "Code, answer only"),
                    ("code", "Code, reasoning on"), ("prose-nothink", "Prose, answer only"),
                    ("prose", "Prose, reasoning on")):
    table += f"| {label} | " + " | ".join(f"{mean(case, s, 'tps'):.1f}" for s in (1, 2, 4, 8)) + " |\n"
text = text[:start] + table + "\n" + text[end:]
text = re.sub(r"- \*\*Real-text prefill:\*\* [^\n]*\n",
              f"- **Real-text prefill:** {k(src[4096])} tok/s at 4K, {k(src[65536])} at 64K, "
              f"{k(src[200000])} at 200K.\n", text, count=1)
old = "## Quality\n\n"
new = old + (
    "- **Indexer ties make runs irreproducible.** B12X's tiled radix top-k (the\n"
    "  full-width selection of indexer layers 2, 8 and 14) keeps an arbitrary\n"
    "  subset of the positions tied at its 512th score through shared-memory\n"
    "  atomics: two identical full-row runs chose different sets for 78% of layer\n"
    "  2's rows, about 5 of 512 positions each (at most 36;\n"
    "  `experiments/2026-09-29-r5l`, `check_summary.py`). Tied positions carry equal\n"
    "  scores, but the choice changes which keys attention reads and likely explains\n"
    "  temperature-0 outputs differing between identical requests. The technical\n"
    "  report sets no tie rule; DeepSeek's reference (`inference/model.py`,\n"
    "  `torch.topk`) is repeatable, and dgpp (docs/inspiration.md) pins exact ties\n"
    "  to the lower index. `experiments/2026-09-29-topk-ties` adds that rule to\n"
    "  B12X (patch 0003 there): selections become exact (zero differences between\n"
    "  runs or against the split), prefill and decode cost nothing measurable and\n"
    "  acceptance does not change. Whole outputs still differ between identical\n"
    "  requests (6 of 6 distinct at one stream), so other nondeterminism remains.\n"
    "  Promote it with the next image (r5m).\n"
)
assert old in text
text = text.replace(old, new, 1)
todo.write_text(text)
print("promoted", BASE)
