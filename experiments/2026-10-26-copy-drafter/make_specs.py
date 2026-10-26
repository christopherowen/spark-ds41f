#!/usr/bin/env python3
"""Write the cluster A/B's arms and specs (DSpark alone against DSpark with copy
drafts, both on the copy build) from the recipe.

usage: make_specs.py IMAGE [VLLM_TREE]

Every arm is the recipe's shape (config/cluster.json: tp4-ring, tp3-triangle) on the
copy build's IMAGE with the copy recipe's sizing:

- the capture ladder ends at the decode-row envelope, max_num_seqs x 6 plus the
  lookahead's extra rows (16 drafts: 11; 64: 59);
- the batched-token budget grows by the same rows, so the long-prefill threshold (the
  budget less the rows) and every prefill chunk stay the release's;
- the control's speculative config is the recipe's; the copy arm adds
  ``copy_drafts`` (min_match 16, max_drafts 16, initial_drafts 8), the long-copy arm
  max_drafts 64.

The AgentX arms are the same with the replay's two serving edits (port 8001 and
--enable-prompt-tokens-details, experiments/2026-10-25-agentx, #574). Rerun when the
recipe or the build changes; nothing here runs anything.
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import recipe  # noqa: E402

EXPERIMENT = "experiments/2026-10-26-copy-drafter"
SHAPES = {"tp4": "tp4-ring", "tp3": "tp3-triangle"}
BLOCK = 5
COPY = {"min_match": 16, "initial_drafts": 8}


def flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def arm(shape: str, image: str, tree: str | None, lookahead: int | None, notes: str):
    cluster = json.loads((ROOT / "config" / "cluster.json").read_text())
    resolved = recipe.resolve(cluster, SHAPES[shape])
    args = resolved["serve_args"]
    seqs = int(flag(args, "--max-num-seqs"))
    extra = (lookahead or 16) - BLOCK
    rows = seqs * (1 + BLOCK) + extra
    budget = int(flag(args, "--max-num-batched-tokens")) + extra
    threshold = flag(args, "--long-prefill-token-threshold")
    assert int(threshold) == budget - rows, (threshold, budget, rows)
    compilation = json.loads(flag(args, "--compilation-config"))
    compilation["cudagraph_capture_sizes"] = recipe.capture_sizes(rows)
    speculative = json.loads(flag(args, "--speculative-config"))
    if lookahead is not None:
        speculative["copy_drafts"] = dict(COPY, max_drafts=lookahead)
    container = {"image": image}
    if tree is not None:
        container["expected_labels"] = {"local.spark-ds41f.vllm.tree": tree}
    return {
        "notes": notes,
        "extends": {"config": "config/cluster.json", "shape": SHAPES[shape]},
        "set": {"container": container},
        "serve_options": {
            "--max-num-batched-tokens": str(budget),
            "--max-cudagraph-capture-size": str(rows),
            "--compilation-config": json.dumps(compilation, separators=(",", ":")),
            "--long-prefill-token-threshold": threshold,
            "--speculative-config": json.dumps(speculative, separators=(",", ":")),
        },
    }


def agentx_arm(base: dict, shape: str) -> dict:
    """The arm with AgentX's serving edits, as experiments/2026-10-25-agentx's
    production-agentx.json makes them: the whole serve_args, port 8001 and the
    prompt-token details."""
    cluster = json.loads((ROOT / "config" / "cluster.json").read_text())
    resolved = recipe.overlay(base, recipe.resolve(cluster, SHAPES[shape]))
    args = list(resolved["serve_args"])
    args[args.index("--port") + 1] = "8001"
    args.insert(args.index("--enable-prefix-caching") + 1, "--enable-prompt-tokens-details")
    out = copy.deepcopy(base)
    out["notes"] = base["notes"] + " AgentX: port 8001 and --enable-prompt-tokens-details (#574)."
    out["set"]["serve_args"] = args
    out["set"]["deployment"] = {"ready_url": "http://127.0.0.1:8001/v1/models"}
    del out["serve_options"]
    return out


def main() -> None:
    image = sys.argv[1]
    tree = sys.argv[2] if len(sys.argv) > 2 else None
    arms = {}
    for shape in SHAPES:
        arms[f"control-{shape}"] = arm(shape, image, tree, None, (
            f"DSpark alone on the copy build at {SHAPES[shape]}: the recipe with the copy "
            "recipe's ladder and budget (the long-prefill threshold unchanged), no copy_drafts."))
        arms[f"copy-{shape}"] = arm(shape, image, tree, 16, (
            f"DSpark with copy drafts on the copy build at {SHAPES[shape]}: copy_drafts "
            "min_match 16, max_drafts 16, initial_drafts 8; the control's sizing."))
    arms["copy64-tp4"] = arm("tp4", image, tree, 64, (
        "The long-copy lab arm at tp4-ring on the same build: copy_drafts max_drafts 64, "
        "its own sizing (the ladder to max_num_seqs x 6 + 59 rows, the budget 59 more). "
        "The long-prefill threshold is the release's, so its text compares with every "
        "other arm's."))
    for name in ("control-tp4", "copy-tp4"):
        arms[f"agentx-{name}"] = agentx_arm(arms[name], "tp4")
    for name, body in arms.items():
        (HERE / f"{name}.json").write_text(json.dumps(body, indent=1) + "\n")

    determinism = [
        ["experiments/2026-10-12-r6g/token_determinism_text.py",
         ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
        ["experiments/2026-10-12-r6g/token_determinism_text.py",
         ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
        ["copy_identity.py", ["--prompts", "12"], "copy-identity"],
        ["copy_identity.py", ["--prompts", "12", "--thinking"], "copy-identity-thinking"],
        ["scripts/distinct_streams.py", ["--streams", "8", "--samples", "4", "--tokens", "256"],
         "c8-distinct"],
    ]
    specs = {}
    for shape in SHAPES:
        specs[f"hybrid-gate-{shape}"] = {
            "experiment": EXPERIMENT, "run": f"copy-drafter-gate-{shape}",
            "jobs": [{"kind": "sync"}, {
                "kind": "measure", "bench": False, "bracket": False, "extras": determinism,
                "arms": [{"config": f"control-{shape}.json", "label": "dspark"},
                         {"config": f"copy-{shape}.json", "label": "copy"},
                         {"config": f"copy-{shape}.json", "label": "copy-again"}]}]}
        measured = [{"config": f"copy-{shape}.json", "label": "copy"},
                    {"config": f"control-{shape}.json", "label": "dspark"}]
        if shape == "tp4":
            measured.append({"config": "copy64-tp4.json", "label": "copy64"})
        measured.append({"config": f"copy-{shape}.json", "label": "copy-again"})
        specs[f"hybrid-measure-{shape}"] = {
            "experiment": EXPERIMENT, "run": f"copy-drafter-measure-{shape}",
            "jobs": [{"kind": "sync"}, {
                "kind": "measure", "suite": ["release", "single-stream", "sampled"],
                "arms": measured}, {
                "kind": "measure", "bench": False, "bracket": False,
                "extras": [
                    ["experiments/2026-10-23-next-release/acceptance/copy_probe.py",
                     ["--prompts", "12"], "copy-probe"],
                    ["copy_identity.py", ["--prompts", "12", "--thinking"],
                     "copy-probe-thinking"]],
                "arms": measured}]}
    specs["agentx-tp4"] = {
        "experiment": EXPERIMENT, "run": "copy-drafter-agentx-tp4",
        "jobs": [{"kind": "agentx", "arm": {"config": f"agentx-{name}.json", "label": label},
                  "points": [1, 8], "duration": 3600, "seed": 42,
                  "dataset": "semianalysis_cc_traces_weka_062126"}
                 for name, label in (("copy-tp4", "copy"), ("control-tp4", "dspark"))]}
    for name, body in specs.items():
        (HERE / f"{name}.json").write_text(json.dumps(body, indent=1) + "\n")
    print(", ".join(sorted([*arms, *specs])))


if __name__ == "__main__":
    main()
