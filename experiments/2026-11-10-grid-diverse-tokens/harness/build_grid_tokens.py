"""Build the engine grid's fixed token sample (servebench/grid_tokens.json) and the
reference forward's sequences from one recorded model step of the servebench corpus.

usage: build_grid_tokens.py MODEL_JSON_GZ GRID_TOKENS_OUT SEQUENCES_OUT [--seed 0]

The source is a ``servebench model`` result (``servebench-model/1``) over the default
prompt set: every prompt's rendered ids and the server's greedy reply. A seeded draw picks
16 distinct prompts. Request ``r`` of the grid reads 16 consecutive reply tokens of its
prompt, the ones at absolute positions 64-79 of prompt + reply, so every window sits past
its prompt (41-55 tokens) inside the reply: decode rows, as serving verifies them.

The reference sequences are the same 16 prompts' first 80 tokens (prompt, then reply up to
the window's end) and one sequence of token 0 (the grid's old rows), for ref_routes.py:
routes at positions 64-79 are the routes the window's rows take after their real context.
"""

import argparse
import gzip
import hashlib
import json
import random

REQUESTS = 16
WINDOW = 16
CONTEXT = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("grid_tokens")
    ap.add_argument("sequences")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raw = open(args.source, "rb").read()
    run = json.loads(gzip.decompress(raw))
    assert run["schema"] == "servebench-model/1", run["schema"]
    assert run["workload"]["prompts"] == "default", run["workload"]
    sequences = run["generation"]["sequences"]
    usable = [i for i, s in enumerate(sequences)
              if len(s["prompt_ids"]) < CONTEXT and len(s["prompt_ids"]) + len(s["output_ids"]) >= CONTEXT + WINDOW]
    picked = random.Random(args.seed).sample(usable, REQUESTS)
    windows, refs = [], []
    for i in picked:
        s = sequences[i]
        full = list(s["prompt_ids"]) + list(s["output_ids"])
        windows.append({"prompt": s["id"], "offset": CONTEXT - len(s["prompt_ids"]),
                        "token_ids": full[CONTEXT:CONTEXT + WINDOW]})
        refs.append(full[:CONTEXT + WINDOW])
    grid = {
        "version": 2,
        "description": ("The engine grid's decode rows (grid version 2): request r of a shape takes "
                        "the first rows of windows[r % 16], 16 consecutive tokens of the server's "
                        "greedy reply to one default prompt, so the MoE routes the grid's rows as it "
                        "routes text. Fixed: every run and every build feeds these ids."),
        "source": {"result": "spark-ds41f-1107 release suite (TP4), model step",
                   "label": run.get("label"), "started_utc": run.get("started_utc"),
                   "file_sha256": hashlib.sha256(raw).hexdigest(),
                   "prompts": run["workload"]["prompts"], "prompts_sha256": run["workload"]["prompts_sha256"],
                   "seed": args.seed, "requests": REQUESTS, "window": WINDOW,
                   "first_position": CONTEXT},
        "windows": windows,
    }
    with open(args.grid_tokens, "w") as fh:
        json.dump(grid, fh, indent=1)
        fh.write("\n")
    with open(args.sequences, "w") as fh:
        json.dump({"context": CONTEXT, "window": WINDOW, "prompts": [w["prompt"] for w in windows],
                   "sequences": refs + [[0] * (CONTEXT + WINDOW)],
                   "labels": [f"text-{w['prompt']}" for w in windows] + ["token0"]}, fh)


if __name__ == "__main__":
    main()
