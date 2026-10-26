#!/usr/bin/env python3
"""One prompt under several chunkings through a decoder stack on the TileLang kernels:
are the rows the first decode step and the drafter read the single chunk's, bit for
bit?

usage: chunking_check.py [--expect identical|record]

The stack is vLLM's tests/kernels/attention/test_deepseek_v41_ced_replay.py (mounted
at /b/ced_replay_stack.py): the model state's CED staging, the decoder's token
metadata, the window metadata and the sparse MLA, over three layers. It runs on
whichever vLLM tree the bundle mounts. On next-candidate the model state takes the
recipe's nonfinal-skip policy (DSpark at five drafts: 199 rows, chunks aligned at the
threshold); on ced-prompt-anchor it has none.

Prints one JSON line a case and a summary. Each line carries a SHA-256 of the rows and
of the single chunk's, so two trees' runs compare bit for bit. ``--expect identical``
exits 1 unless every case is bit-identical to its single chunk and finite; ``record``
only records.
"""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/b")

import ced_replay_stack as stack  # noqa: E402
from vllm.models.deepseek_v4_1.nvidia import model_state as model_state_module  # noqa: E402

WINDOW = stack.CED_WINDOW
PER_CHUNK = hasattr(model_state_module, "_nonfinal_skip_policy")
TP3, TP4 = 8192 - 8 * 6, 16384 - 16 * 6


def stager(prefix, threshold):
    cls = model_state_module.DeepseekV41ModelState
    state = cls.__new__(cls)
    state.ced_state = stack.CEDState(4, 8192, (128, 256, 384, 512, 8192), "cuda")
    state._ced_prompt_logprobs, state._ced_prefix_start = set(), {"prompt": prefix}
    if PER_CHUNK:
        state._ced_skip_min_rows, state._ced_chunk_align = 199, threshold
        state._ced_mm_requests, state._ced_expected_chunk = set(), {}

    def stage(start, rows, prompt_end, starts, seq):
        state._stage_ced(
            SimpleNamespace(
                num_reqs=1,
                req_ids=["prompt"],
                query_start_loc=starts,
                seq_lens=seq,
                num_scheduled_tokens=np.array([rows]),
                num_computed_prefill_tokens_np=np.array([start]),
                prefill_len_np=np.array([prompt_end]),
                has_prefill=True,
            )
        )
        return state.ced_state

    return stage


def digest(rows) -> str:
    return hashlib.sha256(rows.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def run(prompt_end, threshold, prefix):
    chunks = stack.aligned_chunks(threshold, prefix, prompt_end)
    outputs = stack.decoder_stack(prompt_end, chunks, stager(prefix, threshold))
    return chunks, stack.replayed(outputs, max(prefix, prompt_end - WINDOW), prompt_end)


def main() -> int:
    expect = sys.argv[sys.argv.index("--expect") + 1] if "--expect" in sys.argv else "record"
    cases = []
    # Small prompts: 1,024 rows plus a final chunk of 1, 64, 128 or 129 rows.
    for final in (1, 64, 128, 129):
        cases += [(1024 + final, t, h) for t, h in ((512, 0), (384, 0), (512, 448))]
    # A short first chunk after a prefix hit that the floor cuts (64 rows keep 40).
    cases.append((600, 512, 448))
    # The serving shapes' thresholds: TP3 (8,144) and TP4 (16,288).
    for final in (1, 64, 128, 129):
        cases += [(TP3 + final, TP3, 0), (TP4 + final, TP3, 0), (TP4 + final, TP4, 0)]
    singles = {}
    rows_by_case = {}
    failures = 0
    for prompt_end, threshold, prefix in cases:
        if prompt_end not in singles:
            singles[prompt_end] = run(prompt_end, 1 << 30, 0)[1]
        chunks, rows = run(prompt_end, threshold, prefix)
        rows_by_case[(prompt_end, threshold, prefix)] = rows
        single = singles[prompt_end]
        identical = torch.equal(rows, single)
        finite = bool(torch.isfinite(rows).all())
        failures += not (identical and finite)
        print(json.dumps({
            "tree": "next-candidate" if PER_CHUNK else "ced-prompt-anchor",
            "prompt": prompt_end, "threshold": threshold, "prefix": prefix,
            "chunks": [rows for _, rows in chunks],
            "identical_to_single_chunk": identical, "finite": finite,
            "max_abs_difference": float((rows - single).abs().max()),
            "rows_differing": int((rows != single).any(-1).sum()),
            "digest": digest(rows), "single_chunk_digest": digest(single),
        }), flush=True)
    # TP3 against TP4 on the same prompts.
    for final in (1, 64, 128, 129):
        a = rows_by_case[(TP4 + final, TP3, 0)]
        b = rows_by_case[(TP4 + final, TP4, 0)]
        print(json.dumps({"tp3_vs_tp4_prompt": TP4 + final,
                          "identical": torch.equal(a, b)}), flush=True)
    print(json.dumps({"summary": "chunking check", "per_chunk_anchor": PER_CHUNK,
                      "cases": len(cases), "not_identical_or_not_finite": failures}),
          flush=True)
    return 1 if expect == "identical" and failures else 0


if __name__ == "__main__":
    sys.exit(main())
