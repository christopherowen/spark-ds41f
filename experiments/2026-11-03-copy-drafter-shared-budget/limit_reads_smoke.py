"""Smoke check of overlay/limit-reads' counters: stale budget reads,
their resolution once the copy lands, the unresolved path and the periodic line."""
import logging
from types import SimpleNamespace

import numpy as np
import torch

from vllm.config.speculative import CopyDraftsConfig
from vllm.v1.worker.gpu.spec_decode import copy_drafts
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
from vllm.v1.worker.gpu.states import RequestState

states = RequestState(max_num_reqs=4, max_model_len=256, max_num_batched_tokens=256,
                      num_speculative_steps=16, vocab_size=1000, device=torch.device("cuda"))
drafter = copy_drafts.CopyDrafter(CopyDraftsConfig(min_match=16, max_drafts=16, initial_drafts=8),
                                  states, num_draft_steps=5, device=torch.device("cuda"))
states.add_request("a", 10, list(range(10)), 10, 100)
states.add_request("b", 10, list(range(10)), 10, 100)
slot_a, slot_b = states.req_id_to_index["a"], states.req_id_to_index["b"]

manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
manager.num_speculative_steps = 16
manager._stale_confidences = [SimpleNamespace(np=np.full((4, 16), 0.9, dtype=np.float32))]
manager._stale_idx = 0
manager.req_states = states
manager._max_total_logits = 1 << 20
manager.num_bonus_tokens = 1
manager.row_cost = 0.02
manager._slot_draft_limits = drafter.sync_slot_draft_limits


class Event:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done


lines = []
copy_drafts.logger.warning = lambda fmt, *args: lines.append(fmt % args)
mirror = drafter.slot_draft_limits.np
mirror[slot_a], mirror[slot_b] = 16, 5  # what the budget will read (stale)
staged = np.full(4, 16, dtype=np.int32)
staged[slot_a], staged[slot_b] = 8, 16  # the landed proposal: a narrower, b wider
event = Event()
drafter._limits_event, drafter._limits_staging_np = event, staged
manager.get_num_tokens({"a": 17, "b": 17}, {"a": [-1] * 16, "b": [-1] * 16})
r = drafter._reads
assert (r["reads"], r["queued"], r["stale"], r["resolved"]) == (1, 1, 1, 0), r
event.done = True
drafter.sync_slot_draft_limits()  # not the budget: publishes and resolves
assert (r["reads"], r["resolved"], r["consequential"]) == (1, 1, 1), r
assert (r["wide_slots"], r["placeholder_rows"]) == (1, 8), r
assert (r["narrow_slots"], r["dropped_drafts"]) == (1, 11), r
assert mirror[slot_a] == 8 and mirror[slot_b] == 16

# A stale read whose staging buffer is reused before its copy lands: unresolved.
late = Event()
drafter._limits_event, drafter._limits_staging_np = late, staged
manager.get_num_tokens({"a": 17}, {"a": [-1] * 16})
drafter._resolve_stale_reads(reused=staged)
assert r["unresolved"] == 1 and not drafter._stale_reads, r

# Fresh reads up to the periodic line.
drafter._limits_event = None
for _ in range(copy_drafts._READS_EVERY - r["reads"]):
    manager.get_num_tokens({"a": 17}, {"a": [-1] * 16})
assert len(lines) == 1 and lines[0].startswith("COPY-LIMITS reads=1024 queued=2 stale=2"), lines
print(lines[0])
print("limit-reads smoke OK")
