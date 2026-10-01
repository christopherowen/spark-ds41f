"""attn-exact5 -> attn-exact6: record the LM head's input rows and logits (tags 33, 34) and each
step's logit-row batch indices, so the analysis follows a row through to its logits."""
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
shutil.copytree(src, dst, dirs_exist_ok=True)


def edit(name, subs, append=None):
    path = f"{dst}/{name}"
    s = open(path).read()
    for old, new in subs:
        assert s.count(old) == 1, (name, old, s.count(old))
        s = s.replace(old, new)
    if append:
        s = s.rstrip("\n") + "\n" + append
    open(path, "w").write(s)


edit("checksum_debug.py", [(
    "def record_ced(indices: torch.Tensor) -> None:\n",
    '''def record_logit_rows(indices: torch.Tensor) -> None:
    """Record the batch rows whose logits the step computes (the LM head's rows)."""
    n = min(int(indices.numel()), _ROWS)
    log = _log("logit", indices.device, 2 + _ROWS, _SCHEDULE_CAPACITY, _ROWS)
    stage = log.stage[0]
    stage.zero_()
    stage[0].fill_(_step[0] - 1)
    stage[1].fill_(int(indices.numel()))
    stage[2 : 2 + n].copy_(indices.reshape(-1)[:n])
    log.append(0)


def record_ced(indices: torch.Tensor) -> None:
''')])
edit("model_runner.py", [(
    "                self.input_buffers.positions, self.input_buffers.input_ids,\n"
    "            )\n",
    "                self.input_buffers.positions, self.input_buffers.input_ids,\n"
    "            )\n"
    "            checksum_debug.record_logit_rows(logits_indices)\n",
)])
edit("model41.py", [], append='''

# Debug (SPARK3_MOE_CHECKSUM_DIR): the LM head's input rows and logits, as layer 41.
_LM_HEAD_OWNER = type("LmHead", (), {"prefix": "language_model.model.layers.41.lm_head"})()
_upstream_compute_logits = DeepseekV41LLMForCausalLM.compute_logits


def _traced_compute_logits(self, hidden_states):
    from vllm.model_executor.layers.fused_moe.runner import checksum_debug

    logits = _upstream_compute_logits(self, hidden_states)
    if checksum_debug.ENABLED and logits is not None:
        checksum_debug.record_attn(_LM_HEAD_OWNER, 33, hidden_states)
        checksum_debug.record_attn(_LM_HEAD_OWNER, 34, logits)
    return logits


DeepseekV41LLMForCausalLM.compute_logits = _traced_compute_logits
''')
print("ok")
