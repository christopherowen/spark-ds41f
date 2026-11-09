#!/usr/bin/env python3
"""make_r1.py TREE: in a vLLM checkout at original #971 (0782 applied on 0781), rewrite the
working tree to #971 round 1: keep the copy_apply survival gate and the single-request tile cap,
drop the dead-row zero-confidence floor (rationale 3) and its tests. Exact-text edits; every
edit must match once. The files 0782 does not change keep HEAD's content."""
import subprocess, sys
from pathlib import Path

tree = Path(sys.argv[1])
def show(rev, path):
    return subprocess.run(["git", "-C", str(tree), "show", f"{rev}:{path}"], check=True,
                          capture_output=True, text=True).stdout
def sub(text, old, new, path):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one match, found {n}: {old[:80]!r}")
    return text.replace(old, new)

# 1. model_runner.py and test_prepare_kernels.py: 0782's only hunks there are the dead-row ones
#    (copy drafter declares dead rows; the test of that declaration). Restore 0781's versions.
for p in ("vllm/v1/worker/gpu/model_runner.py", "tests/v1/worker/test_prepare_kernels.py"):
    (tree / p).write_text(show("HEAD~1", p))

# 2. adaptive_verification.py: drop _ZERO_CONFIDENCE_LOG_TAU and restore dead_row_args.
p = "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
t = show("HEAD", p)
t = sub(t, "SINGLE_REQ_TILE_SPILL_SCALE = 2.0\n_ZERO_CONFIDENCE_LOG_TAU = -1e30\n",
        "SINGLE_REQ_TILE_SPILL_SCALE = 2.0\n", p)
new_dead = '''        """Arguments for the fused dead-row marking, or None when disabled.

        The confidences are the current drafts', broadcast from rank 0, so
        every rank marks the same rows. On wide copy-draft steps
        (``step_steps > num_draft_steps``), trailing ``-1`` placeholder rows
        from a shrunk proposal carry ``confidence == 0`` (``log(0) = -inf``),
        so a finite floor ``_ZERO_CONFIDENCE_LOG_TAU`` masks them as padding
        even when ``dead_rows_tau`` is 0.
        """
        if not getattr(self, "_has_admitted_drafts", True):
            return None
        tau = self.effective_dead_rows_tau()
        has_copy_tail = getattr(self, "step_steps", 0) > getattr(
            self, "num_draft_steps", self.num_speculative_steps
        )
        if tau <= 0 and not has_copy_tail:
            return None
        log_tau = float(np.log(tau)) if tau > 0 else _ZERO_CONFIDENCE_LOG_TAU
        return DeadRows(
            self._confidence_probs,
            is_padding,
            self._dead_row_cut[:num_reqs],
            log_tau,
        )
'''
old_dead = '''        """Arguments for the fused dead-row marking, or None when disabled.

        The confidences are the current drafts', broadcast from rank 0, so
        every rank marks the same rows.
        """
        tau = self.effective_dead_rows_tau()
        if tau <= 0 or not getattr(self, "_has_admitted_drafts", True):
            return None
        return DeadRows(
            self._confidence_probs,
            is_padding,
            self._dead_row_cut[:num_reqs],
            float(np.log(tau)),
        )
'''
base = show("HEAD~1", p)
assert base.count(old_dead) == 1, "0781's dead_row_args text not found"
t = sub(t, new_dead, old_dead, p)
(tree / p).write_text(t)

# 3. test_adaptive_verification.py: keep the tile-cap test's assertions verbatim; drop its
#    dead_row_args block (the removed part) and the name's "masks_shrunk_tail".
p = "tests/v1/spec_decode/test_adaptive_verification.py"
t = show("HEAD", p)
t = sub(t, "def test_single_request_tile_spill_caps_16th_copy_draft_at_16_rows_and_masks_shrunk_tail() -> None:\n",
        "def test_single_request_tile_spill_caps_16th_copy_draft_at_16_rows() -> None:\n", p)
dead_block = '''    assert total_c2 == 34

    # On wide copy steps (step_steps=16 > num_draft_steps=5), dead_row_args returns a
    # sentinel log_tau=-1e30 even when dead_rows_tau is 0.0 so -1 placeholder rows
    # (confidence 0.0 -> log(0) = -inf < -1e30) are masked as padding.
    mgr_c1._confidence_probs = torch.ones((1, 16), dtype=torch.float32)
    mgr_c1._dead_row_cut = torch.zeros(1, dtype=torch.int32)
    mgr_c1._has_admitted_drafts = True
    is_padding = torch.zeros(17, dtype=torch.bool)
    mgr_c1.step_steps = 5
    assert mgr_c1.dead_row_args(is_padding, 1) is None
    mgr_c1.step_steps = 16
    dead = mgr_c1.dead_row_args(is_padding, 1)
    assert dead is not None
    assert dead.log_tau == pytest.approx(-1e30, rel=1e-5)
'''
t = sub(t, dead_block, "    assert total_c2 == 34\n", p)
(tree / p).write_text(t)
print("r1 edits applied")
