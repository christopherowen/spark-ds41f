"""Write screen_zero_broadcast_sync_rowsfix.py: PR #980's screen with one change, the
rejection_sample call's draft_sampled laid out as [num_logits] as the worker does."""
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()
old = """                sampled_tokens, num_sampled = rejection_sample(
                    proc_logits,
                    draft_logits,
                    draft_tokens.view(-1),
"""
new = """                # rejection_sample reads draft_sampled as [num_logits], each
                # request's drafts at its rows 1..num_steps (the worker passes
                # input_ids[logits_indices]); draft_tokens.view(-1) is only
                # num_reqs * num_steps long.
                draft_rows = torch.zeros(
                    (num_reqs, 1 + num_steps), dtype=draft_tokens.dtype, device=device
                )
                draft_rows[:, 1:] = draft_tokens
                sampled_tokens, num_sampled = rejection_sample(
                    proc_logits,
                    draft_logits,
                    draft_rows.view(-1),
"""
assert src.count(old) == 1, "call site not found exactly once"
open(dst_path, "w").write(src.replace(old, new))
