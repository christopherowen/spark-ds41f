"""Write screen_zero_broadcast_sync_rowsfix_valid.py from the rowsfix screen: compare
only the sampled tokens rejection_sample wrote.

rejection_sample returns sampled = new_empty(num_reqs, num_steps + 1) and writes
row r's entries [0, num_sampled[r]) only; the rest is uninitialized memory, which
the worker never reads. The screen records the whole row. This copy records the
row with entries at or past num_sampled set to -1 (mismatch_sampled_steps), and the
raw row as before (mismatch_sampled_raw_steps), so both counts are reported."""
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


sub(
    "        hist_sampled = torch.empty((steps_verify, num_reqs, 1 + num_steps), dtype=torch.int64, device=device)\n",
    "        hist_sampled = torch.empty((steps_verify, num_reqs, 1 + num_steps), dtype=torch.int64, device=device)\n"
    "        hist_sampled_raw = torch.empty((steps_verify, num_reqs, 1 + num_steps), dtype=torch.int64, device=device)\n",
)
sub(
    "                num_accepted = (num_sampled - 1).clamp_min(0)\n",
    "                raw_sampled = sampled_tokens.clone()\n"
    "                written = (\n"
    "                    torch.arange(1 + num_steps, device=device)[None, :]\n"
    "                    < num_sampled.to(torch.int64)[:, None]\n"
    "                )\n"
    "                sampled_tokens = torch.where(\n"
    "                    written, sampled_tokens, torch.full_like(sampled_tokens, -1)\n"
    "                )\n"
    "                num_accepted = (num_sampled - 1).clamp_min(0)\n",
)
sub(
    "            hist_sampled[step].copy_(sampled_tokens.view(num_reqs, 1 + num_steps))\n",
    "            hist_sampled[step].copy_(sampled_tokens.view(num_reqs, 1 + num_steps))\n"
    "            hist_sampled_raw[step].copy_(\n"
    "                (sampled_tokens if is_greedy else raw_sampled).view(num_reqs, 1 + num_steps)\n"
    "            )\n",
)
sub(
    "        mm_s1, _ = count_step_mismatches(hist_sampled)\n",
    "        mm_s1, _ = count_step_mismatches(hist_sampled)\n"
    "        mismatch_sampled_raw_steps, _ = count_step_mismatches(hist_sampled_raw)\n",
)
sub(
    '            "mismatch_sampled_steps": mismatch_sampled_steps,\n',
    '            "mismatch_sampled_steps": mismatch_sampled_steps,\n'
    '            "unchecked_sampled_raw_steps": mismatch_sampled_raw_steps,\n',
)
open(dst_path, "w").write(s)
