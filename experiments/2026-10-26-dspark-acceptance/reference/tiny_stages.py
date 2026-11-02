"""drafter_probe.py's batched drafter against the reference's sequential loop, stage by stage, on
the --tiny random model.

usage: python3 -I tiny_stages.py REFDIR    (REFDIR holds the snapshot's inference/ modules)

Prints, at one decode step: the context projection against the reference ring's contents, each
stage's attention output given the ring's own contents (bit-identical expected), and the drafts.
Then the whole run with one-row context projections (the loop's GEMM shapes), one anchor a pass.
"""
import sys
from pathlib import Path

PROBE = str(Path(__file__).resolve().parent)
sys.argv = ["drafter_probe.py", "--tiny", sys.argv[1], "/dev/null"]
sys.path.insert(0, PROBE)
import drafter_probe as P  # noqa: E402
import torch  # noqa: E402

M = P.M
torch.set_grad_enabled(False)
model = M.Transformer(P.ARGS)
P.load_module(model, "", skip=("vision.", "aligner.", "ffn.experts."))
g = torch.Generator().manual_seed(7)
seq = torch.randint(0, 3900, (200,), generator=g).tolist()
prompt = 140

captured = {}
orig = M.DSparkAttention.forward


def spy(self, x, start_pos, main_x):
    if start_pos > 0:
        captured.setdefault(start_pos, {})[self.layer_id] = {
            "x": x.clone(), "main_x": main_x.clone(),
        }
    out = orig(self, x, start_pos, main_x)
    if start_pos > 0:
        captured[start_pos][self.layer_id]["ring"] = self.window_kv_cache[:1].clone()
        captured[start_pos][self.layer_id]["out"] = out.clone()
    return out


M.DSparkAttention.forward = spy
with torch.inference_mode():
    _, _, mh = model(torch.tensor([seq[:prompt]]))
    model.forward_spec(torch.tensor([seq[prompt]]), mh)
    hidden = [mh[0]]
    ref = {}
    for i in range(prompt, len(seq) - 1):
        _, _, mh = model(torch.tensor([seq[i : i + 1]]), i)
        hidden.append(mh[0])
        out, logits, conf = model.forward_spec(torch.tensor([seq[i + 1]]), mh, i)
        ref[i + 1] = (out[0, 1:].tolist(), logits[0].clone(), conf[0].clone())
M.DSparkAttention.forward = orig
main_hidden = torch.cat(hidden, dim=0)

stages = P.build_stages()
first = stages[0]
with torch.inference_mode():
    main_x = first.main_norm(first.main_proj(main_hidden.unsqueeze(0)))[0]
    # main_x per position vs the reference's per-step main_x (decode positions)
    i = 170
    ref_mx = captured[i][P.ARGS.n_layers]["main_x"][0, 0]
    print("main_x diff at", i, (main_x[i].float() - ref_mx.float()).abs().max().item(), ref_mx.float().abs().max().item())
    for j, blk in enumerate(stages):
        lid = P.ARGS.n_layers + j
        ctx = P.context_kv(blk.attn, main_x)
        ring = captured[i][lid]["ring"][0]  # slots
        win = blk.attn.window_size
        pos = torch.arange(i - win + 1, i + 1)
        slots = pos % win
        d = (ctx[pos].float() - ring[slots].float()).abs().max().item()
        print(f"stage {j}: ctx vs ring max abs {d:.3g} (scale {ring.float().abs().max().item():.3g})")
        # attention with the reference's own x and ring contents, through my batched function
        x = captured[i][lid]["x"]  # [1, B, dim]
        ring_ctx = torch.zeros(i + 1, ring.size(-1), dtype=ring.dtype)
        ring_ctx[pos] = ring[slots]
        mine = P.dspark_attention_many(blk.attn, x.clone(), torch.tensor([i + 1]), ring_ctx)
        ref_out = captured[i][lid]["out"]
        print(f"stage {j}: attention out diff {(mine.float() - ref_out.float()).abs().max().item():.3g} "
              f"(scale {ref_out.float().abs().max().item():.3g})")
    # whole drafter at anchor i+1 from my path
    rec = P.draft(stages, main_hidden, seq, [i + 1], P.head_chunks())
    print("drafts mine", rec["drafts"][0], "ref", ref[i + 1][0], "margins", rec["margin"][0])

# Exactness: context projections one row at a time (the loop's M=1 GEMMs), one anchor a pass.
def ctx_rows(attn, main_x):
    rd = attn.rope_head_dim
    rows = []
    for p in range(main_x.size(0)):
        kv = attn.kv_norm(attn.wkv(main_x[p : p + 1].unsqueeze(0)))[0]
        P.rotate(kv[..., -rd:], attn.freqs_cis[p : p + 1])
        M.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
        rows.append(kv)
    return torch.cat(rows)


P.context_kv = ctx_rows
with torch.inference_mode():
    same = 0
    anchors = sorted(ref)
    # main_x one row at a time too
    first = stages[0]
    mx_rows = torch.cat([first.main_norm(first.main_proj(main_hidden[p : p + 1].unsqueeze(0)))[0] for p in range(main_hidden.size(0))])
    # feed main_x directly by bypassing main_proj: wrap draft's use via a hidden that projects to itself
    saved = (first.main_proj, first.main_norm)
    first.main_proj = torch.nn.Identity()
    first.main_norm = torch.nn.Identity()
    P.ARGV.anchor_chunk = 1
    rec = P.draft(stages, mx_rows, seq, anchors, P.head_chunks())
    first.main_proj, first.main_norm = saved
    for n, a in enumerate(anchors):
        same += rec["drafts"][n] == ref[a][0]
    err = max(abs(x - y) for n, a in enumerate(anchors) for x, y in zip(rec["confidence"][n], ref[a][2].float().tolist()))
    print("row-exact: drafts identical", same, "/", len(anchors), "max confidence logit error", err)
