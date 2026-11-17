"""Offline test-then-train study of end-of-turn Markov W1 learning on dumped turns.

Reads the turns the observe stage wrote (``VLLM_DSPARK_TURN_LEARNING_DUMP``), in the
order they finished, with the checkpoint's own drafter tensors: the shared head, the
drafter's final norm and its Markov W1 and W2. For each turn:

1. the replay check: the Markov chain over the kept rows with the drafted tokens as its
   inputs, against the drafts the serving step made (the single-GPU arithmetic here is
   not the serving kernels', so near ties can differ; the agreement rate is reported);
2. teacher-forced acceptance (leading positions of each five-token block whose argmax is
   the committed token) under the loaded W1 and under each learner's W1, before the
   learner sees the turn;
3. each learner's update on the turn: full-vocabulary cross-entropy of the committed
   token, gradient into the W1 rows of the chain's inputs, on an FP32 master copy.

Prints one JSON line a turn and a summary line. Nothing is written elsewhere.
"""

import argparse
import glob
import json
import math
import os

import torch
from safetensors import safe_open

H_TENSORS = {
    "head": "head.weight",
    "norm": "mtp.2.norm.weight",
    "w1": "mtp.2.markov_head.embed.weight",
    "w2": "mtp.2.markov_head.head.weight",
}


def load_tensors(snapshot: str, device: str) -> tuple[dict, float]:
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for key, name in H_TENSORS.items():
        with safe_open(os.path.join(snapshot, index[name]), "pt", device="cpu") as f:
            out[key] = f.get_tensor(name).to(device)
    config = json.load(open(os.path.join(snapshot, "config.json")))
    eps = float(config.get("text_config", config).get("rms_norm_eps", 1e-6))
    return out, eps


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(torch.bfloat16)


def base_logits(hidden: torch.Tensor, t: dict, eps: float, chunk: int = 320) -> torch.Tensor:
    rows = hidden.reshape(-1, hidden.shape[-1])
    out = []
    for s in range(0, rows.shape[0], chunk):
        normed = rms_norm(rows[s : s + chunk], t["norm"], eps)
        out.append(normed @ t["head"].t())  # BF16 logits, FP32 accumulation
    return torch.cat(out).view(*hidden.shape[:-1], -1)


def markov_logits(base: torch.Tensor, prev: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor):
    """base [R, V] BF16 plus the Markov bias of prev ([R]) through W1 (BF16) and W2."""
    bias = w1[prev].to(torch.bfloat16) @ w2.t()
    return base.float() + bias.float()


def chain(base: torch.Tensor, ids: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, *, teacher: bool):
    """Argmax at each of the five positions: the chain's input at position i is the
    label before it (teacher) or the previous argmax (replay)."""
    rows, n = base.shape[0], base.shape[1]
    preds = torch.empty((rows, n), dtype=torch.int64, device=base.device)
    prev = ids[:, 0].long()
    for i in range(n):
        z = markov_logits(base[:, i], prev, w1, w2)
        preds[:, i] = z.argmax(-1)
        prev = ids[:, i + 1].long() if teacher else preds[:, i]
    return preds


def accepted(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Leading positions of each block whose prediction is its label."""
    ok = (preds == labels.long()) & (labels >= 0)
    return ok.int().cumprod(dim=1).sum(dim=1)


def ce_step(base, ids, labels, master, w2, lr, method="full", shards=4, chunk=256):
    """One SGD step of the turn's cross-entropy into W1's rows: over the full
    vocabulary (``full``), or over each vocabulary shard's maximum and the committed
    token (``shard4``, the candidates serving's shard pairs already hold)."""
    rows, n = labels.shape
    prev_all = ids[:, :n].reshape(-1).long()
    y_all = labels.reshape(-1).long()
    z_rows = base.reshape(rows * n, -1)
    keep = y_all >= 0
    idx = keep.nonzero().squeeze(1)
    if idx.numel() == 0:
        return 0.0
    w1 = master.to(torch.bfloat16)
    loss_sum = 0.0
    grad_rows = torch.zeros((idx.numel(), master.shape[1]), dtype=torch.float32, device=master.device)
    for s in range(0, idx.numel(), chunk):
        sel = idx[s : s + chunk]
        z = markov_logits(z_rows[sel], prev_all[sel], w1, w2)
        y = y_all[sel]
        if method == "full":
            logp = torch.log_softmax(z, dim=-1)
            loss_sum += float(-logp.gather(1, y[:, None]).sum())
            p = logp.exp()
            p[torch.arange(sel.numel(), device=p.device), y] -= 1.0
            grad_rows[s : s + chunk] = p @ w2.float()
            continue
        vocab = z.shape[1]
        width = -(-vocab // shards)
        cands = torch.stack(
            [z[:, k * width : min((k + 1) * width, vocab)].argmax(-1) + k * width for k in range(shards)],
            dim=1,
        )
        cands = torch.cat([cands, y[:, None]], dim=1)  # [R, shards + 1]
        zc = z.gather(1, cands)
        dup = (cands[:, :shards] == y[:, None])  # the label among the maxima: count it once
        zc[:, :shards] = zc[:, :shards].masked_fill(dup, float("-inf"))
        logp = torch.log_softmax(zc, dim=-1)
        loss_sum += float(-logp[:, -1].sum())
        coef = logp.exp()
        coef[:, -1] -= 1.0
        grad_rows[s : s + chunk] = torch.einsum("rc,rcd->rd", coef, w2.float()[cands])
    master.index_add_(0, prev_all[idx], grad_rows, alpha=-lr)
    return loss_sum / idx.numel()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turns", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--lrs", default="0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--methods", default="full,shard4")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    t, eps = load_tensors(args.snapshot, device)
    w1_loaded, w2 = t["w1"], t["w2"]
    lrs = [(m, float(x)) for m in args.methods.split(",") if m for x in args.lrs.split(",") if x]
    masters = {key: w1_loaded.float().clone() for key in lrs}
    files = sorted(glob.glob(os.path.join(args.turns, "*.pt")))
    if args.limit:
        files = files[: args.limit]
    print(json.dumps({
        "turn_files": len(files),
        "w1_row_norm_median": float(w1_loaded.float().norm(dim=1).median()),
        "w2_row_norm_median": float(w2.float().norm(dim=1).median()),
        "eps": eps,
    }), flush=True)
    totals = {"blocks": 0, "labeled": 0, "loaded": 0, "replay_checked": 0, "replay_equal": 0,
              "learned": {f"{m}:{lr}": 0 for m, lr in lrs}}
    tokens = 0
    for path in files:
        rec = torch.load(path, map_location="cpu")
        hidden = rec["hidden"].to(device)
        ids = rec["ids"].to(device)
        labels = rec["labels"].to(device)
        drafts = rec["drafts"].to(device).long()
        active = rec["meta"][:, 1].to(device)
        keep = (labels >= 0).any(dim=1)
        if not bool(keep.any()):
            continue
        hidden, ids, labels, drafts, active = hidden[keep], ids[keep], labels[keep], drafts[keep], active[keep]
        with torch.no_grad():
            base = base_logits(hidden, t, eps)
            replay = chain(base, ids, w1_loaded, w2, teacher=False)
            pos = torch.arange(replay.shape[1], device=device)[None, :] < active[:, None]
            loaded_acc = accepted(chain(base, ids, w1_loaded, w2, teacher=True), labels)
            row = {"file": os.path.basename(path), "blocks": int(labels.shape[0]),
                   "labeled": int((labels >= 0).sum()),
                   "replay_equal": float(((replay == drafts) | ~pos).all(dim=1).float().mean()),
                   "loaded": int(loaded_acc.sum())}
            totals["replay_checked"] += int(pos.sum())
            totals["replay_equal"] += int(((replay == drafts) & pos).sum())
            for key in lrs:
                name = f"{key[0]}:{key[1]}"
                w1 = masters[key].to(torch.bfloat16)
                acc = accepted(chain(base, ids, w1, w2, teacher=True), labels)
                row[name] = int(acc.sum())
                totals["learned"][name] += int(acc.sum())
            for key in lrs:
                name = f"{key[0]}:{key[1]}"
                row["loss " + name] = round(ce_step(base, ids, labels, masters[key], w2, key[1], method=key[0]), 4)
                drift = (masters[key] - w1_loaded.float()).norm(dim=1) / w1_loaded.float().norm(dim=1).clamp_min(1e-6)
                row["drift " + name] = round(float(drift.max()), 4)
        totals["blocks"] += row["blocks"]
        totals["labeled"] += row["labeled"]
        totals["loaded"] += row["loaded"]
        tokens += row["labeled"]
        row["cum_tokens"] = tokens
        row["cum_loaded"] = totals["loaded"]
        row["cum_learned"] = {k: v for k, v in totals["learned"].items()}
        print(json.dumps(row), flush=True)
    summary = dict(totals)
    summary["replay_agreement"] = totals["replay_equal"] / max(totals["replay_checked"], 1)
    summary["gain_pct"] = {k: 100.0 * (v - totals["loaded"]) / max(totals["loaded"], 1)
                           for k, v in totals["learned"].items()}
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
