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

``blockseen`` is the rule a shadow learner in serving can run: the block-maxima rule
on the positions up to each block's first miss only (``live_candidates``). Each
learner also gets the shadow score serving can compute from those maxima alone
beside the exact teacher-forced one, so the proxy is calibrated: ``proxy_accepted`` over the
seen positions only, and ``tf_proxy_accepted``, teacher-forced over every drafted position
from the same maxima, for the loaded and the learned table alike.

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


def markov_logits(base: torch.Tensor, prev: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                  rounded: bool = False):
    """base [R, V] BF16 plus the Markov bias of prev ([R]) through W1 (BF16) and W2; with
    ``rounded``, each sum rounded to BF16 as the serving step rounds it (its argmax then
    takes the first of tied tokens, as the step does)."""
    bias = w1[prev].to(torch.bfloat16) @ w2.t()
    z = base.float() + bias.float()
    return z.to(torch.bfloat16).float() if rounded else z


def chain(base: torch.Tensor, ids: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, *, teacher: bool,
          rounded: bool = False):
    """Argmax at each of the five positions: the chain's input at position i is the
    label before it (teacher) or the previous argmax (replay)."""
    rows, n = base.shape[0], base.shape[1]
    preds = torch.empty((rows, n), dtype=torch.int64, device=base.device)
    prev = ids[:, 0].long()
    for i in range(n):
        z = markov_logits(base[:, i], prev, w1, w2, rounded=rounded)
        preds[:, i] = z.argmax(-1)
        prev = ids[:, i + 1].long() if teacher else preds[:, i]
    return preds


def accepted(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Leading positions of each block whose prediction is its label."""
    ok = (preds == labels.long()) & (labels >= 0)
    return ok.int().cumprod(dim=1).sum(dim=1)


def block_candidates(z: torch.Tensor, shards: int = 4, block: int = 2048) -> torch.Tensor:
    """Each 2048-wide block's argmax within each vocabulary shard ([R, blocks]): the
    candidates the draft step's greedy partials already hold, at no extra compute."""
    vocab = z.shape[1]
    width = -(-vocab // shards)
    out = []
    for k in range(shards):
        lo, hi = k * width, min((k + 1) * width, vocab)
        for b in range(lo, hi, block):
            e = min(b + block, hi)
            out.append(z[:, b:e].argmax(-1) + b)
    return torch.stack(out, dim=1)


def ce_step(base, ids, labels, master, w2, lr, method="full", shards=4, chunk=256, live_w1=None,
            train_mask=None, accum=None):
    """One SGD step of the turn's cross-entropy into W1's rows: over the full
    vocabulary (``full``), or over each vocabulary shard's maximum and the committed
    token (``shard4``, the candidates serving's shard pairs already hold)."""
    rows, n = labels.shape
    prev_all = ids[:, :n].reshape(-1).long()
    y_all = labels.reshape(-1).long()
    z_rows = base.reshape(rows * n, -1)
    keep = y_all >= 0
    if train_mask is not None:
        keep = keep & train_mask.reshape(-1)
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
        if method == "live4":
            z_live = markov_logits(z_rows[sel], prev_all[sel], live_w1, w2)
            cands = torch.stack(
                [z_live[:, k * width : min((k + 1) * width, vocab)].argmax(-1) + k * width
                 for k in range(shards)], dim=1)
            cands = torch.cat([cands, y[:, None]], dim=1)
            zc = z.gather(1, cands)
            dup = (cands[:, :shards] == y[:, None])
            zc[:, :shards] = zc[:, :shards].masked_fill(dup, float("-inf"))
            logp = torch.log_softmax(zc, dim=-1)
            loss_sum += float(-logp[:, -1].sum())
            coef = logp.exp()
            coef[:, -1] -= 1.0
            grad_rows[s : s + chunk] = torch.einsum("rc,rcd->rd", coef, w2.float()[cands])
            continue
        if method == "block":
            # The live step's block maxima (loaded W1), and the label only when it is
            # among them: serving keeps no head logit for any other token.
            z_live = markov_logits(z_rows[sel], prev_all[sel], live_w1, w2)
            maxima = block_candidates(z_live, shards)
            present = (maxima == y[:, None]).any(dim=1)
            zc = z.gather(1, maxima)
            target = (maxima == y[:, None]).float().argmax(dim=1)
            logp = torch.log_softmax(zc, dim=-1)
            loss_sum += float(-(logp.gather(1, target[:, None]).squeeze(1) * present).sum())
            coef = logp.exp()
            coef[torch.arange(sel.numel(), device=coef.device), target] -= 1.0
            coef = coef * present[:, None]
            grad_rows[s : s + chunk] = torch.einsum("rc,rcd->rd", coef, w2.float()[maxima])
            continue
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
    if accum is None:
        master.index_add_(0, prev_all[idx], grad_rows, alpha=-lr)
        return loss_sum / idx.numel()
    # AdaGrad a row: the turn's summed gradient of each row, scaled by the root of
    # that row's running sum of squared gradient norms (frequent rows step less).
    rows_g = torch.zeros_like(master)
    rows_g.index_add_(0, prev_all[idx], grad_rows)
    touched = rows_g.abs().sum(dim=1) > 0
    accum[touched] += rows_g[touched].pow(2).sum(dim=1)
    scale = lr / (accum[touched].sqrt() + 1e-6)
    master[touched] -= scale[:, None] * rows_g[touched]
    return loss_sum / idx.numel()


def live_candidates(base, ids, labels, active, w1_loaded, w2):
    """What a shadow learner in serving sees of a turn: at each position, the live
    step's block maxima under the loaded table (the sums rounded to BF16, as the
    greedy partials keep them) and their values; the positions up to each block's
    first miss under those maxima, within its drafted positions (there the step's
    input was the committed token, so its maxima are teacher forcing's)."""
    rows, n = labels.shape
    out = []
    hits = torch.zeros((rows, n), dtype=torch.bool, device=base.device)
    for i in range(n):
        prev = ids[:, i].long()
        z_live = markov_logits(base[:, i], prev, w1_loaded, w2).to(torch.bfloat16).float()
        maxima = block_candidates(z_live)
        values = z_live.gather(1, maxima)
        pred = maxima.gather(1, values.argmax(dim=1, keepdim=True)).squeeze(1)
        hits[:, i] = (pred == labels[:, i].long()) & (labels[:, i] >= 0)
        out.append((prev, maxima, values))
    drafted = torch.arange(n, device=base.device)[None, :] < active[:, None]
    seen = torch.cat([torch.ones_like(hits[:, :1]), hits.int().cumprod(dim=1)[:, :-1].bool()], dim=1)
    return out, seen & drafted & (labels >= 0)


def proxy_accepted(cands, labels, mask, master, w1_loaded, w2):
    """The shadow score: the leading run of seen positions where the argmax over the
    live maxima of their values under the learner's table (the loaded values plus the
    change of the input's W1 row projected on the maxima's W2 rows) is the committed
    token. With the loaded table it is the drafts' accepted run."""
    rows, n = labels.shape
    hits = torch.zeros((rows, n), dtype=torch.bool, device=labels.device)
    for i, (prev, maxima, values) in enumerate(cands):
        v = values
        if master is not None:
            # As serving would round it with the learned table: the step's sum less
            # the loaded bias, plus the learned (BF16) row's bias, each bias rounded
            # to BF16 as the projection writes it, the sum rounded again.
            rows_w2 = w2[maxima].float()
            b0 = torch.einsum("rd,rcd->rc", w1_loaded[prev].float(), rows_w2).to(torch.bfloat16).float()
            bl = torch.einsum("rd,rcd->rc", master[prev].to(torch.bfloat16).float(), rows_w2)
            bl = bl.to(torch.bfloat16).float()
            v = (values - b0 + bl).to(torch.bfloat16).float()
        pred = maxima.gather(1, v.argmax(dim=1, keepdim=True)).squeeze(1)
        hits[:, i] = pred == labels[:, i].long()
    return (hits & mask).int().cumprod(dim=1).sum(dim=1)


def draft_candidates(base, ids, drafts, labels, active, w1_loaded, w2):
    """The live step's block maxima at every drafted position, as serving keeps them:
    under the step's own input (the anchor, then the previous draft), the sums rounded
    to BF16; and the positions a teacher-forced score reads (drafted and labeled)."""
    rows, n = labels.shape
    out = []
    for i in range(n):
        prev_d = ids[:, 0].long() if i == 0 else drafts[:, i - 1].long()
        z_live = markov_logits(base[:, i], prev_d, w1_loaded, w2, rounded=True)
        maxima = block_candidates(z_live)
        out.append((prev_d, ids[:, i].long(), maxima, z_live.gather(1, maxima)))
    drafted = torch.arange(n, device=base.device)[None, :] < active[:, None]
    return out, drafted & (labels >= 0)


def tf_proxy_accepted(dc, labels, valid, w1_loaded, w2, master=None):
    """The teacher-forced shadow score: at each drafted position, each live maximum's
    logit with the step's input bias replaced by the committed input's under a table
    (the loaded one, or the learner's as BF16), biases and sum rounded to BF16 as
    serving rounds them; the leading run of positions whose best maximum is the
    committed token. Both tables are scored by this one rule over the same maxima."""
    hits = []
    for prev_d, prev_l, maxima, values in dc:
        rows_w2 = w2[maxima].float()
        b0 = torch.einsum("rd,rcd->rc", w1_loaded[prev_d].float(), rows_w2).to(torch.bfloat16).float()
        src = w1_loaded[prev_l].float() if master is None else master[prev_l].to(torch.bfloat16).float()
        bt = torch.einsum("rd,rcd->rc", src, rows_w2).to(torch.bfloat16).float()
        z = (values - b0 + bt).to(torch.bfloat16).float()
        pred = maxima.gather(1, z.argmax(dim=1, keepdim=True)).squeeze(1)
        hits.append(pred == labels[:, len(hits)].long())
    return (torch.stack(hits, dim=1) & valid).int().cumprod(dim=1).sum(dim=1)


def cap_drift(master, loaded, cap):
    """Each row's displacement from the loaded row, at most ``cap`` of its norm."""
    if cap <= 0:
        return
    delta = master - loaded
    limit = cap * loaded.norm(dim=1, keepdim=True)
    norm = delta.norm(dim=1, keepdim=True)
    scale = torch.where(norm > limit, limit / norm.clamp_min(1e-12), torch.ones_like(norm))
    master.copy_(loaded + delta * scale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turns", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--lrs", default="0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--methods", default="full,shard4,block,live4")
    ap.add_argument("--opt", default="sgd", choices=("sgd", "adagrad"))
    ap.add_argument("--max-drift", type=float, default=0.0,
                    help="cap each row's displacement at this fraction of its loaded norm (0: none)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pattern", default="*.pt", help="turn files to read, in name order")
    args = ap.parse_args()
    device = "cuda"
    t, eps = load_tensors(args.snapshot, device)
    w1_loaded, w2 = t["w1"], t["w2"]
    lrs = [(m, float(x)) for m in args.methods.split(",") if m for x in args.lrs.split(",") if x]
    masters = {key: w1_loaded.float().clone() for key in lrs}
    accums = {key: (torch.zeros(w1_loaded.shape[0], device=device) if args.opt == "adagrad" else None)
              for key in lrs}
    files = sorted(glob.glob(os.path.join(args.turns, args.pattern)))
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
            # Positions up to each block's first miss under the loaded table: there the
            # chain's input is the committed token, so the draft step saw these maxima.
            tf_loaded = chain(base, ids, w1_loaded, w2, teacher=True)
            hit = ((tf_loaded == labels.long()) & (labels >= 0)).int()
            prefix_mask = torch.cat([torch.ones_like(hit[:, :1]), hit.cumprod(dim=1)[:, :-1]], dim=1).bool()
            pos = torch.arange(replay.shape[1], device=device)[None, :] < active[:, None]
            loaded_acc = accepted(chain(base, ids, w1_loaded, w2, teacher=True), labels)
            loaded_r = int(accepted(chain(base, ids, w1_loaded, w2, teacher=True, rounded=True), labels).sum())
            totals["rounded_loaded"] = totals.get("rounded_loaded", 0) + loaded_r
            # How much of the turn the compact candidates cover: the committed token among
            # the live step's block maxima, at each labeled position.
            n = labels.shape[1]
            flat_y = labels.reshape(-1).long()
            flat_prev = ids[:, :n].reshape(-1).long()
            flat_base = base.reshape(-1, base.shape[-1])
            lab = (flat_y >= 0).nonzero().squeeze(1)
            covered = 0
            for c0 in range(0, lab.numel(), 256):
                sel = lab[c0 : c0 + 256]
                z_live = markov_logits(flat_base[sel], flat_prev[sel], w1_loaded, w2)
                covered += int((block_candidates(z_live) == flat_y[sel][:, None]).any(dim=1).sum())
            row = {"file": os.path.basename(path), "blocks": int(labels.shape[0]),
                   "labeled": int((labels >= 0).sum()),
                   "replay_equal": float(((replay == drafts) | ~pos).all(dim=1).float().mean()),
                   "loaded": int(loaded_acc.sum()),
                   "label_in_block_maxima": covered}
            totals["covered"] = totals.get("covered", 0) + covered
            totals["replay_checked"] += int(pos.sum())
            totals["replay_equal"] += int(((replay == drafts) & pos).sum())
            cands, seen = live_candidates(base, ids, labels, active, w1_loaded, w2)
            dc, tf_valid = draft_candidates(base, ids, drafts, labels, active, w1_loaded, w2)
            tf_loaded = int(tf_proxy_accepted(dc, labels, tf_valid, w1_loaded, w2).sum())
            row["tf_proxy_loaded"] = tf_loaded
            totals["tf_proxy_loaded"] = totals.get("tf_proxy_loaded", 0) + tf_loaded
            proxy_loaded = int(proxy_accepted(cands, labels, seen, None, w1_loaded, w2).sum())
            row["proxy_loaded"] = proxy_loaded
            totals["proxy_loaded"] = totals.get("proxy_loaded", 0) + proxy_loaded
            totals["seen"] = totals.get("seen", 0) + int(seen.sum())
            for key in lrs:
                name = f"{key[0]}:{key[1]}"
                w1 = masters[key].to(torch.bfloat16)
                acc = accepted(chain(base, ids, w1, w2, teacher=True), labels)
                row[name] = int(acc.sum())
                totals["learned"][name] += int(acc.sum())
                acc_r = int(accepted(chain(base, ids, w1, w2, teacher=True, rounded=True), labels).sum())
                row["rounded " + name] = acc_r
                totals.setdefault("rounded", {}).setdefault(name, 0)
                totals["rounded"][name] += acc_r
                proxy = int(proxy_accepted(cands, labels, seen, masters[key], w1_loaded, w2).sum())
                row["proxy " + name] = proxy
                totals.setdefault("proxy", {}).setdefault(name, 0)
                totals["proxy"][name] += proxy
                tfp = int(tf_proxy_accepted(dc, labels, tf_valid, w1_loaded, w2, masters[key]).sum())
                row["tf_proxy " + name] = tfp
                totals.setdefault("tf_proxy", {}).setdefault(name, 0)
                totals["tf_proxy"][name] += tfp
            for key in lrs:
                name = f"{key[0]}:{key[1]}"
                mask = {"live4": prefix_mask, "blockseen": seen}.get(key[0])
                method = "block" if key[0] == "blockseen" else key[0]
                row["loss " + name] = round(ce_step(base, ids, labels, masters[key], w2, key[1], method=method, live_w1=w1_loaded,
                                                      train_mask=mask, accum=accums[key]), 4)
                cap_drift(masters[key], w1_loaded.float(), args.max_drift)
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
    summary["label_in_block_maxima"] = totals.get("covered", 0) / max(totals["labeled"], 1)
    summary["gain_pct"] = {k: 100.0 * (v - totals["loaded"]) / max(totals["loaded"], 1)
                           for k, v in totals["learned"].items()}
    base_r = max(totals.get("rounded_loaded", 0), 1)
    summary["rounded_gain_pct"] = {k: 100.0 * (v - base_r) / base_r
                                   for k, v in totals.get("rounded", {}).items()}
    base_tf = max(totals.get("tf_proxy_loaded", 0), 1)
    summary["tf_proxy_gain_pct"] = {k: 100.0 * (v - base_tf) / base_tf
                                    for k, v in totals.get("tf_proxy", {}).items()}
    base_proxy = max(totals.get("proxy_loaded", 0), 1)
    summary["proxy_gain_pct"] = {k: 100.0 * (v - base_proxy) / base_proxy
                                 for k, v in totals.get("proxy", {}).items()}
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
