"""Model behaviour: what a build computes, judged without timing.

- **Generation.** Every prompt once, greedily, at a fixed concurrency, recording each
  request's token ids, with the server's speculative-decoding counters per draft position
  read before and after. On a batch-invariant server greedy decoding makes the text, and
  with it the acceptance, a property of the build rather than of the run.
- **Teacher-forced scoring.** Token sequences (by default a base run's prompts and
  responses) are sent as prompts with ``prompt_logprobs``. For each response position it
  records the given token's log-probability and rank, plus the top K. Every build scores the
  same tokens, so two builds compare position by position: KL divergence, top-1 agreement,
  perplexity. A build scoring its own greedy text should rank every given token first,
  which checks that the prefill path agrees with the decode path that wrote it.

KL divergence is estimated from each position's top K, the rest of the vocabulary being
one bucket; a token outside the candidate's top K is given the candidate's K-th
log-probability, an upper bound, so the estimate errs low.
"""

import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request
import uuid

from . import prom, stats

SCHEMA = "servebench-model/1"
ACCEPTED_PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos"
DRAFTED_PER_POSITION = "vllm:spec_decode_num_draft_tokens_per_pos"


def _post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _in_parallel(items: list, work, concurrency: int) -> list:
    """``work`` over ``items`` with ``concurrency`` in flight, each starting as one ends;
    results in item order."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(work, items))


def _positions(base_url: str) -> dict | None:
    """Per-position counters; Prometheus clients may expose them with a ``_total`` suffix."""
    text = prom.fetch(base_url)
    if text is None:
        return None
    out = {}
    for key, name in (("accepted", ACCEPTED_PER_POSITION), ("drafted", DRAFTED_PER_POSITION)):
        out[key] = prom.by_label(text, name + "_total", "position") or prom.by_label(text, name, "position")
    return out


def acceptance(start: dict | None, end: dict | None, positions_start: dict | None,
               positions_end: dict | None) -> dict:
    """Counter deltas as acceptance: mean acceptance length (tokens per verification
    step), draft tokens verified per step, and per draft position the acceptance rate and
    how often that position was verified at all."""
    if not start or not end:
        return {}
    delta = {key: end[key] - start[key] for key in end if key in start}
    drafts = delta.get("drafts", 0)
    out = {key: delta.get(key) for key in ("drafts", "draft_tokens", "accepted_tokens",
                                           "generation_tokens", "steps")}
    if drafts:
        out["acceptance_length"] = 1 + delta.get("accepted_tokens", 0) / drafts
        out["verified_per_draft"] = delta.get("draft_tokens", 0) / drafts
    if delta.get("draft_tokens"):
        out["accepted_per_verified"] = delta.get("accepted_tokens", 0) / delta["draft_tokens"]
    if positions_start and positions_end:
        keys = sorted(set(positions_end["accepted"]) | set(positions_end["drafted"]), key=int)
        rows = []
        for key in keys:
            drafted = positions_end["drafted"].get(key, 0) - positions_start["drafted"].get(key, 0)
            accepted = positions_end["accepted"].get(key, 0) - positions_start["accepted"].get(key, 0)
            rows.append({"position": int(key), "drafted": drafted, "accepted": accepted,
                         "rate": accepted / drafted if drafted else None,
                         "coverage": drafted / drafts if drafts else None})
        out["per_position"] = rows
    return out


def generate(base_url: str, model: str, prompt_set: list[dict], *, concurrency: int = 16,
             max_tokens: int = 256, fixed_length: bool = False, extra_body: dict | None = None,
             timeout: float = 1800.0, require_idle: bool = True) -> dict:
    """Every prompt once, greedily; token ids per request and acceptance over the pass."""
    base = base_url.rstrip("/")
    if require_idle:
        prom.require_idle(base, "a model-behaviour run")

    def one(item: tuple[int, dict]) -> dict:
        index, prompt = item
        body = {"model": model, "messages": [{"role": "user", "content": prompt["text"]}],
                "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True,
                "cache_salt": uuid.uuid4().hex}
        if fixed_length:
            body["ignore_eos"] = True
            body["min_tokens"] = max_tokens
        body.update(extra_body or {})
        try:
            answer = _post(base + "/v1/chat/completions", body, timeout)
        except Exception as error:  # noqa: BLE001 - recorded per request
            return {"id": index, "error": f"{type(error).__name__}: {error}"[:300]}
        choice = answer["choices"][0]
        record = {"id": index, "prompt_ids": answer.get("prompt_token_ids"),
                  "output_ids": choice.get("token_ids"), "finish_reason": choice.get("finish_reason")}
        if record["prompt_ids"] is None or record["output_ids"] is None:
            record["error"] = "no token ids in the reply (the server needs return_token_ids)"
        return record

    counters, positions = prom.scrape(base, prom.VLLM), _positions(base)
    began = time.monotonic()
    sequences = _in_parallel(list(enumerate(prompt_set)), one, concurrency)
    seconds = time.monotonic() - began
    return {"seconds": round(seconds, 2),
            "acceptance": acceptance(counters, prom.scrape(base, prom.VLLM), positions, _positions(base)),
            "sequences": sequences}


def corpus_digest(sequences: list[dict]) -> str:
    usable = sorted((s for s in sequences if not s.get("error")), key=lambda s: s["id"])
    return hashlib.sha256(json.dumps([[s["prompt_ids"], s["output_ids"]] for s in usable]).encode()).hexdigest()


def score(base_url: str, model: str, sequences: list[dict], *, top: int = 20,
          concurrency: int = 8, timeout: float = 1800.0) -> list[dict]:
    """Teacher-forced log-probabilities of every sequence's response tokens."""
    base = base_url.rstrip("/")

    def one(sequence: dict) -> dict:
        start = len(sequence["prompt_ids"])
        tokens = sequence["prompt_ids"] + sequence["output_ids"]
        body = {"model": model, "prompt": tokens, "max_tokens": 1, "temperature": 0,
                "prompt_logprobs": top, "cache_salt": uuid.uuid4().hex}
        try:
            rows = _post(base + "/v1/completions", body, timeout)["choices"][0]["prompt_logprobs"]
        except Exception as error:  # noqa: BLE001 - recorded per sequence
            return {"id": sequence["id"], "error": f"{type(error).__name__}: {error}"[:300]}
        scored = {"id": sequence["id"], "tokens": sequence["output_ids"], "logprob": [], "rank": [],
                  "top_ids": [], "top_logprobs": []}
        for position in range(start, len(tokens)):
            entry = rows[position] or {}
            given = entry.get(str(tokens[position]))
            if given is None:
                return {"id": sequence["id"], "error": f"position {position}: no log-probability for the given token"}
            ranked = sorted(((int(key), value) for key, value in entry.items()
                             if value.get("rank") is not None and value["rank"] <= top),
                            key=lambda pair: pair[1]["rank"])
            scored["logprob"].append(given["logprob"])
            scored["rank"].append(given.get("rank"))
            scored["top_ids"].append([key for key, _ in ranked])
            scored["top_logprobs"].append([value["logprob"] for _, value in ranked])
        return scored

    usable = [s for s in sequences if not s.get("error") and s.get("output_ids")]
    return _in_parallel(usable, one, concurrency)


def self_agreement(scores: list[dict]) -> dict:
    """How often the scored token is the top-ranked one, and the corpus perplexity. On a
    build's own greedy text the first should be 1.0."""
    ranks = [r for s in scores if not s.get("error") for r in s["rank"]]
    logprobs = [lp for s in scores if not s.get("error") for lp in s["logprob"]]
    if not ranks:
        return {}
    return {"positions": len(ranks), "rank1_fraction": sum(r == 1 for r in ranks) / len(ranks),
            "perplexity": math.exp(-sum(logprobs) / len(logprobs))}


def kl_divergence(p_logprobs: dict[int, float], q_logprobs: dict[int, float], q_floor: float) -> float:
    """KL(P || Q) over P's listed tokens plus one bucket for the rest of the vocabulary;
    a token Q does not list gets ``q_floor`` (Q's lowest listed log-probability)."""
    total = p_mass = q_mass = 0.0
    for token, p_lp in p_logprobs.items():
        q_lp = q_logprobs.get(token, q_floor)
        p = math.exp(p_lp)
        total += p * (p_lp - q_lp)
        p_mass += p
        q_mass += math.exp(q_lp)
    p_rest = 1.0 - p_mass
    if p_rest > 1e-9:
        total += p_rest * math.log(p_rest / max(1.0 - q_mass, 1e-9))
    return max(total, 0.0)


def compare_scores(base: list[dict], candidate: list[dict]) -> dict:
    """Position by position over sequences whose tokens are equal in both runs."""
    by_id = {s["id"]: s for s in base if not s.get("error")}
    kls, per_sequence, deltas = [], [], []
    identical = top1 = skipped = 0
    base_sum = candidate_sum = 0.0
    for cand in candidate:
        other = by_id.get(cand.get("id"))
        if cand.get("error") or other is None or other["tokens"] != cand["tokens"]:
            skipped += 1
            continue
        sequence_kls = []
        for i, token in enumerate(cand["tokens"]):
            b_ids, b_lps, c_ids, c_lps = (other["top_ids"][i], other["top_logprobs"][i],
                                          cand["top_ids"][i], cand["top_logprobs"][i])
            same = b_ids == c_ids and b_lps == c_lps and other["logprob"][i] == cand["logprob"][i]
            identical += same
            top1 += bool(b_ids and c_ids and b_ids[0] == c_ids[0])
            if same:
                kl = 0.0
            else:
                p = dict(zip(b_ids, b_lps))
                p.setdefault(token, other["logprob"][i])
                q = dict(zip(c_ids, c_lps))
                q.setdefault(token, cand["logprob"][i])
                kl = kl_divergence(p, q, min(c_lps) if c_lps else cand["logprob"][i])
            sequence_kls.append(kl)
            deltas.append(cand["logprob"][i] - other["logprob"][i])
            base_sum += other["logprob"][i]
            candidate_sum += cand["logprob"][i]
        if sequence_kls:
            kls.extend(sequence_kls)
            per_sequence.append(sum(sequence_kls) / len(sequence_kls))
    positions = len(kls)
    if not positions:
        return {"sequences": 0, "skipped": skipped, "positions": 0}
    base_ppl, cand_ppl = math.exp(-base_sum / positions), math.exp(-candidate_sum / positions)
    return {"sequences": len(per_sequence), "skipped": skipped, "positions": positions,
            "identical_fraction": identical / positions, "top1_agreement": top1 / positions,
            "kl": {"mean": sum(kls) / positions, **stats.percentiles(kls, (99,)), "max": max(kls),
                   "per_sequence": stats.describe(per_sequence)},
            "logprob_delta": {"mean": sum(deltas) / positions,
                              "mean_abs": sum(abs(d) for d in deltas) / positions,
                              "max_abs": max(abs(d) for d in deltas)},
            "perplexity": {"base": base_ppl, "candidate": cand_ppl,
                           "change_pct": 100.0 * (cand_ppl / base_ppl - 1.0)}}


def compare_generation(base: dict, candidate: dict) -> dict:
    """Which responses are token-identical, where the others first diverge, and the
    acceptance of each run."""
    by_id = {s["id"]: s for s in base["sequences"] if not s.get("error")}
    identical, divergence = 0, []
    compared = 0
    for cand in candidate["sequences"]:
        other = by_id.get(cand["id"])
        if cand.get("error") or other is None:
            continue
        compared += 1
        a, b = other["output_ids"], cand["output_ids"]
        if a == b:
            identical += 1
            continue
        divergence.append(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
    acceptance_rows = []
    for key in ("acceptance_length", "verified_per_draft", "accepted_per_verified"):
        x, y = base["acceptance"].get(key), candidate["acceptance"].get(key)
        if x and y:
            acceptance_rows.append({"metric": key, "base": x, "candidate": y,
                                    "change_pct": 100.0 * (y / x - 1.0)})
    rates = []
    base_positions = {row["position"]: row for row in base["acceptance"].get("per_position", [])}
    for row in candidate["acceptance"].get("per_position", []):
        other = base_positions.get(row["position"])
        if other and other.get("rate") is not None and row.get("rate") is not None:
            rates.append({"position": row["position"], "base": other["rate"], "candidate": row["rate"]})
    return {"compared": compared, "identical": identical,
            "divergence": stats.percentiles(divergence, (10, 50, 90)) if divergence else {},
            "acceptance": acceptance_rows, "per_position": rates}


WORKLOAD_KEYS = ("prompts_sha256", "prompt_count", "concurrency", "max_tokens", "fixed_length", "top")


def compare(base: dict, candidate: dict) -> dict:
    warnings = [f"{key}: {base['workload'].get(key)} vs {candidate['workload'].get(key)}"
                for key in WORKLOAD_KEYS if base["workload"].get(key) != candidate["workload"].get(key)]
    result = {"warnings": warnings, "generation": compare_generation(base["generation"], candidate["generation"])}
    if base.get("scores") and candidate.get("scores"):
        if base["corpus"]["sha256"] != candidate["corpus"]["sha256"]:
            warnings.append("the runs scored different corpora: only sequences with equal tokens compare "
                            "(score the candidate with --corpus <base run>)")
        result["scores"] = compare_scores(base["scores"], candidate["scores"])
    return result


def format_comparison(result: dict, base_label: str, candidate_label: str) -> str:
    lines = [f"model behaviour: {base_label} -> {candidate_label}"]
    lines += [f"  warning: {w}" for w in result["warnings"]]
    g = result["generation"]
    lines.append(f"  greedy responses identical: {g['identical']}/{g['compared']}"
                 + (f"; the others diverge at token p10/p50/p90 {g['divergence'].get('p10')}/"
                    f"{g['divergence'].get('p50')}/{g['divergence'].get('p90')}" if g["divergence"] else ""))
    for row in g["acceptance"]:
        lines.append(f"  {row['metric']:24s} {row['base']:8.3f} -> {row['candidate']:8.3f}  {row['change_pct']:+.2f}%")
    if g["per_position"]:
        lines.append("  acceptance by draft position: " + ", ".join(
            f"{r['position']}: {r['base']:.3f}->{r['candidate']:.3f}" for r in g["per_position"]))
    s = result.get("scores")
    if s and s.get("positions"):
        kl = s["kl"]
        lines.append(f"  teacher-forced over {s['positions']} positions in {s['sequences']} sequences"
                     + (f" ({s['skipped']} skipped)" if s["skipped"] else ""))
        lines.append(f"    identical distributions {100 * s['identical_fraction']:.2f}%, "
                     f"top-1 agreement {100 * s['top1_agreement']:.3f}%")
        lines.append(f"    KL mean {kl['mean']:.3e} (±{kl['per_sequence'].get('ci95') or 0:.1e} over sequences), "
                     f"p99 {kl.get('p99', 0):.3e}, max {kl['max']:.3e} nats")
        lines.append(f"    given-token log-probability change mean {s['logprob_delta']['mean']:+.2e}, "
                     f"mean |change| {s['logprob_delta']['mean_abs']:.2e}, max {s['logprob_delta']['max_abs']:.2e}")
        p = s["perplexity"]
        lines.append(f"    perplexity {p['base']:.5f} -> {p['candidate']:.5f} ({p['change_pct']:+.3f}%)")
    elif s is not None:
        lines.append(f"  teacher-forced: no comparable sequences ({s.get('skipped', 0)} skipped)")
    return "\n".join(lines)
