"""Cold prefill: time to first token on real-text prompts of fixed sizes.

Each prompt is a long document (Python standard-library source from a fixed seed, so
every build reads the same text), sized by the server's tokenizer and sent with a fresh
cache salt, so nothing of it is cached. The rate is prompt tokens over time to first
token. Repeats go round the sizes in turn, so drift spreads over every size.
"""

import uuid

from . import client, stats
from . import load as loads
from . import prompts

SCHEMA = "servebench-prefill/1"
QUESTION = "Summarize the document above in one sentence."


def document(base_url: str, model: str, size: int, seed: int, vllm: bool) -> tuple[str, int | None]:
    """A document of about ``size`` tokens and its token count (rescaled once by the
    server's tokenizer when it has one)."""
    text = prompts.source_text(size, seed)
    counted = loads.token_count(base_url, model, text) if vllm else None
    if counted:
        text = prompts.source_text(size, seed, chars_per_token=len(text) / counted)
        counted = loads.token_count(base_url, model, text)
    return text, counted


def run(base_url: str, model: str, sizes: list[int], *, repeats: int = 3, vllm: bool = True,
        extra_body: dict | None = None, log=lambda message: None) -> list[dict]:
    samples = []
    for repeat in range(repeats):
        for index, size in enumerate(sizes):
            text, counted = document(base_url, model, size, seed=1000 * repeat + index, vllm=vllm)
            body = {"model": model, "messages": [{"role": "user", "content": f"{text}\n\n{QUESTION}"}],
                    "max_tokens": 1, "temperature": 0}
            if vllm:
                body["cache_salt"] = uuid.uuid4().hex
            body.update(extra_body or {})
            record = client.stream_chat(base_url, body, continuous_usage=vllm, timeout=max(900.0, size / 500))
            ttft = record["first"] - record["sent"] if record["first"] is not None else None
            tokens = record["prompt_tokens"] or counted
            sample = {"size": size, "repeat": repeat, "prompt_tokens": tokens, "ttft_s": ttft,
                      "tps": tokens / ttft if tokens and ttft else None, "error": record["error"]}
            samples.append(sample)
            log(f"servebench: prefill {size} repeat {repeat + 1}/{repeats}: "
                + (f"{sample['tps']:.0f} tok/s" if sample["tps"] else f"failed ({record['error']})"))
    return samples


def summarize(samples: list[dict]) -> list[dict]:
    rows = []
    for size in dict.fromkeys(s["size"] for s in samples):
        ok = [s for s in samples if s["size"] == size and s["tps"]]
        rows.append({"size": size, "prompt_tokens": ok[0]["prompt_tokens"] if ok else None,
                     "tps": stats.describe([s["tps"] for s in ok]), "samples": [round(s["tps"], 2) for s in ok],
                     "failed": sum(1 for s in samples if s["size"] == size and not s["tps"])})
    return rows


def compare(base: list[dict], candidate: list[dict], *, tolerance_pct: float = 0.5) -> dict:
    """Size by size: prefill tok/s change with its Welch interval (higher is better)."""
    by_size = {row["size"]: row for row in base}
    rows, warnings = [], []
    for row in candidate:
        other = by_size.get(row["size"])
        if other is None:
            warnings.append(f"prefill {row['size']}: not in the base run")
            continue
        if other["prompt_tokens"] != row["prompt_tokens"]:
            warnings.append(f"prefill {row['size']}: {other['prompt_tokens']} vs {row['prompt_tokens']} prompt tokens")
        test = stats.welch(other["samples"], row["samples"])
        if test is None:
            continue
        rows.append({"metric": f"prefill {row['size']:,} tok/s", "base": other["tps"]["mean"],
                     "candidate": row["tps"]["mean"], "change_pct": test["change_pct"],
                     "ci95_pct": test["ci95_pct"],
                     "verdict": stats.verdict(test["change_pct"], test["ci95_pct"], True, tolerance_pct)})
    return {"warnings": warnings, "rows": rows}
