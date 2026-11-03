#!/usr/bin/env python3
"""Scan probe receipts for short-motif repetition bursts in generated token ids.

Reports, per request: adjacent-copy runs of motifs of 1-4 tokens, the dominant
motif, burst onset position, and text around the first and strongest burst.
"""
import argparse
import collections
import json


def motif_runs(ids, size):
    runs = []
    i = 0
    while i + size <= len(ids):
        motif = tuple(ids[i:i + size])
        count = 1
        j = i + size
        while j + size <= len(ids) and tuple(ids[j:j + size]) == motif:
            count += 1
            j += size
        if count >= 3:
            runs.append((i, count, motif))
        i = j if count > 1 else i + 1
    return runs


def bursts(ids, size, min_copies=2, min_bursts=3):
    """Separated bursts: adjacent copies, possibly interrupted by other text."""
    positions = []
    i = 0
    while i + size <= len(ids):
        motif = tuple(ids[i:i + size])
        count = 1
        j = i + size
        while j + size <= len(ids) and tuple(ids[j:j + size]) == motif:
            count += 1
            j += size
        if count >= min_copies:
            positions.append((i, count, motif))
        i = j if count > 1 else i + 1
    by_motif = collections.defaultdict(list)
    for pos, count, motif in positions:
        by_motif[motif].append((pos, count))
    out = []
    for motif, occ in by_motif.items():
        if len(occ) >= min_bursts:
            out.append((motif, occ))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--tokenizer", default="/Users/christopherowen/projects/spark-ds41f/.work/loop-analysis/tokenizer.json")
    parser.add_argument("--context", type=int, default=30, help="tokens of context to show around onset")
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    data = json.load(open(args.receipt))
    for req in data["requests"]:
        name = req["name"]
        resp = req.get("response", {})
        choice = (resp.get("choices") or [{}])[0]
        ids = choice.get("token_ids") or []
        if not ids:
            print(f"== {name}: no token ids ==")
            continue
        msg = choice.get("message", {})
        reasoning = msg.get("reasoning_content") or ""
        report = {
            "name": name,
            "finish_reason": choice.get("finish_reason"),
            "output_tokens": len(ids),
            "reasoning_chars": len(reasoning),
            "answer_chars": len(msg.get("content") or ""),
        }
        best = None
        for size in (1, 2, 3, 4):
            for pos, count, motif in motif_runs(ids, size):
                key = count * size
                if best is None or key > best[0]:
                    best = (key, size, pos, count, motif)
        if best:
            _, size, pos, count, motif = best
            report["longest_adjacent_run"] = {
                "motif": tok.decode(list(motif), skip_special_tokens=False),
                "motif_ids": list(motif),
                "copies": count,
                "onset_token": pos,
                "fraction_of_output": round(count * size / len(ids), 3),
            }
            lo = max(0, pos - args.context)
            report["text_at_onset"] = tok.decode(ids[lo:pos + count * size + args.context], skip_special_tokens=False)
        # separated burst families (the screenshot pattern)
        fam = bursts(ids, 2)
        fam.sort(key=lambda x: -sum(c for _, c in x[1]))
        report["burst_families_size2"] = [
            {
                "motif": tok.decode(list(m), skip_special_tokens=False),
                "occurrences": len(occ),
                "total_copies": sum(c for _, c in occ),
                "first_at": occ[0][0],
                "span_tokens": occ[-1][0] - occ[0][0],
            }
            for m, occ in fam[:5]
        ]
        report["tail_text"] = tok.decode(ids[-160:], skip_special_tokens=False)
        print(json.dumps(report, indent=1, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
