#!/usr/bin/env python3
"""Compare complete recorded generation and top/given-token scores without inference."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

GENERATION_FIELDS = ("id", "prompt_ids", "output_ids", "finish_reason")
SCORE_FIELDS = ("id", "tokens", "logprob", "rank", "top_ids", "top_logprobs")


def canonical(value):
    # JSON preserves the sign of zero; disallow NaN/Inf instead of comparing
    # rounded scores, numeric tolerances or only the top-ranked token.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def rows(value, fields):
    result = {}
    for row in value:
        if row.get("error") or row["id"] in result:
            raise ValueError("failed or duplicate sequence")
        selected = {field: row[field] for field in fields}
        canonical(selected)
        result[row["id"]] = selected
    if not result:
        raise ValueError("empty sequence set")
    return result


def checked(value):
    generation = rows(value["generation"]["sequences"], GENERATION_FIELDS)
    scores = rows(value["scores"], SCORE_FIELDS)
    count = value["workload"]["prompt_count"]
    if len(generation) != count or len(scores) != count or set(scores) != set(generation):
        raise ValueError("incomplete generation or scoring coverage")
    if value["corpus"]["sequences"] != count:
        raise ValueError("incomplete corpus")
    positions = 0
    for key, score in scores.items():
        tokens = score["tokens"]
        if not tokens or tokens != generation[key]["output_ids"]:
            raise ValueError("scores are not the generated sequence")
        if any(len(score[field]) != len(tokens) for field in SCORE_FIELDS[2:]):
            raise ValueError("incomplete position coverage")
        if any(not ids or len(ids) != len(lps)
               for ids, lps in zip(score["top_ids"], score["top_logprobs"])):
            raise ValueError("incomplete top-token vectors")
        positions += len(tokens)
    return generation, scores, positions


def compare(reference, candidate):
    bg, bs, bp = checked(reference)
    cg, cs, cp = checked(candidate)
    generation_differences = sorted(set(bg) ^ set(cg) | {
        key for key in bg.keys() & cg.keys() if canonical(bg[key]) != canonical(cg[key])})
    score_differences = sorted(set(bs) ^ set(cs) | {
        key for key in bs.keys() & cs.keys() if canonical(bs[key]) != canonical(cs[key])})
    workload_equal = canonical(reference["workload"]) == canonical(candidate["workload"])
    corpus_equal = reference["corpus"]["sha256"] == candidate["corpus"]["sha256"]
    return {"scope": "recorded generated tokens and all returned top/given-token scores; not the full vocabulary",
            "workload_identical": workload_equal, "corpus_identical": corpus_equal,
            "reference_sequences": len(bg), "candidate_sequences": len(cg),
            "reference_positions": bp, "candidate_positions": cp,
            "generation_differences": generation_differences, "score_differences": score_differences,
            "reference_generation_sha256": digest(bg), "candidate_generation_sha256": digest(cg),
            "reference_scores_sha256": digest(bs), "candidate_scores_sha256": digest(cs),
            "identical": bool(workload_equal and corpus_equal and bp == cp
                              and not generation_differences and not score_differences)}


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read(512 * 1024**2 + 1)
    if len(raw) > 512 * 1024**2:
        raise ValueError("model report exceeds decoded bound")
    return json.loads(raw)


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite evidence")
    provenance = json.loads(args.provenance.read_text())
    reference_sha = file_digest(args.reference)
    if reference_sha != provenance["model_sha256"]:
        raise ValueError("recorded reference differs from pinned evidence")
    reference = load(args.reference)
    receipt = {"schema_version": 1, "reference_sha256": reference_sha,
               "provenance_sha256": hashlib.sha256(args.provenance.read_bytes()).hexdigest(),
               "analysis_sha256": file_digest(Path(__file__)),
               "command": sys.argv, "inference_requests": 0, "candidates": []}
    for path in args.candidate:
        row = {"path": str(path), "sha256": file_digest(path)}
        try:
            row.update(compare(reference, load(path)))
        except (ValueError, KeyError, TypeError) as error:
            row.update(identical=False, error=f"{type(error).__name__}: {error}")
        receipt["candidates"].append(row)
    receipt["all_identical"] = all(row["identical"] for row in receipt["candidates"])
    with args.out.open("x") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    return int(not receipt["all_identical"])


if __name__ == "__main__":
    raise SystemExit(main())
