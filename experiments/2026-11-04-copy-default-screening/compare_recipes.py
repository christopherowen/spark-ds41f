#!/usr/bin/env python3
"""Compare both new boots with both saved boots of each earlier recipe."""
import argparse
import json
from pathlib import Path
import subprocess

from analyze_functional import PIN, ROOT, load, require, sha
from analyze_performance import AA_FLOOR_PCT, compare_results, read_json, suite


def compare(snapshot, performance):
    raw = snapshot / "raw/results/private/suites"
    saved = ROOT / "experiments/2026-11-03-candidate-38-tp3-qualification/runs/2026-11-03-dgx1-3"
    current_labels = ("candidate-39", "candidate-39-repeat")
    judge = lambda a, b: compare_results(a, b, 3.0)
    result = {}
    for recipe, labels in (("default", ("candidate-38", "candidate-38-repeat")),
                           ("copy", ("candidate-38-copy", "candidate-38-copy-repeat"))):
        base = saved / ("performance-" + recipe)
        publication = load(base / "publication-receipt.json")
        checked = {}
        for name, record in publication["published"].items():
            path = base / name
            digest = sha(path.read_bytes())
            require(digest == record["sha256"], f"saved reference hash differs: {path}")
            relative = str(path.relative_to(ROOT))
            require(sha(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:{relative}"])) == digest,
                    f"saved reference changed since execution: {relative}")
            checked[name] = digest
        original = load(base / "analysis.json")
        comparisons = {}
        for number, name in enumerate(("release-tp3", "single-stream", "sampled")):
            run = "candidate-39-release-tp3" + ("-" + name if number else "")
            a, b = [raw / run / label for label in current_labels]
            x, y = [base / label / name for label in labels]
            aa = suite.compare(read_json, str(a), str(b), judge)
            reference_aa = suite.compare(read_json, str(x), str(y), judge)
            pairs = {}
            headlines = {}
            for i, reference in enumerate((x, y), 1):
                first = suite.compare(read_json, str(reference), str(a), judge)
                repeat = suite.compare(read_json, str(reference), str(b), judge)
                pairs[f"reference_boot{i}_to_candidate_boot1"] = first
                pairs[f"reference_boot{i}_to_candidate_boot2"] = repeat
                headlines[f"reference_boot{i}_candidate_repeat_floor"] = suite.with_floor(
                    first, aa, floors=AA_FLOOR_PCT, reference_to_repeat=repeat)
                headlines[f"reference_boot{i}_saved_reference_repeat_floor"] = suite.with_floor(
                    first, reference_aa, floors=AA_FLOOR_PCT, reference_to_repeat=repeat)
            adverse = []
            for pair, outcome in pairs.items():
                for step, record in outcome["steps"].items():
                    if "skipped" not in record:
                        adverse.extend({"pair": pair, "step": step, **row}
                                       for row in record[1].get("rows", [])
                                       if row["verdict"].startswith("worse"))
            comparisons[name] = {"pairs": pairs, "headlines": headlines,
                                 "candidate_repeat": aa, "saved_reference_repeat": reference_aa,
                                 "adverse_rows": adverse}
        model_matches = []
        for old_boot in original["boots"]:
            old_digest = old_boot["suites"]["release-tp3"]["steps"]["model"]["corpus_sha256"]
            for new_boot in performance["boots"]:
                new_digest = new_boot["suites"]["release-tp3"]["steps"]["model"]["corpus_sha256"]
                model_matches.append({"reference": old_boot["label"], "candidate": new_boot["label"],
                                      "reference_digest": old_digest, "candidate_digest": new_digest,
                                      "matches": old_digest == new_digest})
        result[recipe] = {"verified_reference_files": checked, "comparisons": comparisons,
                          "model_corpus_digest_matches": model_matches,
                          "limits": ["Every pairing is retained; the two earlier recipes are not pooled.",
                                     "Headlines are assessed against both candidate and saved-reference repeat floors.",
                                     "Saved model text is omitted; corpus digest equality is reported separately.",
                                     "Tail percentiles without an interval are observations, not a proven gain."]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("performance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite a comparison")
    result = compare(args.snapshot, load(args.performance))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({recipe: {"reference_files": len(data["verified_reference_files"]),
                               "all_model_corpus_digests_match": all(r["matches"] for r in data["model_corpus_digest_matches"]),
                               "adverse_rows": {k: len(v["adverse_rows"]) for k, v in data["comparisons"].items()}}
                      for recipe, data in result.items()}))


if __name__ == "__main__":
    main()
