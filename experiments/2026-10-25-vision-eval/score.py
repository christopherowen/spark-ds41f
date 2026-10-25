#!/usr/bin/env python3
"""Score vision_eval.py's answers and compare two builds item by item.

usage: score.py ITEMS_JSONL ARM_JSONL [OTHER_ARM_JSONL]

Each benchmark's own metric, as lmms-eval computes it:
- OCRBench: the answer is contained in the response (case-insensitive; spaces removed for
  handwritten mathematical expressions);
- DocVQA: ANLS (normalized Levenshtein similarity, 0 below 0.5);
- ChartQA: relaxed accuracy (numbers within 5%, otherwise an exact match);
- TextVQA: VQA accuracy over the ten human answers;
- RealWorldQA: an exact match.

With two arms: each benchmark's mean for both, how many answers differ, how many items
score better or worse in the second arm, and a seeded bootstrap 95% interval of the mean
difference (second minus first).
"""
import json
import random
import re
import sys


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def anls(prediction: str, answers: list[str]) -> float:
    def norm(s):
        return " ".join(s.strip().lower().split())
    best = 0.0
    for answer in answers:
        a, p = norm(answer), norm(prediction)
        distance = levenshtein(a, p) / max(len(a), len(p), 1)
        best = max(best, 1 - distance if distance < 0.5 else 0.0)
    return best


def relaxed(prediction: str, answer: str) -> float:
    def number(text):
        try:
            return float(text.rstrip("%")) / 100 if text.endswith("%") else float(text)
        except ValueError:
            return None
    p = prediction.strip().rstrip(".")
    pf, tf = number(p), number(answer)
    if pf is not None and tf:
        return float(abs(pf - tf) / abs(tf) <= 0.05)
    return float(p.lower() == answer.lower())


def ocrbench(prediction: str, answers: list[str], category: str) -> float:
    for answer in answers:
        if category == "Handwritten Mathematical Expression Recognition":
            a = answer.strip().replace("\n", " ").replace(" ", "")
            p = prediction.strip().replace("\n", " ").replace(" ", "")
        else:
            a = answer.lower().strip().replace("\n", " ")
            p = prediction.lower().strip().replace("\n", " ")
        if a in p:
            return 1.0
    return 0.0


_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!*#%^&$~'|:]")


def vqa_norm(text: str) -> str:
    text = text.lower().replace("\n", " ").replace("\t", " ").strip()
    text = _PUNCT.sub("", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)
    words = [_NUMBERS.get(w, w) for w in text.split() if w not in _ARTICLES]
    return " ".join(words)


def vqa(prediction: str, answers: list[str]) -> float:
    p = vqa_norm(prediction)
    normed = [vqa_norm(a) for a in answers]
    scores = []
    for i in range(len(normed)):
        others = normed[:i] + normed[i + 1:]
        scores.append(min(1.0, sum(a == p for a in others) / 3))
    return sum(scores) / len(scores)


def exact(prediction: str, answers: list[str]) -> float:
    p = prediction.strip().rstrip(".").strip().lower()
    return float(any(p == a.strip().lower() for a in answers))


def score(item: dict, prediction: str) -> float:
    name, answers = item["dataset"], item["answers"]
    if name == "ocrbench":
        return ocrbench(prediction, answers, item["category"])
    if name == "docvqa":
        return anls(prediction, answers)
    if name == "chartqa":
        return relaxed(prediction, answers[0])
    if name == "textvqa":
        return vqa(prediction, answers)
    return exact(prediction, answers)


def load(path: str) -> dict:
    out = {}
    for line in open(path):
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if "dataset" in record:
            out[(record["dataset"], record["id"])] = record
    return out


def bootstrap(differences: list[float], seed: int = 0, rounds: int = 10000) -> tuple:
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main() -> None:
    items = {(i["dataset"], i["id"]): i for i in map(json.loads, open(sys.argv[1]))}
    arms = [load(path) for path in sys.argv[2:]]
    for arm, path in zip(arms, sys.argv[2:]):
        failed = [key for key, record in arm.items() if "failed" in record]
        missing = [key for key in items if key not in arm]
        if failed or missing:
            print(f"{path}: {len(failed)} failed, {len(missing)} missing")
    datasets = sorted({key[0] for key in items})
    for name in datasets + ["all"]:
        keys = [k for k in items if (name == "all" or k[0] == name)
                and all("answer" in arm.get(k, {}) for arm in arms)]
        scores = [[score(items[k], arm[k]["answer"]) for k in keys] for arm in arms]
        line = f"{name:12s} n={len(keys):5d} " + " ".join(
            f"{100 * sum(s) / len(s):6.2f}" for s in scores)
        if len(arms) == 2:
            differ = sum(arms[0][k]["answer"] != arms[1][k]["answer"] for k in keys)
            deltas = [b - a for a, b in zip(*scores)]
            better = sum(d > 0 for d in deltas)
            worse = sum(d < 0 for d in deltas)
            low, high = bootstrap(deltas)
            line += (f"  answers differ {differ:4d}  better {better:3d} worse {worse:3d}"
                     f"  diff {100 * sum(deltas) / len(deltas):+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]")
        print(line)


if __name__ == "__main__":
    main()
