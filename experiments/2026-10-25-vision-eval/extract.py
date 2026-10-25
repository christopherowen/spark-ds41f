#!/usr/bin/env python3
"""The evaluation items, from the benchmarks' downloaded parquet files.

usage: extract.py RAW_DIR OUT_DIR   (needs pyarrow: uv run --with pyarrow python -I ...)

RAW_DIR/<dataset>/*.parquet as fetch.sh downloads them. OUT_DIR gets images/<dataset>/ and
items.jsonl: dataset, id, image (relative path), prompt (the question with its benchmark's
answer instruction, as lmms-eval asks it), answers, category. Samples are seeded, so the
items are the same on every run.
"""
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

SHORT = "\nAnswer the question using a single word or phrase."
SAMPLE = 300


def rows(directory: Path) -> list[dict]:
    out = []
    for path in sorted(directory.glob("*.parquet")):
        out.extend(pq.read_table(path).to_pylist())
    return out


def sample(items: list, count: int | None, seed: int) -> list:
    if count is None or len(items) <= count:
        return items
    return [items[i] for i in sorted(random.Random(seed).sample(range(len(items)), count))]


def image_file(image: dict, stem: Path) -> Path:
    data = image["bytes"]
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        suffix = ".png"
    elif data[:2] == b"\xff\xd8":
        suffix = ".jpg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":  # RealWorldQA's
        suffix = ".webp"
    else:
        raise SystemExit(f"{stem}: unrecognised image format")
    path = stem.with_suffix(suffix)
    path.write_bytes(data)
    return path


# dataset: (sample size, id, prompt, answers, category)
DATASETS = {
    "ocrbench": (None, lambda r, i: str(i), lambda r: r["question"],
                 lambda r: list(r["answer"]), lambda r: r["question_type"]),
    "docvqa": (SAMPLE, lambda r, i: str(r["questionId"]), lambda r: r["question"] + SHORT,
               lambda r: list(r["answers"]), lambda r: ",".join(r["question_types"] or [])),
    "chartqa": (SAMPLE, lambda r, i: str(i), lambda r: r["question"] + SHORT,
                lambda r: [r["answer"]], lambda r: r["type"]),
    "textvqa": (SAMPLE, lambda r, i: str(r["question_id"]), lambda r: r["question"] + SHORT,
                lambda r: list(r["answers"]), lambda r: ""),
    "realworldqa": (None, lambda r, i: str(i), lambda r: r["question"],
                    lambda r: [r["answer"]], lambda r: ""),
}


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    items = []
    for name, (count, ident, prompt, answers, category) in DATASETS.items():
        table = rows(raw / name)
        indexed = sample(list(enumerate(table)), count, seed=20261025)
        images = out / "images" / name
        images.mkdir(parents=True, exist_ok=True)
        for index, row in indexed:
            item_id = ident(row, index)
            path = image_file(row["image"], images / item_id)
            items.append({"dataset": name, "id": item_id,
                          "image": str(path.relative_to(out)), "prompt": prompt(row),
                          "answers": answers(row), "category": category(row)})
        print(f"{name}: {len(indexed)} of {len(table)}", flush=True)
    with (out / "items.jsonl").open("w") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    print(f"{len(items)} items", flush=True)


if __name__ == "__main__":
    main()
