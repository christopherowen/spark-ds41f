#!/usr/bin/env python3
"""Answers to public vision benchmarks' questions, one JSON line an item, for scoring.

usage: vision_eval.py BASE_URL DATA_DIR [--concurrency N] [--tokens T]

DATA_DIR holds items.jsonl (extract.py: dataset, id, image file, question, answers) and
the images. Each item is asked once at temperature 0 with thinking off, the image first,
then the question with its benchmark's answer instruction. One JSON line an item (the
item's id and dataset, the answer text, the finish reason), then a summary line. Scoring
is score.py's, against the answers, offline. A request that fails is recorded and the
script exits 1 at the end.
"""
import argparse
import base64
import concurrent.futures
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("data_dir")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    data = Path(args.data_dir)
    items = [json.loads(line) for line in (data / "items.jsonl").read_text().splitlines()]
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]

    def ask(item: dict) -> dict:
        image = data / item["image"]
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
        url = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode()
        body = {"model": model, "max_tokens": args.tokens, "temperature": 0,
                "cache_salt": uuid.uuid4().hex, "chat_template_kwargs": {"thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": item["prompt"]}]}]}
        request = urllib.request.Request(base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        record = {"dataset": item["dataset"], "id": item["id"]}
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                choice = json.load(response)["choices"][0]
        except urllib.error.HTTPError as error:
            return {**record, "failed": error.code,
                    "message": error.read().decode(errors="replace")[:300]}
        return {**record, "answer": choice["message"]["content"],
                "finish_reason": choice["finish_reason"]}

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for record in pool.map(ask, items):
            failed += "failed" in record
            print(json.dumps(record), flush=True)
    print(json.dumps({"summary": "vision eval", "items": len(items), "failed": failed}),
          flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
