"""Replay legitimate output through the scheduler's check_stop, token by
token, and report where the loop handling would act on it.

Two kinds of input:
- --files ROOT...: every text file under the roots (code, docs, JSON,
  patches), cut into 2,048-token chunks, each replayed once as reasoning and
  once as a visible answer;
- --jsonl FILE...: recorded messages, one JSON object a line with
  "reasoning" and/or "content" (strings): the reasoning is replayed as
  reasoning, then the end marker, then the content as the answer, as the
  scheduler sees one reply.

A reasoning break is check_stop forcing the end of reasoning; an answer stop
is check_stop finishing the request with FINISHED_REPETITION. On legitimate
output every one is a false positive.

Run it in the serving image whose vllm you want to measure (or with the
changed files mounted over it):
  python3 replay.py --tokenizer TOKENIZER_JSON --out OUT.json --files ROOT...
  python3 replay.py --tokenizer TOKENIZER_JSON --out OUT.json --jsonl FILE...
"""

import argparse
import json
import pathlib
import random
import time

from tokenizers import Tokenizer

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request

START, END = 200001, 200002  # outside the vocabulary: never in the text
CHUNK = 2048
SUFFIXES = {".py", ".md", ".json", ".patch", ".sh", ".txt", ".toml", ".yaml",
            ".cu", ".cpp", ".h", ".c"}


def replay(reasoning: list[int], answer: list[int] | None):
    """Feed one reply; return (kind, position, loop_info) at the first action,
    or None if check_stop never acts on it."""
    req = Request(request_id="replay", prompt_token_ids=[1, 2, START],
                  sampling_params=SamplingParams(max_tokens=1 << 22), pooling_params=None)
    stream = list(reasoning) + ([END] + answer if answer is not None else [])
    for pos, tok in enumerate(stream):
        prev = len(req._output_token_ids)
        req.append_output_token_ids(tok)
        req.update_reasoning_markers((START,), (END,), prev_num_output=prev)
        if check_stop(req, max_model_len=1 << 23):
            return ("answer_stop" if req.thinking_output_start < 0 else "reasoning_stop",
                    pos, req.last_loop_info)
        # What follows a forced end was never recorded: stop at the break.
        if req.force_thinking_end:
            return ("reasoning_break", pos, req.last_loop_info)
    return None


def text_files(roots: list[str], limit: int, seed: int = 0) -> list[pathlib.Path]:
    files = [p for root in roots for p in pathlib.Path(root).rglob("*")
             if p.is_file() and p.suffix in SUFFIXES and ".git" not in p.parts
             and p.stat().st_size < 2_000_000]
    random.Random(seed).shuffle(files)
    return files[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--jsonl", nargs="*", default=[])
    parser.add_argument("--limit-files", type=int, default=1500)
    args = parser.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids  # noqa: E731

    stats = {"units": 0, "tokens": 0, "events": {}, "examples": []}
    t0 = time.time()

    def record(source: str, ids: list[int], result) -> None:
        if result is None:
            return
        kind, pos, info = result
        stats["events"][kind] = stats["events"].get(kind, 0) + 1
        lo = max(0, pos - 40)
        stats["examples"].append({"source": source, "kind": kind, "pos": pos,
                                  "info": list(info or ()),
                                  "context": tok.decode(ids[lo:pos + 1])[-240:]})

    for path in text_files(args.files, args.limit_files) if args.files else []:
        try:
            ids = encode(path.read_text(errors="strict"))
        except (UnicodeDecodeError, OSError):
            continue
        for c in range(0, len(ids), CHUNK):
            chunk = ids[c:c + CHUNK]
            if len(chunk) < 64:
                continue
            stats["units"] += 1
            stats["tokens"] += len(chunk)
            record(f"{path}#{c // CHUNK}:reasoning", chunk, replay(chunk, None))
            record(f"{path}#{c // CHUNK}:answer", chunk, replay([], chunk))

    for name in args.jsonl:
        for line_no, line in enumerate(open(name), 1):
            message = json.loads(line)
            reasoning = encode(message.get("reasoning") or "")
            content = encode(message.get("content") or "")
            stats["units"] += 1
            stats["tokens"] += len(reasoning) + len(content)
            record(f"{name}:{line_no}", reasoning + [END] + content,
                   replay(reasoning, content))

    stats["seconds"] = round(time.time() - t0, 1)
    pathlib.Path(args.out).write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "examples"}))


if __name__ == "__main__":
    main()
