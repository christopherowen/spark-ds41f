#!/usr/bin/env python3
"""Prompt token ids and mm hashes of the #978 bench's workloads, per turn.

Imports the PR's bench (bench_mm_prefix_fastpath.py at 67792978) for its DS4.1
processor and images, and records, for turns 1..3 of each workload, the sha256
of the prompt token ids, the mm hashes and the placeholder ranges, so the base
and candidate arms can be compared. Usage: identity_978.py <label> <out.json>
"""
import hashlib
import json
import sys

sys.path.insert(0, "/opt/exp")
import bench_mm_prefix_fastpath as bench  # noqa: E402
from vllm.multimodal.media.image import ImageMediaIO  # noqa: E402
from vllm.multimodal.processing.context import TimingContext  # noqa: E402
from vllm.multimodal.processing.inputs import ProcessorInputs  # noqa: E402

WORKLOADS = [
    ("1302x1302_jpeg_1img", 1302, 1302, "JPEG", "image/jpeg", 1),
    ("1302x1302_jpeg_4img", 1302, 1302, "JPEG", "image/jpeg", 4),
    ("1920x1080_png_1img", 1920, 1080, "PNG", "image/png", 1),
    ("3840x2160_png_1img", 3840, 2160, "PNG", "image/png", 1),
    ("4032x3024_jpeg_4img", 4032, 3024, "JPEG", "image/jpeg", 4),
]


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main() -> None:
    label, out_path = sys.argv[1], sys.argv[2]
    proc = bench._build_ds41_processor()
    media_io = ImageMediaIO()
    tokenizer = proc.info.get_tokenizer()
    result = {"label": label, "workloads": {}}
    for name, w, h, fmt, mtype, count in WORKLOADS:
        b64 = [bench._make_encoded_image(w, h, fmt, seed=100 + i) for i in range(count)]
        turns = []
        for turn in range(1, 4):
            text = (
                "System: analyze images.\n"
                + "".join(f"Image {i}: {bench.IMAGE_PLACEHOLDER}\n" for i in range(count))
                + "".join(f"Turn {t} question and assistant reply.\n" for t in range(1, turn + 1))
            )
            loaded = [media_io.load_base64(mtype, b) for b in b64]
            items = proc.info.parse_mm_data({"image": loaded})
            out = proc.apply(ProcessorInputs(prompt=tokenizer.encode(text), mm_data_items=items),
                             TimingContext())
            placeholders = [(p.offset, p.length) for p in out["mm_placeholders"]["image"]]
            turns.append({
                "turn": turn,
                "prompt_tokens": len(out["prompt_token_ids"]),
                "prompt_token_ids_sha256": digest(list(out["prompt_token_ids"])),
                "mm_hashes": list(out["mm_hashes"]["image"]),
                "placeholders_sha256": digest(placeholders),
            })
        result["workloads"][name] = turns
        print(f"[{label}] {name}: " + ", ".join(
            f"t{t['turn']}={t['prompt_tokens']}:{t['prompt_token_ids_sha256'][:8]}" for t in turns),
            flush=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
