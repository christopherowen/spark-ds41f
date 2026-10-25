#!/usr/bin/env python3
"""Host memory of the API server's image path, outside a serving engine (a kernel-local bundle).

usage: image_host_memory.py --config CONFIG.json [--budget-gib 48]

Runs inside the serving image on one node, CPU only. For K requests of N images at one source
size it does what the head's API server does with an OpenAI chat request:

- the parsed request holds one base64 string per image;
- vLLM's decoder (``ImageMediaIO.load_base64``, the chat endpoint's call for a ``data:`` URL)
  runs every image of every request on one pool of ``VLLM_MEDIA_LOADING_THREAD_COUNT`` (8)
  threads, so the requests decode at once;
- DS4.1's processor (``DeepseekV4VLProcessor``) then turns each request's images into ViT
  patches, one request at a time (the renderer has one multimodal worker).

Everything stays referenced until the case ends: an upper bound on what a request holds while
it waits. Each case runs in a fresh child process, so its peak (VmHWM) is its own; peaks are
reset between phases (``/proc/self/clear_refs``). One JSON line per case, then a summary.
Cases whose predicted peak exceeds --budget-gib are skipped and listed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGES = Path("/tmp/site-landmines-images")
# Square at the processor's token budget: the largest source it encodes without shrinking
# (31 x 31 tokens of 42 px, 994 tokens, 93 x 93 patches); a 12 MP and a 48 MP camera photo; and
# the largest square vLLM decodes (VLLM_MAX_IMAGE_PIXELS, 178,956,970 pixels by default).
SIZES = {"token-max": (1302, 1302), "12mp": (4032, 3024), "48mp": (8064, 6048), "vllm-max": (13376, 13376)}
# jpeg: a photo (RGB, a smooth field plus noise, quality 90). png-rgba: a graphic with an alpha
# channel (smooth, no noise): its decode converts RGBA to RGB, the largest transient.
FORMATS = {"jpeg": "image/jpeg", "png-rgba": "image/png"}


def cases() -> list[dict]:
    out = []
    for size in SIZES:
        for fmt in FORMATS:
            out.append({"size": size, "format": fmt, "requests": 1, "images": 1})
    for size in ("token-max", "12mp"):
        out += [{"size": size, "format": "jpeg", "requests": 1, "images": n} for n in (4, 8, 16, 32)]
    out += [{"size": "vllm-max", "format": fmt, "requests": 1, "images": 4} for fmt in FORMATS]
    out += [{"size": "token-max", "format": "jpeg", "requests": k, "images": n}
            for k in (4, 8) for n in (4, 8, 16, 32)]
    out += [{"size": "12mp", "format": "jpeg", "requests": k, "images": n} for k in (4, 8) for n in (4, 8)]
    out.append({"size": "vllm-max", "format": "png-rgba", "requests": 4, "images": 4})
    return out


def predicted_peak_bytes(case: dict) -> int:
    """Decoded RGB for every image, plus one RGBA decode and conversion per pool thread."""
    width, height = SIZES[case["size"]]
    pixels, count = width * height, case["requests"] * case["images"]
    return count * pixels * 3 + min(8, count) * pixels * 8


def generate(size: str, fmt: str) -> Path:
    """The base64 text of one encoded image of this size and format (written once)."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    path = IMAGES / f"{size}.{fmt}.b64"
    if path.exists():
        return path
    width, height = SIZES[size]
    rng = np.random.default_rng(20261026)
    field = Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)).resize(
        (width, height), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    if fmt == "jpeg":
        pixels = np.asarray(field, dtype=np.int16)
        pixels += rng.integers(-12, 13, pixels.shape, dtype=np.int16)
        Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(buffer, "JPEG", quality=90)
    else:
        alpha = Image.linear_gradient("L").resize((width, height))
        rgba = field.convert("RGBA")
        rgba.putalpha(alpha)
        rgba.save(buffer, "PNG")
    IMAGES.mkdir(parents=True, exist_ok=True)
    path.write_text(base64.b64encode(buffer.getvalue()).decode())
    return path


def status() -> dict[str, int]:
    """VmRSS and VmHWM in KiB."""
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            out[key] = int(value.split()[0])
    return out


def reset_peak() -> bool:
    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


def mib(kib: int) -> float:
    return round(kib / 1024, 1)


def measure(case: dict, config_path: str) -> dict:
    """One case, in this (fresh) process."""
    from vllm.models.deepseek_v41.common.mm_preprocess import (
        DeepseekV4VLProcessor,
        num_image_tokens,
    )
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    vision = json.loads(Path(config_path).read_text())["vision_config"]
    processor = DeepseekV4VLProcessor(DeepseekV41Config(vision_config=vision))
    media = ImageMediaIO(image_mode="RGB")
    source = IMAGES / f"{case['size']}.{case['format']}.b64"
    media_type = FORMATS[case["format"]]
    count = case["requests"] * case["images"]
    phases = {}
    base = status()["VmRSS"]
    resettable = reset_peak()

    started = time.perf_counter()
    bodies = [[source.read_text() for _ in range(case["images"])] for _ in range(case["requests"])]
    phases["read"] = (time.perf_counter() - started, status())

    reset_peak()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [[pool.submit(media.load_base64, media_type, data) for data in body] for body in bodies]
        decoded = [[future.result() for future in request] for request in futures]
    phases["decode"] = (time.perf_counter() - started, status())

    reset_peak()
    started = time.perf_counter()
    outputs = [processor(text="", images=[item.media for item in request]) for request in decoded]
    phases["preprocess"] = (time.perf_counter() - started, status())

    grids = outputs[0]["llm_grid"].tolist()
    patches = sum(output["patches"].numel() * output["patches"].element_size() for output in outputs)
    record = {**case, "pixels": SIZES[case["size"]][0] * SIZES[case["size"]][1],
              "base64_mib": round(source.stat().st_size / 2**20, 2), "rss_base_mib": mib(base),
              "peaks_per_phase": resettable, "patches_mib": round(patches / 2**20, 1),
              "tokens_per_image": num_image_tokens(*grids[0]),
              "patches_per_image": int(outputs[0]["patches"].shape[0] // case["images"])}
    for name, (seconds, state) in phases.items():
        record[name] = {"seconds": round(seconds, 2), "held_mib": mib(state["VmRSS"] - base),
                        "peak_mib": mib(state["VmHWM"] - base)}
    peak = max(phase["peak_mib"] for phase in (record["read"], record["decode"], record["preprocess"]))
    record["peak_mib"] = peak
    record["peak_per_image_mib"] = round(peak / count, 1)
    record["held_per_image_mib"] = round(record["preprocess"]["held_mib"] / count, 1)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="the checkpoint's config.json (vision_config)")
    parser.add_argument("--budget-gib", type=float, default=48.0)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(measure(json.loads(args.child), args.config)), flush=True)
        return 0

    planned = cases()
    for size, fmt in sorted({(case["size"], case["format"]) for case in planned}):
        started = time.perf_counter()
        path = generate(size, fmt)
        print(f"# generated {path.name}: {path.stat().st_size / 2**20:.2f} MiB of base64 "
              f"in {time.perf_counter() - started:.1f} s", flush=True)
    records, failed = [], 0
    for case in planned:
        predicted = predicted_peak_bytes(case) / 2**30
        if predicted > args.budget_gib:
            print(json.dumps({**case, "skipped": f"predicted peak {predicted:.1f} GiB > {args.budget_gib:g}"}))
            continue
        child = subprocess.run([sys.executable, __file__, "--config", args.config, "--child", json.dumps(case)],
                               text=True, capture_output=True)
        lines = [line for line in child.stdout.splitlines() if line.startswith("{")]
        if child.returncode or not lines:
            failed += 1
            print(json.dumps({**case, "error": (child.stderr or child.stdout).strip()[-2000:]}), flush=True)
            continue
        records.append(json.loads(lines[-1]))
        print(lines[-1], flush=True)

    print("# summary: case | MiB base64 | peak MiB (per image) | held MiB (per image) | decode s | preprocess s")
    for record in records:
        print(f"# {record['size']:>9} {record['format']:<8} {record['requests']}x{record['images']:<3} | "
              f"{record['base64_mib']:8.2f} | {record['peak_mib']:9.1f} ({record['peak_per_image_mib']:7.1f}) | "
              f"{record['preprocess']['held_mib']:9.1f} ({record['held_per_image_mib']:7.1f}) | "
              f"{record['decode']['seconds']:7.2f} | {record['preprocess']['seconds']:7.2f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
