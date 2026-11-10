#!/usr/bin/env python3
"""Where a cold (turn-1) image request's extra time goes under round 2's fast path: the same
fresh images through ImageMediaIO.load_bytes and the DS4.1 processor's apply(), with the fast
path off (header cache off, processor context without the flag: candidate-54's behaviour on
this tree) and on (header cache on, flag on), alternating per image. CPU only; the processor
is built as #978's bench builds it, with the context flag in place of vllm_config.
Usage: cold_path_microbench.py <label> <out.json> [images per workload]
"""
import json
import statistics
import sys
import time
from types import SimpleNamespace

import pybase64

sys.path.insert(0, "/opt/exp")
import bench_mm_prefix_fastpath as bench  # noqa: E402
from vllm.multimodal.media import image as media_image  # noqa: E402
from vllm.multimodal.media.image import ImageMediaIO  # noqa: E402
from vllm.multimodal.processing.context import TimingContext  # noqa: E402
from vllm.multimodal.processing.inputs import ProcessorInputs  # noqa: E402

label, out_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30


def processor(flag):
    proc = bench._build_ds41_processor()
    ctx = {k: v for k, v in vars(proc.info.ctx).items() if k != "vllm_config"}
    ctx["mm_prefix_metadata_cache"] = flag
    proc.info.ctx = SimpleNamespace(**ctx)
    proc.prefix_metadata_cache_enabled = proc.cache is None and flag
    return proc


procs = {"off": processor(False), "on": processor(True)}
media_io = ImageMediaIO()
tok = procs["off"].info.get_tokenizer()
res = {"label": label, "images": N, "workloads": {}}
for name, w, h, fmt in (("1302x1302_jpeg", 1302, 1302, "JPEG"), ("1920x1080_png", 1920, 1080, "PNG")):
    times = {m: {"load": [], "apply": []} for m in procs}
    raws = [pybase64.b64decode(bench._make_encoded_image(w, h, fmt, seed=50000 + i)) for i in range(2 * N)]
    for i in range(N):
        for k, mode in enumerate(("off", "on") if i % 2 == 0 else ("on", "off")):
            data = raws[2 * i + k]  # a fresh image for every request (cold)
            media_image.set_validated_image_meta_cache_enabled(mode == "on")
            t0 = time.perf_counter()
            img = media_io.load_bytes(data)
            t1 = time.perf_counter()
            items = procs[mode].info.parse_mm_data({"image": [img]})
            out = procs[mode].apply(
                ProcessorInputs(prompt=tok.encode(f"{bench.IMAGE_PLACEHOLDER}\nQ{i}\n"),
                                mm_data_items=items), TimingContext())
            t2 = time.perf_counter()
            assert not out["mm_kwargs"]["image"][0].is_lazy
            times[mode]["load"].append((t1 - t0) * 1e3)
            times[mode]["apply"].append((t2 - t1) * 1e3)
    data = raws[0]
    parts = {}
    for part, fn in (("prekey", lambda: media_image._image_prekey(data)),
                     ("sha256", lambda: media_image._image_digest(data))):
        fn()
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        parts[part + "_ms"] = round((time.perf_counter() - t0) / 50 * 1e3, 4)
    res["workloads"][name] = {
        "bytes": len(data),
        **{f"{m}_{s}_median_ms": round(statistics.median(v), 3) for m in times for s, v in times[m].items()},
        **parts,
    }
    print(f"[{label}] {name}: {res['workloads'][name]}", flush=True)
media_image.set_validated_image_meta_cache_enabled(False)
json.dump(res, open(out_path, "w"), indent=1)
