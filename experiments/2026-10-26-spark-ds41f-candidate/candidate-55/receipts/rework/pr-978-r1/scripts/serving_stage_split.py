#!/usr/bin/env python3
"""Where the serving path's JPEG time goes, per arm: ImageMediaIO.load_base64 and the
DS4.1 processor's apply() with the context the API server builds (no vllm_config, so
no prefix metadata cache), on turns 1..5 of one conversation, 5 conversations.
Usage: serving_stage_split.py <label> <out.json>
"""
import json
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "/opt/exp")
import bench_mm_prefix_fastpath as bench  # noqa: E402
from vllm.multimodal.media.image import ImageMediaIO  # noqa: E402
from vllm.multimodal.processing.context import TimingContext  # noqa: E402
from vllm.multimodal.processing.inputs import ProcessorInputs  # noqa: E402

label, out_path = sys.argv[1], sys.argv[2]
proc = bench._build_ds41_processor()
proc.info.ctx = SimpleNamespace(**{k: v for k, v in vars(proc.info.ctx).items() if k != "vllm_config"})
media_io = ImageMediaIO()
tok = proc.info.get_tokenizer()
res = {"label": label, "workloads": {}}
for name, w, h, fmt, mtype, count in [("1302x1302_jpeg_1img", 1302, 1302, "JPEG", "image/jpeg", 1),
                                      ("4032x3024_jpeg_1img", 4032, 3024, "JPEG", "image/jpeg", 1),
                                      ("1920x1080_png_1img", 1920, 1080, "PNG", "image/png", 1)]:
    st = {"t1_load": [], "t1_apply": [], "t2_load": [], "t2_apply": [], "type": None}
    for conv in range(6):
        b64 = [bench._make_encoded_image(w, h, fmt, seed=1000 * conv + 100 + i) for i in range(count)]
        for turn in range(1, 6):
            text = "".join(f"Image {i}: {bench.IMAGE_PLACEHOLDER}\n" for i in range(count)) + "Q\n" * turn
            t0 = time.perf_counter()
            imgs = [media_io.load_base64(mtype, b) for b in b64]
            t1 = time.perf_counter()
            items = proc.info.parse_mm_data({"image": imgs})
            proc.apply(ProcessorInputs(prompt=tok.encode(text), mm_data_items=items), TimingContext())
            t2 = time.perf_counter()
            if conv == 0:
                continue
            k = "t1" if turn == 1 else "t2"
            st[f"{k}_load"].append((t1 - t0) * 1e3)
            st[f"{k}_apply"].append((t2 - t1) * 1e3)
            media = getattr(imgs[0], "media", imgs[0])
            st["type"] = type(media).__name__
    res["workloads"][name] = {k: (round(statistics.median(v), 2) if isinstance(v, list) else v) for k, v in st.items()}
    print(f"[{label}] {name}: {res['workloads'][name]}", flush=True)
json.dump(res, open(out_path, "w"), indent=1)
