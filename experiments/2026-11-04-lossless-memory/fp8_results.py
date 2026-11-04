"""Strict receipt gates for the exact-byte GPU microbenchmark."""

import hashlib
import math
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
SOURCES = ("fp8_probe.py", "fp8_probe.cu", "fp8_probe_frame.py",
           "collect_weights.py", "memory_codec.py")
FIXED_SOURCES = ("fp8_fixed_probe.py", "fp8_fixed_probe.cu", "fp8_fixed_frame.py",
                 "collect_weights.py", "memory_codec.py")
FIXED_EXACT_FLAG = "gpu_all_encodings_and_exception_boundaries_exact"


def source_hashes(sources=SOURCES):
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in sources}


def validate(native, verdict, *, lanes, image, revision, axis="lanes",
             exact_flag="gpu_all_encodings_and_long_codes_exact", sources=SOURCES):
    if (verdict.get("passed") is not True or verdict.get("exit") != 0
            or verdict.get("image") != image or verdict.get("errors")
            or verdict.get("failed") or native.get("gpu_byte_exact") is not True
            or native.get(exact_flag) is not True
            or native.get("status") != "measured microbenchmark; serving remains unqualified"
            or native.get(axis) != lanes or native.get("model_revision") != revision
            or native.get("sources_sha256") != source_hashes(sources)):
        raise ValueError("incomplete, mismatched or inexact decoder receipt")
    if axis == "threads" and native.get("gpu_consumer_checksums_exact") is not True:
        raise ValueError("fixed-field consumer differs from the immutable source")
    if not native.get("inputs") or native.get("decoded_bytes_per_replay", 0) < 64 * 1024**2:
        raise ValueError("missing checkpoint identity or working set no larger than L2")
    for key in ("host_peak_rss_kib", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        if type(native.get(key)) is not int or native[key] <= 0:
            raise ValueError("missing memory observation")
    for name in ("raw", "encoded"):
        values = native.get("kernel_ms", {}).get(name, [])
        if len(values) < 5 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                  for v in values):
            raise ValueError("missing, nonfinite or invalid timing observations")
        if native.get("allocated_input_bytes", {}).get(name, 0) <= 0:
            raise ValueError("missing actual input allocation")


def summarize(receipts, *, axis="lanes", exact_flag="gpu_all_encodings_and_long_codes_exact"):
    if len(receipts) != 2 or receipts[0]["inputs"] != receipts[1]["inputs"]:
        raise ValueError("need first/repeat observations of the same checkpoint tiles")
    if any(r.get("gpu_byte_exact") is not True or
           r.get(exact_flag) is not True for r in receipts):
        raise ValueError("an inexact reconstruction cannot become a timing win")
    for key in ("sources_sha256", axis, "model_revision", "decoded_bytes_per_replay",
                "allocated_input_bytes", "sample_encoded_bytes_with_metadata"):
        if receipts[0].get(key) != receipts[1].get(key):
            raise ValueError("first/repeat source, layout or allocation differs")
    ratios = [statistics.median(r["kernel_ms"]["encoded"]) /
              statistics.median(r["kernel_ms"]["raw"]) for r in receipts]
    # Resample whole process observations, then the observations within each.
    # This is exploratory uncertainty from two fresh containers, not serving CI.
    rng, bootstrap = random.Random(41), []
    for _ in range(2000):
        values = []
        for receipt in rng.choices(receipts, k=2):
            means = {name: statistics.median(rng.choices(times, k=len(times)))
                     for name, times in receipt["kernel_ms"].items()}
            values.append(means["encoded"] / means["raw"])
        bootstrap.append(statistics.mean(values))
    bootstrap.sort()
    ci = [bootstrap[49], bootstrap[1949]]
    first = receipts[0]
    allocation_ratio = first["allocated_input_bytes"]["encoded"] / first["decoded_bytes_per_replay"]
    return {"scope": "standalone byte consumer; not GEMM or serving performance",
            axis: first[axis], "first_repeat_median_time_ratios": ratios,
            "exploratory_bootstrap_ratio_ci95": ci,
            "encoded_input_over_real_raw_bytes": allocation_ratio,
            "decision": ("retain for fused-consumer investigation"
                         if max(ratios) < 1 and ci[1] < 1 and allocation_ratio < 1
                         else "reject this microdecoder layout"),
            "serving_performance_verified": False}
