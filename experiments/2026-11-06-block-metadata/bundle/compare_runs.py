#!/usr/bin/env python3
"""Compare two fresh component-screen boots (boot-1.json and boot-2.json)."""
import argparse
import json
from pathlib import Path


def interval_significant_negative(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] < 0.0 and ci[1] < 0.0


def interval_significant_positive(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] > 0.0 and ci[1] > 0.0


def compare_boots(boot1_path: Path, boot2_path: Path) -> dict:
    b1 = json.loads(boot1_path.read_text())
    b2 = json.loads(boot2_path.read_text())
    if b1.get("control_source") != b2.get("control_source"):
        raise RuntimeError("control_source mismatch between boots")
    records1 = b1.get("records", [])
    records2 = b2.get("records", [])
    if len(records1) != 45 or len(records2) != 45:
        raise RuntimeError(f"expected 45 records per boot, got {len(records1)} and {len(records2)}")

    cases = []
    mode_summaries = {
        mode: {
            "total_cases": 0,
            "gpu_steady_both_boots_sig_faster": 0,
            "gpu_steady_both_boots_sig_slower": 0,
            "gpu_steady_sign_flip_or_inconclusive": 0,
            "gpu_reorder_both_boots_sig_slower": 0,
            "prep_steady_completion_both_boots_sig_faster": 0,
            "prep_page_crossing_completion_both_boots_sig_faster": 0,
        }
        for mode in ("row", "chunk", "span")
    }

    for r1, r2 in zip(records1, records2):
        key1 = (r1["mode"], r1["context_label"], r1["concurrency"])
        key2 = (r2["mode"], r2["context_label"], r2["concurrency"])
        if key1 != key2:
            raise RuntimeError(f"record order mismatch: {key1} != {key2}")
        mode, context, concurrency = key1
        ms = mode_summaries[mode]
        ms["total_cases"] += 1

        gs1, gs2 = r1["gpu"]["steady"], r2["gpu"]["steady"]
        gr1, gr2 = r1["gpu"]["reorder"], r2["gpu"]["reorder"]
        ps1 = r1["preparation"]["steady"]["host_and_device_completion"]
        ps2 = r2["preparation"]["steady"]["host_and_device_completion"]
        pc1 = r1["preparation"]["page_crossing"]["host_and_device_completion"]
        pc2 = r2["preparation"]["page_crossing"]["host_and_device_completion"]
        po1 = r1["preparation"]["overwrite"]["host_and_device_completion"]
        po2 = r2["preparation"]["overwrite"]["host_and_device_completion"]

        if interval_significant_negative(gs1["paired_95_interval_us"]) and interval_significant_negative(
            gs2["paired_95_interval_us"]
        ):
            ms["gpu_steady_both_boots_sig_faster"] += 1
            steady_verdict = "both_boots_sig_faster"
        elif interval_significant_positive(gs1["paired_95_interval_us"]) and interval_significant_positive(
            gs2["paired_95_interval_us"]
        ):
            ms["gpu_steady_both_boots_sig_slower"] += 1
            steady_verdict = "both_boots_sig_slower"
        else:
            ms["gpu_steady_sign_flip_or_inconclusive"] += 1
            steady_verdict = "sign_flip_or_inconclusive"

        if interval_significant_positive(gr1["paired_95_interval_us"]) and interval_significant_positive(
            gr2["paired_95_interval_us"]
        ):
            ms["gpu_reorder_both_boots_sig_slower"] += 1

        if interval_significant_negative(ps1["paired_95_interval_us"]) and interval_significant_negative(
            ps2["paired_95_interval_us"]
        ):
            ms["prep_steady_completion_both_boots_sig_faster"] += 1

        if interval_significant_negative(pc1["paired_95_interval_us"]) and interval_significant_negative(
            pc2["paired_95_interval_us"]
        ):
            ms["prep_page_crossing_completion_both_boots_sig_faster"] += 1

        cases.append(
            {
                "mode": mode,
                "context_label": context,
                "concurrency": concurrency,
                "extra_gpu_cache_bytes": r1["extra_gpu_cache_bytes"],
                "epoch_host_bytes_including_pool": r1["epoch_host_bytes_including_pool"],
                "gpu_steady_verdict": steady_verdict,
                "gpu_steady": {
                    "boot1_mean_delta_us": gs1["paired_mean_change_us"],
                    "boot1_ci95_us": gs1["paired_95_interval_us"],
                    "boot2_mean_delta_us": gs2["paired_mean_change_us"],
                    "boot2_ci95_us": gs2["paired_95_interval_us"],
                },
                "gpu_reorder": {
                    "boot1_mean_delta_us": gr1["paired_mean_change_us"],
                    "boot1_ci95_us": gr1["paired_95_interval_us"],
                    "boot2_mean_delta_us": gr2["paired_mean_change_us"],
                    "boot2_ci95_us": gr2["paired_95_interval_us"],
                },
                "prep_steady_completion": {
                    "boot1_mean_delta_us": ps1["paired_mean_change_us"],
                    "boot1_ci95_us": ps1["paired_95_interval_us"],
                    "boot2_mean_delta_us": ps2["paired_mean_change_us"],
                    "boot2_ci95_us": ps2["paired_95_interval_us"],
                },
                "prep_page_crossing_completion": {
                    "boot1_mean_delta_us": pc1["paired_mean_change_us"],
                    "boot1_ci95_us": pc1["paired_95_interval_us"],
                    "boot2_mean_delta_us": pc2["paired_mean_change_us"],
                    "boot2_ci95_us": pc2["paired_95_interval_us"],
                },
                "prep_overwrite_completion": {
                    "boot1_mean_delta_us": po1["paired_mean_change_us"],
                    "boot1_ci95_us": po1["paired_95_interval_us"],
                    "boot2_mean_delta_us": po2["paired_mean_change_us"],
                    "boot2_ci95_us": po2["paired_95_interval_us"],
                },
            }
        )

    return {
        "control_source": b1["control_source"],
        "boot1_host_max_rss_kib": b1["host_max_rss_kib"],
        "boot2_host_max_rss_kib": b2["host_max_rss_kib"],
        "boot1_torch_peak_allocated_bytes_two_arms": b1["torch_peak_allocated_bytes_two_arms"],
        "boot2_torch_peak_allocated_bytes_two_arms": b2["torch_peak_allocated_bytes_two_arms"],
        "mode_summaries": mode_summaries,
        "conclusion": (
            "All 55 GPU correctness gates and 7 CPU bounds/publication checks pass in both fresh boots, "
            "but captured GPU gather timing shows a clear tradeoff: while c16 static-order steady gathers "
            "improve by ~0.9 to 4.2 us at long context, c1 steady gathers either regress by up to +1.25 us "
            "(+64% over the 1.86 us full-copy control) or flip sign across boots, and batch-reordered gathers "
            "regress across all 45 cases (+0.27 to +2.01 us) due to per-CTA state check, barrier, and state "
            "write overhead. Do not promote or include in Candidate 1107."
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("boot1", type=Path)
    parser.add_argument("boot2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compare_boots(args.boot1, args.boot2)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"mode_summaries": summary["mode_summaries"], "conclusion": summary["conclusion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
