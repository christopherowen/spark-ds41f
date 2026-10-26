#!/usr/bin/env python3
"""The gated rejection-sampler benchmark's results as a table.

usage: bench_table.py OUTPUT.txt

Reads the JSON result lines of benchmarks/kernels/benchmark_tilelang_rejection.py in a
kernel-lab bundle's output.txt: per case, the medians of ours and the baseline's graph
replays (microseconds a call), and the 95% interval of the paired difference (ours
minus the baseline); "slower" where the interval lies above 0 (the gate).
"""
import json
import sys


def main() -> None:
    print("| Kernel | Rows | Dtype | Temperatures | Dead rows | Ours (us) | Baseline (us) "
          "| Difference, 95% (us) | Gate |")
    print("| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |")
    for line in open(sys.argv[1]):
        if not line.startswith('{"kernel"'):
            continue
        r = json.loads(line)
        lo, hi = r["interval_us"]
        print(f"| {r['kernel']} | {r['rows']} | {r['dtype']} | {r['temperatures']} "
              f"| {'yes' if r.get('dead_rows') else 'no'} | {r['tilelang_us']:.2f} "
              f"| {r['baseline_us']:.2f} | {lo:+.2f} to {hi:+.2f} "
              f"| {'slower' if r['slower'] else 'pass'} |")


if __name__ == "__main__":
    main()
