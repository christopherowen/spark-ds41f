"""Choose the row-band WO budgets and the FFN/NEXT budgets from a sweep's grid results, by the rule in README.md.

usage: compose.py <suite run directory holding own-ref, own-ref-again, own-wo*, own-ffn*, own-next*>
"""
import json
import math
import re
import sys
from pathlib import Path

BANDS = ((1, 3), (4, 16), (17, 32), (33, 64), (65, 96))
REF_MB = {"wo": 15, "ffn": 12, "next": 12}
GRID_MARGIN = 0.0042


def load(root: Path, arm: str) -> dict:
    return {s["label"]: s for s in json.loads((root / arm / "grid.json").read_text())["engine"]}


def band_of(n: int) -> tuple[int, int] | None:
    return next((b for b in BANDS if b[0] <= n <= b[1]), None)


def main() -> None:
    root = Path(sys.argv[1])
    arms = {p.name: load(root, p.name) for p in sorted(root.iterdir()) if (p / "grid.json").exists()}
    refs = [arms["own-ref"], arms["own-ref-again"]]
    labels = list(refs[0])
    shape = {l: tuple(map(int, re.search(r"(\d+)x(\d+)", l).groups())) for l in labels}  # requests, rows

    def val(a: dict, l: str, key: str) -> float:
        return a[l][key]["mean"]

    def band_sum(a: dict, band: tuple[int, int]) -> float:
        return (sum(val(a, l, "forward_ms") for l in labels if band_of(shape[l][1]) == band)
                + sum(val(a, l, "drafter_ms") for l in labels if band_of(shape[l][0]) == band))

    def geomean(a: dict) -> float:
        return math.exp(sum(math.log(val(a, l, "step_ms")) for l in labels) / len(labels))

    levels = {k: {int(re.search(r"\d+", n).group()): a for n, a in arms.items() if n.startswith(f"own-{k}")}
              for k in ("wo", "ffn", "next")}
    print(f"{root.name}: {len(arms)} arms, {len(labels)} shapes; reference boots differ by "
          f"{100 * (geomean(refs[1]) / geomean(refs[0]) - 1):+.2f}% on the grid")

    print("\nWO: band sum (target forwards of the band's shapes + drafter times of the shapes whose requests are in it), ms")
    wo_levels = sorted(levels["wo"])
    print("%-8s %9s %9s | %s | chosen" % ("band", "ref", "ref-again", " ".join("%8s" % f"wo{x}" for x in wo_levels)))
    chosen = {}
    for band in BANDS:
        r = [band_sum(a, band) for a in refs]
        if not r[0]:
            continue
        sums = {x: band_sum(levels["wo"][x], band) for x in wo_levels}
        best = min(sums, key=sums.get)
        take = sums[best] < min(r) - abs(r[0] - r[1])
        if take:
            chosen[band] = best
        cells = " ".join("%8s" % ("%+.2f%%" % (100 * (sums[x] / (sum(r) / 2) - 1))) for x in wo_levels)
        print("%-8s %9.3f %9.3f | %s | %s" % ("%d-%d" % band, r[0], r[1], cells,
                                               f"{best} MB" if take else "15 MB (no value beats both reference boots by their spread)"))

    print("\nFFN and NEXT: grid geometric mean against the reference boots")
    g = [geomean(a) for a in refs]
    other = {}
    for k in ("ffn", "next"):
        for x, a in sorted(levels[k].items()):
            gm = geomean(a)
            slower = [("%d-%d" % b) for b in BANDS if band_sum(refs[0], b) and band_sum(a, b) > max(band_sum(r, b) for r in refs)]
            take = gm < min(g) * (1 - GRID_MARGIN) and not slower
            print("  %s%-3d %+.2f%% against the mean reference; bands slower than both reference boots: %s%s"
                  % (k, x, 100 * (gm / (sum(g) / 2) - 1), ", ".join(slower) or "none", "  -> taken" if take else ""))
            if take and (k not in other or gm < other[k][1]):
                other[k] = (x, gm)

    ref_mean = {l: {key: sum(val(a, l, key) for a in refs) / 2 for key in ("step_ms", "forward_ms", "drafter_ms")} for l in labels}
    logs = 0.0
    for l in labels:
        step = ref_mean[l]["step_ms"]
        fb, db = band_of(shape[l][1]), band_of(shape[l][0])
        if fb in chosen:
            step += val(levels["wo"][chosen[fb]], l, "forward_ms") - ref_mean[l]["forward_ms"]
        if db in chosen:
            step += val(levels["wo"][chosen[db]], l, "drafter_ms") - ref_mean[l]["drafter_ms"]
        logs += math.log(step / ref_mean[l]["step_ms"])
    bands = ",".join("%d-%d=%d" % (*b, x) for b, x in sorted(chosen.items()))
    print(f"\nVLLM_DS41_L2_PREFETCH_WO_BANDS={bands or '(unset)'}")
    for k, (x, _) in other.items():
        print(f"VLLM_DS41_L2_PREFETCH_{k.upper()}_MB={x}")
    print(f"predicted grid change from the WO bands alone: {100 * (math.exp(logs / len(labels)) - 1):+.2f}%")


if __name__ == "__main__":
    main()
