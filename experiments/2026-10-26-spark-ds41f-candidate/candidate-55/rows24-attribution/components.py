"""Step, forward, drafter and remainder (ms) per shape: mean of each side's boots, and the difference."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); a_names = sys.argv[2].split(","); b_names = sys.argv[3].split(",")
def load(n): return {s["label"]: s for s in json.loads((root / n / "grid.json").read_text())["engine"]}
def mean(names, label, key):
    xs = [load(n)[label][key]["mean"] for n in names]; return sum(xs) / len(xs)
labels = [s["label"] for s in json.loads((root / a_names[0] / "grid.json").read_text())["engine"]]
order = sorted(labels, key=lambda l: (int(l.split("x")[1].split()[0]), l))
print("%-24s %7s %7s %7s | %7s %7s %7s | d step  d fwd   d draft d rest   (B-A, ms; %% of A step)" % ("shape", "A step", "fwd", "draft", "B step", "fwd", "draft"))
for l in order:
    a = [mean(a_names, l, k) for k in ("step_ms", "forward_ms", "drafter_ms")]; b = [mean(b_names, l, k) for k in ("step_ms", "forward_ms", "drafter_ms")]
    ds, df, dd = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    print("%-24s %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | %+6.3f %+6.3f %+6.3f %+6.3f  (%+.2f%%)" % (l.replace("decode ", ""), *a, *b, ds, df, dd, ds - df - dd, 100 * ds / a[0]))
