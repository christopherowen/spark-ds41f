"""Per-arm step ms for the 24- and 32-row shapes (mean of 100 replays), from a suite run dir."""
import json, sys
from pathlib import Path
want = ["decode 4x24 rows @4K", "decode 8x24 rows @4K", "decode 8x24 rows @64K", "decode 8x24 rows @256K", "decode 16x32 rows @4K", "decode 16x48 rows @4K", "decode 8x48 rows @4K", "decode 12x72 rows @4K", "decode 16x96 rows @4K"]
arms = sorted([p for p in Path(sys.argv[1]).iterdir() if (p / "grid.json").exists()], key=lambda p: (p / "grid.json").stat().st_mtime)
ref = sys.argv[2] if len(sys.argv) > 2 else None
data = {a.name: {s["label"]: s["step_ms"]["mean"] for s in json.loads((a / "grid.json").read_text())["engine"]} for a in arms}
print("%-34s" % "arm" + "".join("%10s" % w.replace("decode ", "").replace(" rows ", "") for w in want))
for a, v in data.items():
    print("%-34s" % a + "".join("%10.3f" % v[w] for w in want))
if ref:
    def pair(n):
        xs = [data[k] for k in (n, n + "-again") if k in data]
        return {w: sum(x[w] for x in xs) / len(xs) for w in want}, len(xs)
    base, nb = pair(ref)
    print("\nchange against %s (mean of its %d boots), %%" % (ref, nb))
    for a in [a for a in data if not a.endswith("-again")]:
        v, n = pair(a)
        print("%-30s %d  " % (a, n) + "".join("%+10.2f" % (100 * (v[w] / base[w] - 1)) for w in want))
