"""changed_lines.py <vllm-src> <base> <head> <trace.json>

Lines a step-cost trace executed that the base..head diff added or changed
(new-side line numbers), per component."""
import json
import re
import subprocess
import sys

src, base, head, trace = sys.argv[1:5]
diff = subprocess.run(
    ["git", "-C", src, "diff", "-U0", base, head, "--", "vllm"],
    capture_output=True, text=True, check=True,
).stdout
added, cur = {}, None
for line in diff.splitlines():
    if line.startswith("+++ b/"):
        cur = line[6:]
        added.setdefault(cur, set())
    m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
    if m and cur:
        start, n = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        added[cur].update(range(start, start + n))
t = json.load(open(trace))
out = {}
for comp, files in t["lines"].items():
    hits = {f: sorted(set(lines) & added.get(f, set())) for f, lines in files.items()}
    out[comp] = {f: v for f, v in hits.items() if v}
print(json.dumps({
    "base": base, "head": head, "trace": trace,
    "changed_lines_by_file": {f: len(v) for f, v in added.items()},
    "executed_changed_lines": out,
}, indent=1))
