"""audit_attrs.py PATCH REPO BASE_REV: every attribute name the patch's added non-test lines
access (obj.attr, getattr(obj, "attr")), and where the base tree (BASE_REV, before the
patch) and the patched tree (HEAD) define it (def/class attribute/assignment/kwarg)."""
import re, subprocess, sys, json
patch, repo, base = sys.argv[1:4]
added = {}
cur = None
for line in open(patch, encoding="utf-8"):
    if line.startswith("+++ b/"):
        cur = line[6:].strip()
        continue
    if cur and line.startswith("+") and not line.startswith("+++") and not cur.startswith("tests/"):
        body = line[1:]
        code = body.split("#", 1)[0]
        for m in re.finditer(r"\.([A-Za-z_][A-Za-z0-9_]*)", code):
            added.setdefault(m.group(1), set()).add(cur)
        for m in re.finditer(r"getattr\(\s*[^,]+,\s*\"([A-Za-z_][A-Za-z0-9_]*)\"", code):
            added.setdefault(m.group(1), set()).add(cur + " (getattr)")
def defined(rev, name):
    pat = (rf"(def {name}\b|class {name}\b|\.{name}\s*(:[^=]*)?=[^=]|^\s+{name}\s*(:[^=]*)?=[^=]|"
           rf"\b{name}\s*=\s*|import .*\b{name}\b)")
    out = subprocess.run(["git", "-C", repo, "grep", "-n", "-E", pat, rev, "--", "vllm/"],
                         capture_output=True, text=True).stdout
    return [l.split(":", 2)[1] + ":" + l.split(":", 3)[2] for l in out.splitlines()][:3]
rows = []
for name in sorted(added):
    b = defined(base, name)
    h = defined("HEAD", name)
    rows.append({"attr": name, "used_in": sorted(added[name]), "base_defs": len(b) > 0,
                 "base_example": b[:2], "patched_defs": len(h) > 0, "patched_example": h[:2]})
for r in rows:
    flag = "OK  " if r["base_defs"] else ("PATCH" if r["patched_defs"] else "MISSING")
    print(f"{flag:8} {r['attr']:40} {', '.join(r['used_in'])}  base={r['base_example'][:1]} head={r['patched_example'][:1]}")
