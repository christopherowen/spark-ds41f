#!/usr/bin/env python3
"""check_verify.py DIR: compare the SHA-256 prefixes each arm's container saw (verify-<arm>.txt)
with overlay-sha.txt (columns: image d5778ab5 | original c53 c0e25ff1 | r1 | path)."""
import sys
d = sys.argv[1]
exp = {}
for ln in open(f"{d}/overlay-sha.txt"):
    parts = ln.split()
    img, orig, r1, path = parts[0], parts[1], parts[2], parts[3]
    exp[path] = {"base": img, "orig": orig, "r1": img if r1 == "(image)" else r1}
# base arm: the no1009 files are the only overlay; other files are the image's
no1009 = {}
import hashlib, os
for ln in open(f"{d}/no1009-files.txt"):
    f = ln.strip(); no1009[f] = hashlib.sha256(open(f"{d}/ovl/no1009/{f}", "rb").read()).hexdigest()[:12]
for arm in ("base", "r1", "orig"):
    seen = {}
    for ln in open(f"{d}/verify-{arm}.txt"):
        p = ln.split()
        if len(p) == 2 and len(p[0]) == 12 and all(c in "0123456789abcdef" for c in p[0]):
            seen[p[1]] = p[0]
    bad = []
    for path, e in exp.items():
        want = no1009.get(path, e["base"]) if arm == "base" else e[arm]
        if seen.get(path) != want:
            bad.append((path, seen.get(path), want))
    print(arm, f"{len(seen)} files seen,", "all equal to expected" if not bad else f"MISMATCH {bad}")
