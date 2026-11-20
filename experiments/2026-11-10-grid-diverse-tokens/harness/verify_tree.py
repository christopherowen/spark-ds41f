# usage: python3 /h/verify_tree.py LSTREE_FILE [ROOT]
# Checks every tracked file of a git tree (``git ls-tree -r <tree>`` output) against the
# files the container sees under ROOT (default /opt/spark-ds41f/candidate/vllm): git blob
# SHA-1 of regular files, link target of symlinks; submodules are skipped.
import hashlib
import os
import sys

lst, root = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "/opt/spark-ds41f/candidate/vllm")
ok = bad = missing = skipped = 0
bad_paths, missing_paths = [], []
for line in open(lst):
    meta, path = line.rstrip("\n").split("\t", 1)
    mode, kind, sha = meta.split()
    p = os.path.join(root, path)
    if kind != "blob":
        skipped += 1
        continue
    if mode == "120000":
        if os.path.islink(p):
            data = os.readlink(p).encode()
        else:
            missing += 1
            missing_paths.append(path)
            continue
    else:
        if not os.path.isfile(p):
            missing += 1
            missing_paths.append(path)
            continue
        data = open(p, "rb").read()
    got = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if got == sha:
        ok += 1
    else:
        bad += 1
        bad_paths.append(path)
print(f"tree list: {lst}")
print(f"root: {root}")
print(f"match: {ok}  differ: {bad}  missing: {missing}  skipped(non-blob): {skipped}")
for p in bad_paths:
    print(f"DIFFER  {p}")
for p in missing_paths[:50]:
    print(f"MISSING {p}")
if len(missing_paths) > 50:
    print(f"... {len(missing_paths) - 50} more missing")
