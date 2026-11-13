import hashlib, os, sys
root = sys.argv[1]
for top in ("vllm", "tests"):
    for d, _, fs in os.walk(os.path.join(root, top)):
        if "__pycache__" in d:
            continue
        for f in fs:
            if f.endswith((".pyc", ".so")):
                continue
            p = os.path.join(d, f)
            if os.path.islink(p):
                continue
            data = open(p, "rb").read()
            h = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            print(h, os.path.relpath(p, root))
