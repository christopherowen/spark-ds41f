# usage (in the image, no GPU): python3 /h/sass.py CACHE_DIR
# For every cached TileLang cubin: each kernel's name, the SHA-256 of its SASS (cuobjdump
# -sass; address and encoding comments removed) and its register count, one line per kernel.
import glob
import hashlib
import os
import re
import subprocess
import sys

os.environ["PATH"] = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin:" + os.environ["PATH"]
rows = set()
for f in glob.glob(f"{sys.argv[1]}/cache/*/cuda-binaries/*/kernel.cubin"):
    sass = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", f], capture_output=True, text=True).stdout
    res = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", f], capture_output=True, text=True).stdout
    regs = dict(re.findall(r"Function (\S+):\s*REG:(\d+)", res))
    for chunk in re.split(r"\n\s*Function : ", sass)[1:]:
        name, _, body = chunk.partition("\n")
        body = re.sub(r"/\*[0-9a-fx]+\*/", "", body)
        body = re.sub(r"\s+", " ", body)
        rows.add(f"{name.strip()} {hashlib.sha256(body.encode()).hexdigest()[:16]} REG:{regs.get(name.strip(), '?')}")
for r in sorted(rows):
    print(r)
