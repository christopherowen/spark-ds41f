# usage (in the image, no GPU): python3 /h/screenregs.py CACHE_DIR NAME
# For each cached cubin defining kernel NAME: registers, launch bound (EIATTR max threads)
# and the distinct DEPBAR.LE depths of its SASS (the cp.async pipeline drain).
import glob
import os
import re
import subprocess
import sys

os.environ["PATH"] = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin:" + os.environ["PATH"]
cu = "/usr/local/cuda/bin/cuobjdump"
for f in glob.glob(f"{sys.argv[1]}/cache/*/cuda-binaries/*/kernel.cubin"):
    res = subprocess.run([cu, "--dump-resource-usage", f], capture_output=True, text=True).stdout
    m = re.search(rf"Function {sys.argv[2]}_kernel:\s*REG:(\d+)", res)
    if not m:
        continue
    elf = subprocess.run([cu, "-elf", f], capture_output=True, text=True).stdout
    mt = re.search(r"EIATTR_MAX_THREADS[\s\S]{0,200}?0x([0-9a-f]+)", elf)
    sass = subprocess.run([cu, "-sass", f], capture_output=True, text=True).stdout
    dep = sorted(set(int(x) for x in re.findall(r"DEPBAR\.LE SB0, 0x([0-9a-f]+)", sass)), key=int) if False else sorted({int(x, 16) for x in re.findall(r"DEPBAR\.LE SB0, 0x([0-9a-f]+)", sass)})
    print(f"REG:{m.group(1)} max_threads:{int(mt.group(1), 16) if mt else '?'} depbar_le:{dep} {os.path.basename(os.path.dirname(f))[:12]}")
