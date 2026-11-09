#!/bin/bash
# setup.sh: image identity, the image's tree check, the two overlays (no GPU).
set -euo pipefail
B=$HOME/spark-lab/agents/candidate-53-own-marks
cd $B
IMG=$(docker image inspect spark-ds41f:candidate-50 --format '{{.Id}}')
echo "$IMG" > image-id
echo "host=$(hostname) image=$IMG created=$(docker image inspect $IMG --format '{{.Created}}') utc=$(date -u +%FT%TZ)"
docker image inspect $IMG --format '{{json .Config.Labels}}' | tr ',' '\n' | grep -i -E "tree|head|vllm|patch" || true
C50=d5778ab56647a20d94710ecb94cfd9a4de27bee9; C53=c0e25ff14f7cbf760973002447e50ba507b987f9
# 1. Every tracked file of candidate-50's tree (d5778ab5) against the image's /opt/spark-ds41f/candidate/vllm
git -C vllm-c53 ls-tree -r -z $C50 | python3 -c '
import sys
rows = [r.split(b"\t", 1) for r in sys.stdin.buffer.read().split(b"\0") if r]
rows = [(m.split(), p.decode()) for m, p in rows]
rows = [(m[2].decode(), p) for m, p in rows if m[1] == b"blob"]
open("c50-tracked.txt", "w").write("".join(p + "\n" for _, p in rows))
open("c50-blobs.txt", "w").write("".join(h + "  " + p + "\n" for h, p in rows))
' 
docker run --rm --network none --entrypoint python3 -v $B/c50-tracked.txt:/list:ro $IMG -c '
import hashlib, os
os.chdir("/opt/spark-ds41f/candidate/vllm")
for f in open("/list").read().splitlines():
    if os.path.islink(f):
        d = os.readlink(f).encode()
    elif os.path.isfile(f):
        d = open(f, "rb").read()
    else:
        print("MISSING  " + f); continue
    print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest() + "  " + f)
' > image-blobs.txt
if diff -q c50-blobs.txt image-blobs.txt >/dev/null; then echo "image tree: all $(wc -l < c50-tracked.txt) tracked files of $C50 equal"; else echo "image tree differs:"; diff c50-blobs.txt image-blobs.txt | head -20; fi
# 2. Overlays
git -C vllm-c53 diff --name-only $C50 $C53 > changed-files.txt
rm -rf ovl; mkdir -p ovl/c53 ovl/no1009
while read -r f; do mkdir -p ovl/c53/$(dirname $f); git -C vllm-c53 show $C53:$f > ovl/c53/$f; done < changed-files.txt
N=no1009/experiments/2026-10-26-spark-ds41f-candidate/candidate-50/no1009/overlay
(cd $N && find . -type f | sed 's,^\./,,' | sort) > no1009-files.txt
while read -r f; do mkdir -p ovl/no1009/$(dirname $f); cp $N/$f ovl/no1009/$f; done < no1009-files.txt
while read -r f; do
  s1=$(git -C vllm-c53 show $C50:$f | sha256sum | cut -c1-12); s2=$(sha256sum < ovl/c53/$f | cut -c1-12)
  s0=$(git -C vllm-c53 show 14c5a2dc924541ce4eda44d5b39d811be15f2363:$f | sha256sum | cut -c1-12)
  echo "$s1 $s0 $s2 $f"
done < changed-files.txt > overlay-sha.txt
echo "# columns: image (candidate-50 d5778ab5) | base without 0778 (14c5a2dc) | candidate-53 (c0e25ff1) | path" | cat - overlay-sha.txt > overlay-sha.tmp && mv overlay-sha.tmp overlay-sha.txt
cat overlay-sha.txt
wc -l changed-files.txt no1009-files.txt
