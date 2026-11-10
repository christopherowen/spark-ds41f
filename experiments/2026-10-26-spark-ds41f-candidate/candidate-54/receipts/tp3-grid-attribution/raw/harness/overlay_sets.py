"""For each arm tree: the non-test files whose blob differs from the candidate-50 image
(out/image-c50-blobs.txt), and for each no<PR> tree the non-test files that differ from
candidate-54's tree. Writes out/ovl-<arm>.txt and out/rev-<arm>.txt."""
import json
import subprocess
import sys

W = "/home/chris/spark-lab/agents/tp3-grid-attribution"
img = {}
for line in open(f"{W}/out/image-c50-blobs.txt"):
    h, p = line.split(maxsplit=1)
    img[p.strip()] = h


def blobs(tree_dir):
    out = subprocess.run(["git", "-C", tree_dir, "ls-tree", "-r", "HEAD", "--", "vllm", "tests"],
                         capture_output=True, text=True, check=True).stdout
    d = {}
    for line in out.splitlines():
        meta, p = line.split("\t", 1)
        d[p] = meta.split()[2]
    return d


arms = ["c54"] + sys.argv[1:]
c54 = blobs(f"{W}/c54")
summary = {}
for arm in arms:
    t = c54 if arm == "c54" else blobs(f"{W}/trees/{arm}")
    vs_img = sorted(p for p, h in t.items() if img.get(p) != h and not p.startswith("tests/"))
    open(f"{W}/out/ovl-{arm}.txt", "w").write("\n".join(vs_img) + "\n")
    rev = sorted(p for p in set(t) | set(c54) if t.get(p) != c54.get(p) and not p.startswith("tests/"))
    if arm != "c54":
        open(f"{W}/out/rev-{arm}.txt", "w").write("\n".join(rev) + "\n")
    summary[arm] = {"vs_image": len(vs_img), "vs_c54_nontest": rev}
print(json.dumps(summary, indent=1))
