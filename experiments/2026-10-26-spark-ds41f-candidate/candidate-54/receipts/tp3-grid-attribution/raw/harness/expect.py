"""out/expect-<arm>.json: {path: blob in the arm's tree} for every non-test file that
differs between the candidate-50 image and candidate-54's tree (the 22 files), so a boot
checks mounted and unmounted files alike."""
import json
import subprocess
import sys

W = "/home/chris/spark-lab/agents/tp3-grid-attribution"
universe = [p for p in open(f"{W}/out/c54-vs-image-c50.txt").read().split() if not p.startswith("tests/")]
for arm in sys.argv[1:]:
    tree = f"{W}/c54" if arm == "c54" else f"{W}/trees/{arm}"
    out = subprocess.run(["git", "-C", tree, "ls-tree", "HEAD", "--", *universe],
                         capture_output=True, text=True, check=True).stdout
    d = {}
    for line in out.splitlines():
        meta, p = line.split("\t", 1)
        d[p] = meta.split()[2]
    assert len(d) == len(universe), (arm, len(d))
    json.dump(d, open(f"{W}/out/expect-{arm}.json", "w"), indent=1)
    print(arm, len(d))
