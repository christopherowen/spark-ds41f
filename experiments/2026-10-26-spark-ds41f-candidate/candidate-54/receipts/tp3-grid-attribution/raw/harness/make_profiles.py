"""Write candidate-54/tp3-switched-no<PR>.json and -dev.json for each overlay directory,
as candidate-53's tp3-switched-no960-no998 profiles are written."""
import json
import os
import sys

ROOT = sys.argv[1]  # .../tp3-grid-attribution
EXP = "experiments/2026-10-26-spark-ds41f-candidate/candidate-54"
NOTES = {
    "no961": "candidate-54 without #961 (0781, paced 64 KiB L2 prefetch): the same image, with the 2 non-test files that differ when 0781 is reverted from candidate-54's tree (5f021d9a), mounted from no961/overlay. No other candidate-54 patch touches them.",
    "no963": "candidate-54 without #963 round 1 (0782, pack12 plane loads and per-column vocab exceptions): the same image, with the 2 non-test files that differ when 0782 is reverted from candidate-54's tree (5f021d9a), mounted from no963/overlay. Shared files (gemm.py with #1016 r1, #991 r1 and #987; vocab.py with #997 and #1001 r1) keep the other PRs' changes.",
    "no987": "candidate-54 without #987 (0783, WO-B WIDE_DECODE 128 rows): the same image, with the 2 non-test files that differ when 0783 is reverted from candidate-54's tree (5f021d9a), mounted from no987/overlay. gemm.py keeps the other PRs' changes.",
    "no1001": "candidate-54 without #1001 round 1 (0784, 128-row vocab head tile stages): the same image, with the 1 non-test file that differs when 0784 is reverted from candidate-54's tree (5f021d9a), mounted from no1001/overlay. vocab.py keeps the other PRs' changes.",
    "no1030": "candidate-54 without #1030 (0785, moe_swiglu_cast BF16 rounding): the same image, with the 1 non-test file that differs when 0785 is reverted from candidate-54's tree (5f021d9a), mounted from no1030/overlay. tilelang/moe.py keeps #966's change. #1030 declares a temperature-0 output change, so this arm's outputs differ from candidate-54's by construction.",
    "no960": "candidate-54 without #960 round 1 (0788, FULL-graph attention/CED metadata replay): the same image, with the 5 non-test files that differ when 0788 is reverted from candidate-54's tree (5f021d9a), mounted from no960/overlay. Shared files (model_runner.py, dflash/speculator.py, nvidia/model_state.py) keep the other PRs' changes.",
    "no998": "candidate-54 without #998 round 1 (0791, compact packed vocab tables and expand display carve-out weights): the same image, with the 6 non-test files that differ when 0791 is reverted from candidate-54's tree (5f021d9a), mounted from no998/overlay. Shared files (model_runner.py, packed_bf16.py, fused_moe/tilelang.py) keep the other PRs' changes.",
    "no963-no1030": "candidate-54 without #963 round 1 (0782) and #1030 (0785), grouped as the two kernel-level suspects their own marks checked at TP3 shapes: the same image, with the 3 non-test files that differ when 0785 and then 0782 are reverted from candidate-54's tree (5f021d9a), mounted from no963-no1030/overlay. Shared files (gemm.py, vocab.py, tilelang/moe.py) keep the other PRs' changes. #1030 declares a temperature-0 output change, so this arm's outputs differ from candidate-54's by construction.",
    "noC51": "candidate-54 without candidate-51's four patches (0781 #961, 0782 #963 r1, 0783 #987, 0784 #1001 r1): the same image, with the 5 non-test files that differ when 0781..0784 are reverted from candidate-54's tree (5f021d9a), mounted from noC51/overlay. Shared files keep the other PRs' changes.",
}
base = os.path.join(ROOT, EXP)
for arm in sorted(NOTES):
    ov = os.path.join(base, arm, "overlay")
    if not os.path.isdir(ov):
        continue
    files = sorted(os.path.relpath(os.path.join(d, f), ov) for d, _, fs in os.walk(ov) for f in fs)
    mounts = [["{checkout}/cache", "/cache", "rw"]]
    mounts += [[f"{{checkout}}/{EXP}/{arm}/overlay/{p}", f"/opt/spark-ds41f/candidate/vllm/{p}", "ro"] for p in files]
    prof = {"notes": NOTES[arm],
            "extends": {"config": f"{EXP}/tp3-switched.json"},
            "set": {"container": {"mounts": mounts}}}
    dev = {"notes": f"tp3-switched-{arm}.json with the dev-mode collective RPC, for the engine grid.",
           "extends": {"config": f"{EXP}/tp3-switched-{arm}.json"},
           "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}}}
    json.dump(prof, open(os.path.join(base, f"tp3-switched-{arm}.json"), "w"), indent=2)
    open(os.path.join(base, f"tp3-switched-{arm}.json"), "a").write("\n")
    json.dump(dev, open(os.path.join(base, f"tp3-switched-{arm}-dev.json"), "w"), indent=2)
    open(os.path.join(base, f"tp3-switched-{arm}-dev.json"), "a").write("\n")
    print(arm, len(files))
