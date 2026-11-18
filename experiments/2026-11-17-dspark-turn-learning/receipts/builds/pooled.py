"""Every grid of the 2026-11-17 jobs (to 01:05 on 11-18) on tp4-switched, by arm, against the pool of that day's learner-off boots
(the same profile and image in every job); run from the repository root."""
import json, math, statistics

R = "experiments/2026-11-17-dspark-turn-learning/receipts/"
ARMS = {  # arm: [(receipt dir, boot labels)]
    "learner off": [("offstep-screen", ["off-1", "off-2"]), ("comm-attribution", ["off-1", "off-2"]),
                    ("exchange", ["off-1", "off-2"]), ("reserve", ["off-1", "off-2"]), ("streams", ["off-1", "off-2"]),
                    ("stepops", ["off-1", "off-2"]), ("builds", ["off-1", "off-2"]), ("commthreads-2", ["off-1"]),
                    ("split", ["off-1", "off-2"])],
    "memory held from load (150, 350 MiB)": [("reserve", ["reserve150-1", "reserve150-2", "reserve350-1", "reserve350-2"])],
    "three bare streams": [("streams", ["streams3-1", "streams3-2"])],
    "the learner's NCCL operations at load, step's communicator": [("stepops", ["learnerops-1", "learnerops-2"])],
    "BF16 all-reduces at load, step's communicator": [("stepops", ["bf16ops-1", "bf16ops-2"])],
    "a second communicator made at load, idle": [("comm-attribution", ["comm-1", "comm-2"]), ("streams", ["commpad-1", "commpad-2"])],
    "a second communicator made at load, destroyed": [("commthreads-2", ["destroyed-1"])],
    "learner, its own communicator made at load (RoCE)": [("offstep-screen", ["offstep-1", "offstep-2"]), ("comm-attribution", ["offstep-1", "offstep-2"])],
    "learner, its own communicator made at load (Socket)": [("exchange", ["socket-1", "socket-2"])],
    "learner, the step's communicator": [("exchange", ["step-1", "step-2"]), ("builds", ["shared-1", "shared-2"])],
    "learner, its own communicator made at its first exchange": [("builds", ["lazy-1", "lazy-2"]), ("split", ["lazy-1", "lazy-2"])],
    "learner, end of turn skipped (the capture alone)": [("split", ["capture-1", "capture-2"])],
}


def gm(path):
    rows = json.load(open(path))["engine"]
    return math.exp(statistics.mean(math.log(r["step_ms"]["mean"]) for r in rows))


pool = [gm(f"{R}{d}/{b}/grid.json") for d, bs in ARMS["learner off"] for b in bs]
base = statistics.mean(pool)
sd = statistics.stdev(pool)
print(f"learner off: {len(pool)} boots, {base:.4f} ms (sd {sd:.4f} ms, {100 * sd / base:.2f}%)")
for arm, sources in ARMS.items():
    if arm == "learner off":
        continue
    values = [gm(f"{R}{d}/{b}/grid.json") for d, bs in sources for b in bs]
    print(f"{100 * (statistics.mean(values) / base - 1):+6.2f}%  {arm} ({len(values)} boots: "
          + ", ".join(f"{100 * (v / base - 1):+.2f}" for v in values) + ")")
