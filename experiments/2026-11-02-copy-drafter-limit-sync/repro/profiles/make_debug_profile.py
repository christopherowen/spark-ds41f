import json

base = json.load(open("config/site-cand37-ring2-copy8.local.json"))
base["notes"] = "debug arm: copy8 + instrumented copy_drafts.py overlay (bounds checks)"
container = base.setdefault("set", {}).setdefault("container", {})
container["mounts"] = [[
    "$HOME/copy-debug/copy_drafts.py",
    "/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/copy_drafts.py",
    "ro",
]]
with open("config/site-cand37-ring2-copy8-debug.local.json", "w") as fh:
    json.dump(base, fh, indent=1)
print("wrote debug profile")
