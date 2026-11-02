import json

base = json.load(open("config/site-cand37-ring2-copy8.local.json"))
base["notes"] = "debug arm: copy8 + tilelang_copy.py padding 0 instead of -1"
container = base.setdefault("set", {}).setdefault("container", {})
container["mounts"] = [[
    "$HOME/copy-debug/tilelang_copy.py",
    "/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py",
    "ro",
]]
with open("config/site-cand37-ring2-copy8-pad0.local.json", "w") as fh:
    json.dump(base, fh, indent=1)
print("wrote debug profile")
