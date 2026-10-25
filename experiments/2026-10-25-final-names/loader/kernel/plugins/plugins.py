#!/usr/bin/env python3
"""What VLLM_PLUGINS loads in the image, and which loader `--load-format direct` gets
(a kernel-lab script).

usage: plugins.py

Lists the image's vllm.general_plugins entry points. Then, in a fresh interpreter for each
setting (unset, empty, b12x_loader), it loads the general plugins as every vLLM process
does and prints:
- the plugins loaded;
- the B12X modules they imported;
- the class get_model_loader returns for `direct` and for `b12x`.

It passes when an empty VLLM_PLUGINS loads no plugin and imports no B12X module, and
`direct` is DirectModelLoader under every setting.
"""

import json
import os
import subprocess
import sys
from importlib.metadata import entry_points

PROBE = r"""
import json, sys
from vllm.plugins import load_general_plugins, load_plugins_by_group, DEFAULT_PLUGINS_GROUP
loaded = sorted(load_plugins_by_group(DEFAULT_PLUGINS_GROUP))
load_general_plugins()
b12x = sorted(name for name in sys.modules if name.split(".")[0] == "b12x")
from vllm.config.load import LoadConfig
from vllm.model_executor import model_loader
direct = type(model_loader.get_model_loader(LoadConfig(load_format="direct"))).__module__
registered = model_loader._LOAD_FORMAT_TO_MODEL_LOADER.get("b12x")
print(json.dumps({"loaded": loaded, "b12x_modules_after_plugins": len(b12x),
                  "direct_loader": direct, "b12x_registered_by": getattr(registered, "__module__", None)}))
"""


def main() -> int:
    points = {point.name: point.value for point in entry_points(group="vllm.general_plugins")}
    print(json.dumps({"vllm.general_plugins": points}), flush=True)
    results = {}
    for label, value in (("unset", None), ("empty", ""), ("b12x_loader", "b12x_loader")):
        env = {key: item for key, item in os.environ.items() if key != "VLLM_PLUGINS"}
        if value is not None:
            env["VLLM_PLUGINS"] = value
        process = subprocess.run([sys.executable, "-c", PROBE], env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
        results[label] = json.loads(lines[-1]) if lines else {"error": process.returncode}
        print(json.dumps({"VLLM_PLUGINS": label, **results[label]}), flush=True)
    direct = "vllm.model_executor.model_loader.direct_loader"
    ok = (results["empty"].get("loaded") == [] and results["empty"].get("b12x_modules_after_plugins") == 0
          and all(result.get("direct_loader") == direct for result in results.values()))
    print(json.dumps({"summary": "plugins", "passed": ok}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
