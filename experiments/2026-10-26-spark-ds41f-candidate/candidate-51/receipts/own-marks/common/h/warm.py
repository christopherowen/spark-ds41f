# usage: python3 /h/warm.py SCRIPT [args...]
# At least 250 ms of GPU warm-up (BF16 matmuls), then SCRIPT runs unchanged as __main__
# with sys.argv = [SCRIPT, args...].
import runpy
import sys

sys.path.insert(0, "/h")
from omlib import gpu_warmup  # noqa: E402

gpu_warmup(250.0)
sys.path.pop(0)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
