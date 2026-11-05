"""Assert the actual image source, then run the independent native top-k test."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import vllm.models.deepseek_v4_1.tilelang.indexer as indexer

path = Path(indexer.__file__).resolve()
if not path.is_relative_to(Path("/opt/spark-ds41f/candidate/vllm")):
    raise RuntimeError("top-k module did not load from the built candidate source")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != os.environ["EXPECTED_TOPK_SOURCE_SHA256"]:
    raise RuntimeError("top-k module bytes differ from the canonical prepared source")
print(json.dumps({"module": str(path), "source_sha256": digest}), flush=True)
runpy.run_path("/qualification/reproducer.py", run_name="__main__")
