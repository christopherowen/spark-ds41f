"""Reuse the published exclusive isolated sp8 runner."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent/'2026-11-04-clustered-indexer/sp8_run.py'
spec = importlib.util.spec_from_file_location('isolated_sp8',source)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HERE = HERE

if __name__ == '__main__':
    raise SystemExit(runner.main())
