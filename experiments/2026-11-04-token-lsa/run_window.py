"""Same coordinated idle Spaceport protocol, using this round's bundles."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent/'2026-11-04-semantic-indexer/run_window.py'
spec = importlib.util.spec_from_file_location('kernel_window',source)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HERE = HERE

if __name__ == '__main__':
    runner.main()
