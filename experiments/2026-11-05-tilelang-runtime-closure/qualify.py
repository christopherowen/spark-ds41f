"""Assemble final qualification; retain historical observations in runs/."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("qualify_final.py")), run_name="__main__")
