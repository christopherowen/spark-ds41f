"""Unit test package for spark-ds41f."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "scripts", ROOT / "tests"):
    _path = str(_entry)
    if _path not in sys.path:
        sys.path.insert(0, _path)
