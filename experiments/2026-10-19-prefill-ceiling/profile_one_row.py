#!/usr/bin/env python3
"""Single-token target forwards for the CED cost comparison."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench.profile import one_row  # noqa: E402
print(json.dumps(one_row(sys.argv[1])))
