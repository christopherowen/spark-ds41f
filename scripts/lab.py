#!/usr/bin/env python3
"""Experiment windows, runs and kernel-lab jobs on the Spark cluster."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark_ds41f.lab import *  # noqa: E402, F403
from spark_ds41f.lab import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
