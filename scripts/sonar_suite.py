#!/usr/bin/env python3
"""`sonar suite`: a deployment suite (config/suites/) or a Sonar built-in.

usage: scripts/sonar_suite.py URL [--suite NAME --out DIR [--corpus FILE] [--label L]]

Lab extras call ``python3 SCRIPT URL ARGS...``; this runs the pinned Sonar
(spark_ds41f/sonar.py; the pin is config/sonar.json).
A --suite NAME of config/suites/ runs that file, so its digest is the recorded references'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_ds41f import sonar  # noqa: E402

if __name__ == "__main__":
    sys.exit(sonar.exec_command('suite', sonar.suite_arguments(sys.argv[1:]), reasoning_off=False))
