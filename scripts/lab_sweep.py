#!/usr/bin/env python3
"""Write a one-variable environment sweep as lab arms and a run spec under experiments/.

usage: scripts/lab_sweep.py --base config/cluster-tp4.json --variable NAME --values V1,V2,...
           --experiment experiments/<dir> --run NAME [--streams 1,4,16] [--samples 3]
           [--tokens 256] [--profile lean] [--label PREFIX]

Each value becomes <experiment>/sweep-<prefix><value>.json: the base profile with only
environment[NAME] set to that value. The run spec <experiment>/<run>.json holds one
"measure" job: the base profile itself first (label "base", bracketed, so it is measured
again at the end to show drift), then one arm per value. Each arm gets the profile's
bench and, for each --streams count, scripts/distinct_streams.py (that many concurrent
streams with distinct prompts). A value equal to the base profile's own setting is
skipped, since the base arm measures it; a variable the base does not set runs at its
built-in default in the base arm, so leave that default out of --values.

Commit the experiment and sync the node checkouts before running it: every node's
checkout must hold the arm configs.

  scripts/lab.py run <experiment>/<run>.json --production-config <base>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALUE = re.compile(r"^[A-Za-z0-9._-]+$")


def label_prefix(variable: str) -> str:
    """The variable's last word, lower case: VLLM_L2_PREFETCH_GRID gives "grid"."""
    return variable.rsplit("_", 1)[-1].lower()


def sweep(base: dict, base_path: str, variable: str, values: list[str], experiment: str, run: str,
          streams: list[int], samples: int = 3, tokens: int = 256, profile: str = "lean",
          prefix: str | None = None) -> dict[str, dict]:
    """The files of a sweep, by repository-relative path."""
    if not isinstance(base.get("environment"), dict):
        raise ValueError(f"{base_path} has no environment object")
    if not values or len(set(values)) != len(values):
        raise ValueError("values must be given once each")
    bad = [value for value in values if not VALUE.match(value)]
    if bad:
        raise ValueError(f"values must be plain words or numbers: {bad}")
    if not streams or any(count < 1 for count in streams):
        raise ValueError("streams must be positive counts")
    prefix = label_prefix(variable) if prefix is None else prefix
    current = base["environment"].get(variable)
    files: dict[str, dict] = {}
    arms = [{"config": base_path, "label": "base"}]
    for value in values:
        if value == current:
            continue
        arm = json.loads(json.dumps(base))
        arm["environment"][variable] = value
        name = f"sweep-{prefix}{value}.json"
        files[f"{experiment}/{name}"] = arm
        arms.append({"config": name, "label": f"{prefix}{value}"})
    if len(arms) < 2:
        raise ValueError("every value equals the base profile's setting; nothing to compare")
    extras = [["scripts/distinct_streams.py",
               ["--streams", str(count), "--samples", str(samples), "--tokens", str(tokens)],
               f"c{count}-distinct"] for count in streams]
    files[f"{experiment}/{run}.json"] = {
        "experiment": experiment, "run": run,
        "jobs": [{"kind": "measure", "profile": profile, "bracket": True, "extras": extras, "arms": arms}],
    }
    return files


def check_destination(experiment: str) -> Path:
    """The experiment directory, which must lie under experiments/."""
    path = (ROOT / experiment).resolve()
    if not path.is_relative_to((ROOT / "experiments").resolve()) or path == (ROOT / "experiments").resolve():
        raise ValueError("sweeps belong in a directory under experiments/")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="repository-relative base profile")
    parser.add_argument("--variable", required=True)
    parser.add_argument("--values", required=True, help="comma-separated")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--streams", default="1,4,16", help="comma-separated concurrent stream counts")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--profile", default="lean", choices=("lean", "full"))
    parser.add_argument("--label", help="arm label prefix (default: the variable's last word)")
    args = parser.parse_args(argv)
    try:
        check_destination(args.experiment)
        base = json.loads((ROOT / args.base).read_text())
        files = sweep(base, args.base, args.variable, [v.strip() for v in args.values.split(",") if v.strip()],
                      args.experiment.rstrip("/"), args.run, [int(s) for s in args.streams.split(",")],
                      args.samples, args.tokens, args.profile, args.label)
        existing = [path for path in files if (ROOT / path).exists()]
        if existing:
            raise ValueError(f"refusing to overwrite {existing}")
    except (ValueError, OSError) as error:
        raise SystemExit(f"lab_sweep: {error}") from error
    for path, content in files.items():
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as handle:
            json.dump(content, handle, indent=2)
            handle.write("\n")
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
