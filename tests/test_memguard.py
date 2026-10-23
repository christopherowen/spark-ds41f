"""Host-independent behavioral tests for the fail-closed memory guard."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MEMGUARD = ROOT / "scripts" / "memguard.sh"
MEMGUARD_START = ROOT / "scripts" / "memguard-start.sh"
loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader("spark", loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)


class MemguardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.state = self.root / "state"
        self.actions = self.root / "actions"
        self.ready = self.root / "ready"
        self._write_command(
            "docker",
            """#!/usr/bin/env bash
set -eu
case "$1" in
  inspect)
    state=$(cat "$FAKE_STATE")
    if [[ "$state" == absent ]]; then
      exit 1
    fi
    printf '%s\n' "$state"
    ;;
  kill)
    printf 'kill %s\n' "$*" >>"$FAKE_ACTIONS"
    printf 'false\n' >"$FAKE_STATE"
    ;;
  *)
    exit 2
    ;;
esac
""",
        )
        self._write_command(
            "awk",
            """#!/usr/bin/env bash
set -eu
if [[ -n "${FAKE_MEM_SEQUENCE:-}" ]]; then
  counter="$FAKE_STATE.calls"
  calls=$(cat "$counter" 2>/dev/null || printf '0')
  printf '%s\n' "$((calls + 1))" >"$counter"
  IFS=',' read -r -a values <<<"$FAKE_MEM_SEQUENCE"
  index=$calls
  if ((index >= ${#values[@]})); then
    index=$((${#values[@]} - 1))
  fi
  printf '%s\n' "$((values[index] * 1024 * 1024))"
  exit 0
fi
printf '%s\n' "$FAKE_MEM_AVAILABLE_KIB"
""",
        )
        self._write_command(
            "logger",
            """#!/usr/bin/env bash
set -eu
printf 'logger %s\n' "$*" >>"$FAKE_ACTIONS"
""",
        )
        self._write_command(
            "sleep",
            """#!/usr/bin/env bash
set -eu
state=$(cat "$FAKE_STATE")
if [[ -n "${FAKE_SLEEP_LIMIT:-}" ]]; then
  counter="$FAKE_STATE.sleeps"
  pauses=$(cat "$counter" 2>/dev/null || printf '0')
  pauses=$((pauses + 1))
  printf '%s\n' "$pauses" >"$counter"
  if ((pauses >= FAKE_SLEEP_LIMIT)); then
    printf 'false\n' >"$FAKE_STATE"
  fi
  exit 0
fi
case "$state" in
  absent) printf 'true\n' >"$FAKE_STATE" ;;
  true) printf 'false\n' >"$FAKE_STATE" ;;
esac
""",
        )
        self._write_command(
            "sudo",
            """#!/usr/bin/env bash
set -eu
printf 'sudo %s\n' "$*" >>"$FAKE_ACTIONS"
if [[ "$*" == *" systemd-run "* ]]; then
  for argument in "$@"; do
    case "$argument" in
      --setenv=READY_FILE=*)
        ready=${argument#--setenv=READY_FILE=}
        : >"$ready"
        ;;
    esac
  done
fi
""",
        )

    def _write_command(self, name: str, body: str) -> None:
        command = self.bin / name
        command.write_text(body, encoding="utf-8")
        command.chmod(0o755)

    def _run(
        self,
        *,
        state: str,
        available_gib: int,
        wait_seconds: int,
        phase: str = "startup",
        available_kib: int | None = None,
        warn_gib: int | None = None,
        mem_sequence: list[int] | None = None,
        sleep_limit: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run the guard. ``mem_sequence`` gives MemAvailable (GiB) sample by sample, the
        last value repeating; ``sleep_limit`` stops the container after that many pauses."""
        self.state.write_text(f"{state}\n", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:{environment['PATH']}",
                "CONTAINER_NAME": "candidate",
                "THRESHOLD_GIB": "5",
                "INTERVAL": "0.25",
                "MEMGUARD_PHASE": phase,
                "WAIT_FOR_CONTAINER_SECONDS": str(wait_seconds),
                "READY_FILE": str(self.ready),
                "FAKE_STATE": str(self.state),
                "FAKE_ACTIONS": str(self.actions),
                "FAKE_MEM_AVAILABLE_KIB": str(available_kib if available_kib is not None
                                              else available_gib * 1024 * 1024),
            }
        )
        if warn_gib is not None:
            environment["MEMGUARD_WARN_GIB"] = str(warn_gib)
        if mem_sequence is not None:
            environment["FAKE_MEM_SEQUENCE"] = ",".join(str(value) for value in mem_sequence)
        if sleep_limit is not None:
            environment["FAKE_SLEEP_LIMIT"] = str(sleep_limit)
        return subprocess.run(
            [str(MEMGUARD)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

    def test_prelaunch_guard_is_ready_before_container_and_tracks_its_lifetime(
        self,
    ) -> None:
        result = self._run(state="absent", available_gib=16, wait_seconds=120)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.ready.exists())
        self.assertFalse(self.actions.exists())

    def test_running_container_is_killed_immediately_below_threshold(self) -> None:
        result = self._run(state="true", available_gib=4, wait_seconds=0)

        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.ready.exists())
        self.assertIn("kill kill --signal KILL candidate", self.actions.read_text())

    def test_a_sample_short_by_1344_kib_kills_at_once(self) -> None:
        # A fixed line: one sample, however close, kills (#410's 1.3 MiB miss still kills).
        result = self._run(state="true", available_gib=0, available_kib=5 * 1024 * 1024 - 1344,
                           wait_seconds=0, phase="steady")

        self.assertEqual(result.returncode, 1)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertIn("kill kill --signal KILL candidate", actions)
        self.assertIn("killing candidate during steady: MemAvailable 5241536 KiB below 5242880 KiB "
                      "(short by 1344 KiB)", actions)

    def test_the_warn_line_never_delays_the_kill(self) -> None:
        # Warned at 6 GiB, then the first sample below 5 GiB kills: no second sample is read.
        result = self._run(state="true", available_gib=16, wait_seconds=0, phase="steady",
                           warn_gib=8, mem_sequence=[6, 4, 16, 16], sleep_limit=10)

        self.assertEqual(result.returncode, 1)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertEqual(actions.count("kill kill --signal KILL candidate"), 1)
        self.assertEqual(actions.count("low headroom on candidate during steady"), 1)
        self.assertIn("killing candidate during steady: MemAvailable 4194304 KiB below 5242880 KiB "
                      "(short by 1048576 KiB); warn line 8388608 KiB", actions)
        self.assertEqual(Path(f"{self.state}.calls").read_text(encoding="utf-8").strip(), "2")
        # A first sample below the kill line kills before any warning.
        self.actions.unlink()
        Path(f"{self.state}.calls").unlink()
        Path(f"{self.state}.sleeps").unlink()
        result = self._run(state="true", available_gib=16, wait_seconds=0, phase="steady",
                           warn_gib=8, mem_sequence=[4, 16], sleep_limit=10)
        self.assertEqual(result.returncode, 1)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertIn("kill kill --signal KILL candidate", actions)
        self.assertNotIn("low headroom", actions)

    def test_prelaunch_guard_rejects_low_memory_without_starting_container(
        self,
    ) -> None:
        result = self._run(state="absent", available_gib=4, wait_seconds=120)

        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.ready.exists())
        self.assertNotIn("kill --signal", self.actions.read_text())

    def _start(self, *arguments: str, container: str | None = None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:{environment['PATH']}",
                "XDG_RUNTIME_DIR": str(self.root),
                "FAKE_ACTIONS": str(self.actions),
            }
        )
        if container is not None:  # the container's state, for the running mode's check
            self.state.write_text(f"{container}\n", encoding="utf-8")
            environment["FAKE_STATE"] = str(self.state)
        return subprocess.run(
            [str(MEMGUARD_START), *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

    def test_start_wrapper_confirms_protected_prelaunch_unit(self) -> None:
        # An arm overlay: the guard's settings come from the resolved configuration
        # (r6m's first boot failed to arm when the script read the arm's file).
        cluster = spark.read_cluster("experiments/2026-10-14-r6m/r6m-release.json")
        result = self._start(*spark.memguard_settings(cluster, "startup", "prelaunch"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode prelaunch", result.stdout)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertIn("--property=OOMScoreAdjust=-1000", actions)
        self.assertIn("--property=MemoryMin=64M", actions)
        self.assertIn("--property=CPUWeight=1000", actions)
        self.assertIn("--setenv=INTERVAL=0.25", actions)
        self.assertIn("--setenv=WAIT_FOR_CONTAINER_SECONDS=120", actions)
        self.assertIn("--setenv=CONTAINER_NAME=dsv41-karmic-kraken", actions)
        self.assertIn("--setenv=MEMGUARD_WARN_GIB=0", actions)
        self.assertIn("no warn line", result.stdout)

    def test_low_headroom_warns_once_per_crossing(self) -> None:
        # Kill line 5 GiB, warn line 8: two crossings below 8 log two lines, and nothing dies.
        result = self._run(state="true", available_gib=16, wait_seconds=0, phase="steady",
                           warn_gib=8, mem_sequence=[6, 6, 16, 7, 6], sleep_limit=5)

        self.assertEqual(result.returncode, 0, result.stderr)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertNotIn("kill --signal", actions)
        self.assertEqual(actions.count("low headroom on candidate during steady"), 2)
        self.assertIn("MemAvailable 6291456 KiB below the 8388608 KiB warn line (kill line 5242880 KiB)",
                      actions)

    def test_no_warn_line_logs_nothing_above_the_kill_line(self) -> None:
        result = self._run(state="true", available_gib=16, wait_seconds=0, phase="steady",
                           mem_sequence=[6, 6], sleep_limit=2)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.actions.exists())

    def test_warn_line_must_exceed_the_kill_line(self) -> None:
        for warn_gib in (5, 4):
            result = self._run(state="true", available_gib=16, wait_seconds=0, phase="steady",
                               warn_gib=warn_gib)
            self.assertEqual(result.returncode, 2)
            self.assertIn("MEMGUARD_WARN_GIB must exceed THRESHOLD_GIB", result.stderr)
        result = self._run(state="true", available_gib=16, wait_seconds=0, phase="steady", warn_gib="x")
        self.assertEqual(result.returncode, 2)
        self.assertIn("MEMGUARD_WARN_GIB must be a non-negative integer", result.stderr)

    def test_start_wrapper_passes_the_steady_warn_line(self) -> None:
        cluster = spark.read_cluster("experiments/2026-10-14-r6m/r6m-release.json")
        cluster["host"]["memguard_warn_gib"] = 4

        steady = self._start(*spark.memguard_settings(cluster, "steady"), container="true")
        self.assertEqual(steady.returncode, 0, steady.stderr)
        self.assertIn("warn line 4 GiB", steady.stdout)
        self.assertIn("--setenv=MEMGUARD_WARN_GIB=4", self.actions.read_text(encoding="utf-8"))
        # The startup guard runs at the launch line (5 GiB), above a 4 GiB warn line: it gets none.
        self.assertEqual(spark.memguard_settings(cluster, "startup", "prelaunch")[-1], "0")
        startup = self._start(*spark.memguard_settings(cluster, "startup", "prelaunch")[:-1], "4")
        self.assertEqual(startup.returncode, 2)
        self.assertIn("valid only for the steady memguard", startup.stderr)

    def test_doctor_validates_the_warn_line(self) -> None:
        nodes = json.loads((ROOT / "config/nodes.example.json").read_text(encoding="utf-8"))
        lock = spark.read_json("upstreams.lock.json")

        def findings(warn_gib):
            cluster = spark.read_cluster("config/cluster.json", "tp3-triangle")
            if warn_gib is not None:
                cluster["host"]["memguard_warn_gib"] = warn_gib
            with mock.patch.object(spark, "tracked_secret_problems", return_value=[]):
                return [f for f in spark.local_doctor(cluster, nodes, lock) if "memguard_warn_gib" in f]

        self.assertEqual(findings(None), [])  # absent: no warn line
        self.assertEqual(findings(0), [])
        self.assertEqual(findings(4), [])
        self.assertEqual(findings(3), ["host.memguard_warn_gib must exceed host.memguard_gib"])
        for bad in (-1, 4.5, True, "4"):
            self.assertEqual(findings(bad), ["host.memguard_warn_gib must be a non-negative integer"], bad)

    def test_start_wrapper_refuses_a_config_path(self) -> None:
        result = self._start("config/cluster.json", "startup", "prelaunch")

        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertFalse(self.actions.exists())

    def test_steady_settings_have_no_container_wait(self) -> None:
        cluster = spark.read_cluster("experiments/2026-10-14-r6m/r6m-release.json")
        name, phase, gib, interval, wait, mode, warn = spark.memguard_settings(cluster, "steady")
        self.assertEqual((name, phase, wait), ("dsv41-karmic-kraken", "steady", "0"))
        self.assertEqual(gib, str(cluster["host"]["memguard_gib"]))
        self.assertEqual(interval, str(cluster["host"]["memguard_interval_seconds"]))
        self.assertEqual((mode, warn), ("running", "0"))  # no warn line unless the host sets one


if __name__ == "__main__":
    unittest.main()
