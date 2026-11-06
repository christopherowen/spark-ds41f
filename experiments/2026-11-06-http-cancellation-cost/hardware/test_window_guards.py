"""Negative checks: a foreign hold and a changed shape must never reach apply."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import restore_window as window


class WindowGuards(unittest.TestCase):
    def test_foreign_hold_cannot_beat_or_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hold = root / "hold.json"
            hold.write_text(
                json.dumps(
                    {
                        "holder": "another-task",
                        "window": "foreign",
                        "heartbeat": "unchanged",
                    }
                )
            )
            original = hold.read_bytes()
            state = {"hold_file": str(hold), "window": "ours"}
            path = root / "state.json"
            path.write_text(json.dumps(state))
            with patch.object(window.subprocess, "run") as execute:
                with self.assertRaises(RuntimeError):
                    window.beat(state)
                with self.assertRaises(RuntimeError):
                    window.restore(path)
                execute.assert_not_called()
            self.assertEqual(hold.read_bytes(), original)

    def test_changed_shape_never_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hold = root / "hold.json"
            hold.write_text(
                json.dumps(
                    {
                        "holder": window.HOLDER,
                        "window": "ours",
                        "heartbeat": window.stamp(),
                    }
                )
            )
            state = {
                "hold_file": str(hold),
                "window": "ours",
                "checkout": str(root),
                "nodes": [],
                "active_profile": "candidate.json",
                "restore_profile": "restore.json",
            }
            path = root / "state.json"
            path.write_text(json.dumps(state))
            response = SimpleNamespace(
                returncode=0, stdout="live:  candidate\nshape: CHANGED\n", stderr=""
            )
            with patch.object(
                window.subprocess, "run", return_value=response
            ) as execute:
                with self.assertRaises(RuntimeError):
                    window.restore(path)
                self.assertEqual(execute.call_count, 1)
                self.assertNotIn("--apply", execute.call_args.args[0])
            self.assertTrue(hold.exists())
            self.assertIn("restore_error", json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
