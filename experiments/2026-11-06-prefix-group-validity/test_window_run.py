import subprocess
import sys
import unittest
from unittest.mock import patch
import window_run


class WindowRunTest(unittest.TestCase):
    def test_successful_owned_command_returns_its_status(self):
        argv = ['window_run.py', '--window-state', 'unused', '--', sys.executable, '-c', 'pass']
        with patch.object(sys, 'argv', argv), patch.object(window_run, 'require_window', return_value={}):
            self.assertEqual(window_run.main(), 0)

    def test_lost_lease_terminates_the_owned_child(self):
        argv = ['window_run.py', '--window-state', 'unused', '--', sys.executable,
                '-c', 'import time; time.sleep(5)']
        actual_popen = subprocess.Popen
        children = []
        def start(*args, **kwargs):
            child = actual_popen(*args, **kwargs)
            children.append(child)
            return child
        with patch.object(sys, 'argv', argv), patch.object(window_run.subprocess, 'Popen', side_effect=start):
            with patch.object(window_run, 'require_window', side_effect=[{}, RuntimeError('lease lost')]):
                with self.assertRaisesRegex(RuntimeError, 'lease lost'):
                    window_run.main()
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())


if __name__ == '__main__':
    unittest.main()
