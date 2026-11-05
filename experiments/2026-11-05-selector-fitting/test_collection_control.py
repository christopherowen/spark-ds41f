"""Test the exact DATA-prefix contract used by the native remote helper."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('native_collection',Path(__file__).with_name('collect.py'))
collection = importlib.util.module_from_spec(spec);spec.loader.exec_module(collection)


class ControlTests(unittest.TestCase):
    def test_remote_data_prefix_arms_disarms_and_hashes_without_stdin(self):
        with tempfile.TemporaryDirectory() as home,patch('pathlib.Path.home',return_value=Path(home)):
            target = Path(home)/'projects/spark-ds41f/cache/selector-fitting-20261105'
            def run(data):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    try:exec('DATA='+repr(data)+'\n'+collection.ARM,{})
                    except SystemExit:pass
                return json.loads(output.getvalue())
            self.assertTrue(run({'action':'check'})['ok'])
            self.assertEqual(target.stat().st_mode&0o777,0o700)
            case = {'case_id':'fit-a','prompt_sha256':'a'*64}
            run({'action':'arm','case':case})
            self.assertEqual(json.loads((target/'arm.json').read_text()),case)
            with self.assertRaises(FileExistsError):run({'action':'arm','case':case})
            run({'action':'disarm'});run({'action':'disarm'})
            for i in range(12):(target/f'fixture-{i}.pt').write_bytes(b'fixed private input')
            result = run({'action':'manifest'})
            self.assertEqual(len(result),12)
            self.assertTrue(all(row['bytes'] == len(b'fixed private input') and len(row['sha256']) == 64 for row in result.values()))


if __name__ == '__main__':unittest.main()
