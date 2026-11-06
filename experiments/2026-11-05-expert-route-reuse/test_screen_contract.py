from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from screen_contract import verify_contract_before


class BeforeContractTests(unittest.TestCase):
    def probe(self):
        return ET.parse(Path(__file__).parent / 'runs/pytest-format-probe.xml')

    def validate(self, tree, code=1):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.xml'
            tree.write(path)
            return verify_contract_before(path, code)

    def test_real_pytest_report_without_type_is_accepted(self):
        tree = self.probe()
        self.assertTrue(all(case.find('failure').get('type') is None for case in tree.iter('testcase')))
        self.assertEqual(self.validate(tree), 4)

    def test_compiler_failure_is_not_accepted_from_displayed_source(self):
        tree = self.probe()
        failure = next(tree.iter('testcase')).find('failure')
        failure.set('message', 'RuntimeError: compiler failed')
        failure.text = 'source mentions rasterization2DRow<3> and AssertionError'
        with self.assertRaisesRegex(RuntimeError, 'does not reproduce'):
            self.validate(tree)

    def test_setup_skip_or_wrong_exit_is_rejected(self):
        for tag in ('error', 'skipped'):
            tree = self.probe()
            ET.SubElement(next(tree.iter('testcase')), tag)
            with self.assertRaises(RuntimeError):
                self.validate(tree)
        with self.assertRaises(RuntimeError):
            self.validate(self.probe(), code=2)

    def test_wrong_case_count_or_identity_is_rejected(self):
        tree = self.probe()
        case = next(tree.iter('testcase')); case.set('name', 'different_test')
        with self.assertRaises(RuntimeError):
            self.validate(tree)
        tree = self.probe(); suite = next(tree.iter('testsuite'))
        suite.remove(next(tree.iter('testcase')))
        with self.assertRaises(RuntimeError):
            self.validate(tree)


if __name__ == '__main__':
    unittest.main()
