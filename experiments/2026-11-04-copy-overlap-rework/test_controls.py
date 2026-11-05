"""Offline rejection checks for image authority and the functional phase gate."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


round_control = module('copy_overlap_round', Path(__file__).with_name('run_round.py'))
collector = module('runtime_collector', ROOT / 'scripts/collect_runtime.py')


def passing_image():
    names = ('build-prepare', 'build-check', 'build-image', 'build-image-apply',
             'gate-manifest', 'gate-bundle', 'release-tests', 'topk-initcheck')
    source_hash = json.loads(Path(__file__).with_name('runs').joinpath('preflight/topk-source.json').read_text())['sha256']
    return {'image': 'sha256:' + 'a' * 64, 'commit': 'b' * 40, 'ended': '2026-11-04T00:00:00Z',
            'passed': True, 'gate': {'passed': True, 'summary': '8074 passed, 360 skipped'},
            'conditions': {'halted': [], 'fan_stop_failures': []},
            'steps': [{'log': name + '.log', 'exit': 0} for name in names],
            'source': {'vllm_head': round_control.EXPECTED_HEAD,
                       'expected_labels': {'local.spark-ds41f.vllm.tree': round_control.EXPECTED_TREE}},
            'topk_initcheck': {'passed': True, 'source_sha256': source_hash}}


class ImageGateTests(unittest.TestCase):
    def test_accept_completed_actual_image_gates(self):
        round_control.validate_image_gate(passing_image())

    def test_reject_failed_or_incomplete_gates(self):
        for change in ('no_tests', 'bad_source', 'bad_sanitizer', 'halt', 'failed_step'):
            image = passing_image()
            if change == 'no_tests': image['gate']['summary'] = '360 skipped'
            if change == 'bad_source': image['source']['vllm_head'] = 'c' * 40
            if change == 'bad_sanitizer': image['topk_initcheck']['source_sha256'] = '0' * 64
            if change == 'halt': image['conditions']['halted'].append('lost telemetry')
            if change == 'failed_step': image['steps'][-1]['exit'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                round_control.validate_image_gate(image)

    def test_resume_only_safe_unmeasured_scheduling_close(self):
        image = passing_image()
        old = {'ended': image['ended'], 'commit': image['commit'], 'restored': True,
               'error': 'RuntimeError: window changed or closure requested',
               'steps': [{'log': 'image-and-gate.log', 'exit': 0}]}
        round_control.validate_gate_resume(old, image)
        for key, value in (('restore_error', 'failure'), ('management_loss_latched', True)):
            changed = copy.deepcopy(old); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                round_control.validate_gate_resume(changed, image)
        changed = copy.deepcopy(old); changed['steps'].append({'log': 'functional.log', 'exit': 0})
        with self.assertRaises(ValueError): round_control.validate_gate_resume(changed, image)


class FunctionalGateTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(Path(__file__).with_name('functional.json').read_text())
        self.lines = [f'{Path(script).name} {arm["label"]} exit 0'
                      for arm in self.spec['jobs'][0]['arms']
                      for script, _, _ in self.spec['jobs'][0]['extras']]

    def test_accept_all_ordered_exits(self):
        self.assertEqual(len(round_control.validate_functional_exits('\n'.join(self.lines), self.spec)['ordered_exits']), 40)

    def test_reject_c39_collector_failure_despite_parent_zero(self):
        lines = [line.replace('exit 0', 'exit 1') if line.startswith('collect_runtime.py') else line
                 for line in self.lines]
        with self.assertRaises(ValueError): round_control.validate_functional_exits('\n'.join(lines), self.spec)

    def test_reject_missing_or_reordered_exits(self):
        for lines in (self.lines[:-1], list(reversed(self.lines))):
            with self.assertRaises(ValueError): round_control.validate_functional_exits('\n'.join(lines), self.spec)


class CollectorTests(unittest.TestCase):
    def test_explicit_sha_required(self):
        for value in (None, 'spark-ds41f:candidate-41'):
            with self.assertRaises(ValueError):
                collector.expected_image(argparse.Namespace(expected_image_id=value, image_receipt=None), {})
        value = 'sha256:' + 'a' * 64
        self.assertEqual(collector.expected_image(argparse.Namespace(expected_image_id=value, image_receipt=None), {}), value)

    def test_receipt_must_pass_and_bind_requested_source(self):
        image = passing_image()
        cluster = {'container': {'expected_labels': image['source']['expected_labels']}}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(collector, 'ROOT', Path(directory)):
            path = Path(directory) / 'results/private/image.json'; path.parent.mkdir(parents=True)
            args = argparse.Namespace(expected_image_id=None, image_receipt='results/private/image.json')
            path.write_text(json.dumps(image))
            self.assertEqual(collector.expected_image(args, cluster), image['image'])
            image['gate']['passed'] = False; path.write_text(json.dumps(image))
            with self.assertRaises(ValueError): collector.expected_image(args, cluster)
            image['gate']['passed'] = True
            image['source']['expected_labels'] = {}; path.write_text(json.dumps(image))
            with self.assertRaises(ValueError): collector.expected_image(args, cluster)


if __name__ == '__main__':
    unittest.main()
