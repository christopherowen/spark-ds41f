"""Offline rejection checks for image authority and the functional phase gate."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


round_control = module('topk_round', Path(__file__).with_name('run_round.py'))
collector = module('runtime_collector', ROOT / 'scripts/collect_runtime.py')
policy = round_control.policy


def passing_image():
    names = ('build-prepare', 'build-check', 'build-image', 'build-image-apply',
             'gate-manifest', 'gate-bundle', 'release-tests', 'topk-initcheck')
    source_hash = json.loads(Path(__file__).with_name('runs').joinpath('preflight/topk-source.json').read_text())['sha256']
    token, commit, container_id = 'c' * 32, 'b' * 40, 'd' * 64
    pin = policy.load_plan()['builder_worker']['image']
    name = policy.worker_name(commit, token)
    plan = policy.build_resources.builder_plan(name, pin)
    worker = {'owner': policy.OWNER, 'owner_token': token, 'commit': commit, 'window': 'dgx1-test',
              'builder': name, 'container_id': container_id, 'created': True, 'passed': True,
              'requested_image': pin,
              'actual_image': {'Id': pin.split('@')[1], 'RepoDigests': [pin], 'Os': 'linux', 'Architecture': 'arm64'},
              'validation': {'passed': True, 'container_id': container_id, 'image_id': pin.split('@')[1],
                  'plan': plan,
                  'effective_host_config': {'CpuQuota': 200000, 'CpuPeriod': 100000,
                      'Memory': 34359738368, 'MemorySwap': 34359738368},
                  'cgroup': {'cpu.max': '200000 100000', 'memory.max': '34359738368', 'memory.swap.max': '0'}}}
    return {'image': 'sha256:' + 'a' * 64, 'commit': 'b' * 40, 'ended': '2026-11-04T00:00:00Z',
            'passed': True, 'gate': {'passed': True, 'summary': '8104 passed, 360 skipped'},
            'conditions': {'halted': [], 'fan_stop_failures': [],
                'thermal': {'dgx4': {'max_gpu_c': 64, 'min_headroom_c': 23, 'max_system_c': 74.6,
                                     'max_power_w': 220, 'thermal_slowdown_ms': 0, 'power_cap_ms': 0}},
                'memory': {'dgx4': {'min_available_gib': 79.3, 'swap_growth_gib': 0}},
                'cooling': [{'node': 'dgx4', 'final_c': 54}], 'fan_curve': [{'node': 'dgx4'}]},
            'steps': [{'log': name + '.log', 'exit': 0} for name in names],
            'source': {'vllm_head': round_control.EXPECTED_HEAD,
                       'tilelang_head': round_control.TILELANG_HEAD,
                       'expected_labels': {'local.spark-ds41f.vllm.tree': round_control.EXPECTED_TREE,
                                           'local.spark-ds41f.tilelang.tree': round_control.TILELANG_TREE}},
            'release_cache': {'absent_before': True, 'image': 'sha256:' + 'a' * 64,
                              'host_root': '/home/swank/.cache/spark-lab/qualification/candidate-46-' + 'a' * 64,
                              'generated_bundle_sha256': 'e' * 64, 'actual_bundle_sha256': 'f' * 64},
            'gate_manifest': {'files': 138, 'sha256': 'd' * 64,
                              'required_frontend_files': ['tests/v1/engine/test_request_submission_cleanup.py',
                                                          'tests/v1/engine/test_utility_call_cleanup.py']},
            'topk_initcheck': {'passed': True, 'source_sha256': source_hash},
            'native_gate_audit': {'passed': True, 'files': 138, 'expected_files': 138,
                                  'passed_cases': 8104, 'failed_cases': 0, 'error_cases': 0,
                                  'results_sha256': '1' * 64},
            'native_regression_audit': {'passed': True, 'invariant_passed': True, 'invariant_occurrences': 1,
                'critical_files': {name: {'passed': True} for name in (
                    'tests/v1/engine/test_request_submission_cleanup.py',
                    'tests/v1/engine/test_utility_call_cleanup.py',
                    'tests/v1/worker/test_gpu_tilelang_logprob_softmax.py',
                    'tests/v1/worker/test_gpu_verified_logprob_token_ids.py')}},
            'resource_limits': {'source_prepare': policy.PREP_LIMITS,
                                'release_gate': policy.GATE_LIMITS,
                                'topk_initcheck': policy.INITCHECK_LIMITS},
            'builder_worker': worker,
            'builder_worker_stop': {'passed': True, 'owner_token': token, 'builder': name,
                'container_id': container_id, 'cache_state_preserved': True, 'global_builder_selection_changed': False}}


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

    def test_reject_legacy_shared_cache_or_incomplete_frontend_coverage(self):
        for change in ('old_cache', 'wrong_image', 'wrong_root', 'partial_manifest', 'missing_regression', 'duplicate_step'):
            image = passing_image()
            if change == 'old_cache': image['release_cache']['absent_before'] = False
            if change == 'wrong_image': image['release_cache']['image'] = 'sha256:' + 'c' * 64
            if change == 'wrong_root': image['release_cache']['host_root'] = '/tmp/candidate-46-' + 'a' * 64
            if change == 'partial_manifest': image['gate_manifest']['files'] = 136
            if change == 'missing_regression': image['gate_manifest']['required_frontend_files'].pop()
            if change == 'duplicate_step': image['steps'].insert(1, image['steps'][0])
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
        for logs in (['image-and-gate.log', 'image-and-gate.log'], ['seed-image.log', 'image-and-gate.log']):
            changed = copy.deepcopy(old); changed['steps'] = [{'log': item, 'exit': 0} for item in logs]
            with self.assertRaises(ValueError): round_control.validate_gate_resume(changed, image)

    def test_reject_incomplete_native_files_or_resource_evidence(self):
        for change in ('native_missing', 'native_failed', 'resource_unbounded', 'worker_missing'):
            image = passing_image()
            if change == 'native_missing': image['native_gate_audit']['files'] = 137
            if change == 'native_failed': image['native_gate_audit']['failed_cases'] = 1
            if change == 'resource_unbounded': image['resource_limits'] = {}
            if change == 'worker_missing': image['builder_worker'] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                round_control.validate_image_gate(image)

    def test_reject_client_only_cap_wrong_worker_and_missing_stop(self):
        for change in ('unbounded_worker', 'wrong_pin', 'wrong_stop', 'global_selection'):
            image = passing_image()
            if change == 'unbounded_worker': image['builder_worker']['validation']['cgroup']['memory.max'] = 'max'
            if change == 'wrong_pin': image['builder_worker']['actual_image']['RepoDigests'] = []
            if change == 'wrong_stop': image['builder_worker_stop']['container_id'] = 'e' * 64
            if change == 'global_selection': image['builder_worker_stop']['global_builder_selection_changed'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                round_control.validate_image_gate(image)


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
        for value in (None, 'spark-ds41f:candidate-46'):
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


class ScopeAndPolicyTests(unittest.TestCase):
    def setUp(self):
        self.token = 'a' * 32
        self.holder = 'spark-lab: exact C46 owned window'
        self.window = 'dgx1-test'
        self.commit = 'b' * 40
        self.controller = {'host': 'dgx1', 'pid': 123, 'argv': ['python3', str(Path(__file__).with_name('run_round.py'))],
                           'path': str(Path(__file__).with_name('run_round.py')), 'start_ticks': 100,
                           'commit': self.commit, 'published_commit': self.commit, 'start_commit': self.commit}
        self.watchdog = {'host': 'dgx1', 'pid': 124, 'ppid': 123,
                        'argv': ['python3', str(policy.HERE / 'c46_window.py'), 'watchdog', '--window', self.window,
                                 '--holder', self.holder, '--owner-token', self.token, '--commit', self.commit,
                                 '--restore-profile', round_control.RESTORE],
                        'path': str(policy.HERE / 'c46_window.py'), 'start_ticks': 101,
                        'commit': self.commit, 'published_commit': self.commit, 'start_commit': self.commit}
        self.scope = policy.scope(self.token, self.holder, self.window,
                                  self.controller, round_control.RESTORE, self.watchdog)
        self.hold = {'holder': self.holder, 'window': self.window, 'resource_scope': self.scope}

    def test_exact_scope_accepts_only_own_holder_window_token(self):
        self.assertEqual(self.scope['token'], self.window)
        self.assertEqual(self.scope['restore'], {'profile': round_control.RESTORE, 'commit': self.commit,
            'image': policy.RESTORE_IMAGE, 'nodes': ['dgx1', 'dgx2', 'dgx3']})
        policy.assert_owned(self.hold, self.window, self.holder, self.token)
        for key, value in (('holder', 'spark-lab: foreign'), ('window', 'other')):
            hold = copy.deepcopy(self.hold); hold[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                policy.assert_owned(hold, self.window, self.holder, self.token)
        with self.assertRaises(RuntimeError):
            policy.assert_owned(self.hold, self.window, self.holder, 'b' * 32)

    def test_reject_descriptive_or_unrelated_watchdog_metadata(self):
        for change in ('description', 'wrong_parent', 'missing_start', 'wrong_commit'):
            hold = copy.deepcopy(self.hold)
            if change == 'description': hold['resource_scope']['watchdog'] = 'a watchdog exists'
            if change == 'wrong_parent': hold['resource_scope']['watchdog']['ppid'] = 999
            if change == 'missing_start': hold['resource_scope']['watchdog'].pop('start_ticks')
            if change == 'wrong_commit': hold['resource_scope']['watchdog']['start_commit'] = 'f' * 40
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                policy.assert_owned(hold, self.window, self.holder, self.token)

    def test_reject_foreign_local_node_cpu_or_gpu_claims(self):
        policy.validate_local_records({'own.json': self.hold}, self.token, self.window)
        for path in ('node-hold.json', 'cpu-claim.json', 'gpu-claim.json'):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                policy.validate_local_records({path: {'holder': 'foreign'}}, self.token, self.window)

    def test_contract_preserves_sampled_again_and_c8_limitation(self):
        contract = policy.load_plan()['semantic_validator']['contract']
        self.assertEqual(contract['sampled_again'], {'required': True, 'expected': []})
        self.assertIn('observational limitation', contract['c8_distinct_identical_false']['classification'])
        self.assertTrue(contract['temperature_zero_and_model_and_other_declared_live_gates']['required'])

    def test_source_packet_refuses_apply_and_semantics_are_not_exit_only(self):
        with mock.patch.object(policy, 'load_plan', return_value={}):
            with self.assertRaises(RuntimeError): policy.require_launch_ready()
        # The connected semantic barrier's zero-exit/false-result failure
        # proofs live in test_semantic_barriers.py; it is no longer a stub.
        with mock.patch.object(policy.os, 'uname', return_value=SimpleNamespace(nodename='offline-control')):
            with self.assertRaises(RuntimeError):
                policy.create_validate_owned_worker({}, Path('/unused'), self.commit, self.window, self.holder, self.token)

    def test_cleanup_refuses_another_owner_before_any_docker_command(self):
        with mock.patch.object(policy.subprocess, 'run') as command, self.assertRaises(RuntimeError):
            policy.stop_owned_worker(passing_image()['builder_worker'], Path('/unused'), 'f' * 32)
        command.assert_not_called()

    def test_worker_limits_are_read_at_actual_pid_cgroup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); proc = root / 'proc'; hierarchy = root / 'cgroup'
            (proc / '7').mkdir(parents=True)
            (proc / '7/cgroup').write_text('0::/system.slice/owned.scope\n')
            own = hierarchy / 'system.slice/owned.scope'; own.mkdir(parents=True)
            (hierarchy / 'memory.max').write_text('max')
            for field, value in {'cpu.max': '200000 100000', 'memory.max': '34359738368', 'memory.swap.max': '0'}.items():
                (own / field).write_text(value)
            result = policy.worker_cgroup(7, proc, hierarchy)
            self.assertEqual(result['host_cgroup_path'], '/system.slice/owned.scope')
            self.assertEqual(result['memory.max'], '34359738368')

    def test_capture_binds_actual_direct_child_watchdog_birth_and_argv(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            (proc / '123/task/123').mkdir(parents=True)
            (proc / '123/task/123/children').write_text('124')
            for pid, ppid, script, extra, start in (
                (123, 1, policy.HERE / 'run_round.py', ['--apply'], 100),
                (124, 123, policy.HERE / 'c46_window.py', self.watchdog['argv'][2:], 101)):
                directory = proc / str(pid); directory.mkdir(exist_ok=True)
                fields = ['S', str(ppid)] + ['0'] * 17 + [str(start)]
                (directory / 'stat').write_text(f'{pid} (python3) ' + ' '.join(fields))
                (directory / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in ['python3', str(script), *extra]) + b'\0')
                (directory / 'cwd').symlink_to(policy.ROOT)
            with mock.patch.object(policy.os, 'getpid', return_value=123):
                controller, watchdog = policy.capture_control_processes(self.commit, self.watchdog['argv'], 124, proc)
            self.assertEqual((controller['pid'], controller['start_ticks']), (123, 100))
            self.assertEqual((watchdog['pid'], watchdog['ppid'], watchdog['start_ticks']), (124, 123, 101))
            self.assertEqual(watchdog['argv'][-2:], ['--restore-profile', policy.RESTORE_PROFILE])

    def test_actual_builder_caps_reject_missing_quota_and_gpu_assignment(self):
        resource = policy.build_resources
        pin = policy.load_plan()['builder_worker']['image']
        name = policy.worker_name(self.commit, self.token)
        container = {'Id': 'd' * 64, 'Name': '/buildx_buildkit_' + name + '0', 'Image': pin.split('@')[1],
                     'State': {'Running': True, 'Pid': 7},
                     'Config': {'Image': pin, 'Env': ['NVIDIA_VISIBLE_DEVICES=void']},
                     'Args': ['--allow-insecure-entitlement=network.host'],
                     'HostConfig': {'CpuQuota': 200000, 'CpuPeriod': 100000, 'Memory': 34359738368,
                                    'MemorySwap': 34359738368, 'Runtime': 'runc', 'Privileged': True,
                                    'DeviceRequests': None, 'Devices': []}}
        buildx = {'Name': name, 'Driver': 'docker-container', 'Nodes': [{'Name': name + '0'}]}
        cgroup = {'cpu.max': '200000 100000', 'memory.max': '34359738368', 'memory.swap.max': '0'}
        result = resource.validate_worker(name, pin, buildx, [container], resource.CONFIG.read_bytes(), cgroup)
        self.assertTrue(result['passed'])
        self.assertTrue(result['gpu_assignment']['privileged'])
        self.assertIn('do not prove', result['gpu_assignment']['limit'])
        for key, value in (('CpuQuota', 0), ('Memory', 0), ('MemorySwap', -1), ('Runtime', 'nvidia'),
                           ('DeviceRequests', [{'Count': -1}])):
            changed = copy.deepcopy(container); changed['HostConfig'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resource.validate_worker(name, pin, buildx, [changed], resource.CONFIG.read_bytes(), cgroup)
        self.assertNotIn('--use', resource.builder_plan(name, pin)['create_argv'])

    def test_native_per_file_audit_rejects_duplicate_missing_and_failure(self):
        manifest = {'files': [f'tests/test_{n}.py' for n in range(138)]}
        rows = [{'file': p, 'status': 'ok', 'passed': 1, 'skipped': 0,
                 'failed': 0, 'error': 0, 'failures': []} for p in manifest['files']]
        encode = lambda records: '\n'.join(json.dumps(row) for row in records)
        self.assertTrue(policy.audit_native_files(manifest, encode(rows))['passed'])
        failed = copy.deepcopy(rows); failed[0]['failed'] = 1
        for records in (rows[:-1], rows + [rows[0]], failed):
            self.assertFalse(policy.audit_native_files(manifest, encode(records))['passed'])

    def test_full_two_boot_protocol_and_bounded_gate_containers(self):
        functional = json.loads(Path(__file__).with_name('functional.json').read_text())
        performance = json.loads(Path(__file__).with_name('performance.json').read_text())
        self.assertEqual(len(functional['jobs'][0]['extras']) * len(functional['jobs'][0]['arms']), 40)
        self.assertEqual(performance['jobs'][0]['suite'], ['release-tp3', 'single-stream', 'sampled'])
        self.assertEqual([a['label'] for a in performance['jobs'][0]['arms']],
                         ['candidate-46', 'candidate-46-repeat'])
        self.assertEqual(performance['jobs'][0]['reference'], 'manifests/suites/2026-10-30-spark-ds41f-1030-tp3')
        for kind, size in (('release', '32g'), ('initcheck', '16g')):
            args = policy.container_limits(kind)
            self.assertEqual(args[args.index('--cpus') + 1], '2')
            self.assertEqual(args[args.index('--memory') + 1], size)
            self.assertEqual(args[args.index('--memory-swap') + 1], size)
            self.assertEqual(args[args.index('--pids-limit') + 1], '512')
            self.assertIn('MAX_JOBS=2', args)


if __name__ == '__main__':
    unittest.main()
