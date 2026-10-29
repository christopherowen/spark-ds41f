"""CPU-only switched deployment contract; no remote or GPU execution."""
import copy
import unittest
from test_topology import spark, topology


def fixture(count=4, shape='tp4-ring'):
    cluster = spark.read_cluster('config/cluster.json', shape)
    cluster['fabric'] = {'transport': 'oneshot-switched', 'topology': 'switched'}
    cluster.pop('benchmark_reference', None)
    cluster['environment'] = {k: v for k, v in cluster['environment'].items()
                              if k not in topology.RING_ENV and k not in (
                                  'NCCL_SWITCHLESS_BIDIRECTIONAL', 'NCCL_MIN_TRAFFIC_PER_CHANNEL',
                                  'NCCL_THREAD_THRESHOLDS')}
    cluster['distributed']['master_addr'] = '192.0.2.11'
    nodes = {'nodes': []}
    for rank, name in enumerate(('dgx1', 'dgx2', 'dgx3', 'dgx4')[:count]):
        suffix = '1' if rank < 2 else '0'
        pair = [f'rocep1s0f{suffix}', f'roceP2p1s0f{suffix}']
        nodes['nodes'].append(dict(name=name, rank=rank, head=rank == 0,
            management_ip=f'192.0.2.{11+rank}', roce_gid_index=3,
            socket_interface='ens1f1np1' if rank < 2 else 'ens1f0np0',
            roce_peer_hcas={str(p): pair[:] for p in range(count) if p != rank},
            roce_subnets=dict(zip(pair, ['10.50.1.0/24', '10.50.2.0/24']))))
    return cluster, nodes


class SwitchedTopologyTest(unittest.TestCase):
    def test_switched_contract_rejects_unsafe_lane_or_policy_changes(self):
        changes = [
            lambda c, n: c['fabric'].pop('topology'),
            lambda c, n: c['environment'].update(NCCL_SWITCHLESS_BIDIRECTIONAL='0'),
            lambda c, n: c['environment'].update(NCCL_MIN_TRAFFIC_PER_CHANNEL='1'),
            lambda c, n: c['environment'].update(NCCL_THREAD_THRESHOLDS='0 0 0'),
            lambda c, n: c['environment'].update(NCCL_ALGO='Ring'),
            lambda c, n: c['serve_args'].append('--disable-custom-all-reduce'),
            lambda c, n: n['nodes'][0]['roce_peer_hcas']['1'].reverse(),
            lambda c, n: n['nodes'][0]['roce_peer_hcas']['1'].__setitem__(1, 'rocep1s0f1'),
            lambda c, n: n['nodes'][0]['roce_subnets'].clear(),
            lambda c, n: n['nodes'][1].update(management_ip=n['nodes'][0]['management_ip']),
            lambda c, n: n['nodes'][0].update(roce_gid_index=-1),
            lambda c, n: n['nodes'][0]['roce_peer_hcas'].pop('2'),
            lambda c, n: n['nodes'][0].update(rank=1),
            lambda c, n: n['nodes'][0].update(head=False),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                cluster, nodes = fixture()
                change(cluster, nodes)
                self.assertTrue(topology.problems(cluster, nodes))

    def test_switched_render_selects_exact_hcas_and_fabric_bootstrap(self):
        cluster, nodes = fixture()
        for node in nodes['nodes']:
            node['socket_interface'] = 'ens1f1np1' if node['rank'] < 2 else 'ens1f0np0'
            env = spark.expected_environment(cluster, node)
            pair = next(iter(node['roce_peer_hcas'].values()))
            self.assertEqual(env['NCCL_IB_HCA'], '=' + ','.join(pair))
            self.assertEqual(env['GLOO_SOCKET_IFNAME'], node['socket_interface'])
            self.assertEqual(env['NCCL_SOCKET_IFNAME'], '=' + node['socket_interface'])
            self.assertEqual(env['SPARK_DEPLOYMENT_TRANSPORT'], 'oneshot-switched')
            self.assertEqual(env['VLLM_HOST_IP'], node['management_ip'])
            self.assertEqual(env['SPARKNET_ROCE_TOPOLOGY'], 'direct')
            self.assertEqual(spark.roce_gid_setting(cluster, node), (3, sorted(pair)))

    def test_traffic_class_marks_both_backends_alike(self):
        cluster, nodes = fixture()
        for node in nodes['nodes']:
            node['roce_traffic_class'] = 106  # DSCP 26 << 2 | ECT(0)
        self.assertEqual(topology.problems(cluster, nodes), [])
        for node in nodes['nodes']:
            env = spark.expected_environment(cluster, node)
            self.assertEqual(env['NCCL_IB_TC'], '106')
            self.assertEqual(env['SPARKNET_ROCE_TRAFFIC_CLASS'], '106')
        cluster['fabric']['transport'] = 'nccl-switched'
        env = spark.expected_environment(cluster, nodes['nodes'][0])
        self.assertEqual(env['NCCL_IB_TC'], '106')
        self.assertNotIn('SPARKNET_ROCE_TRAFFIC_CLASS', env)

    def test_traffic_class_left_out_renders_nothing(self):
        cluster, nodes = fixture()
        self.assertEqual(topology.problems(cluster, nodes), [])
        env = spark.expected_environment(cluster, nodes['nodes'][0])
        self.assertNotIn('NCCL_IB_TC', env)
        self.assertNotIn('SPARKNET_ROCE_TRAFFIC_CLASS', env)

    def test_traffic_class_rejects_bad_mixed_or_duplicated_values(self):
        def every(value):
            return lambda c, n: [node.update(roce_traffic_class=value) for node in n['nodes']]
        changes = [
            every(256),
            every(-1),
            every('106'),
            every(True),
            every([106]),
            lambda c, n: n['nodes'][1].update(roce_traffic_class=106),
            lambda c, n: (every(106)(c, n), n['nodes'][2].update(roce_traffic_class=104)),
            lambda c, n: (every(106)(c, n), c['environment'].update(NCCL_IB_TC='106')),
            lambda c, n: (every(106)(c, n), c['environment'].update(SPARKNET_ROCE_TRAFFIC_CLASS='106')),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                cluster, nodes = fixture()
                change(cluster, nodes)
                self.assertTrue(any('roce_traffic_class' in p for p in topology.problems(cluster, nodes)))

    def test_candidate_generation_preserves_explicit_switched_transport(self):
        cluster, nodes = fixture()
        result = topology.candidate(cluster, nodes, 'switched.json')
        self.assertEqual(result['fabric'], cluster['fabric'])
        self.assertFalse(result['deployment']['launch_enabled'])
        self.assertNotIn('benchmark_reference', result)
        self.assertNotIn('--disable-custom-all-reduce', result['serve_args'])

    def test_nccl_switched_is_capability_rejected_on_locked_image(self):
        cluster, nodes = fixture()
        cluster['fabric']['transport'] = 'nccl-switched'
        cluster['deployment']['launch_enabled'] = False
        findings = spark.local_doctor(cluster, nodes, spark.read_json('upstreams.lock.json'))
        self.assertTrue(any('no runtime support' in str(p) for p in findings), findings)

    def test_missing_or_wifi_bootstrap_interface_is_rejected_before_render(self):
        for interface in (None, '', 'wlan0', 'eth0,eth1'):
            cluster, nodes = fixture()
            if interface is None:
                nodes['nodes'][0].pop('socket_interface')
            else:
                nodes['nodes'][0]['socket_interface'] = interface
            self.assertTrue(any('socket_interface' in p for p in topology.problems(cluster, nodes)))

    def test_explicit_switched_clique_reuses_pair_without_weakening_ring(self):
        cluster, nodes = fixture()
        self.assertEqual(topology.problems(cluster, nodes), [])
        self.assertEqual(topology.one_shot_topology('oneshot-switched'), 'direct')
        self.assertFalse(topology.neighbour_ring('oneshot-switched'))
        cluster['fabric']['transport'] = 'oneshot-direct'
        self.assertTrue(topology.problems(cluster, nodes))


    def test_three_node_switched_map_is_accepted(self):
        # Validation only: a clique of three peers is a valid switched map. No
        # three-node switched deployment is qualified; docs/switched-fabric.md
        # says so and the collective probe is the gate.
        cluster, nodes = fixture(count=3, shape='tp3-triangle')
        self.assertEqual(topology.problems(cluster, nodes), [])
        self.assertEqual(len(nodes['nodes'][0]['roce_peer_hcas']), 2)

    def test_nccl_switched_needs_plain_nccl_and_no_sparknet_settings(self):
        cluster, nodes = fixture()
        cluster['fabric']['transport'] = 'nccl-switched'
        cluster['environment'] = {k: v for k, v in cluster['environment'].items()
                                  if not k.startswith('SPARKNET_')}
        self.assertTrue(topology.problems(cluster, nodes), 'custom all-reduce must be off')
        cluster['serve_args'].append('--disable-custom-all-reduce')
        self.assertEqual(topology.problems(cluster, nodes), [])
        cluster['environment']['SPARKNET_ROCE_TIMEOUT_MS'] = '10000'
        self.assertTrue(topology.problems(cluster, nodes), 'sparknet settings do not apply')

    def test_virtual_rendezvous_interfaces_are_rejected(self):
        for interface in ('lo', 'docker0', 'br-2f19', 'veth7c', 'tailscale0', 'wlp1s0'):
            with self.subTest(interface=interface):
                cluster, nodes = fixture()
                for node in nodes['nodes']:
                    node['socket_interface'] = interface
                self.assertTrue(topology.problems(cluster, nodes))

    def test_switched_render_pins_exact_hcas_and_declares_the_transport(self):
        cluster, nodes = fixture()
        for node in nodes['nodes']:
            env = spark.expected_environment(cluster, node)
            pair = next(iter(node['roce_peer_hcas'].values()))
            self.assertEqual(env['NCCL_IB_HCA'], '=' + ','.join(pair))
            self.assertEqual(env['NCCL_SOCKET_IFNAME'], '=' + node['socket_interface'])
            self.assertEqual(env['SPARK_DEPLOYMENT_TRANSPORT'], 'oneshot-switched')
            self.assertEqual(env['SPARKNET_ROCE_TOPOLOGY'], 'direct')
            # The switchless bidirectional control is absent, not zeroed.
            self.assertNotIn('NCCL_SWITCHLESS_BIDIRECTIONAL', env)

    def test_bootstrap_address_may_live_off_the_fabric(self):
        # Management addressing is a site choice: a switched map whose API head
        # is reachable over a management network is valid.
        cluster, nodes = fixture()
        self.assertEqual(topology.problems(cluster, nodes), [])
        for node in nodes['nodes']:
            node['management_ip'] = node['management_ip'].replace('192.0.2.', '198.51.100.')
        cluster['distributed']['master_addr'] = '198.51.100.11'
        self.assertEqual(topology.problems(cluster, nodes), [])


if __name__ == '__main__':
    unittest.main()
