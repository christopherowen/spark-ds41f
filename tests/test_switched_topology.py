"""CPU-only switched deployment contract; no remote or GPU execution.

The node map's rules and the fabric environment are sparknet's (its own tests cover
the rails, rendezvous interfaces and traffic class); these tests cover what serving
adds on a switch and that the render is sparknet's at the lock's revision.
"""
import copy
import unittest
from test_topology import spark, topology


def fixture(count=4, shape='tp4-ring'):
    cluster = spark.read_cluster('config/cluster.json', shape)
    cluster['fabric'] = {'transport': 'oneshot-switched', 'topology': 'switched'}
    cluster.pop('benchmark_reference', None)
    ring = {**topology.fabric(cluster).render.RING_ENV, **topology.VLLM_RING_ENV}
    cluster['environment'] = {k: v for k, v in cluster['environment'].items()
                              if k not in ring and k not in (
                                  'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS',
                                  'NCCL_SWITCHLESS_BIDIRECTIONAL', 'NCCL_MIN_TRAFFIC_PER_CHANNEL',
                                  'NCCL_THREAD_THRESHOLDS')}
    cluster['distributed']['master_addr'] = '192.0.2.11'
    nodes = {'schema_version': 1, 'nodes': []}
    for rank, name in enumerate(('dgx1', 'dgx2', 'dgx3', 'dgx4')[:count]):
        suffix = '1' if rank < 2 else '0'
        rails = [f'rocep1s0f{suffix}', f'roceP2p1s0f{suffix}']
        nodes['nodes'].append(dict(name=name, rank=rank, head=rank == 0,
            management_ip=f'192.0.2.{11+rank}', roce_gid_index=3,
            management_interface='ens1f1np1' if rank < 2 else 'ens1f0np0',
            roce_hcas=rails, roce_subnets=dict(zip(rails, ['10.50.1.0/24', '10.50.2.0/24']))))
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
            lambda c, n: n['nodes'][0]['roce_hcas'].reverse(),
            lambda c, n: n['nodes'][0]['roce_hcas'].__setitem__(1, n['nodes'][0]['roce_hcas'][0]),
            lambda c, n: n['nodes'][0]['roce_subnets'].clear(),
            lambda c, n: n['nodes'][1].update(management_ip=n['nodes'][0]['management_ip']),
            lambda c, n: n['nodes'][0].update(roce_gid_index=-1),
            lambda c, n: n['nodes'][0].update(roce_peer_hcas={'1': n['nodes'][0]['roce_hcas']}),
            lambda c, n: n['nodes'][0].update(rank=1),
            lambda c, n: n['nodes'][0].update(head=False),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                cluster, nodes = fixture()
                change(cluster, nodes)
                self.assertTrue(topology.problems(cluster, nodes))

    def test_switched_render_is_sparknets_and_declares_the_transport(self):
        cluster, nodes = fixture()
        render = topology.fabric(cluster).render
        for node in nodes['nodes']:
            env = spark.expected_environment(cluster, node)
            rendered = render.node_environment(nodes, node, transport='oneshot-switched',
                                               base=cluster['environment'])
            self.assertEqual(env, {**rendered, 'VLLM_HOST_IP': node['management_ip'],
                                   'SPARK_DEPLOYMENT_TRANSPORT': 'oneshot-switched'})
            self.assertEqual(env['NCCL_IB_HCA'], '=' + ','.join(node['roce_hcas']))
            self.assertEqual(env['SPARKNET_ROCE_HCA'], ','.join(node['roce_hcas']))
            self.assertEqual(env['GLOO_SOCKET_IFNAME'], node['management_interface'])
            self.assertEqual(env['SPARKNET_ROCE_TOPOLOGY'], 'direct')
            # The switchless bidirectional control is absent, not zeroed.
            self.assertNotIn('NCCL_SWITCHLESS_BIDIRECTIONAL', env)
            self.assertEqual(spark.roce_gid_setting(cluster, node), (3, sorted(node['roce_hcas'])))

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

    def test_explicit_switched_clique_without_weakening_ring(self):
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

    def test_nccl_switched_needs_plain_nccl_and_no_sparknet_settings(self):
        cluster, nodes = fixture()
        cluster['fabric']['transport'] = 'nccl-switched'
        cluster['environment'] = {k: v for k, v in cluster['environment'].items()
                                  if not k.startswith('SPARKNET_')}
        self.assertTrue(topology.problems(cluster, nodes), 'custom all-reduce must be off')
        cluster['serve_args'].append('--disable-custom-all-reduce')
        self.assertEqual(topology.problems(cluster, nodes), [])
        env = spark.expected_environment(cluster, nodes['nodes'][0])
        self.assertNotIn('SPARKNET_ROCE_HCA', env)
        cluster['environment']['SPARKNET_ROCE_TIMEOUT_MS'] = '10000'
        self.assertTrue(topology.problems(cluster, nodes), 'sparknet settings do not apply')

    def test_bootstrap_address_may_live_off_the_fabric(self):
        # Management addressing is a site choice: a switched map whose API head
        # is reachable over a management network is valid.
        cluster, nodes = fixture()
        self.assertEqual(topology.problems(cluster, nodes), [])
        for node in nodes['nodes']:
            node['management_ip'] = node['management_ip'].replace('192.0.2.', '198.51.100.')
        cluster['distributed']['master_addr'] = '198.51.100.11'
        self.assertEqual(topology.problems(cluster, nodes), [])

    def test_site_access_fields_stay_out_of_the_fabric(self):
        cluster, nodes = fixture()
        for node in nodes['nodes']:
            node.update(ssh_host=f"{node['name']}.local", transfer_host=node['management_ip'])
        self.assertEqual(topology.problems(cluster, nodes), [])
        self.assertTrue(topology.problems(cluster, copy.deepcopy(nodes) | {
            'nodes': [{**n, 'unknown_field': 1} for n in nodes['nodes']]}))


if __name__ == '__main__':
    unittest.main()
