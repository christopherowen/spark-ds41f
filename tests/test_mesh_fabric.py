"""Check the two-hop fabric against endpoint QP routes, not just snapshots."""
import argparse
import copy
import importlib.machinery
import importlib.util
import ipaddress
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import mesh_fabric as fabric
import mesh_probe
import topology
loader = importlib.machinery.SourceFileLoader('spark3_mesh_test', str(ROOT/'bin/spark3'))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
EXPERIMENT = 'experiments/2026-10-02-rocenante-mesh4'


class MeshTest(unittest.TestCase):
    def setUp(self):
        self.nodes = json.loads((ROOT/EXPERIMENT/'nodes.example.json').read_text())
        self.cluster = json.loads((ROOT/EXPERIMENT/'cluster.json').read_text())
        self.by_rank = {n['rank']: n for n in self.nodes['nodes']}

    def test_logical_mesh_preserves_physical_map_and_stripe_reciprocity(self):
        original = copy.deepcopy(self.nodes)
        plans = fabric.build_plan(self.nodes)
        for rank, node in self.by_rank.items():
            peers = topology.logical_peer_hcas(self.cluster, node)
            self.assertEqual(set(peers), {str(p) for p in range(4) if p != rank})
            for peer, hcas in node['roce_peer_hcas'].items():
                self.assertEqual(peers[peer], hcas)
            dest = (rank+2)%4
            reverse = topology.logical_peer_hcas(self.cluster, self.by_rank[dest])[str(rank)]
            for lane in range(2):
                source_hca = peers[str(dest)][lane]
                target_hca = reverse[lane]
                source_ip = str(ipaddress.ip_interface(node['mesh_ports'][source_hca]['address']).ip)
                dest_ip = str(ipaddress.ip_interface(self.by_rank[dest]['mesh_ports'][target_hca]['address']).ip)
                route = plans[rank]['routes'][lane]
                self.assertEqual(route['add'][3], dest_ip+'/32')
                self.assertEqual(route['add'][-1], source_ip)
                marker = plans[rank]['markers'][lane]
                self.assertEqual(marker, dict(device=source_hca, source_ip=source_ip, destination_ip=dest_ip))
                rules = [r for p in plans.values() for r in p['rules'] if r['origin']==rank and r['lane']==lane]
                self.assertEqual(len(rules),1)
                self.assertIn(self.by_rank[dest]['mesh_ports'][target_hca]['mac'], rules[0]['add'])
                self.assertIn('skip_sw', rules[0]['add'])
                self.assertNotIn('ip', rules[0]['add'])
                self.assertNotIn('udp', rules[0]['add'])
            self.assertEqual(len(plans[rank]['rules']), 2)
        self.assertEqual(self.nodes, original)

    def test_invalid_geometry_rejected_before_host_actions(self):
        for mutate in (
            lambda n: n['mesh_ports'].pop(next(iter(n['mesh_ports']))),
            lambda n: n['mesh_ports']['rocep1s0f0'].update(pci='0002:01:00.0'),
            lambda n: n['mesh_ports']['rocep1s0f0'].update(address='10.99.0.1/24'),
            lambda n: n['mesh_ports']['rocep1s0f0'].update(netdev='bad;command'),
        ):
            nodes = copy.deepcopy(self.nodes); mutate(nodes['nodes'][0])
            with self.assertRaises(ValueError): fabric.build_plan(nodes)
        self.assertEqual(topology.problems(self.cluster, self.nodes), [])
        self.cluster['environment']['B12X_ROCE_TOPOLOGY']='ring4'
        self.assertTrue(topology.problems(self.cluster, self.nodes))

    def test_doctor_checks_candidate_source_and_prevents_unqualified_serving(self):
        cluster, nodes, lock = spark3.configuration(argparse.Namespace(cluster_config=EXPERIMENT+'/cluster.json'))
        errors, _ = spark3.split_findings(spark3.local_doctor(cluster,nodes,lock))
        self.assertEqual(errors, [])
        cluster['deployment']['launch_enabled']=True
        errors, _ = spark3.split_findings(spark3.local_doctor(cluster,nodes,lock))
        self.assertTrue(any('bounded collective' in e for e in errors))

    def test_probe_owns_fabric_lifetime_and_checks_correct_runtime(self):
        command = spark3.collective_probe_command(self.cluster, self.nodes, self.by_rank[0], 29581)
        self.assertEqual(command[:2], ['sudo','python3'])
        self.assertTrue(command[2].endswith('/scripts/mesh_probe.py'))
        self.assertIn('B12X_ROCE_TOPOLOGY=mesh4', command)
        self.assertIn('--name=spark3-collective-probe', command)

    def facts(self):
        node = self.by_rank[0]
        result = {'ports': {}, 'errors': []}
        for h,p in node['mesh_ports'].items():
            addr = str(ipaddress.ip_interface(p['address']).ip)
            result['ports'][h] = dict(pci=p['pci'],netdev=p['netdev'],mac=p['mac'],
                addresses=[p['address']], mtu=9000,up=True,hw_tc=True,
                parameters={k:v for k,(_,v) in fabric.PARAMETERS.items()},
                eswitch={'mode':'legacy','inline-mode':'none','encap-mode':'basic'},
                gid='::ffff:'+addr,gid_type='RoCE v2',rdma_mtu='4096')
        return result

    def test_doctor_is_read_only_and_separates_correction_commands(self):
        facts = self.facts(); node = self.by_rank[0]
        self.assertEqual(fabric.findings(node, facts), ([],[]))
        for p in facts['ports'].values(): p['parameters']['hairpin_queue_size']=1024
        with mock.patch.object(fabric.subprocess,'run',side_effect=AssertionError('must not execute fixes')):
            errors, commands = fabric.findings(node, facts)
        self.assertEqual(len(errors),4); self.assertEqual(len(commands),4)
        self.assertTrue(all('value 8192 cmode driverinit' in c for c in commands))
        facts['ports'].clear()
        self.assertEqual(len(fabric.findings(node,facts)[0]),4)

    def test_unrelated_or_software_tc_filter_cannot_pass(self):
        rule = fabric.build_plan(self.nodes)[0]['rules'][0]
        own = dict(pref=rule['pref'],kind='flower',options={'in_hw':True})
        self.assertTrue(fabric.offload_ready(rule,[own]))
        for bad in (dict(own,pref=1),dict(own,options={'in_hw':False}),dict(own,options={})):
            self.assertFalse(fabric.offload_ready(rule,[bad]))

    def test_marker_requires_actual_ready_line(self):
        process = mock.Mock(); process.poll.return_value=None
        process.stdout.readline.return_value='error\n'
        with mock.patch.object(mesh_probe.select,'select',return_value=([process.stdout],[],[])):
            with self.assertRaises(RuntimeError): mesh_probe.wait_ready(process,'hca0')
            process.stdout.readline.return_value='READY hca0\n'
            mesh_probe.wait_ready(process,'hca0')
            process.poll.return_value=1
            with self.assertRaises(RuntimeError): mesh_probe.wait_ready(process,'hca0')

    def test_partial_setup_rolls_back_only_successfully_added_state(self):
        plan = fabric.build_plan(self.nodes)[0]
        actions = []
        def execute(command):
            if command == plan['routes'][1]['add']:
                raise RuntimeError('injected second route failure')
            actions.append(command)
        def capture(command):
            if command[:2] == ['docker','ps']: return ''
            if 'qdisc' in command: return '[{"kind":"clsact"}]'
            rules = [dict(pref=r['pref'],kind='flower',options={'in_hw':True})
                     for r in plan['rules'] if r['add'] in actions]
            return json.dumps(rules)
        args = argparse.Namespace(nodes=ROOT/EXPERIMENT/'nodes.example.json',rank=0,
            serving_container='serving',marker=Path('/marker'),
            command=['--','docker','run','--rm','--name=spark3-collective-probe'])
        with mock.patch.object(mesh_probe.os,'geteuid',return_value=0), \
             mock.patch.object(mesh_probe,'open',mock.mock_open()), \
             mock.patch.object(mesh_probe.fcntl,'flock'), \
             mock.patch.object(fabric,'inventory',return_value=self.facts()), \
             mock.patch.object(fabric,'capture',side_effect=capture), \
             mock.patch.object(mesh_probe,'execute',side_effect=execute), \
             mock.patch.object(mesh_probe.signal,'signal'):
            with self.assertRaisesRegex(RuntimeError,'second route failure'):
                mesh_probe.run(args)
        self.assertIn(plan['routes'][0]['delete'],actions)
        self.assertNotIn(plan['routes'][1]['delete'],actions)
        for rule in plan['rules']: self.assertIn(rule['delete'],actions)
        self.assertFalse(any('qdisc' in command for command in actions))


if __name__ == '__main__': unittest.main()
