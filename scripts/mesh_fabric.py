#!/usr/bin/env python3
"""Four-node NIC-forwarded fabric: pure plans and read-only host inventory.

Physical cable order is rank order. The generated rules change Ethernet only;
RoCEnante's endpoint QPs retain reliable delivery and the original payload.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
import re
import shlex
import subprocess

PARAMETERS = {"flow_steering_mode": ("runtime", "hmfs"),
              "hairpin_num_queues": ("driverinit", 4),
              "hairpin_queue_size": ("driverinit", 8192)}


def site_problems(nodes):
    errors, macs, ips = [], set(), set()
    entries = nodes.get("nodes", [])
    if len(entries) != 4 or sorted(n.get("rank", -1) for n in entries) != list(range(4)):
        return ["mesh4 requires four ranks in cable order"]
    for n in entries:
        ports = n.get("mesh_ports", {})
        required = {h for pair in n.get("roce_peer_hcas", {}).values() for h in pair}
        if set(ports) != required or len(ports) != 4:
            errors.append(f"{n['name']}: mesh_ports must describe all four cable HCAs")
            continue
        devices = set()
        for h, p in ports.items():
            try:
                if not re.fullmatch(r'[A-Za-z0-9_.-]{1,15}', p['netdev']) or p['netdev'] in devices:
                    raise ValueError('invalid or duplicate netdev')
                devices.add(p['netdev'])
                if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', p['pci']):
                    raise ValueError('invalid PCI address')
                if not re.fullmatch(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}', p['mac']) or int(p['mac'][:2],16) & 1 or p['mac'] in macs:
                    raise ValueError('invalid or duplicate unicast MAC')
                macs.add(p['mac'])
                addr = ipaddress.IPv4Interface(p['address'])
                if addr.network != ipaddress.IPv4Network(n['roce_subnets'][h]) or addr.ip in ips or addr.ip in (addr.network.network_address, addr.network.broadcast_address):
                    raise ValueError('duplicate IP or wrong cable subnet')
                ips.add(addr.ip)
            except (KeyError, ValueError, TypeError) as e:
                errors.append(f"{n['name']}/{h}: {e}")
        routes = n['roce_peer_hcas']
        try:
            prev, nxt = routes[str((n['rank']-1)%4)], routes[str((n['rank']+1)%4)]
            if len(prev) != 2 or len(nxt) != 2:
                raise ValueError('requires two stripes per cable')
            for lane in range(2):
                # Two functions of one CX7 PCI device, on the same PCI root.
                if ports[prev[lane]]['pci'].split('.')[0] != ports[nxt[lane]]['pci'].split('.')[0]:
                    raise ValueError('hairpin ingress/egress must share a PCI device on each lane')
            if ports[prev[0]]['pci'].split(':')[0] == ports[prev[1]]['pci'].split(':')[0]:
                raise ValueError('the two lanes must use different PCI roots')
        except (KeyError, ValueError) as e:
            errors.append(f"{n['name']}: {e}")
    return errors


def build_plan(nodes):
    errors = site_problems(nodes)
    if errors:
        raise ValueError('\n'.join(errors))
    by_rank = {n['rank']: n for n in nodes['nodes']}
    result = {r: {'rank': r, 'routes': [], 'rules': [], 'markers': []} for r in range(4)}
    for rank in range(4):
        dest = (rank + 2) % 4
        via = ((rank+1)%4, (rank-1)%4) if rank < dest else ((rank-1)%4, (rank+1)%4)
        for lane, mid in enumerate(via):
            def endpoint(r, peer):
                h = by_rank[r]['roce_peer_hcas'][str(peer)][lane]
                return h, by_rank[r]['mesh_ports'][h]
            hca, source = endpoint(rank, mid)
            _, ingress = endpoint(mid, rank)
            _, egress = endpoint(mid, dest)
            _, target = endpoint(dest, mid)
            ip = lambda p: str(ipaddress.IPv4Interface(p['address']).ip)
            route = ['ip', 'route', 'add', ip(target)+'/32', 'via', ip(ingress),
                     'dev', source['netdev'], 'src', ip(source)]
            undo = route.copy(); undo[2] = 'del'
            result[rank]['routes'].append({'add': route, 'delete': undo, 'destination': dest, 'lane': lane})
            # A namespace reserved for this experiment; add refuses collisions.
            pref = str(48100 + rank*2 + lane)
            identity = ['tc', 'filter', 'add', 'dev', ingress['netdev'], 'ingress',
                        'protocol', '0x88b5', 'pref', pref, 'handle', pref]
            add = identity + ['flower', 'skip_sw', 'src_mac', source['mac'], 'dst_mac', ingress['mac'],
                    'action', 'pedit', 'ex', 'munge', 'eth', 'type', 'set', '0x0800', 'pipe',
                    'action', 'pedit', 'ex', 'munge', 'eth', 'dst', 'set', target['mac'], 'pipe',
                    'action', 'mirred', 'egress', 'redirect', 'dev', egress['netdev']]
            delete = identity.copy(); delete[2] = 'delete'; delete += ['flower']
            result[mid]['rules'].append({'add': add, 'delete': delete, 'pref': int(pref),
                'ingress': ingress['netdev'], 'origin': rank, 'destination': dest, 'lane': lane})
            result[rank]['markers'].append({'device': hca, 'source_ip': ip(source), 'destination_ip': ip(target)})
    return result


def capture(command):
    p = subprocess.run(command, text=True, capture_output=True, timeout=15)
    if p.returncode:
        raise ValueError(f"{shlex.join(command)}: {p.stderr.strip()}")
    return p.stdout


def inventory(node):
    """Read settings; never create a flow, change an interface or reload a driver."""
    result = {'ports': {}, 'errors': []}
    for hca, wanted in node.get('mesh_ports', {}).items():
        try:
            device = Path('/sys/class/infiniband') / hca
            pci = (device/'device').resolve().name
            netdevs = sorted(p.name for p in (device/'device/net').iterdir())
            if wanted['netdev'] not in netdevs:
                raise ValueError(f'{hca}: expected netdev is absent')
            netdev = wanted['netdev']
            link = json.loads(capture(['ip', '-j', 'address', 'show', 'dev', netdev]))[0]
            values = {}
            for key, (cmode, _) in PARAMETERS.items():
                raw = json.loads(capture(['sudo', '-n', 'devlink', '-j', 'dev', 'param', 'show', 'pci/'+pci, 'name', key]))
                entries = raw['param']['pci/'+pci][0]['values']
                values[key] = next(v['value'] for v in entries if v['cmode'] == cmode)
            eswitch = json.loads(capture(['sudo', '-n', 'devlink', '-j', 'dev', 'eswitch', 'show', 'pci/'+pci]))['dev']['pci/'+pci]
            gid_dir = device/'ports/1'
            gid_index = str(node.get('roce_gid_index', 3))
            result['ports'][hca] = dict(pci=pci, netdev=netdev, mac=link['address'],
                mtu=link['mtu'], up=link.get('operstate') == 'UP',
                addresses=[f"{a['local']}/{a['prefixlen']}" for a in link['addr_info'] if a['family']=='inet'],
                parameters=values, eswitch=eswitch,
                hw_tc=bool(re.search(r'^hw-tc-offload: on$', capture(['ethtool','-k',netdev]),re.M)),
                gid=(gid_dir/'gids'/gid_index).read_text().strip(),
                gid_type=(gid_dir/'gid_attrs/types'/gid_index).read_text().strip(),
                rdma_mtu=re.search(r'active_mtu:\s+(\d+)', capture(['ibv_devinfo','-d',hca,'-i','1'])).group(1))
        except (OSError, ValueError, KeyError, IndexError, AttributeError, StopIteration, subprocess.TimeoutExpired) as e:
            result['errors'].append(str(e))
    return result


def findings(node, facts):
    errors = [f"{node['name']}: {e}" for e in facts.get('errors', [])]
    corrections = []
    for h, p in node.get('mesh_ports', {}).items():
        actual = facts.get('ports', {}).get(h)
        label = f"{node['name']}/{h}"
        if actual is None:
            errors.append(label+': no NIC inventory'); continue
        for field in ('pci','netdev','mac'):
            if actual.get(field) != p[field]: errors.append(f'{label}: {field} differs from node map')
        if p['address'] not in actual.get('addresses', []): errors.append(label+': expected cable address missing')
        if actual.get('mtu') != 9000 or not actual.get('up'): errors.append(label+': requires link UP at MTU 9000')
        if not actual.get('hw_tc'): errors.append(label+': hardware TC offload is disabled')
        if actual.get('eswitch') != {'mode':'legacy','inline-mode':'none','encap-mode':'basic'}:
            errors.append(label+': requires legacy eswitch, inline none, encapsulation basic')
        try:
            if actual.get('gid_type') != 'RoCE v2' or ipaddress.IPv6Address(actual['gid']).ipv4_mapped != ipaddress.IPv4Interface(p['address']).ip:
                raise ValueError()
        except (KeyError, ValueError): errors.append(label+': selected GID must be RoCE v2 for the configured cable IP')
        if actual.get('rdma_mtu') != '4096': errors.append(label+': active RDMA MTU must be 4096')
        for key, (cmode, value) in PARAMETERS.items():
            if actual.get('parameters', {}).get(key) != value:
                errors.append(f'{label}: {key} must be {value}')
                corrections.append(shlex.join(['sudo','devlink','dev','param','set','pci/'+p['pci'], 'name',key,'value',str(value),'cmode',cmode]))
    return errors, corrections


def offload_ready(rule, filters):
    """A software fallback or an unrelated offloaded filter cannot pass."""
    return any(f.get('pref') == rule['pref'] and f.get('kind') == 'flower'
               and f.get('options', {}).get('in_hw') is True for f in filters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan','doctor','inventory'])
    parser.add_argument('--nodes', type=Path)
    parser.add_argument('--rank', type=int, choices=range(4))
    parser.add_argument('--node-json', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.node_json:
        print(json.dumps(inventory(json.loads(args.node_json)))); return 0
    if not args.nodes: parser.error('--nodes is required')
    nodes = json.loads(args.nodes.read_text())
    plan = build_plan(nodes)
    if args.action == 'plan':
        print(json.dumps(plan if args.rank is None else plan[args.rank], indent=2)); return 0
    if args.rank is None: parser.error('--rank is required')
    node = next(n for n in nodes['nodes'] if n['rank'] == args.rank)
    facts = inventory(node)
    if args.action == 'inventory': print(json.dumps(facts, indent=2)); return 0
    errors, corrections = findings(node, facts)
    print(json.dumps({'problems': errors, 'correction_commands': corrections,
        'note': 'driverinit changes require a coordinated reload/reboot with all RDMA users stopped; doctor changes nothing'}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
