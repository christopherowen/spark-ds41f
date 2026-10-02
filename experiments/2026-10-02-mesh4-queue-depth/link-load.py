#!/usr/bin/env python3
"""Count directed-link payload work for the current equal two-stripe mesh plan."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import mesh_fabric

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--nodes', type=Path, required=True)
args = parser.parse_args()
nodes = json.loads(args.nodes.read_text())
plan = mesh_fabric.build_plan(nodes)
by_rank = {node['rank']: node for node in nodes['nodes']}
links = defaultdict(list)
# For a payload divisible by 32 bytes, each neighbor receives half on each
# physical lane. Opposite-peer traffic adds half on each of its two hops.
for rank, node in by_rank.items():
    for peer in ((rank - 1) % 4, (rank + 1) % 4):
        for lane in range(2):
            links[lane, rank, peer].append({'origin': rank, 'destination': peer,
                                           'class': 'direct', 'payload_fraction': 0.5})
for middle, entry in plan.items():
    for rule in entry['rules']:
        origin, dest, lane = (rule[k] for k in ('origin', 'destination', 'lane'))
        for a, b in ((origin, middle), (middle, dest)):
            links[lane, a, b].append({'origin': origin, 'destination': dest,
                                      'class': 'forwarded', 'payload_fraction': 0.5})
rows = [{'lane': lane, 'source': by_rank[a]['name'], 'destination': by_rank[b]['name'],
         'payload_fraction': sum(flow['payload_fraction'] for flow in flows),
         'flows': flows} for (lane, a, b), flows in sorted(links.items())]
assert len(rows) == 16
assert sum(row['payload_fraction'] for row in rows) == 16
print(json.dumps({'assumptions': 'One identical divisible-by-32 payload per rank; two equal stripes; direct and forwarded all-to-all; excludes flags, ACKs and headers. Counts link work, not measured utilization or elapsed time.',
                  'links': rows, 'maximum_payload_fraction': max(row['payload_fraction'] for row in rows),
                  'minimum_payload_fraction': min(row['payload_fraction'] for row in rows),
                  'average_payload_fraction': 1.0}, indent=2))
