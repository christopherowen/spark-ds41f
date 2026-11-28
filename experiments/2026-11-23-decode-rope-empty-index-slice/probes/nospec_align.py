"""Per request: the token produced by its first decode step on an empty-slice graph size (TP8), vs
its first token/logprob difference from idle. Concurrent passes are found as connected groups of
requests that shared steps, taken in time order and paired with the pass files in launch order.
usage: nospec_align.py TRACE OUT.json IDLE.json PASS.json [PASS.json...]   (pass files in launch order)"""
import json, sys, collections
S = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
EMPTY = {25, 26, 27, 28, 33, 34, 35, 41, 42, 49}
seq = collections.defaultdict(list); first = {}; parent = {}
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for s in S:
    if str(s['req_ids'][0]).startswith('_warmup'): continue
    q = s['qsl']
    for i, r in enumerate(s['req_ids']):
        parent.setdefault(r, r); first.setdefault(r, s['t'])
        if q[i + 1] - q[i] == 1 and not s['prefilling'][i]:
            seq[r].append((s['input_ids'][q[i]], s['n_pad'], s['cg_mode']))
    for r in s['req_ids'][1:]: parent[find(r)] = find(s['req_ids'][0])
groups = collections.defaultdict(list)
for r in parent: groups[find(r)].append(r)
comps = sorted((g for g in groups.values() if len(g) > 1), key=lambda g: min(first[r] for r in g))
idle = json.load(open(sys.argv[3]))['records']
files = sys.argv[4:]
OUT = {'empty_slice_rows_tp8': sorted(EMPTY), 'passes': []}
comps = comps[-len(files):]
for p, comp in zip(files, comps):
    recs = json.load(open(p))['records']
    res = collections.Counter(); odd = []
    ids = set(comp)
    steps = collections.Counter(s['n_pad'] for s in S if s['cg_mode'] == 'FULL' and s['n_pad'] in EMPTY and s['req_ids'][0] in ids)
    for pi, (x, y) in enumerate(zip(idle, recs)):
        toks = [int(str(t).split(':')[-1]) for t in y['tokens']]
        cand = [r for r in comp if len(seq[r]) >= 1 and [t for t, _, _ in seq[r]] == toks[:len(seq[r])] and len(toks) - len(seq[r]) in (0, 1)]
        if len(cand) != 1: res['unmatched'] += 1; odd.append((pi, 'cands', len(cand))); continue
        e = next((j + 1 for j, (_, npad, m) in enumerate(seq[cand[0]]) if m == 'FULL' and npad in EMPTY), None)
        m = min(len(x['tokens']), len(toks))
        f = next((k for k in range(m) if x['tokens'][k] != y['tokens'][k] or x['token_logprobs'][k] != y['token_logprobs'][k]), None)
        if f is None and e is None: res['identical, never on an empty-slice size'] += 1
        elif f is None: res['identical despite an empty-slice step'] += 1; odd.append((pi, 'e', e))
        elif f == e: res['first diff = token made by its first empty-slice step'] += 1
        else: res['first diff elsewhere'] += 1; odd.append((pi, 'f', f, 'e', e))
    name = p.split('/')[-1]
    OUT['passes'].append({'file': name, 'requests': len(comp), 'results': dict(res),
                          'decode_steps_on_empty_slice_graphs': {str(k): v for k, v in sorted(steps.items())}})
    print(f"{name} ({len(comp)} reqs): {dict(res)} steps on empty-slice graphs {dict(steps)} {odd[:5]}")
json.dump(OUT, open(sys.argv[2], 'w'), indent=1, sort_keys=True)
