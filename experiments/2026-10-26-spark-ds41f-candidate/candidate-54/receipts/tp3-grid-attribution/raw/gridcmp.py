import json,sys
def load(p):
    g=json.load(open(p))
    out={}
    for e in g['engine']:
        d={}
        for k in ('step_ms','forward_ms','drafter_ms'):
            if k in e and e[k]: d[k]=e[k]['mean']
        out[e['label']]=d
    return out
paths=sys.argv[1:]
gs=[load(p) for p in paths]
labels=list(gs[0].keys())
for k in ('step_ms','forward_ms','drafter_ms'):
    print('==',k)
    for l in labels:
        vals=[g.get(l,{}).get(k) for g in gs]
        s='  '.join(f'{v:7.2f}' if v is not None else '   -   ' for v in vals)
        base=vals[0]
        d='  '.join(f'{(v-base):+6.2f}' if (v is not None and base) else '' for v in vals[1:])
        print(f'{l:28s} {s}   {d}')
