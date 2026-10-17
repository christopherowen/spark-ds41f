"""Compare battery runs; exclude counting ceiling from means, as v41bench's footer does.

Usage: compare.py <dirA> <dirB> [labelA labelB]
"""
import json, re, sys, statistics as st, pathlib
a, b = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
la, lb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else (a.name, b.name)

def idle(d):
    out = {}
    for line in (d / 'idletest.txt').read_text(encoding='utf-8').splitlines():
        m = re.match(r'(\w) (\w+) [^:]+: (\d+) tok, TTFT ([\d.]+)s, ([\d.]+) tok/s after first token, ([\d.]+) tok/step, step ms p50\s+([\d.]+)', line)
        if m: out[m.group(1) + ' ' + m.group(2)] = (float(m.group(5)), float(m.group(6)), float(m.group(7)))
    return out

def bench(d):
    lv, pf = {}, {}
    for line in (d / 'bench.txt').read_text(encoding='utf-8').splitlines():
        m = re.match(r'\s+C(\d+) (\w+)\s+agg\s+([\d.]+) tok/s\s+per-stream ([\d.]+)', line)
        if m and m.group(2) != 'ceiling_count': lv.setdefault(int(m.group(1)), {})[m.group(2)] = (float(m.group(3)), float(m.group(4)))
        m = re.match(r"\s+prefill\s+\d+: .*'prompt_tokens': (\d+).*'prefill_tok_s': ([\d.]+)", line)
        if m: pf[int(m.group(1))] = float(m.group(2))
    return lv, pf

def qeval(d):
    f = next(d.glob('qeval-*.json')); q = json.loads(f.read_text(encoding='utf-8'))
    r = {x['id']: x for x in q['results']}
    return q, r

ia, ib = idle(a), idle(b)
print(f"## Idle step time (idletest)\n| run | {la} tok/s, tok/step, ms | {lb} tok/s, tok/step, ms |\n|---|---|---|")
for k in ia:
    if k in ib: print(f"| {k} | {ia[k][0]:.1f}, {ia[k][1]:.2f}, {ia[k][2]:.1f} | {ib[k][0]:.1f}, {ib[k][1]:.2f}, {ib[k][2]:.1f} |")

(lva, pfa), (lvb, pfb) = bench(a), bench(b)
print(f"\n## v41bench (aggregate tok/s, mean over 8 categories / per-stream mean; counting ceiling excluded)\n| C | {la} | {lb} | change |\n|---:|---:|---:|---:|")
for c in sorted(lva):
    if c not in lvb: continue
    ma = st.mean(v[0] for v in lva[c].values()); mb = st.mean(v[0] for v in lvb[c].values())
    pa = st.mean(v[1] for v in lva[c].values()); pb = st.mean(v[1] for v in lvb[c].values())
    print(f"| {c} | {ma:.1f} / {pa:.1f} | {mb:.1f} / {pb:.1f} | {100*(mb/ma-1):+.1f}% |")
print(f"\n### per category, C1 per-stream and C16 aggregate\n| category | C1 {la} | C1 {lb} | C16 {la} | C16 {lb} |\n|---|---:|---:|---:|---:|")
for cat in lva.get(1, {}):
    print(f"| {cat} | {lva[1][cat][1]:.1f} | {lvb.get(1,{}).get(cat,(0,0))[1]:.1f} | {lva.get(16,{}).get(cat,(0,0))[0]:.1f} | {lvb.get(16,{}).get(cat,(0,0))[0]:.1f} |")
print(f"\n## Cold prefill (tok/s)\n| prompt tokens | {la} | {lb} |\n|---:|---:|---:|")
for k in sorted(set(pfa) | set(pfb)):
    print(f"| {k:,} | {pfa.get(k, '-')} | {pfb.get(k, '-')} |")
na, nb = json.loads((a / 'needle.json').read_text(encoding='utf-8')), json.loads((b / 'needle.json').read_text(encoding='utf-8'))
print(f"\n## Salted needle\n{la}: {na['prompt_tokens']:,} tok, pass={na['pass']}, {na['prefill_tok_s']} tok/s ({na['total_s']} s)\n{lb}: {nb['prompt_tokens']:,} tok, pass={nb['pass']}, {nb['prefill_tok_s']} tok/s ({nb['total_s']} s)")
(qa, ra), (qb, rb) = qeval(a), qeval(b)
pa_ = [k for k, v in ra.items() if v.get('pass')]; pb_ = [k for k, v in rb.items() if v.get('pass')]
print(f"\n## qeval (75 tasks, greedy, concurrency 1)\n| | {la} | {lb} |\n|---|---:|---:|")
print(f"| passed | {len(pa_)}/{len(ra)} | {len(pb_)}/{len(rb)} |")
print(f"| token-cap hits | {sum(1 for v in ra.values() if v.get('finish_reason')=='length')} | {sum(1 for v in rb.values() if v.get('finish_reason')=='length')} |")
print(f"| median tok/s | {qa['median_tok_s']} | {qb['median_tok_s']} |")
print(f"| wall s | {qa['wall_s']} | {qb['wall_s']} |")
ct_a = sum(v.get('completion_tokens') or 0 for v in ra.values()); ct_b = sum(v.get('completion_tokens') or 0 for v in rb.values())
rc_a = sum(v.get('reasoning_chars') or 0 for v in ra.values()); rc_b = sum(v.get('reasoning_chars') or 0 for v in rb.values())
th = [k for k in ra if (ra[k].get('reasoning_chars') or 0) > 0 or (rb.get(k, {}).get('reasoning_chars') or 0) > 0]
print(f"| completion tokens, all tasks | {ct_a:,} | {ct_b:,} ({100*(ct_b/ct_a-1):+.1f}%) |")
print(f"| reasoning chars, {len(th)} thinking tasks | {rc_a:,} | {rc_b:,} ({100*(rc_b/max(rc_a,1)-1):+.1f}%) |")
fixed = sorted(k for k in ra if not ra[k].get('pass') and rb.get(k, {}).get('pass'))
broke = sorted(k for k in ra if ra[k].get('pass') and not rb.get(k, {}).get('pass'))
print(f"\nfixed by {lb}: {fixed}\nbroken by {lb}: {broke}\nmissed by both: {sorted(k for k in ra if not ra[k].get('pass') and not rb.get(k, {}).get('pass'))}")
for k in broke + fixed:
    print(f"  {k}: {la} -> {(ra[k].get('why') or '')!s:.80} | {lb} -> {(rb[k].get('why') or '')!s:.80}")
va = (a / 'vision.txt').read_text(encoding='utf-8').strip().splitlines()[-1]; vb = (b / 'vision.txt').read_text(encoding='utf-8').strip().splitlines()[-1]
print(f"\n## Vision + tools\n{la}: {va}\n{lb}: {vb}")

if (b / 'metrics-before-idle.txt').exists() and (b / 'metrics-after-idle.txt').exists():
    def metric(path, name):
        pattern = re.compile(r'^' + re.escape(name) + r'\{[^\n]*\}\s+([\d.eE+-]+)$', re.M)
        return sum(float(value) for value in pattern.findall(path.read_text(encoding='utf-8')))
    names = {'requests': 'vllm:request_success_total', 'tokens': 'vllm:generation_tokens_total',
             'drafts': 'vllm:spec_decode_num_drafts_total', 'accepted': 'vllm:spec_decode_num_accepted_tokens_total'}
    delta = {key: metric(b / 'metrics-after-idle.txt', name) - metric(b / 'metrics-before-idle.txt', name)
             for key, name in names.items()}
    assert delta['requests'] == 5, 'idle counter window includes unexpected requests'
    assert delta['tokens'] == delta['drafts'] + delta['accepted'], 'speculative token accounting mismatch'
    delta['tokens_per_step'] = delta['tokens'] / delta['drafts']
    print(f"\n## Idle DSpark counter cross-check\n{json.dumps(delta, sort_keys=True)}")
    print('Five idle-test requests only; generation tokens = accepted draft tokens + draft steps.')
