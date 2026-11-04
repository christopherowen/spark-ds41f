"""Compare four matched request boots, retaining uncertainty and fidelity gates."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parent/'runs/serving-window-02'
ARMS=['lsa-a','control-a','control-b','lsa-b']
NOTE=('Two fresh boots per arm; Student-t intervals use boot medians, df=1. '
      'Normality is unverified at n=2. Ratios use Bonferroni97.5% component '
      'intervals for joint95%; no positive ratio interval when a component crosses zero.')


def ci(values,q=.975):
    center=statistics.mean(values)
    delta=math.tan(math.pi*(q-.5))*abs(values[0]-values[1])/2
    return [center-delta,center+delta]


reports={a:json.loads((ROOT/a/'bench.json').read_text()) for a in ARMS}
waves={a:[json.loads(x) for x in (ROOT/a/'requests.jsonl').read_text().splitlines()] for a in ARMS}
reference=waves['control-a']
identities=lambda rows:[(x['label'],[r['model_input_sha256'] for r in x['requests']]) for x in rows]
assert all(identities(waves[a])==identities(reference) for a in ARMS),'model inputs differ; no matched comparison'
assert len(reference)==41
output_differences=[]
for index,base in enumerate(reference):
    for a in ARMS:
        actual=waves[a][index]
        for stream,(x,y) in enumerate(zip(base['requests'],actual['requests'])):
            assert x['ok'] and y['ok'],(a,index,stream,'failed request')
            assert x['prompt_tokens']==y['prompt_tokens'],(a,index,stream,'tokenized inputs differ')
            if (x['output_sha256'],x['completion_tokens'],x['finish_reason'])!=(y['output_sha256'],y['completion_tokens'],y['finish_reason']):
                output_differences.append({'arm':a,'wave':index,'label':base['label'],'stream':stream,
                    'control_output':x['output_sha256'],'candidate_output':y['output_sha256']})
points=[]
labels=list(dict.fromkeys(x['label'] for x in reference if not x['label'].startswith('quality ')))
for label in labels:
    values={};ttft={}
    for a in ARMS:
        selected=[x for x in waves[a] if x['label']==label]
        if not label.startswith('prefill '):
            assert len(selected)==5
            selected=selected[1:]  # native discarded warmup
        else:assert len(selected)==3
        requests=[r for wave in selected for r in wave['requests']]
        values[a]=statistics.median(r['elapsed_s'] for r in requests)
        ttft[a]=statistics.median(r['ttft_s'] for r in requests)
    control=[values[a] for a in ['control-a','control-b']]
    candidate=[values[a] for a in ['lsa-a','lsa-b']]
    left,right=ci(control,.9875),ci(candidate,.9875)
    ratio=statistics.mean(control)/statistics.mean(candidate)
    points.append({'workload':label,'actual_prompt_tokens':[r['prompt_tokens'] for wave in reference if wave['label']==label for r in wave['requests']], 'metric':'native raw API request elapsed seconds; median per boot',
        'control_boot_medians_s':control,'lsa_boot_medians_s':candidate,
        'control_boot_median_t95_s':ci(control),'lsa_boot_median_t95_s':ci(candidate),
        'request_speedup_control_over_lsa':ratio,
        'lsa_latency_change_pct':100*(1/ratio-1),
        'speedup_joint95_interval':[left[0]/right[1],left[1]/right[0]] if min(left+right)>0 else None,
        'control_ttft_boot_medians_s':[ttft[a] for a in ['control-a','control-b']],
        'lsa_ttft_boot_medians_s':[ttft[a] for a in ['lsa-a','lsa-b']]})
for a in ARMS:
    d=reports[a]
    assert d['suites']['quality']['passed']==5
    assert d['identity']['client_commit']=='bc004ccedbcf9266fb52f6496d0b2975e1c8df43'
    assert all(x['image_id']=='sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713' for x in d['identity']['nodes'].values())
    assert all(len(v)==4 for v in d['suites']['decode']['samples'].values())
result={'schema_version':1,'kind':'paired real-weight TP3-switched request comparison; no released build remeasured',
        'arms':ARMS,'runtime_commit':'bc004ccedbcf9266fb52f6496d0b2975e1c8df43',
        'single_variable':'VLLM_DS41_CLUSTERED_SEARCH=0/1 on identical C36 image and guard',
        'all_inputs_match':True,'request_waves_per_boot':len(reference),
        'requests_per_boot':sum(len(x['requests']) for x in reference),
        'input_sequence_sha256':hashlib.sha256(json.dumps(identities(reference),sort_keys=True).encode()).hexdigest(),
        'all_outputs_match':not output_differences,'output_differences':output_differences,
        'quality_passes_per_boot':5,'points':points,'uncertainty_note':NOTE,
        'scope':'Nominal32K/131K prefill classes contain actual28095–30739/116664–121877 tokens. Five quality requests, 256-token prose/code at c1/c4/c8 with one discarded warmup and four fixed samples; three cold source-prefill samples at32K/131K. No mixed-traffic, full distribution or promotion gate.',
        'conditions':'Same existing Spaceport4KiB kernel and stock fan control; retain native cooling/thermal/memory observations. These hosts do not establish equivalence with archived64KiB release hardware.',
        'pilot':'serving-window-01 retained separately; not used in matched comparison',
        'released_timing_repeated':False,'promoted':False}
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Matched request comparison','',
       'Both arms use the same C36 image and exact scratch-read guard. The only runtime variable is clustered search.',
       '',f'Inputs match across all four boots; output digests match: **{not output_differences}**.',
       '', '| Workload | Control seconds, two boots | LSA seconds, two boots | LSA latency change |',
       '| --- | --- | --- | --- |']
for p in points:
    fmt=lambda v:' / '.join(f'{x:.4f}' for x in v)
    lines.append(f"| {p['workload']} | {fmt(p['control_boot_medians_s'])} | {fmt(p['lsa_boot_medians_s'])} | {p['lsa_latency_change_pct']:+.1f}% |")
lines+=['',NOTE,'','Positive latency change means slower. Full intervals and per-boot TTFT are in `comparison.json`.','',
        'Original native reports and request receipts retain every sample, memory/thermal observation and error. No release was remeasured or promoted.']
(ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
