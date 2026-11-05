"""Summarize paired trials; keep process and between-process uncertainty separate."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent

def ci(values, critical):
    mean = statistics.mean(values)
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "low": mean-margin, "high": mean+margin}

def percent(interval):
    return {key: 100*math.expm1(value) for key,value in interval.items()}

def main():
    runs=[]
    for number in (1,2):
        folder=ROOT/'runs'/f'run-{number}'
        verdict=json.loads((folder/'verdict.json').read_text())
        assert verdict['passed'], verdict
        records=[json.loads(line) for line in (folder/'output.txt').read_text().splitlines() if line.startswith('{')]
        assert records[-1].get('passed'), records[-1]
        cases={(r['n'],r['rows'],r['tp3_last_rank']):r for r in records if 'samples_us' in r}
        assert len(cases)==21, len(cases)
        runs.append(cases)
    assert runs[0].keys()==runs[1].keys()
    results=[]
    for key in sorted(runs[0]):
        for mode in ('l2','cold','prefetch'):
            for variant in ('fused64','fused128','fused256','split_qkv','split_kvq','split_parallel','split_q','standalone','projection_only'):
                processes=[]
                process_log_means=[]
                for cases in runs:
                    arms=cases[key]['samples_us'][mode]
                    fused, candidate=arms['fused'], arms[variant]
                    assert len(fused)==len(candidate)==31
                    ratios=[math.log(b/a) for a,b in zip(fused,candidate)]
                    delta=[b-a for a,b in zip(fused,candidate)]
                    # Two-sided 95% t interval, df=30; paired by shuffled trial.
                    interval=ci(ratios,2.042272456)
                    process_log_means.append(interval['mean'])
                    processes.append({'fused_median_us':statistics.median(fused),
                                      'variant_median_us':statistics.median(candidate),
                                      'paired_delta_us_95ci':ci(delta,2.042272456),
                                      'paired_delta_pct_95ci':percent(interval)})
                results.append({'n':key[0],'rows':key[1],'tp3_last_rank':key[2],
                                'mode':mode,'variant':variant,'processes':processes,
                                # Two fresh processes only: df=1 deliberately gives
                                # a broad interval instead of pooling away drift.
                                'between_process_delta_pct_95ci':percent(ci(process_log_means,12.706204736))})
    output={'interpretation':'Positive delta is slower than the current fused boundary. projection_only is an empirical precomputed-input lower reference, not an implementable candidate or a rigorous bound. These are isolated kernel chains, not serving boots or full-step results.',
            'method':'31 shuffled paired trials per condition in each of two fresh containers. Within-process two-sided t intervals use paired log ratios (df=30); between-process intervals use the two process means (df=1). No released build was benchmarked.',
            'comparisons':results}
    (ROOT/'runs/analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    for mode in ('l2','cold','prefetch'):
        for variant in ('fused64','fused128','fused256','split_qkv','split_kvq','split_parallel','split_q','standalone','projection_only'):
            selected=[r for r in results if r['mode']==mode and r['variant']==variant]
            means=[r['between_process_delta_pct_95ci']['mean'] for r in selected]
            wins=[(r['n'],r['rows'],r['tp3_last_rank']) for r in selected if r['between_process_delta_pct_95ci']['high']<0]
            losses=sum(r['between_process_delta_pct_95ci']['low']>0 for r in selected)
            print(json.dumps({'mode':mode,'variant':variant,'delta_pct_range':[min(means),max(means)],'wins_95ci':wins,'losses_95ci':losses,'cases':len(selected)}))

if __name__=='__main__': main()
