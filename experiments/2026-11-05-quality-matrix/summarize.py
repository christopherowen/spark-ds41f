"""Summarize public receipts only; no model text, programs or serving calls."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

from check_quality import check

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('boot_intervals',HERE.parent/'2026-11-05-answer-quality/summarize.py')
intervals=importlib.util.module_from_spec(spec);spec.loader.exec_module(intervals)
LABELS=('native-a','lsa-a','lsa-b','native-b')


def main():
    p=argparse.ArgumentParser();p.add_argument('window',type=Path)
    args=p.parse_args();root=args.window;target=root/'comparison.json'
    assert not target.exists(),'refuse to overwrite evidence'
    source=json.loads((root/'source.json').read_text());benchmark=source['benchmark']
    if benchmark=='longbench':
        subprocess.run([sys.executable,str(HERE.parent/'2026-11-05-answer-quality/summarize.py'),str(root)],check=True)
        return
    generations={};grades={};missing=[]
    for label in LABELS:
        generation=root/('answers-'+label+'.json')
        grade=root/('grade-'+label)/'receipt.json' if benchmark=='code' else generation
        if not generation.exists() or not grade.exists():missing.append(label);continue
        generations[label]=json.loads(generation.read_text());grades[label]=json.loads(grade.read_text())
        if not generations[label].get('complete') or not grades[label].get('complete'):missing.append(label)
    report={'benchmark':benchmark,'source':source,'complete':not missing,'promotion_qualified':False,
        'request_timing':'all generation including thinking and unsuccessful requests; boot, cooling and CPU grading recorded separately',
        'uncertainty':'two boot intervals assume normal boot means with df1; unique tasks are not multiplied by repeat boots',
        'restored':json.loads((root/'restored.json').read_text()) if (root/'restored.json').exists() else None}
    if missing:
        report.update(missing_or_incomplete=missing,conclusion='Incomplete matched cohort; no speed or quality qualification.')
    else:
        report['quality']=check(generations,grades,benchmark)
        report['per_boot']={};rates={}
        for label in LABELS:
            run=generations[label];rows=run['records'];correct=sum(r['correct'] for r in grades[label]['records'])
            wall=run['wall_seconds'];rate=correct/wall;rates[label]=rate
            row={'tasks':len(rows),'correct':correct,'accuracy':correct/len(rows),'wall_seconds':wall,
                 'correct_answers_per_second':rate,'median_request_seconds':statistics.median(r['seconds'] for r in rows),
                 'natural_completions':sum(r['status']=='complete' and r['finish_reason']=='stop' for r in rows),
                 'server_metrics':run['server_metrics']}
            if benchmark=='ruler':row['mean_official_score']=sum(r['score'] for r in grades[label]['records'])/len(rows)
            grading=root/('grade-'+label+'-exit.json')
            if grading.exists():row['separate_cpu_grading_seconds']=json.loads(grading.read_text())['seconds']
            report['per_boot'][label]=row
        report['wall_time']=intervals.speed([generations[n]['wall_seconds'] for n in ('native-a','native-b')],
                                           [generations[n]['wall_seconds'] for n in ('lsa-a','lsa-b')])
        old=[rates[n] for n in ('native-a','native-b')];new=[rates[n] for n in ('lsa-a','lsa-b')]
        a,b=intervals.interval(old),intervals.interval(new)
        report['useful_throughput']={'lsa_over_native':statistics.mean(new)/statistics.mean(old) if statistics.mean(old)>0 else None,
            'joint95_ratio_interval':[b[0]/a[1],b[1]/a[0]] if min(a+b)>0 else None}
        conditions={label:json.loads((root/('answers-'+label+'-conditions.json')).read_text()) for label in LABELS}
        report['conditions_passed']=all(not c['halted'] and not c['fan_stop_failures'] and len(c['thermal'])==4
            and not any(t.get('thermal_slowdown_ms') for t in c['thermal'].values()) for c in conditions.values())
        report['memory']={label:c['memory'] for label,c in conditions.items()}
        report['conclusion']='Inspect every quality loss, latency distribution, workload and condition before a superiority decision.'
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
