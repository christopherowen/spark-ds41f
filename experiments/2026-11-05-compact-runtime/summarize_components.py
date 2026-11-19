"""Two-process component intervals from public receipts; no serving work."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('component_intervals',HERE.parent/'2026-11-05-answer-quality/summarize.py')
intervals=importlib.util.module_from_spec(spec);spec.loader.exec_module(intervals)


def main():
    p=argparse.ArgumentParser();p.add_argument('window',type=Path)
    root=p.parse_args().window;target=root/'component-comparison.json'
    assert not target.exists(),'refuse to overwrite evidence'
    paths=[root/(label+'.json') for label in ('timing-01','timing-02')]
    runs=[json.loads(path.read_text()) for path in paths]
    assert all(r['passed'] and not r['request_performance_measured'] for r in runs)
    identity=lambda row:(row['file'],row['sha256'],row['width'],row['page'],row['rows'],row['source'])
    assert [identity(r) for r in runs[0]['cases']]==[identity(r) for r in runs[1]['cases']]
    report={'complete':True,'request_performance_measured':False,'model_quality_tested':False,
        'input_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'scope':runs[0]['scope'],'cases':[],
        'uncertainty':'conditional two-process component joint95 ratio intervals assume normal process medians (unverified), df1; not a simultaneous guarantee over all shapes, model requests or unseen documents'}
    for a,b in zip(runs[0]['cases'],runs[1]['cases']):
        native=[statistics.median(r['native_before']['samples_ms']+r['native_after']['samples_ms']) for r in (a,b)]
        candidate=[r['compact']['median_ms'] for r in (a,b)]
        row={k:a[k] for k in ('file','sha256','page','width','rows','source','visible_range')}
        row['milliseconds']=intervals.speed(native,candidate)
        report['cases'].append(row)
    report['construction']=[r['construction'] for r in runs]
    report['memory_observations']=[r['memory_observations'] for r in runs]
    report['conclusion']='Component evidence only; inspect single-row regressions and required maintenance. Independent final-answer, request-speed and memory/admission qualification remains required.'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(report['cases']),'out':str(target),'model_quality_tested':False}))


if __name__=='__main__':main()
