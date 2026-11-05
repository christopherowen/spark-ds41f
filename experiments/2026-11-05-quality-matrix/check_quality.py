"""Per-task final-answer gates for matched two-boot code or RULER receipts."""
import json


def canonical_check(receipt):
    """Check the pinned reference behavior without relaxing any model test."""
    assert receipt['complete'] and len(receipt['records'])==164
    failures=[r for r in receipt['records'] if not r['correct']]
    if failures:
        assert len(failures)==1 and failures[0]['id']=='HumanEval/32',failures
        row=failures[0]
        assert row['base']['passed'] and row['base']['tests']==row['base']['passed_tests']==100
        assert not row['plus']['passed'] and row['plus']['status']=='fail'
        assert row['plus']['tests']==row['plus']['executed']==788 and row['plus']['passed_tests']==781
        assert row['code_sha256']=='5fc21103508369bd5cd470d9713bf65c8c80e3b202c20a54f62e13a034a4c1b7'
    return {'passed':True,'canonical_correct':164-len(failures),'tasks':164,
            'known_reference_failures':[r['id'] for r in failures],
            'model_tests_changed':False,'model_tasks_omitted':[],
            'scope':'reference-program validation only; every model still faces all official base and plus tests'}


def check(generations,grades,benchmark):
    assert set(generations)==set(grades)=={'native-a','lsa-a','lsa-b','native-b'}
    first=generations['native-a'];ids=[r['id'] for r in first['records']]
    assert ids and len(set(ids))==len(ids)
    for label,run in generations.items():
        assert run['complete'] and [r['id'] for r in run['records']]==ids
        assert [(r['id'],r['prompt_sha256']) for r in run['records']]==[(r['id'],r['prompt_sha256']) for r in first['records']]
        key='source_lock' if benchmark=='code' else 'manifest_sha256'
        assert run[key]==first[key]
        assert grades[label]['complete'] and [r['id'] for r in grades[label]['records']]==ids
    losses=[];wins=[];incomplete=[];unstable=[]
    for i,identity in enumerate(ids):
        rows={label:g['records'][i] for label,g in generations.items()}
        scores={label:g['records'][i]['correct'] if benchmark=='code' else g['records'][i]['score'] for label,g in grades.items()}
        for label,row in rows.items():
            if row['status']!='complete' or row['finish_reason']!='stop':incomplete.append({'id':identity,'arm':label})
        for arm in ('native','lsa'):
            if scores[arm+'-a']!=scores[arm+'-b']:unstable.append({'id':identity,'arm':arm})
        old=max(scores['native-a'],scores['native-b']);new=min(scores['lsa-a'],scores['lsa-b'])
        if new<old:losses.append(identity)
        if new>old:wins.append(identity)
    return {'passed':not(losses or incomplete or unstable),'benchmark':benchmark,'tasks':len(ids),
            'wins':wins,'losses':losses,'incomplete':incomplete,'unstable':unstable,
            'promotion_qualified':False,'scope':'observed per-task no-loss only; no guarantee about untested inputs'}


def early(native,candidate,benchmark):
    assert native['complete'] and candidate['complete']
    assert [(r['id'],r.get('prompt_sha256')) for r in native['records']]==[(r['id'],r.get('prompt_sha256')) for r in candidate['records']]
    key='correct' if benchmark=='code' else 'score'
    return [a['id'] for a,b in zip(native['records'],candidate['records']) if b[key]<a[key]]
