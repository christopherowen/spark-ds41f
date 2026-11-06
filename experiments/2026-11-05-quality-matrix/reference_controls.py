"""Reuse a published complete pair of unchanged same-day native controls."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from timing_conditions import fan_curve_ready


def dependency_paths(root,benchmark):
    from spark_ds41f.config import read_cluster
    paths=set()
    def configs(relative):
        if relative in paths:return
        paths.add(relative);raw=json.loads((root/relative).read_text())
        if 'extends' in raw:configs(raw['extends']['config'])
    configs('experiments/2026-11-05-quality-matrix/native.json')
    cluster=read_cluster('experiments/2026-11-05-quality-matrix/native.json')
    paths.add(cluster['nodes_config'])
    paths.add(cluster['upstreams_config'])
    for source,*_ in cluster['container']['mounts']:
        if source.startswith('{checkout}/'):paths.add(source.removeprefix('{checkout}/'))
    paths.update(['spark_ds41f/config.py','scripts/recipe.py','scripts/topology.py','spark_ds41f/bench.py',
                  'experiments/2026-11-05-quality-matrix/measure.py','experiments/2026-11-05-quality-matrix/timing_conditions.py','experiments/2026-11-05-answer-quality/api_metrics.py'])
    folder='2026-11-05-answer-quality' if benchmark=='longbench' else '2026-11-05-context-quality'
    paths.add('experiments/'+folder+'/evaluate.py')
    if benchmark=='longbench':
        paths.update('experiments/'+folder+'/'+name for name in ['prepare.py','sources.lock.json'])
    else:
        paths.update(str(p.relative_to(root)) for p in (root/'experiments'/folder).glob('*.set.json'))
    return sorted(paths)


def signature(run,benchmark):
    assert run['complete']
    if benchmark=='longbench':
        phases=run['phases'];assert phases
        groups=[p['records'] for p in phases]
        identity=[run['selection_sha256'],run['dataset_sha256'],run['protocol'],[(p['name'],p['concurrency'],p['cache_reuse']) for p in phases]]
        key=lambda r:(r['id'],r['prompt_sha256'],r['correct'],r['predicted'],r['status'],r['finish_reason'])
    else:
        groups=[run['records']]
        identity=[run['manifest_sha256'],run['full_manifest_sha256'],run['protocol'],run['split'],run['stage']]
        key=lambda r:(r['id'],r['prompt_sha256'],r['score'],r['status'],r['finish_reason'])
    for rows in groups:
        assert rows and len({r['id'] for r in rows})==len(rows)
        assert all(r['status']=='complete' and r['finish_reason']=='stop' for r in rows),'reference controls must have natural completions'
    return identity+[[key(r) for r in rows] for rows in groups]


def load(root,reference,benchmark,split,stage,nodes):
    root=root.resolve()
    assert benchmark in ('longbench','ruler'),'coding needs its own fresh canonical validation and controls'
    assert not reference.is_absolute() and '..' not in reference.parts
    folder=(root/reference).resolve();relative=folder.relative_to(root)
    assert folder.is_dir()
    git=lambda *a:subprocess.check_output(['git',*a],cwd=root)
    git('cat-file','-e','HEAD:'+str(relative/'restored.json'))
    git('cat-file','-e','HEAD:'+str(relative/'source.json'))
    restored=json.loads((folder/'restored.json').read_text());assert restored['restored'] and restored['holds_removed']
    source=json.loads((folder/'source.json').read_text())
    assert source['benchmark']==benchmark and source['split']==split and source['nodes']==nodes
    assert source.get('ruler_stage','full')==stage
    unchanged={}
    for name in dependency_paths(root,benchmark):
        raw=(root/name).read_bytes()
        assert git('show',source['commit']+':'+name)==raw,name+' changed; fresh native controls required'
        unchanged[name]=hashlib.sha256(raw).hexdigest()
    runs={};files={};date=datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    for label in ('native-a','native-b'):
        for name in ['answers-'+label+'.json','answers-'+label+'-conditions.json','benchmark-'+label+'-exit.json']:
            git('cat-file','-e','HEAD:'+str(relative/name));files[name]=(folder/name).read_bytes()
        exit=json.loads(files['benchmark-'+label+'-exit.json'])
        assert exit['exit']==0 and exit['started_at'][:10]==date,'reference controls must be from today UTC'
        conditions=json.loads(files['answers-'+label+'-conditions.json'])
        assert not conditions['halted'] and not conditions['fan_stop_failures'] and len(conditions['thermal'])==len(nodes)
        assert not any(t.get('thermal_slowdown_ms') for t in conditions['thermal'].values())
        assert fan_curve_ready(conditions.get('fan_curve',[]),conditions['thermal']),'reference controls lack the required performance fan curve; fresh qualified controls required'
        runs[label]=json.loads(files['answers-'+label+'.json'])
    assert signature(runs['native-a'],benchmark)==signature(runs['native-b'],benchmark),'unstable or mismatched native controls'
    provenance={'source':str(relative),'source_commit':source['commit'],'same_utc_day':date,
                'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},'unchanged_inputs':unchanged,
                'scope':'published historical native boots reused unchanged; candidate boots are fresh; these arms are not interleaved in the new window'}
    return runs,files,provenance
