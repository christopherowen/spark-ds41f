"""Check retained receipts and reproduce the diagnostic decision table."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOW = HERE/'runs/semantic-window-01'


def read(path):
    return json.loads(path.read_text())


def main():
    closure = read(WINDOW/'restored.json')
    independent = read(WINDOW/'independent-restoration.json')
    assert closure['restored'] and closure['holds_removed'] and independent['passed']
    groups = {}
    for kind in ('projection', 'ranks', 'selection'):
        results = []
        for suffix in ('01', '02'):
            path = WINDOW/f'{kind}-{suffix}'
            verdict, result = read(path/'verdict.json'), read(path/'result.json')
            assert verdict['passed'] and verdict['exit'] == 0 and result['passed']
            assert verdict['commit'] == '3b8397f0e74d867643c7f4441ae01191d6a1b061'
            assert verdict['image'] == 'sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713'
            results.append(result)
        # Compare all numerical/correctness observations. Allocator peaks can
        # differ between fresh processes and are retained separately.
        def canonical(result):
            return [{k: v for k, v in c.items() if not k.startswith('peak_')} for c in result['cases']]
        assert canonical(results[0]) == canonical(results[1]), kind+' repeat differs'
        groups[kind] = results[0]
    rows, gates = [], 0
    for case in groups['selection']['cases']:
        n = case['extent']
        for variant in case['variants']:
            assert variant['native_selection_identical'] and variant['native_bound_dominance']
            if case['source']:
                assert all(variant[k] for k in ('native_source_blocks_identical', 'native_source_count_identical', 'native_reindex_identical'))
            assert variant['seed_keys_scored_once']+variant['additional_keys_to_score'] == variant['total_keys_to_score_once']
            gates += 2
        semantic = next(x for x in case['variants'] if x['semantic'])
        position = next(x for x in case['variants'] if not x['semantic'])
        rows.append({'snapshot': case['snapshot'], 'extent': n, 'source': case['source'],
                     'semantic_ideal_retained_keys': semantic['retained_keys'],
                     'semantic_practical_keys_scored_once': semantic['total_keys_to_score_once'],
                     'semantic_practical_pruned_percent': 100*(1-semantic['total_keys_to_score_once']/n),
                     'position_practical_keys_scored_once': position['total_keys_to_score_once']})
    assert len(rows) == 12 and gates == 48
    projected = groups['projection']['cases']
    assert all(v['retained_keys'] == c['extent'] for c in projected for v in c['variants'] if v['rank'] <= 32)
    query = read(WINDOW/'query-map-cpu-02/result.json')
    assert query['passed'] and len(query['skipped']) == 1
    assert query['skipped'][0]['excluded_from_projection_acceptance']
    assert all(c['retained_at_cpu_cutoff'] == c['extent'] for c in query['cases'] if c['map_rank'] <= 64)
    verdicts = [read(p) for p in WINDOW.glob('*/verdict.json') if p.parent.name in
                ('math-01', 'projection-01', 'projection-02', 'ranks-01', 'ranks-02', 'selection-01', 'selection-02')]
    result = {'schema_version': 1, 'decision': 'retain research; no qualified faster runtime',
              'scope': 'untimed fixed-snapshot diagnostics; no request speed claim',
              'runtime_commit': '3b8397f0e74d867643c7f4441ae01191d6a1b061',
              'fresh_process_repeats_identical': True, 'native_selection_checks': gates,
              'source_block_count_reindex_checks': 12, 'rows': rows,
              'key_projection_through_rank32_retains_every_key': True,
              'key_projection_cases': projected,
              'query_map_source_holdout_rejected': query['skipped'],
              'minimum_host_mem_available_bytes': min(v['memory']['minimum_mem_available_bytes'] for v in verdicts),
              'maximum_thermal_c': max(v['memory']['maximum_thermal_c'] for v in verdicts),
              'restored': closure, 'independent_restoration': 'runs/semantic-window-01/independent-restoration.json'}
    (HERE/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Exact semantic search screen', '',
             'Two fresh C36 processes agree on every diagnostic observation. All48 native',
             'top512 checks and12 source block/count/reindex checks pass. These are twelve',
             'snapshots from one prompt; source queries repeat across ranks. No new full',
             'model serving or request-speed qualification is claimed.', '',
             '| Snapshot | Visible keys | Semantic ideal retained | Practical scored once | Practical pruning |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['snapshot']} | {row['extent']} | {row['semantic_ideal_retained_keys']} | {row['semantic_practical_keys_scored_once']} | {row['semantic_practical_pruned_percent']:.2f}% |")
    lines += ['', 'The practical column includes every exact seed score and reuses it. It is work',
              'accounting from an untimed oracle, not a kernel timing. Native eight-key',
              'expansion loses additional sparsity; the executable native candidate API',
              'has not been changed to gather arbitrary individual keys.', '',
              'Cheap key-PCA maps through32 dimensions retain every key. Rank64 saves at',
              'most0.42% of keys; rank96 saves62.48% on one dense sample but its optimistic',
              'BF16 coefficients/norm already cost196 bytes/key versus68 native packed',
              'bytes. Other rank96 samples retain96.59–100% of keys. Query-fitted maps',
              'through64 dimensions also retain every key on distinct dense queries.',
              'The initial positive source result was training/evaluation overlap and is',
              'excluded: its query is identical across ranks. Both the flawed initial',
              'screen and the corrected exclusion remain retained.', '',
              'Small shape ranks lose pruning. Full seven-vector shapes keep useful dense',
              'pruning but require a128-dimensional center, seven128-dimensional factors,',
              'residual/rounding metadata and query-dependent slope optimization. Even an',
              'optimistic FP32 center/factor representation is4096 bytes per eight-key',
              'group before other metadata, versus544 native packed-key bytes. This is',
              'a cost obstacle, not a measured impossibility result for every exact method.', '',
              'Keep the native runtime. A new kernel/metadata design must demonstrate lower',
              'total cost before another request comparison. No latency confidence interval',
              'is fabricated from deterministic retention counts. C37 was restored on all',
              'three nodes, with independent live/image/source/lease verification.']
    (HERE/'results.md').write_text('\n'.join(lines)+'\n')
    print('all receipts, repeated observations, fidelity gates and closure verified')


if __name__ == '__main__':
    main()
