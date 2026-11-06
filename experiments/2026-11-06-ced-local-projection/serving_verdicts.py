"""Strict functional verdicts for the unchanged published request clients."""
import json
from pathlib import Path

EXPECTED_SCENARIOS = {'c8', 'staggered', 'mixed', 'cached', 'again', 'mixed15'}


def verdict(path, kind):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('{'):
            rows.append(json.loads(line))
    summaries = [row for row in rows if 'summary' in row]
    if len(summaries) != 1:
        return {'passed': False, 'issues': ['exactly one summary is required'], 'summary': None}
    summary = summaries[0]
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    if kind in ('determinism', 'greedy'):
        check(len(rows) == 7, 'determinism record count differs')
        check(summary.get('summary') == 'temperature-0 determinism', 'unexpected summary kind')
        check(summary.get('identical') is True, 'tokens or log probabilities differ')
        check(summary.get('tokens') == 192, 'output budget differs')
        check(summary.get('logprobs') is (kind == 'determinism'), 'logprob mode differs')
        scenarios = summary.get('scenarios', {})
        check(set(scenarios) == EXPECTED_SCENARIOS, 'incomplete determinism scenarios')
        for name, value in scenarios.items():
            check(value.get('token_diffs') == value.get('logprob_diffs') == 0, name + ' differs')
        raw = {row['scenario']: row for row in rows if 'scenario' in row}
        check(set(raw) == EXPECTED_SCENARIOS, 'raw scenarios missing')
        for name, row in raw.items():
            expected = 16 if name == 'mixed15' else 9 if name == 'mixed' else 8
            requests = row.get('requests', [])
            check(len(requests) == expected, name + ' request count differs')
            check(row.get('token_diffs') == row.get('logprob_diffs') == 0, name + ' raw differences')
            for request in requests:
                check(all(request.get(key) is None for key in
                          ('first_token_diff', 'first_char_diff', 'first_logprob_diff'))
                      and request.get('logprobs_equal') is not False, name + ' request differs')
        for key in ('reference_sha256', 'long_sha256', 'reference_text_sha256', 'long_text_sha256'):
            check(isinstance(summary.get(key), str) and len(summary[key]) == 64, key + ' missing')
    elif kind == 'sampled':
        check(len(rows) == 3, 'sampled record count differs')
        check(summary.get('summary') == 'seeded temperature-1 digest', 'unexpected summary kind')
        check(summary.get('identical') is True, 'seeded sampled outputs differ')
        check(summary.get('tokens') == 192 and summary.get('seed') == 20261024, 'sampling identity differs')
        scenarios = summary.get('scenarios', {})
        check(set(scenarios) == {'c8', 'again'}, 'incomplete sampled scenarios')
        for name, value in scenarios.items():
            check(value.get('differing') == [], name + ' sampled differences')
        raw = {row['scenario']: row for row in rows if 'scenario' in row}
        check(set(raw) == {'c8', 'again'}, 'raw sampled scenarios missing')
        for name, row in raw.items():
            check(row.get('differing') == [] and row.get('sha256') == summary.get('reference_sha256'),
                  name + ' sampled digest differs')
        check(isinstance(summary.get('reference_sha256'), str) and len(summary['reference_sha256']) == 64,
              'sampled reference digest missing')
    elif kind == 'ced':
        check(len(rows) == 5, 'CED record count differs')
        check(summary.get('summary') == 'ced cross-shape' and summary.get('passed') is True, 'CED gate failed')
        lengths = summary.get('lengths', {})
        check(set(lengths) == {'8208', '16352'}, 'CED lengths differ')
        for name, value in lengths.items():
            check(value.get('repeats_identical') is True and value.get('cold') is True, name + ' repeat/cache failure')
            for key in ('prompt', 'tokens', 'logprobs'):
                check(isinstance(value.get(key), str) and len(value[key]) == 16, name + '/' + key + ' missing')
        requests = [row for row in rows if 'summary' not in row]
        check(len(requests) == 4, 'CED request count differs')
    else:
        raise ValueError(kind)
    return {'passed': not issues, 'issues': issues, 'summary': summary}


def compare(arms):
    issues = []
    expected = ['control-1', 'control-2', 'candidate-1', 'candidate-2']
    if list(arms) != expected:
        return {'passed': False, 'issues': ['four complete ordered arms are required']}
    for label, gates in arms.items():
        if set(gates) != {'ced', 'determinism', 'greedy', 'sampled'} or not all(g['passed'] for g in gates.values()):
            issues.append(label + ' functional gate failed')
    for kind in ('ced', 'determinism', 'greedy', 'sampled'):
        summaries = [arms[label].get(kind, {}).get('summary') for label in expected]
        if any(summary is None for summary in summaries):
            issues.append(kind + ' missing summary')
            continue
        keys = ('lengths',) if kind == 'ced' else ('reference_sha256',) if kind == 'sampled' else (
            'reference_sha256', 'long_sha256', 'reference_text_sha256', 'long_text_sha256')
        for key in keys:
            if any(summary.get(key) != summaries[0].get(key) for summary in summaries[1:]):
                issues.append(kind + '/' + key + ' differs across boots or images')
    for label, gates in arms.items():
        a, b = gates.get('determinism', {}).get('summary'), gates.get('greedy', {}).get('summary')
        if a and b and any(a.get(key) != b.get(key) for key in ('reference_text_sha256', 'long_text_sha256')):
            issues.append(label + ' logprob and greedy texts differ')
    return {'passed': not issues, 'issues': issues}
