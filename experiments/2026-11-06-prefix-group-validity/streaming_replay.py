#!/usr/bin/env python3
"""Numerical scope check for recovery after actual streaming-input updates.

This scalar NumPy model is not native-kernel or full-checkpoint evidence. It
preserves the sequence of CED prefill floors and ordinary decode steps. Fresh
stateless requests that happen to extend a prior prompt are a different case.
"""
import argparse
import json
from pathlib import Path

import numpy as np

WINDOW, LAYERS = 128, 3


def run(encoder, phases):
    state = np.full((LAYERS + 1, len(encoder)), np.nan, dtype=np.float64)
    for start, end, floor in phases:
        begin = max(start, floor)
        if begin >= end:
            continue
        state[0, begin:end] = encoder[begin:end]
        for layer in range(LAYERS):
            source = state[layer]
            for position in range(begin, end):
                history = source[max(floor, position + 1 - WINDOW):position + 1]
                assert np.isfinite(history).all()
                scores = source[position] * history / np.sqrt(512.)
                maximum = max(0., scores.max())
                weights = np.exp(scores - maximum)
                state[layer + 1, position] = source[position] + np.dot(weights, history) / (
                    weights.sum() + np.exp(-maximum))
    return state


def difference(actual, expected):
    assert np.isfinite(actual[:-1, -WINDOW:]).all()
    assert np.isfinite(expected[:-1, -WINDOW:]).all()
    assert np.isfinite(actual[-1, -1]) and np.isfinite(expected[-1, -1])
    return {
        'final_output_equal': bool(actual[-1, -1] == expected[-1, -1]),
        'final_output_absolute_difference': float(abs(actual[-1, -1] - expected[-1, -1])),
        'live_kv_inputs_equal': bool(np.array_equal(actual[:-1, -WINDOW:], expected[:-1, -WINDOW:])),
        'live_kv_max_absolute_difference': float(np.max(abs(actual[:-1, -WINDOW:] - expected[:-1, -WINDOW:]))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    initial_prompt, generated_before, generated_after = 1024, 257, 17
    for appended in (1, 2, 127, 128, 129, 256):
        update_start = initial_prompt + generated_before
        updated_prompt = update_start + appended
        total = updated_prompt + generated_after
        encoder = np.sin(np.arange(total, dtype=np.float64) * .01) * .5
        phases = [
            (0, initial_prompt, initial_prompt - WINDOW),
            *((p, p + 1, 0) for p in range(initial_prompt, update_start)),
            (update_start, updated_prompt, updated_prompt - WINDOW),
            *((p, p + 1, 0) for p in range(updated_prompt, total)),
        ]
        uninterrupted = run(encoder, phases)
        first_anchor = run(encoder, [(0, total, initial_prompt - WINDOW)])
        latest_anchor = run(encoder, [(0, total, updated_prompt - WINDOW)])
        old_suffix = run(encoder, [(0, total, total - WINDOW)])
        row = {'appended_input_tokens': appended, 'initial_prompt': initial_prompt,
               'generated_before_update': generated_before, 'updated_prompt': updated_prompt,
               'generated_after_update': generated_after, 'phases': phases,
               'immutable_first_anchor': difference(first_anchor, uninterrupted),
               'mutable_latest_anchor': difference(latest_anchor, uninterrupted),
               'last_128_only': difference(old_suffix, uninterrupted)}
        rows.append(row)
    # Distinguish two failures of single-anchor recovery. A one-token update
    # does not truncate the next row's natural window; a >=128-token input
    # starts a fresh private suffix. Intermediate updates retain older layer
    # inputs while imposing a new floor on their newly computed rows.
    assert rows[0]['immutable_first_anchor']['live_kv_inputs_equal']
    assert not rows[0]['mutable_latest_anchor']['live_kv_inputs_equal']
    assert all(r['mutable_latest_anchor']['live_kv_inputs_equal'] for r in rows[3:])
    assert all(not r['immutable_first_anchor']['live_kv_inputs_equal'] for r in rows[1:])
    report = {'kind': 'scalar numerical dependency model; no GPU or checkpoint',
              'window': WINDOW, 'layers': LAYERS, 'passed': True, 'cases': rows}
    with args.out.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({'passed': True, 'cases': len(rows), 'kind': report['kind']}))


if __name__ == '__main__':
    main()
