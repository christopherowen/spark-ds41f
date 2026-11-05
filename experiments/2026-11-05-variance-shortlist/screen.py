"""Untimed variance-margin selection probe; no serving or speed claim."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from metadata import decode
from validate_retention import recall
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan, CANDIDATE_BLOCKS
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import summarize_pages, STATE_BYTES
from vllm.models.deepseek_v4_1.tilelang.direct_shortlist import DirectPageShortlistPlan

BETAS = (0., .01, .05, .2, 1.)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def one(data, betas):
    page, width, rows = data['page_size'], data['width'], len(data['q'])
    source = bool(data['candidate_blocks'])
    assert rows == 64 and width % page == 0
    assert int(data['active'][0]) <= width and int(data['lengths'].max()) <= width
    pool = torch.zeros(len(data['pool']), page*STATE_BYTES+128,
                       dtype=torch.uint8, device='cuda')[:, :page*STATE_BYTES]
    pool[:, :page*68].copy_(data['pool'])
    copied = min(data['table'].shape[1], width//page)
    assert copied*page >= int(data['lengths'].max())
    table = torch.zeros(rows, width//page, dtype=torch.int32, device='cuda')
    table[:, :copied].copy_(data['table'][:, :copied])
    q, scales, weights = [data[k].cuda() for k in ('q', 'scales', 'weights')]
    lengths, active = data['lengths'].cuda(), data['active'].cuda()
    args = [q, scales, weights, pool, table, lengths, active]
    complete = torch.tensor([i for i, n in enumerate(data['layout']['written_prefixes'])
                             if n == page], dtype=torch.int64, device='cuda')
    assert len(complete)
    summarize_pages(page)(pool, pool.view(torch.bfloat16), complete*page+page-1)

    # Required statistics are computed with Torch only for this offline probe.
    # A serving implementation, if justified, must build/update them in-band.
    buckets = page//64
    centers = pool[complete, page*68:page*72].contiguous().view(torch.bfloat16)
    centers = centers.reshape(len(complete), buckets, 128).float()
    labels = pool[complete, page*72:page*72+page//8].long().repeat_interleave(8, 1)
    assert bool((labels < buckets).all())
    keys = decode(pool[complete, :page*64].reshape(-1, page, 64),
                  pool[complete, page*64:page*68].reshape(-1, page, 4))
    assert bool(keys.isfinite().all())
    assigned = centers.gather(1, labels[:, :, None].expand(-1, -1, 128))
    squared = (keys-assigned).square().sum(-1)
    sums = torch.zeros(len(complete), buckets, dtype=torch.float32, device='cuda')
    sizes = torch.zeros_like(sums)
    sums.scatter_add_(1, labels, squared)
    sizes.scatter_add_(1, labels, torch.ones_like(squared))
    assert bool((sizes == 64).all()), 'membership must retain balanced native groups'
    radii = torch.zeros(len(pool), buckets, dtype=torch.float32, device='cuda')
    radii[complete] = (sums/64).sqrt()
    logical_radii = radii[table.long()].reshape(rows, width//64)
    query_factor = (decode(q, scales).norm(dim=-1)*weights.float().abs()).sum(-1)
    margin = query_factor[:, None]*logical_radii
    assert bool(margin.isfinite().all()) and bool((margin >= 0).all())

    native = IndexerV41Plan(width, page, candidate_blocks=source)
    options = {}
    if source:
        options = {'candidate_out': torch.empty(rows, CANDIDATE_BLOCKS, dtype=torch.int32, device='cuda'),
                   'candidate_out_lengths': torch.empty(rows, dtype=torch.int32, device='cuda')}
    expected, _ = native.run(*args, **options)
    expected = expected.clone()
    expected_blocks = options['candidate_out'].clone() if source else None
    plan = DirectPageShortlistPlan(width, page, candidate_blocks=source,
                                  budget=512 if source else 64, row_cap=256)
    scratch = [torch.empty(shape, dtype=dtype, device='cuda')
               for shape, dtype in plan.scratch_specs(rows)]
    b = dict(zip(plan._specs(rows), scratch))
    out = torch.empty(rows, 512, dtype=torch.int32, device='cuda')
    blocks = torch.empty(rows, CANDIDATE_BLOCKS, dtype=torch.int32, device='cuda')
    counts = torch.empty(rows, dtype=torch.int32, device='cuda')
    plan.run(*args, out=out, candidate_out=blocks, candidate_out_lengths=counts, scratch=scratch)
    reference = out.clone()
    reference_blocks = blocks.clone() if source else None
    reference_counts = counts.clone() if source else None

    # Initialize the unused suffix before any Torch arithmetic. Native kernels
    # need only the valid prefix, whereas pointwise Torch operations read it all.
    b['scores'].zero_()
    plan.score(q, scales, weights, pool.view(torch.bfloat16), table, lengths,
               active, b['scores'], b['counts'])
    base = b['scores'].clone()

    def buffers(name, subplan):
        return [b[f'{name}_{i}'] for i in range(len(subplan.scratch_specs(rows)))]

    results = {}
    for beta in betas:
        b['scores'].copy_((base.float()+beta*margin).bfloat16())
        plan.coarse.run(b['scores'], b['counts'], active, b['selected'], b['values'],
                        scratch=buffers('coarse', plan.coarse))
        plan.compact(b['selected'], pool, table, lengths, active,
                     b['candidates'], b['candidate_lengths'])
        _, scores = plan.final.run(q, scales, weights, pool, table, lengths, active,
                                   b['candidates'], b['candidate_lengths'], out=out,
                                   scratch=buffers('final', plan.final))
        if source:
            plan.select_blocks.run(scores, b['candidate_lengths'], active,
                                   b['compact'], b['block_values'], counts=counts,
                                   scratch=buffers('blocks', plan.select_blocks))
            plan.remap(b['compact'], b['candidates'], blocks)
        if beta == 0:
            torch.testing.assert_close(out, reference, atol=0, rtol=0)
            if source:
                torch.testing.assert_close(blocks, reference_blocks, atol=0, rtol=0)
                torch.testing.assert_close(counts, reference_counts, atol=0, rtol=0)
        row = {'native_top512_key_recall': recall(expected, out)}
        if source:
            row['native_source_block_recall'] = recall(expected_blocks, blocks)
        results[str(beta)] = row
    return {'case_id': data['case_id'], 'layer': data['capture_context']['layer_id'],
            'page_size': page, 'source_interface': source, 'rows': rows,
            'min_visible_keys': int(lengths.min()), 'max_visible_keys': int(lengths.max()),
            'zero_margin_matches_original_keys_and_handoff': True, 'variants': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--replay-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    replay = json.loads(args.replay_receipt.read_text())
    assert replay['passed'] and replay['actual_captures']
    paths = sorted(args.inputs.glob('*-rank-0-*.pt'))
    assert len(paths) == 12
    report = {'complete': False, 'betas': BETAS, 'replay_sha256': digest(args.replay_receipt),
              'formula': 'native centroid score + beta * cluster RMS distance * sum_h(abs(w_h) * norm(q_h))',
              'scope': 'untimed selection proxy; RMS margin is not an exclusion bound; no serving change',
              'model_answer_quality_tested': False, 'performance_measured': False,
              'independent_fit_documents': 2, 'independent_validation_documents': 1,
              'validation_note': 'reused diagnostic document, previously inspected for the separate PCA probe; no generalization claim',
              'files': {}}
    try:
        for path in paths:
            assert digest(path) == replay['files'][path.name]['sha256']
        training = [p for p in paths if p.name.startswith(('fit-a-', 'fit-b-'))]
        validation = [p for p in paths if p.name.startswith('validation-a-')]
        assert len(training) == 8 and len(validation) == 4
        for path in training:
            data = torch.load(path, map_location='cpu', weights_only=True)
            assert data['rank'] == 0 and data['case_id'] in ('fit-a', 'fit-b')
            row = one(data, BETAS)
            report['files'][path.name] = row
            args.out.write_text(json.dumps(report, indent=2)+'\n')
            print('CASE '+json.dumps({'file': path.name, 'complete': True}), flush=True)
        objectives = {}
        for beta in BETAS:
            values = []
            for row in report['files'].values():
                metric = 'native_source_block_recall' if row['source_interface'] else 'native_top512_key_recall'
                values.append(row['variants'][str(beta)][metric]['mean'])
            objectives[str(beta)] = sum(values)/len(values)
        # A single global beta, chosen on the fitting documents before reading
        # validation tensors. Ties choose the smallest margin, including zero.
        chosen = max(BETAS, key=lambda beta: (objectives[str(beta)], -beta))
        report.update(training_objectives=objectives, selected_beta=chosen)
        for path in validation:
            data = torch.load(path, map_location='cpu', weights_only=True)
            assert data['rank'] == 0 and data['case_id'] == 'validation-a'
            report['files'][path.name] = one(data, tuple(sorted({0., chosen})))
            args.out.write_text(json.dumps(report, indent=2)+'\n')
            print('CASE '+json.dumps({'file': path.name, 'complete': True}), flush=True)
        report['complete'] = True
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        print('RESULT '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
