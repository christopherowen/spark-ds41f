"""Untimed breakdown of the native top-512 keys the opt-in LSA shortlist drops.

Out-of-band selection proxy on the captured rank-0 operands. The native and LSA
selections come from the real kernels; Torch supplies per-key native scores and
the counterfactual shortlists. No serving change, speed or answer-quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from metadata import decode
from validate_retention import recall
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import STATE_BYTES, summarize_blocks
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerV41Plan
from vllm.models.deepseek_v4_1.tilelang.query_reuse import QueryReusePlan

TOPK = 512
RANK_EDGES = (32, 64, 128, 256, 384, 512)
RECENCY_EDGES = (512, 2048, 8192, 32768)
COMPANY_EDGES = (1, 2, 4, 8)
DEPTH_EDGES = (640, 768, 1024, 2048)
COARSE_EDGES = (1.0, 1.5, 2.0, 4.0, 8.0)
WINDOWS = (512, 2048, 8192)
BUDGET_FACTORS = (2, 4)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bf16(x):
    return x.bfloat16().float()


def scores_of(keys, query, weight):
    """Indexer arithmetic: BF16 dot, ReLU, BF16 weighted term, FP32 serial sum, BF16."""
    dots = bf16(keys @ query.T).clamp_min(0.)
    terms = bf16(dots * weight)
    total = torch.zeros(len(keys), dtype=torch.float32, device=keys.device)
    for head in range(terms.shape[1]):
        total += terms[:, head]
    return bf16(total)


def top_mask(scores, allowed, k=TOPK):
    """Top-k of the allowed keys; equal scores keep the lower position."""
    masked = torch.where(allowed, scores, torch.full_like(scores, float('-inf')))
    order = torch.sort(masked, descending=True, stable=True).indices[:k]
    order = order[masked[order] > float('-inf')]
    mask = torch.zeros_like(allowed)
    mask[order] = True
    return mask


def bucket(values, edges, right=False):
    return torch.bucketize(values, torch.tensor(edges, device=values.device), right=right)


def histogram(index, size):
    return torch.bincount(index, minlength=size).tolist()


def run_kernels(data):
    page, width, rows = data['page_size'], data['width'], len(data['q'])
    source = bool(data['candidate_blocks'])
    native_pool = data['pool'].cuda()
    lsa_pool = torch.zeros(len(native_pool), page * STATE_BYTES + 128,
                           dtype=torch.uint8, device='cuda')[:, :page * STATE_BYTES]
    lsa_pool[:, :page * 68].copy_(native_pool)
    complete = [i for i, n in enumerate(data['layout']['written_prefixes']) if n == page]
    slots = torch.tensor(complete, dtype=torch.int64, device='cuda') * page + page - 1
    summarize_blocks(page)(lsa_pool, lsa_pool.view(torch.bfloat16), slots)
    q, s, w = [data[k].cuda() for k in ('q', 'scales', 'weights')]
    table = torch.zeros(rows, width // page, dtype=torch.int32, device='cuda')
    copied = min(table.shape[1], data['table'].shape[1])
    table[:, :copied].copy_(data['table'][:, :copied])
    lengths, active = data['lengths'].cuda(), data['active'].cuda()
    budget = 1024 if source else 64

    def run(plan, pool):
        out = torch.empty(rows, TOPK, dtype=torch.int32, device='cuda')
        scratch = [torch.empty(shape, dtype=dtype, device='cuda')
                   for shape, dtype in plan.scratch_specs(rows)]
        kwargs = {'out': out, 'scratch': scratch}
        if source:
            kwargs['candidate_out'] = torch.empty(rows, CANDIDATE_BLOCKS, dtype=torch.int32, device='cuda')
            kwargs['candidate_out_lengths'] = torch.empty(rows, dtype=torch.int32, device='cuda')
        plan.run(q, s, w, pool, table, lengths, active, **kwargs)
        torch.cuda.synchronize()
        return out.clone()

    native = run(IndexerV41Plan(width, page, candidate_blocks=source), native_pool)
    lsa = run(QueryReusePlan(width, page, candidate_blocks=source, budget=budget, row_cap=256), lsa_pool)
    return native, lsa, native_pool, q, s, w, table, lengths, active, budget


def one(data):
    page, width, rows = data['page_size'], data['width'], len(data['q'])
    source = bool(data['candidate_blocks'])
    native, lsa, pool, q, s, w, table, lengths, active, budget = run_kernels(data)
    block_budget = budget * 8
    keys_physical = decode(pool[:, :page * 64].reshape(-1, page, 64),
                           pool[:, page * 64:page * 68].reshape(-1, page, 4))
    queries = bf16(decode(q, s))
    weights = w.float()
    limit = min(int(active[0]), width)

    names = ['emulated_baseline'] + [f'window_{n}' for n in WINDOWS]
    names += [f'budget_x{f}' for f in BUDGET_FACTORS] + ['oracle_block_mean_of_key_scores']
    per_row = {name: {'recall': [], 'top64_recall': []} for name in names}
    kernel = {'recall': [], 'top64_recall': [], 'top128_recall': []}
    checks = {'torch_native_agreement': [], 'emulated_lsa_agreement': []}
    totals = {
        'native_by_rank': [0] * (len(RANK_EDGES) + 1), 'missed_by_rank': [0] * (len(RANK_EDGES) + 1),
        'native_by_recency': [0] * (len(RECENCY_EDGES) + 1), 'missed_by_recency': [0] * (len(RECENCY_EDGES) + 1),
        'native_by_company': [0] * (len(COMPANY_EDGES) + 1), 'missed_by_company': [0] * (len(COMPANY_EDGES) + 1),
        'replacement_by_native_rank': [0] * (len(DEPTH_EDGES) + 1),
        'missed_by_block_coarse_rank_over_budget': [0] * (len(COARSE_EDGES) + 1),
        'missed_in_open_page': 0, 'missed': 0, 'native': 0,
    }

    for row in range(rows):
        visible = min(int(lengths[row]), limit)
        assert visible >= TOPK
        pages = -(-visible // page)
        keys = keys_physical[table[row, :pages].long()].reshape(-1, 128)[:visible]
        score = scores_of(keys, queries[row], weights[row])
        position = torch.arange(visible, device='cuda')
        everything = torch.ones(visible, dtype=torch.bool, device='cuda')

        native_ids, lsa_ids = native[row].long(), lsa[row].long()
        assert int(native_ids.min()) >= 0 and int(native_ids.max()) < visible
        assert int(lsa_ids.min()) >= 0 and int(lsa_ids.max()) < visible
        native_mask = torch.zeros(visible, dtype=torch.bool, device='cuda')
        native_mask[native_ids] = True
        lsa_mask = torch.zeros_like(native_mask)
        lsa_mask[lsa_ids] = True
        assert int(native_mask.sum()) == TOPK and int(lsa_mask.sum()) == TOPK

        rank = torch.empty(visible, dtype=torch.long, device='cuda')
        rank[torch.sort(score, descending=True, stable=True).indices] = position

        def measure(mask):
            kept = mask & native_mask
            return float(kept.sum()) / TOPK, float((kept & (rank < 64)).sum()) / 64, kept

        checks['torch_native_agreement'].append(float((top_mask(score, everything) & native_mask).sum()) / TOPK)
        r, r64, kept = measure(lsa_mask)
        kernel['recall'].append(r)
        kernel['top64_recall'].append(r64)
        kernel['top128_recall'].append(float((kept & (rank < 128)).sum()) / 128)

        missed = native_mask & ~lsa_mask
        replacement = lsa_mask & ~native_mask
        recency = visible - 1 - position
        padded = torch.zeros(-(-visible // 8) * 8, dtype=torch.long, device='cuda')
        padded[:visible] = native_mask.long()
        company = padded.reshape(-1, 8).sum(1).repeat_interleave(8)[:visible]
        for name, values, edges, right in (('rank', rank, RANK_EDGES, True),
                                           ('recency', recency, RECENCY_EDGES, True),
                                           ('company', company, COMPANY_EDGES, False)):
            index = bucket(values, edges, right)
            size = len(edges) + 1
            for which, mask in (('native', native_mask), ('missed', missed)):
                counts = histogram(index[mask], size)
                target = totals[f'{which}_by_{name}']
                for i, count in enumerate(counts):
                    target[i] += count
        for i, count in enumerate(histogram(bucket(rank[replacement], DEPTH_EDGES, True), len(DEPTH_EDGES) + 1)):
            totals['replacement_by_native_rank'][i] += count
        totals['missed'] += int(missed.sum())
        totals['native'] += TOPK

        # Counterfactual shortlists, emulated in Torch. Full pages are summarised
        # as BF16 means of eight consecutive keys; the open page is always scored.
        full = visible // page * (page // 8)
        open_page = position >= full * 8
        totals['missed_in_open_page'] += int((missed & open_page).sum())
        if full:
            means = bf16(keys[:full * 8].reshape(full, 8, 128).mean(1))
            coarse = scores_of(means, queries[row], weights[row])
            coarse_order = torch.sort(coarse, descending=True, stable=True).indices
            coarse_rank = torch.empty(full, dtype=torch.long, device='cuda')
            coarse_rank[coarse_order] = torch.arange(full, device='cuda')
            oracle_order = torch.sort(score[:full * 8].reshape(full, 8).mean(1),
                                      descending=True, stable=True).indices
            inside = missed & ~open_page
            ratio = (coarse_rank[position[inside] // 8].float() + 1) / block_budget
            for i, count in enumerate(histogram(bucket(ratio, COARSE_EDGES, True), len(COARSE_EDGES) + 1)):
                totals['missed_by_block_coarse_rank_over_budget'][i] += count

        def shortlist(order, blocks, window=0):
            if full <= blocks:
                return everything
            chosen = torch.zeros(full, dtype=torch.bool, device='cuda')
            chosen[order[:blocks]] = True
            allowed = open_page.clone()
            allowed[:full * 8] = chosen.repeat_interleave(8)
            if window:
                allowed |= position >= visible - window
            return allowed

        variants = {'emulated_baseline': shortlist(coarse_order, block_budget) if full else everything}
        for n in WINDOWS:
            variants[f'window_{n}'] = shortlist(coarse_order, block_budget, n) if full else everything
        for f in BUDGET_FACTORS:
            variants[f'budget_x{f}'] = shortlist(coarse_order, block_budget * f) if full else everything
        variants['oracle_block_mean_of_key_scores'] = shortlist(oracle_order, block_budget) if full else everything
        for name, allowed in variants.items():
            selected = top_mask(score, allowed)
            r, r64, _ = measure(selected)
            per_row[name]['recall'].append(r)
            per_row[name]['top64_recall'].append(r64)
            if name == 'emulated_baseline':
                checks['emulated_lsa_agreement'].append(float((selected & lsa_mask).sum()) / TOPK)

    def summary(values):
        return {'mean': sum(values) / len(values), 'minimum': min(values)}

    return {
        'layer': data['capture_context']['layer_id'], 'case_id': data['case_id'], 'source': source,
        'rows': rows, 'page_size': page, 'min_visible_keys': int(lengths.min()),
        'max_visible_keys': int(lengths.max()), 'budget_pages': budget, 'block_budget': block_budget,
        'fine_key_budget': block_budget * 8,
        'kernel_top512_recall_metric': {k: v for k, v in recall(native, lsa).items() if k != 'per_query'},
        'kernel': {k: summary(v) for k, v in kernel.items()},
        'checks': {k: summary(v) for k, v in checks.items()},
        'totals': totals,
        'counterfactuals': {name: {k: summary(v) for k, v in rowset.items()} for name, rowset in per_row.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--replay-receipt', type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads(args.replay_receipt.read_text())
    assert replay['passed'] and replay['actual_captures']
    paths = sorted(args.inputs.glob('*-rank-0-*.pt'))
    assert len(paths) == 12
    torch.cuda.reset_peak_memory_stats()
    report = {
        'passed': False, 'replay_sha256': digest(args.replay_receipt),
        'scope': 'untimed selection proxy on three documents; adjacent query rows do not add independent documents; '
                 'no serving change, speed or answer-quality result',
        'edges': {'rank': RANK_EDGES, 'recency_keys_from_row_end': RECENCY_EDGES, 'native_keys_in_same_block': COMPANY_EDGES,
                  'replacement_native_rank': DEPTH_EDGES, 'block_coarse_rank_over_budget': COARSE_EDGES,
                  'note': 'rank and recency buckets are [previous edge, edge); company buckets are (previous edge, edge]'},
        'windows_in_keys': WINDOWS, 'budget_factors': BUDGET_FACTORS, 'files': {},
    }
    try:
        for path in paths:
            assert digest(path) == replay['files'][path.name]['sha256']
            data = torch.load(path, map_location='cpu', weights_only=True)
            assert data['rank'] == 0 and len(data['q']) == 64
            report['files'][path.name] = one(data)
            print('CASE ' + json.dumps({'file': path.name, 'recall': report['files'][path.name]['kernel']['recall']}), flush=True)
        report['passed'] = True
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['memory_observations'] = {
            'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'scope': 'standalone tensor allocator only; no model memory qualification'}
        print('RESULT ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
