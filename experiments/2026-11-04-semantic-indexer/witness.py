"""Seed-score reuse accounting; native full scores are an untimed oracle here."""
import torch


def witnessed_keep(proposal, upper, native, source):
    """Propose once, then retain all uncertified keys plus already scored seeds.

    The proposal may be arbitrarily poor. Only exact native seed scores establish
    the witness threshold, and upper bounds certify exclusions. Source witnesses
    use ordinary eight-key block maxima, excluding the forced final block.
    """
    n = len(native)
    seeds = torch.zeros(n, dtype=torch.bool, device=native.device)
    if source:
        tail = (-n) % 8
        estimate = torch.cat((proposal, proposal[-1:].expand(tail))).reshape(-1, 8).amax(1)
        actual = torch.cat((native, native[-1:].expand(tail))).reshape(-1, 8).amax(1)
        ordinary = max(0, len(actual)-1)
        block_ids = estimate[:ordinary].topk(min(2048, ordinary)).indices
        positions = (block_ids[:, None]*8+torch.arange(8, device=native.device)).flatten()
        seeds[positions] = True
        seeds[(n-1)//8*8:] = True
        threshold = actual[block_ids].topk(2047).values[-1] if len(block_ids) >= 2047 else -float('inf')
    else:
        seeds[proposal.topk(min(1024, n)).indices] = True
        threshold = native[seeds].topk(512).values[-1] if int(seeds.sum()) >= 512 else -float('inf')
    keep = (upper >= threshold) | seeds
    return keep, {'seed_keys_scored_once': int(seeds.sum()),
                  'additional_keys_to_score': int((keep & ~seeds).sum()),
                  'total_keys_to_score_once': int(keep.sum()),
                  'native_seed_cutoff': float(threshold), 'seed_scores_reused': True}
