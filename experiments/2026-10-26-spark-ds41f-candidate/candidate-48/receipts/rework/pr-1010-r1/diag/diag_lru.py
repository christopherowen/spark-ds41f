# Diagnostic: where the LRU probe's inversions sit (aged checkpoints vs the
# finished request's own run) for each retention.
import sys
sys.path.insert(0, "/probe")
import probe_prefix_cache_retention as P
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id

for retention in [0, 512, 8192, None]:
    manager, roles = P.make_ced_manager(retention, num_blocks=1024)
    swa = roles["encoder-swa"]
    tokens = list(range(32768))
    r1 = P.make_ced_request("turn-1", tokens[:16384])
    P.run_prefill_and_decode(manager, r1, tokens[16384:16896], tokens_per_step=4)
    r2 = P.make_ced_request("turn-2", tokens[:24576])
    P.run_prefill_and_decode(manager, r2, tokens[24576:25088], tokens_per_step=4)
    h2e = {h: (i + 1) * P.HASH for i, h in enumerate(r2.block_hashes)}
    ent = [(h2e[get_block_hash(b.block_hash)], get_group_id(b.block_hash) == swa)
           for b in manager.block_pool.free_block_queue.get_all_free_blocks() if b.block_hash is not None]
    top = max(e for e, _ in ent)
    rs = next(i for i, (e, _) in enumerate(ent) if e == top)
    inv = [i for i in range(len(ent) - 1)
           if ent[i][0] < ent[i + 1][0] or (ent[i][0] == ent[i + 1][0] and ent[i][1] > ent[i + 1][1])]
    print(f"retention={retention}: cached={len(ent)} aged={rs} (all_swa={all(s for _, s in ent[:rs])}) run={len(ent)-rs}")
    print(f"  inversions={len(inv)} in_aged={sum(1 for i in inv if i < rs - 1)} at_boundary={sum(1 for i in inv if i == rs - 1)} in_run={sum(1 for i in inv if i >= rs)}")
    print("  aged ends:", [e for e, _ in ent[:rs]])
