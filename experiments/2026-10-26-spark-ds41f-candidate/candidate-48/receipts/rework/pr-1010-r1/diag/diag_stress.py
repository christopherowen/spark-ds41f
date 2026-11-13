# Diagnostic: dump the free-queue head before each stress-probe eviction and the
# cached blocks the probe's eviction takes, per profile.
import sys
sys.path.insert(0, "/workspace/experiments/2026-11-08-prefix-cache-retention-probe")
import probe_prefix_cache_retention as P
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id

GROUPS = {0: "gmla", 1: "imla", 2: "eswa", 3: "dswa", 4: "fswa"}

def main():
    turn_schedule = [
        (65536, 389, 4), (70144, 412, 6), (75008, 290, 3), (80128, 395, 5),
        (85248, 401, 4), (90112, 310, 6), (95232, 399, 5), (99584, 512, 4),
    ]
    for profile_name, retention in {"tp3-triangle": 512, "tp4-ring": 8192}.items():
        manager, _ = P.make_ced_manager(retention, num_blocks=2200, max_model_len=131072)
        agent_tokens = {a: [a * 1_000_000 + i for i in range(102400)] for a in range(2)}
        h2pos = {}
        for a in range(2):
            r = P.make_ced_request(f"map-{a}", agent_tokens[a])
            for i, h in enumerate(r.block_hashes):
                h2pos[h] = (a, (i + 1) * P.HASH)
        prev = {0: 0, 1: 0}
        print(f"=== {profile_name} retention={retention}")
        for turn_idx, (plen, dlen, step) in enumerate(turn_schedule):
            if turn_idx in (3, 5, 7):
                fl = manager.block_pool.free_block_queue.get_all_free_blocks()
                ncached = sum(1 for b in fl if b.block_hash is not None)
                nfree = len(fl)
                head = []
                taken = 0
                cached_seen = 0
                for b in fl:
                    taken += 1
                    if b.block_hash is not None:
                        cached_seen += 1
                        a, end = h2pos[get_block_hash(b.block_hash)]
                        head.append(f"{GROUPS[get_group_id(b.block_hash)]}:a{a}@{end}")
                        if cached_seen == 18:
                            break
                print(f" turn {turn_idx+1}: free={nfree} cached_free={ncached} claim={taken} uncached_before={taken-18}")
                print("   evicted:", " ".join(head))
                claimed = manager.block_pool.get_new_blocks(taken)
                for b in claimed:
                    b.ref_cnt -= 1
                manager.block_pool.free_block_queue.prepend_n(claimed)
            for a in range(2):
                seq = agent_tokens[a]
                req = P.make_ced_request(f"agent-{a}-turn-{turn_idx}", seq[:plen])
                hit = P.run_prefill_and_decode(manager, req, seq[plen:plen + dlen],
                                               tokens_per_step=step, prefill_chunk_size=8192)
                if prev[a]:
                    print(f"   turn {turn_idx+1} agent {a}: hit={hit} prev_len={prev[a]} aligned={(prev[a]//256)*256}")
                prev[a] = plen + dlen
        # cached SWA checkpoints for agent 0 left in the pool at the end
        sw = sorted(h2pos[get_block_hash(b.block_hash)][1] for b in manager.block_pool.blocks
                    if b.block_hash is not None and get_group_id(b.block_hash) == 2
                    and h2pos[get_block_hash(b.block_hash)][0] == 0)
        print("  agent0 eswa cached ends (last 40):", sw[-40:])

if __name__ == "__main__":
    main()
