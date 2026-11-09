# SPDX-License-Identifier: Apache-2.0
"""PR #1010 rework, round 2: bound check for the retention probe's two
eviction clauses (0 LRU inversions, stress hit share >= 1107).

Runs on the candidate-48 image with the round-1 core files mounted. It does not
change product code. It replays the probe's two eviction scenarios and, for each
eviction, chooses which cached blocks are evicted according to an ordering
rule, so it can measure every ordering family against the same retained set:

  actual    the free queue as the mounted code builds it (the probe unchanged)
  floor1    aged SWA checkpoints first, each conversation's aged blocks in
            descending token order, then the finished requests' runs: the
            ordering with the fewest probe inversions that still evicts no
            live tail (1 inversion by construction in the LRU scenario)
  oracle0   the best eviction set any 0-inversion order can produce: per
            conversation the evicted blocks must be a prefix of its
            (end desc, MLA before SWA) order; the split k0 + k1 = N between the
            two conversations is chosen to minimise the next turns' loss, with
            the hit measured by the real lookup (get_computed_blocks)
  mono_lru  a natural 0-inversion order: conversations in LRU order, each one
            tail-to-root (agent 0 finished first, so its top goes first)

The LRU part counts the probe's inversions for actual / floor1 / monotone and
draws eviction curves (extension and branch hit after evicting the first k
cached blocks of each order), using temporary hash removal and the real lookup.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time

sys.path.insert(0, "/probe")
import probe_prefix_cache_retention as P  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id  # noqa: E402

GROUPS = {0: "gmla", 1: "imla", 2: "eswa", 3: "dswa", 4: "fswa"}
SWA_GID = 2
SCHED = [
    (65536, 389, 4), (70144, 412, 6), (75008, 290, 3), (80128, 395, 5),
    (85248, 401, 4), (90112, 310, 6), (95232, 399, 5), (99584, 512, 4),
]
EVICT_TURNS = (3, 5, 7)
REF_1107 = {"tp3-triangle": 1124352, "tp4-ring": 1122816}
REUSABLE = 1128008


def inversions(entries):
    """The probe's definition: adjacent (end, is_sparse) pairs out of order."""
    n = 0
    for i in range(len(entries) - 1):
        ea, sa = entries[i]
        eb, sb = entries[i + 1]
        if ea < eb or (ea == eb and sa > sb):
            n += 1
    return n


def wrap_free(manager, held):
    orig = manager.free

    def free(req):
        ids = set()
        for mgr in manager.coordinator.single_type_managers:
            for b in mgr.req_to_blocks.get(req.request_id, ()):
                if not b.is_null:
                    ids.add(b.block_id)
        held[req.request_id] = ids
        return orig(req)

    manager.free = free


def block_keys(pool, b):
    keys = []
    if b.block_hash is not None:
        keys.append(b.block_hash)
    keys.extend(pool.cached_block_hashes_by_block.get(b.block_id, ()))
    return keys


def hit_without(manager, blocks, prompt, name):
    """Hit of a fresh request for `prompt` if `blocks` were evicted."""
    pool = manager.block_pool
    saved = []
    for b in blocks:
        for k in block_keys(pool, b):
            if pool.cached_block_hash_to_block.pop(k, b.block_id) is not None:
                saved.append((k, b))
    try:
        req = P.make_ced_request(name, prompt)
        _, hit, _ = manager.get_computed_blocks(req)
    finally:
        for k, b in saved:
            pool.cached_block_hash_to_block.insert(k, b)
    return hit


def live_ids(manager, prompt, name):
    """Block ids the real lookup returns for `prompt` (the live hit path)."""
    req = P.make_ced_request(name, prompt)
    blocks, _, _ = manager.get_computed_blocks(req)
    return {b.block_id for grp in blocks.blocks for b in grp if not b.is_null}


def mono_key(e):
    # end desc, MLA before SWA at equal end, redundant copies first, then group.
    return (-e["end"], e["sparse"], not e["dup"], e["gid"], e["bid"])


def lru_part():
    out = {}
    for retention in [0, 512, 8192, None]:
        manager, roles = P.make_ced_manager(retention, num_blocks=1024)
        held = {}
        wrap_free(manager, held)
        swa = roles["encoder-swa"]
        tokens = list(range(32768))
        r1 = P.make_ced_request("turn-1", tokens[:16384])
        P.run_prefill_and_decode(manager, r1, tokens[16384:16896], tokens_per_step=4)
        r2 = P.make_ced_request("turn-2", tokens[:24576])
        hit2 = P.run_prefill_and_decode(manager, r2, tokens[24576:25088], tokens_per_step=4)
        h2e = {h: (i + 1) * P.HASH for i, h in enumerate(r2.block_hashes)}
        run_ids = held["turn-2"]
        ent = []
        for b in manager.block_pool.free_block_queue.get_all_free_blocks():
            if b.block_hash is None:
                continue
            gid = get_group_id(b.block_hash)
            ent.append({
                "end": h2e[get_block_hash(b.block_hash)],
                "sparse": gid == swa, "gid": gid, "bid": b.block_id,
                "run": b.block_id in run_ids, "dup": False, "blk": b,
            })
        ext_prompt = tokens[:25088] + tokens[25088:25600]
        live = live_ids(manager, ext_prompt, "ext-live")
        for e in ent:
            e["live"] = e["bid"] in live
        aged = [e for e in ent if not e["run"]]
        run = [e for e in ent if e["run"]]
        first_run = next(i for i, e in enumerate(ent) if e["run"])
        orders = {
            "actual": ent,
            "floor1": sorted([e for e in ent if not e["live"]], key=mono_key)
            + [e for e in ent if e["live"]],
            "monotone": sorted(ent, key=mono_key),
        }
        branch_points = [8192 + 64, 20480 + 64]
        res = {
            "turn2_hit": hit2,
            "cached": len(ent),
            "aged": len(aged),
            "aged_all_swa": all(e["sparse"] for e in aged),
            "aged_contiguous_ahead_of_run": all(not e["run"] for e in ent[:first_run])
            and all(e["run"] for e in ent[first_run:]),
            "run": len(run),
            "live_hit_path_cached": sum(1 for e in ent if e["live"]),
            "aged_on_hit_path": sum(1 for e in aged if e["live"]),
            "inversions": {k: inversions([(e["end"], e["sparse"]) for e in v])
                           for k, v in orders.items()},
            # Alternative scopes for the same adjacent-pair rule, on the actual
            # queue: only the finished request's own run, and per KV group.
            "actual_inversions_within_run_only": inversions(
                [(e["end"], e["sparse"]) for e in run]),
            "actual_inversions_per_group": {
                GROUPS[g]: inversions([(e["end"], e["sparse"]) for e in ent if e["gid"] == g])
                for g in sorted({e["gid"] for e in ent})
            },
            "curves": {},
        }
        for oname in ("actual", "monotone"):
            order = orders[oname]
            curve = {"extension": [], "branch_8256": [], "branch_20544": []}
            prev = {}
            for k in range(0, len(order) + 1):
                blocks = [e["blk"] for e in order[:k]]
                vals = {
                    "extension": hit_without(manager, blocks, ext_prompt, "ext"),
                }
                for bp in branch_points:
                    prompt = tokens[:bp] + [900000 + i for i in range(512)]
                    vals[f"branch_{bp}"] = hit_without(manager, blocks, prompt, "br")
                for key, v in vals.items():
                    if prev.get(key) != v:
                        curve[key].append([k, v])
                        prev[key] = v
            res["curves"][oname] = curve
        out[str(retention)] = res
    return out


def stress(retention, variant, evict_n=18, num_blocks=2200):
    manager, _ = P.make_ced_manager(retention, num_blocks=num_blocks, max_model_len=131072)
    pool = manager.block_pool
    held = {}
    wrap_free(manager, held)
    agent_tokens = {a: [a * 1_000_000 + i for i in range(102400)] for a in range(2)}
    h2pos = {}
    for a in range(2):
        r = P.make_ced_request(f"map-{a}", agent_tokens[a])
        for i, h in enumerate(r.block_hashes):
            h2pos[h] = (a, (i + 1) * P.HASH)

    stray = [0]
    in_evict = [False]
    orig_gnb = pool.get_new_blocks

    def gnb(n):
        if not in_evict[0]:
            node = pool.free_block_queue.fake_free_list_head.next_free_block
            for _ in range(n):
                if node.block_hash is not None:
                    stray[0] += 1
                node = node.next_free_block
        return orig_gnb(n)

    pool.get_new_blocks = gnb

    last_req = {0: None, 1: None}
    prev = {0: 0, 1: 0}
    total_hit = 0
    total_reusable = 0
    evictions = []
    turns = []
    for turn_idx, (plen, dlen, step) in enumerate(SCHED):
        if turn_idx in EVICT_TURNS:
            fl = pool.free_block_queue.get_all_free_blocks()
            ent = []
            hash_count = {}
            for b in fl:
                if b.block_hash is None:
                    continue
                hash_count[b.block_hash] = hash_count.get(b.block_hash, 0) + 1
            live = {a: live_ids(manager, agent_tokens[a][:plen], f"live{a}")
                    for a in range(2)}
            for b in fl:
                if b.block_hash is None:
                    continue
                a, end = h2pos[get_block_hash(b.block_hash)]
                gid = get_group_id(b.block_hash)
                keep = pool.cached_block_hash_to_block.get_one_block(b.block_hash)
                ent.append({
                    "agent": a, "end": end, "gid": gid, "sparse": gid == SWA_GID,
                    "bid": b.block_id, "blk": b,
                    "run": b.block_id in live[a],
                    "dup": hash_count[b.block_hash] > 1
                    and keep is not None and keep.block_id != b.block_id,
                })
            # Per conversation: redundant copies (free to evict; the probe's LRU
            # scenario never has any), then (end desc, MLA before SWA).
            mono = {a: [e for e in ent if e["agent"] == a and e["dup"]]
                    + sorted([e for e in ent if e["agent"] == a and not e["dup"]],
                             key=mono_key)
                    for a in range(2)}
            info = {
                "turn": turn_idx + 1,
                "cached_free": len(ent),
                "aged_by_agent": {a: sum(1 for e in ent if e["agent"] == a and not e["run"])
                                  for a in range(2)},
                "aged_non_swa": sum(1 for e in ent if not e["run"] and not e["sparse"]),
                "redundant_copies": sum(1 for e in ent if e["dup"]),
                # Premise of the bound: cached non-redundant blocks per
                # (conversation, 256-token position).
                "max_blocks_per_position": max(
                    __import__("collections").Counter(
                        (e["agent"], -(-e["end"] // P.PAGE)) for e in ent if not e["dup"]
                    ).values()
                ),
            }
            chosen = None
            if variant == "floor1":
                aged = sorted([e for e in ent if not e["run"]],
                              key=lambda e: (-e["end"], e["sparse"], e["agent"], e["gid"]))
                chosen = aged[:evict_n]
                if len(chosen) < evict_n:
                    rest = sorted([e for e in ent if e["run"]], key=mono_key)
                    chosen += rest[: evict_n - len(chosen)]
            elif variant == "mono_lru":
                order = mono[0] + mono[1]
                chosen = order[:evict_n]
            elif variant == "oracle0":
                losses = {}
                for a in range(2):
                    prompt = agent_tokens[a][:plen]
                    base = hit_without(manager, [], prompt, f"o{a}")
                    losses[a] = [
                        base - hit_without(manager, [e["blk"] for e in mono[a][:k]],
                                           prompt, f"o{a}")
                        for k in range(evict_n + 1)
                    ]
                best = min(range(evict_n + 1),
                           key=lambda k0: (losses[0][k0] + losses[1][evict_n - k0], k0))
                chosen = mono[0][:best] + mono[1][: evict_n - best]
                info["oracle_k0"] = best
                info["oracle_loss_by_k0"] = [losses[0][k] + losses[1][evict_n - k]
                                             for k in range(evict_n + 1)]
                info["loss_agent0_by_k"] = losses[0]
            if chosen is not None:
                q = pool.free_block_queue
                for e in chosen:
                    q.remove(e["blk"])
                q.prepend_n([e["blk"] for e in chosen])
            # The probe's own eviction step.
            fl = pool.free_block_queue.get_all_free_blocks()
            blocks_to_claim = 0
            cached_seen = 0
            evicted = []
            for b in fl:
                blocks_to_claim += 1
                if b.block_hash is not None:
                    cached_seen += 1
                    a, end = h2pos[get_block_hash(b.block_hash)]
                    evicted.append(f"{GROUPS[get_group_id(b.block_hash)]}:a{a}@{end}")
                    if cached_seen == evict_n:
                        break
            in_evict[0] = True
            claimed = pool.get_new_blocks(blocks_to_claim)
            in_evict[0] = False
            for b in claimed:
                b.ref_cnt -= 1
            pool.free_block_queue.prepend_n(claimed)
            info["evicted"] = evicted
            info["evicted_live_run_blocks"] = sum(
                1 for e in ent
                if e["run"] and e["bid"] in {b.block_id for b in claimed}
            )
            evictions.append(info)
        for a in range(2):
            seq = agent_tokens[a]
            name = f"agent-{a}-turn-{turn_idx}"
            req = P.make_ced_request(name, seq[:plen])
            hit = P.run_prefill_and_decode(manager, req, seq[plen:plen + dlen],
                                           tokens_per_step=step, prefill_chunk_size=8192)
            last_req[a] = name
            if prev[a]:
                total_hit += hit
                total_reusable += prev[a]
                turns.append({"turn": turn_idx + 1, "agent": a, "hit": hit,
                              "aligned": prev[a] // P.PAGE * P.PAGE,
                              "after_eviction": turn_idx in EVICT_TURNS})
            prev[a] = plen + dlen
    ceiling = sum(t["aligned"] for t in turns)
    return {
        "retention": retention,
        "variant": variant,
        "evict_n": evict_n,
        "num_blocks": num_blocks,
        "total_hit": total_hit,
        "total_reusable": total_reusable,
        "overall": round(total_hit / total_reusable, 6),
        "ceiling": ceiling,
        "loss_vs_ceiling": ceiling - total_hit,
        "lost_positions": (ceiling - total_hit) // P.PAGE,
        "evicted_turn_hits": [(t["turn"], t["agent"], t["hit"], t["aligned"])
                              for t in turns if t["after_eviction"]],
        "stray_cached_evictions_by_normal_allocation": stray[0],
        "evictions": evictions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--set", default="main", choices=["main", "pre0763"])
    args = ap.parse_args()
    t0 = time.time()
    payload = {"boot": args.boot, "host": socket.gethostname(),
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    payload["set"] = args.set
    payload["lru"] = lru_part()
    st = {}
    if args.set == "pre0763":
        runs = [
            ("tp3-triangle", 512, "actual", 18, 2200), ("tp4-ring", 8192, "actual", 18, 2200),
            ("tp3-triangle", 512, "actual", 36, 2200), ("tp4-ring", 8192, "actual", 36, 2200),
        ]
    else:
        runs = [
        ("tp3-triangle", 512, "actual", 18, 2200), ("tp3-triangle", 512, "floor1", 18, 2200),
        ("tp3-triangle", 512, "oracle0", 18, 2200), ("tp3-triangle", 512, "mono_lru", 18, 2200),
        ("tp4-ring", 8192, "actual", 18, 2200), ("tp4-ring", 8192, "floor1", 18, 2200),
        ("tp4-ring", 8192, "oracle0", 18, 2200), ("tp4-ring", 8192, "mono_lru", 18, 2200),
        ("dense-bound", None, "actual", 18, 3600), ("dense-bound", None, "oracle0", 18, 3600),
        ("tp3-triangle", 512, "actual", 36, 2200), ("tp3-triangle", 512, "oracle0", 36, 2200),
        ("tp4-ring", 8192, "actual", 36, 2200), ("tp4-ring", 8192, "oracle0", 36, 2200),
        ("tp4-ring", 8192, "floor1", 36, 2200),
    ]
    for prof, ret, var, n, nb in runs:
        key = f"{prof}/{var}/evict{n}"
        r = stress(ret, var, n, nb)
        if prof in REF_1107 and n == 18:
            r["minus_1107_tokens"] = r["total_hit"] - REF_1107[prof]
            r["meets_1107"] = r["total_hit"] >= REF_1107[prof]
        st[key] = r
        mbp = max((e.get("max_blocks_per_position", 0) for e in r["evictions"]), default=0)
        print(f"{key}: hit={r['total_hit']} overall={r['overall']:.6%} max_blocks_per_pos={mbp} "
              f"lost_positions={r['lost_positions']} "
              f"loss_vs_ceiling={r['loss_vs_ceiling']} stray={r['stray_cached_evictions_by_normal_allocation']} "
              f"evicted_turns={r['evicted_turn_hits']}", flush=True)
    payload["stress"] = st
    payload["elapsed_s"] = round(time.time() - t0, 1)
    for ret, res in payload["lru"].items():
        print(f"lru retention={ret}: inversions={res['inversions']} aged={res['aged']} "
              f"aged_all_swa={res['aged_all_swa']} aged_ahead={res['aged_contiguous_ahead_of_run']} "
              f"within_run={res['actual_inversions_within_run_only']} "
              f"per_group={res['actual_inversions_per_group']} "
              f"aged_on_hit_path={res['aged_on_hit_path']}")
        for oname, c in res["curves"].items():
            print(f"   {oname}: {json.dumps(c)}")
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=1)
    print("wrote", args.output, "elapsed", payload["elapsed_s"])


if __name__ == "__main__":
    main()
