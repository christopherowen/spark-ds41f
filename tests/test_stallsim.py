"""The concurrency-stall simulator (experiments/2026-10-26-concurrency-stalls) keeps vLLM's
scheduling semantics: aligned chunks, FCFS admission without partial chunks, decode rows in
every step, late admission, the held first token, retention checkpoints and LRU eviction."""

from __future__ import annotations

import dataclasses
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/2026-10-26-concurrency-stalls"))

import stallsim  # noqa: E402
from stallsim import Engine, KVCache, Lineage, Policy, Shape  # noqa: E402


def toy(**changes) -> Shape:
    """A small shape: budget 1,024, two sequences (threshold 1,012), costs that are easy to read."""
    shape = Shape(name="toy", budget=1024, max_seqs=2, pool_blocks=10_000, retention_interval=512,
                  fill_blocks=0, dec_knots=[[1, 0.010], [2, 0.012]], F=0.050, m0=50e-6, m1=0.0,
                  short_e=0.002, short_s=0.0005, alpha=1.0, fe0=0.0, fe1=0.0, engram_first_s=0.0,
                  host_s=0.0, accept_knots=[[1, 2.0], [2, 2.0]])
    return dataclasses.replace(shape, **changes)


def chunk_steps(engine: Engine, request) -> list:
    """(start, chunk, x0) of every step that carried one of the request's chunks."""
    return [(e[3], e[2]) for e in engine.trace if e[0] is request and e[1] == "prefill"]


class TracingEngine(Engine):
    """Records every scheduled entry and every step's total tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []
        self.plans = []
        self.totals = []
        self.decode_counts = []

    def schedule(self):
        plan = super().schedule()
        if plan is not None:
            self.plans.append(plan["entries"])
            self.trace.extend(plan["entries"])
            self.totals.append(sum(e[2] for e in plan["entries"]))
            self.decode_counts.append((plan["n_dec"], plan["prefill"]))
        return plan


class AlignedChunkTest(unittest.TestCase):
    def test_matches_vllm(self):
        engine = Engine(toy())
        threshold = engine.threshold
        self.assertEqual(threshold, 1024 - 2 * 6)
        # vLLM: min(remaining, threshold - computed % threshold)
        for computed, remaining in ((0, 5000), (300, 5000), (1012, 10), (2023, 4000), (0, 100)):
            expected = min(remaining, threshold - computed % threshold)
            self.assertEqual(engine.aligned_chunk(computed, remaining, threshold), expected)

    def test_policy_grid(self):
        engine = Engine(toy(), Policy(chunk_tiers=[(0, 1024), (1, 256)], grid=256))
        self.assertEqual(engine.aligned_chunk(0, 5000, 1024), 1024)
        self.assertEqual(engine.aligned_chunk(100, 5000, 256), 156)
        self.assertEqual(engine.aligned_chunk(100, 5000, 1024), 924)


class SchedulingTest(unittest.TestCase):
    def test_short_request_waits_for_the_long_prompt(self):
        """Production: a long prompt's whole aligned chunk leaves no room, so a short request
        that arrives during it waits for the prompt's last (partial) chunk."""
        engine = TracingEngine(toy(), seed=1)
        long = engine.submit("doc", 0.0, 10 * 1012 + 300, 1, Lineage())
        short = engine.submit("chat", 0.2, 40, 8, Lineage())
        engine.run(100.0)
        long_steps = [x0 for x0, _ in chunk_steps(engine, long)]
        short_steps = [x0 for x0, _ in chunk_steps(engine, short)]
        self.assertEqual(len(short_steps), 1)
        last_long = [e for e in engine.trace if e[0] is long][-1]
        self.assertEqual(last_long[2], 300)  # the partial last chunk
        # the short prompt is admitted with the last chunk, not before
        index_short = next(i for i, e in enumerate(engine.trace) if e[0] is short)
        index_last = engine.trace.index(last_long)
        self.assertGreater(index_short, index_last - 1)
        self.assertGreater(short.first, long.first - 0.5)
        for x0 in long_steps:
            self.assertEqual(x0 % engine.threshold, 0)

    def test_decode_rows_ride_in_every_chunk_step(self):
        engine = TracingEngine(toy(), seed=2)
        stream = engine.submit("stream", 0.0, 20, 10_000, Lineage())
        engine.submit("doc", 0.5, 8 * 1012, 1, Lineage())
        engine.run(60.0)
        chunk_steps_ = [(n, p) for n, p in engine.decode_counts if p >= 1000]
        self.assertGreaterEqual(len(chunk_steps_), 7)
        self.assertTrue(all(n == 1 for n, _ in chunk_steps_))
        self.assertTrue(all(total <= engine.budget for total in engine.totals))
        self.assertIsNotNone(stream.first)

    def test_late_admission_joins_the_step_after_next(self):
        shape = toy(dec_knots=[[1, 0.100], [2, 0.100]], decision_frac=0.7)
        engine = TracingEngine(shape, seed=3)
        engine.submit("stream", 0.0, 20, 10_000, Lineage())
        engine.run(1.0)
        step = engine.steps[-1]
        # arrives after the step in flight passed its decision point
        late = engine.submit("chat", step.start + 0.85 * step.duration, 20, 4, Lineage())
        engine.run(5.0)
        admitted_step = next(s for s in engine.steps if s.start >= late.admitted - 1e-9)
        self.assertGreaterEqual(admitted_step.start, step.start + step.duration * 2 - 1e-6)

    def test_held_first_token(self):
        held = Engine(toy(), seed=4)
        a = held.submit("chat", 0.0, 40, 8, Lineage())
        held.run(5.0)
        free = Engine(toy(), Policy(sink_hold=False), seed=4)
        b = free.submit("chat", 0.0, 40, 8, Lineage())
        free.run(5.0)
        self.assertAlmostEqual(b.first, free.steps[0].duration, places=9)
        self.assertAlmostEqual(a.first, held.steps[0].duration + held.steps[1].duration, places=9)


class CacheTest(unittest.TestCase):
    def test_append_resumes_at_the_prompt_end_checkpoint(self):
        engine = Engine(toy(retention_interval=4096), seed=5)
        lineage = Lineage()
        first = engine.submit("agent", 0.0, 3000, 50, lineage)
        engine.run(50.0)
        second = engine.submit("agent", 60.0, 3050 + 900, 10, lineage)
        engine.run(200.0)
        self.assertEqual(first.hit_tokens, 0)
        self.assertEqual(second.hit_tokens, 3000)  # the previous prompt's end
        self.assertEqual(second.ideal_hit, 3000)

    def test_hit_leaves_replay_rows(self):
        cache = KVCache(10_000, 7, "lru")
        lineage = Lineage()
        for j in range(8):
            cache.acquire(("b", lineage.id, j))
        cache.add_checkpoint(lineage.id, 1024)
        cache.add_checkpoint(lineage.id, 2000)
        hit, _ = cache.hit(lineage, 2048, 128)
        self.assertEqual(hit, 1024)  # 2000 > 2048 - 128
        hit, _ = cache.hit(lineage, 2200, 128)
        self.assertEqual(hit, 2000)

    def test_lru_evicts_checkpoints_first_and_prefix_keeps_them(self):
        """A session's prompt-end checkpoint is made during its prefill, so production's LRU
        queues it ahead of everything freed while the request decodes; under pressure the next
        turn loses the whole prefix. "prefix" queues it with the request's blocks."""
        for eviction, survives in (("lru", False), ("prefix", True)):
            engine = Engine(toy(retention_interval=100_000, max_seqs=4), Policy(eviction=eviction), seed=6)
            session = Lineage()
            engine.submit("agent", 0.0, 2000, 400, session)   # decodes for a while
            engine.submit("chat", 0.5, 2000, 4, Lineage())    # finishes first
            engine.run(100.0)
            # pressure: evict one checkpoint's worth of the oldest cached entries
            engine.cache.make_room(engine.cache.free() + 7)
            hit, _ = engine.cache.hit(session, 2400 + 300, 128)
            self.assertEqual(hit, 2000 if survives else 0, eviction)

    def test_divergent_branch_shares_only_its_prefix(self):
        cache = KVCache(10_000, 7, "lru")
        parent = Lineage()
        for j in range(10):
            cache.acquire(("b", parent.owner_of_block(j), j))
        cache.add_checkpoint(parent.id, 1024)
        cache.add_checkpoint(parent.id, 2304)
        child = Lineage(parent, 1300)
        hit, _ = cache.hit(child, 3000, 128)
        self.assertEqual(hit, 1024)


class PreemptionTest(unittest.TestCase):
    def test_lifo_preemption_recomputes_and_finishes(self):
        shape = toy(pool_blocks=120, max_seqs=2, budget=1024)
        engine = Engine(shape, seed=7)
        a = engine.submit("chat", 0.0, 400, 12_000, Lineage())
        b = engine.submit("chat", 0.0, 400, 12_000, Lineage())
        engine.run(10_000.0)
        self.assertIsNotNone(a.finished_at)
        self.assertIsNotNone(b.finished_at)
        self.assertGreater(engine.preemptions, 0)
        self.assertEqual(b.preemptions, engine.preemptions)


class PolicyTest(unittest.TestCase):
    def long_and_probes(self, policy: Policy):
        engine = TracingEngine(toy(max_seqs=6), policy, seed=8)
        engine.submit("stream", 0.0, 20, 50_000, Lineage())
        long = engine.submit("doc", 1.0, 20 * engine.threshold + 77, 1, Lineage())
        probes = [engine.submit("probe", 1.0 + t, 120, 6, Lineage()) for t in (0.5, 1.5, 2.5)]  # more than the spare rows
        engine.run(200.0)
        return engine, long, probes

    def test_bypass_keeps_the_long_prompts_boundaries(self):
        base, long0, probes0 = self.long_and_probes(Policy())
        for policy in (Policy(order="bypass"), Policy(order="bypass", first_token_priority=True),
                       Policy(order="spf")):
            engine, long, probes = self.long_and_probes(policy)
            ends = [x0 + c for x0, c in chunk_steps(engine, long)]
            self.assertTrue(all(end % engine.threshold == 0 or end == long.target for end in ends), policy)
            self.assertEqual(ends, [x0 + c for x0, c in chunk_steps(base, long0)], policy)
            for probe, probe0 in zip(probes, probes0):
                if probe0.sent < long0.first - 0.3:  # sent while the long prompt prefills
                    self.assertLess(probe.first - probe.sent, 0.5 * (probe0.first - probe0.sent), policy)
                else:
                    self.assertLessEqual(probe.first - probe.sent, probe0.first - probe0.sent + 1e-9, policy)

    def test_two_lanes_livelock_on_aligned_chunks(self):
        """vLLM's two prefill lanes give each long prompt half the budget; its whole aligned
        chunk never fits, and neither lane is released. Releasing the lane fixes it."""
        for release in (False, True):
            engine = Engine(toy(max_seqs=4), Policy(order="rr2", lanes_whole_chunks=release), seed=11)
            engine.livelock_s = 5.0
            engine.submit("stream", 0.0, 20, 4000, Lineage())
            first = engine.submit("doc", 0.5, 6 * engine.threshold, 1, Lineage())
            second = engine.submit("doc", 0.6, 6 * engine.threshold, 1, Lineage())
            engine.run(200.0)
            done = first.first is not None and second.first is not None
            self.assertEqual(done, release, release)
            self.assertEqual(engine.livelocked, not release, release)

    def test_fixed_lanes_rotate_whole_chunks_as_vllm_does(self):
        """vLLM's tests/v1/core/test_aligned_prefill_lanes.py, on the toy shape (threshold 1,000)."""
        engine = TracingEngine(toy(max_seqs=4), Policy(order="rr2", lanes_whole_chunks=True), seed=12)
        for name, tokens in (("a", 3000), ("b", 2500), ("c", 300)):
            engine.submit(name, 0.0, tokens, 1, Lineage())
        engine.run(100.0)
        steps = [{e[0].cls: e[2] for e in entries if e[1] == "prefill"} for entries in engine.plans]
        self.assertEqual(steps[:6], [{"a": 1000}, {"b": 1000}, {"a": 1000}, {"b": 1000}, {"a": 1000},
                                     {"b": 500, "c": 300}])

    def test_tiers_use_the_small_chunk_beside_decodes(self):
        policy = Policy(chunk_tiers=[(0, 1024), (1, 256)], grid=256)
        engine = TracingEngine(toy(), policy, seed=9)
        long = engine.submit("doc", 0.0, 4096, 1, Lineage())
        engine.run(100.0)
        self.assertEqual({c for _, c in chunk_steps(engine, long)}, {1024})
        engine = TracingEngine(toy(), policy, seed=9)
        engine.submit("stream", 0.0, 20, 50_000, Lineage())
        long = engine.submit("doc", 1.0, 4096, 1, Lineage())
        engine.run(100.0)
        self.assertEqual({c for _, c in chunk_steps(engine, long)}, {256})

    def test_summary(self):
        engine = Engine(toy(), seed=10)
        stallsim_rng = random.Random(1)
        for i in range(20):
            engine.submit("chat", stallsim_rng.uniform(0, 5), 30, 20, Lineage())
        engine.run(100.0)
        summary = stallsim.summarize(engine, 0.0, 5.0)
        self.assertEqual(summary["all"]["requests"], 20)
        self.assertEqual(summary["all"]["unfinished"], 0)
        self.assertGreater(summary["output_tok_s"], 0)


class ArmsTest(unittest.TestCase):
    def test_arms_follow_the_recipe(self):
        """The validation arms are the recipe's: regenerate them (make_arms.py) when it moves."""
        import json

        import make_arms

        here = ROOT / "experiments/2026-10-26-concurrency-stalls"
        for name, arm in make_arms.arms().items():
            self.assertEqual(json.loads((here / name).read_text()), arm, name)

    def test_slots_keep_the_chunk_boundaries(self):
        import make_arms

        arms = make_arms.arms()
        for name, threshold in (("slots-24-tp4.json", 16288), ("slots-12-tp3.json", 8144)):
            options = arms[name]["serve_options"]
            rows = int(options["--max-num-seqs"]) * 6
            self.assertEqual(int(options["--max-num-batched-tokens"]) - rows, threshold)
            self.assertEqual(int(options["--max-cudagraph-capture-size"]), rows)


if __name__ == "__main__":
    unittest.main()
