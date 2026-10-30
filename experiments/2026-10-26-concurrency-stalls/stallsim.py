"""A discrete-event model of vLLM's step building on spark-ds41f (DeepSeek V4.1 Flash).

It follows the next-candidate tree's `Scheduler.schedule()` (vllm/v1/core/sched/scheduler.py)
for what production enables:
- the running pass in admission order, then the waiting pass in FCFS order, both against one
  token budget (`--max-num-batched-tokens`);
- every decoding request schedules 1 + 5 draft rows;
- DS4.1 forces aligned prefill chunks (`align_prefill_chunks`, model_executor/models/config.py):
  a chunk ends at the next absolute multiple of `--long-prefill-token-threshold`
  (budget - max_num_seqs x 6) or at the prompt end, and a chunk that does not fit whole waits
  ("no partial chunk"). The waiting pass stops at the first request that does not fit;
- `--max-num-seqs` slots, admission against free KV blocks for the first chunk only
  (`--no-scheduler-reserve-full-isl`), LIFO preemption in the running pass, none in the
  waiting pass, and no admission in a step that preempted;
- the prefix cache: 256-token blocks; freed blocks join an LRU queue tail first; sliding-window
  retention checkpoints of 7 slabs at every multiple of `--prefix-cache-retention-interval` and
  at the prompt end, which join the queue when they are made (during prefill); a hit resumes
  only at a cached checkpoint whose blocks are cached, at most at the prompt length less 128
  replay rows;
- async scheduling with late admission (`VLLM_LATE_ADMISSION_LAYER=30`): step N+1 is decided
  while step N runs, when N reaches layer 30; a request that arrives later joins N+2, and a
  request that finishes in N still has a row in N+1;
- sink rejection: each step's last sampled token is held and verified as draft zero of the
  next step, so a request's first token reaches the user at the end of its first decode step.

Step costs come from `Shape` (calibrated in calibrate.py on the release and experiment
records). `Policy` holds the fixes evaluated in evaluate.py; its defaults are production's.

Standard library only.
"""

from __future__ import annotations

import bisect
import collections
import dataclasses
import heapq
import math
import random

BLOCK = 256


# ---------------------------------------------------------------------------------------------
# Shapes and policies


@dataclasses.dataclass
class Shape:
    """One serving shape: sizing from the recipe, costs from calibrate.py."""

    name: str
    budget: int                  # --max-num-batched-tokens
    max_seqs: int                # --max-num-seqs
    pool_blocks: int             # KV pool in blocks of 256 tokens (scripts/kv_capacity.py)
    retention_interval: int      # --prefix-cache-retention-interval
    fill_blocks: int             # blocks a prompt in flight adds at this budget (#586)
    dec_knots: list              # [(decoding requests, plain decode step s)]
    F: float                     # a full prefill chunk's fixed cost (s)
    m0: float                    # s per prompt token at no context
    m1: float                    # s per prompt token per context token
    short_e: float               # a short prompt's eager-step extra (s)
    short_s: float               # s per short-prompt token
    alpha: float                 # share of a decode step that decode rows add to a chunk step
    dec_extrapolation: float = 1.0  # slope past the last knot, as a share of the last slope
    decision_frac: float = 0.70  # late admission: N+1 is decided this far into step N
    host_s: float = 0.005        # an idle engine: arrival to step start
    fe0: float = 0.025           # frontend and delivery, fixed (s)
    fe1: float = 1.0e-6          # frontend, per prompt token (s)
    engram_first_s: float = 0.030  # first-chunk Engram wait, chunks of 1,024 tokens or more
    # Tokens a decoding request gains a step, by decoding requests in the step: the release
    # suites' tokens a step over concurrency (adaptive verification verifies fewer drafts as
    # the batch grows). Drafts are accepted geometrically with the matching rate.
    accept_knots: list = dataclasses.field(default_factory=lambda: [[1, 2.28], [8, 2.09], [16, 1.93]])
    drafts: int = 5
    window_blocks: int = 46      # a request's sliding-window and ring blocks (#586)
    checkpoint_slabs: int = 7    # a retention checkpoint (#574)
    replay_tokens: int = 128     # CED and drafter replay rows on a hit (2026-10-19-prefix-replay)
    step_overhead_s: float = 0.0  # inside every measured step already; a fix can cut it

    @property
    def rows(self) -> int:
        return 1 + self.drafts

    @property
    def threshold(self) -> int:
        return self.budget - self.max_seqs * self.rows

    def t_dec(self, n: int) -> float:
        """A plain decode step with n decoding requests."""
        if n <= 0:
            return 0.0
        knots = self.dec_knots
        if n <= knots[0][0]:
            return knots[0][1]
        for (n0, t0), (n1, t1) in zip(knots, knots[1:]):
            if n <= n1:
                return t0 + (t1 - t0) * (n - n0) / (n1 - n0)
        (n0, t0), (n1, t1) = knots[-2], knots[-1]
        return t1 + (n - n1) * (t1 - t0) / (n1 - n0) * self.dec_extrapolation

    def tokens_per_step(self, n: int) -> float:
        knots = self.accept_knots
        if n <= knots[0][0]:
            return knots[0][1]
        for (n0, t0), (n1, t1) in zip(knots, knots[1:]):
            if n <= n1:
                return t0 + (t1 - t0) * (n - n0) / (n1 - n0)
        return knots[-1][1]

    def accept_rate(self, n: int) -> float:
        """The geometric acceptance rate whose mean is tokens_per_step(n) - 1 accepted drafts."""
        mean = self.tokens_per_step(n) - 1.0
        lo, hi = 0.0, 0.999
        for _ in range(40):
            q = (lo + hi) / 2
            if sum(q ** k for k in range(1, self.drafts + 1)) < mean:
                lo = q
            else:
                hi = q
        return (lo + hi) / 2

    def step_cost(self, n_dec: int, chunks: list, *, short_e: float | None = None,
                  engram_s: float = 0.0, overhead_cut: float = 0.0) -> float:
        """A step's time: n_dec decoding requests' rows and prefill chunks [(tokens, start)]."""
        prompt = sum(c for c, _ in chunks)
        if not prompt:
            return max(self.t_dec(n_dec) - overhead_cut, 1e-4)
        e = self.short_e if short_e is None else short_e
        context = sum(c * (x0 + c / 2) for c, x0 in chunks)
        short = self.t_dec(max(n_dec, 1)) + e + self.short_s * prompt
        full = self.alpha * self.t_dec(n_dec) + self.F + self.m0 * prompt
        return max(min(short, full) + self.m1 * context + engram_s - overhead_cut, 1e-4)


@dataclasses.dataclass
class Policy:
    """A scheduling and cache configuration. The defaults are production's (spark-ds41f-1025)."""

    name: str = "production"
    # Prefill chunks. None: production (one threshold, chunks aligned at its multiples).
    # Otherwise [(decoding requests at least, threshold)]; chunks end on multiples of `grid`.
    chunk_tiers: list | None = None
    grid: int = 0
    budget_alone: int | None = None    # the step budget when no decode is runnable (memory)
    # Prefill order: "fcfs" (production), "bypass" (prefills that finish within one chunk go
    # first), "spf" (shortest remaining prefill first), "rr2" (vLLM's two round-robin lanes).
    order: str = "fcfs"
    bypass_tokens: int | None = None   # "bypass": a prefill this short is short (default: a chunk)
    first_token_priority: bool = False  # long chunks also wait while a first token is held
    long_wait_cap_s: float | None = None  # a long prefill held back this long goes first
    lanes_whole_chunks: bool = False   # "rr2": vLLM's fix (prefill-lanes-aligned): no share caps an aligned chunk
    alpha_scale: float = 1.0           # decode rows in a prefill step cost this share of today's
    dec_extrapolation: float | None = None  # sensitivity: the decode step's slope past the last knot
    max_seqs: int | None = None
    pool_scale: float = 1.0
    extra_pool_blocks: int = 0
    retention_interval: int | None = None
    eviction: str = "lru"              # "lru" (production) or "prefix" (checkpoints with their blocks)
    engram_prefetch: bool = False
    sink_hold: bool = True
    overhead_cut_s: float = 0.0        # host or collective time removed from every step
    lookup_window: int = 32            # waiting requests a non-FCFS order looks at each step
    interleave_decode_steps: int = 0   # pure graphed decode steps interleaved between long chunks


# ---------------------------------------------------------------------------------------------
# Content identity: lineages


class Lineage:
    """A token sequence that shares positions [0, branch) with its parent."""

    __slots__ = ("id", "parent", "branch")
    _next = 0

    def __init__(self, parent: "Lineage | None" = None, branch: int = 0):
        Lineage._next += 1
        self.id = Lineage._next
        self.parent = parent
        self.branch = branch if parent is not None else 0

    def owner_of_block(self, j: int) -> int:
        node, end = self, (j + 1) * BLOCK
        while node.parent is not None and end <= node.branch:
            node = node.parent
        return node.id

    def owner_of_position(self, c: int) -> int:
        node = self
        while node.parent is not None and c <= node.branch:
            node = node.parent
        return node.id

    def chain(self):
        """(node, highest position it owns for this lineage or None)."""
        node, upper = self, None
        while node is not None:
            yield node, upper
            upper = node.branch if node.parent is not None else None
            node = node.parent


# ---------------------------------------------------------------------------------------------
# KV pool and prefix cache


class KVCache:
    """Blocks held by running requests (live, refcounted), freed blocks still cached (LRU,
    evictable), private blocks (partial tails, window and fill blocks) and checkpoints."""

    def __init__(self, pool_blocks: int, checkpoint_slabs: int, eviction: str):
        self.pool = pool_blocks
        self.slabs = checkpoint_slabs
        self.eviction = eviction
        self.live: dict = {}
        self.lru: collections.OrderedDict = collections.OrderedDict()
        self.evictable = 0
        self.private = 0
        # "prefix": checkpoints of prefixes a running request holds; evicted only after the LRU
        self.soft: collections.OrderedDict = collections.OrderedDict()
        self.soft_size = 0
        # A finished request's partial tail block stays cached: vLLM hashes DS4.1's groups in
        # 64-token units and copies a partial hit on write, so an append resumes inside it.
        self.partial_fill: dict = {}
        self.ideal_fill: dict = {}
        self.ideal: set = set()
        self.positions: dict = collections.defaultdict(list)  # owner -> checkpoint positions made
        self.evicted_blocks = 0
        self.evicted_checkpoints = 0
        self.version = 0

    def free(self) -> int:
        return self.pool - len(self.live) - self.private - self.evictable - self.soft_size

    def contains(self, key) -> bool:
        return key in self.live or key in self.lru or key in self.soft

    def make_room(self, n: int) -> bool:
        while self.free() < n and (self.lru or self.soft):
            source = self.lru if self.lru else self.soft
            key, size = source.popitem(last=False)
            if source is self.lru:
                self.evictable -= size
            else:
                self.soft_size -= size
            if key[0] == "c":
                self.evicted_checkpoints += 1
            else:
                self.evicted_blocks += 1
                self.partial_fill.pop(key, None)
            self.version += 1
        return self.free() >= n

    def keep_partial(self, key, fill: int) -> None:
        """A finished request's partial tail block joins the LRU (it was a private block)."""
        if fill <= 0:
            return
        if key in self.lru:
            self.evictable -= self.lru.pop(key)
        self.partial_fill[key] = fill
        self.ideal_fill[key] = max(fill, self.ideal_fill.get(key, 0))
        self.lru[key] = 1
        self.evictable += 1

    def acquire(self, key) -> None:
        if key in self.live:
            self.live[key] += 1
            return
        if key in self.lru:
            self.evictable -= self.lru.pop(key)
        self.live[key] = 1
        self.ideal.add(key)

    def release(self, key) -> None:
        count = self.live[key] - 1
        if count:
            self.live[key] = count
            return
        del self.live[key]
        self.lru[key] = 1
        self.evictable += 1

    def add_checkpoint(self, owner: int, position: int) -> tuple:
        key = ("c", owner, position)
        positions = self.positions[owner]
        index = bisect.bisect_left(positions, position)
        if index == len(positions) or positions[index] != position:
            positions.insert(index, position)
        self.ideal.add(key)
        if key in self.lru:
            self.evictable -= self.lru.pop(key)
        elif key in self.soft:
            self.soft_size -= self.soft.pop(key)
        if self.eviction == "prefix":
            self.soft[key] = self.slabs
            self.soft_size += self.slabs
        else:
            self.lru[key] = self.slabs
            self.evictable += self.slabs
        return key

    def hold_checkpoint(self, key) -> None:
        """A hit used this checkpoint: production touches it; "prefix" keeps it with the prefix."""
        if self.eviction == "prefix":
            if key in self.lru:
                self.evictable -= self.lru.pop(key)
                self.soft[key] = self.slabs
                self.soft_size += self.slabs
        elif key in self.lru:
            self.lru.move_to_end(key)

    def settle(self, key) -> None:
        if key in self.soft:
            self.soft_size -= self.soft.pop(key)
            self.lru[key] = self.slabs
            self.evictable += self.slabs

    def hit(self, lineage: Lineage, prompt: int, replay: int, ideal: bool = False) -> tuple[int, object]:
        """Tokens a new request skips: the deepest cached checkpoint whose blocks are cached."""
        present = self.ideal.__contains__ if ideal else self.contains
        fills = self.ideal_fill if ideal else self.partial_fill
        limit = prompt - replay
        if limit <= 0:
            return 0, None
        blocks, last = 0, (limit + BLOCK - 1) // BLOCK
        while blocks < last and present(("b", lineage.owner_of_block(blocks), blocks)):
            blocks += 1
        reach = blocks * BLOCK
        if blocks < last:
            reach += fills.get(("p", lineage.owner_of_block(blocks), blocks), 0)
        limit = min(limit, reach)
        best, best_key = 0, None
        for node, upper in lineage.chain():
            positions = self.positions.get(node.id)
            if not positions:
                continue
            low = node.branch if node.parent is not None else 0
            high = limit if upper is None else min(limit, upper)
            index = bisect.bisect_right(positions, high) - 1
            while index >= 0 and positions[index] > max(low, best):
                key = ("c", node.id, positions[index])
                if present(key):
                    best, best_key = positions[index], key
                    break
                index -= 1
        return best, best_key


# ---------------------------------------------------------------------------------------------
# Requests


@dataclasses.dataclass(eq=False)
class Request:
    rid: int
    cls: str
    sent: float
    prompt: int
    max_out: int
    lineage: Lineage
    on_finish: object = None
    meta: dict = dataclasses.field(default_factory=dict)
    arrival: float = 0.0
    status: str = "new"            # waiting, running, finished
    epoch: int = 0                 # bumped by preemption
    computed: int = 0
    target: int = 0                # tokens to compute before decoding
    out: int = 0                   # tokens sampled
    emitted: int = 0
    keyed: int = 0                 # leading blocks held as cache keys
    partial: int = 0               # 1 while a partial tail block is private
    fill: int = 0
    window: bool = False
    checkpoints: list = dataclasses.field(default_factory=list)
    hit_tokens: int = 0
    ideal_hit: int = 0
    recompute: int = 0             # prompt tokens computed that an unbounded cache would have held
    lookup_version: int = -1
    lookup_hit: int = 0
    lookup_key: object = None
    admitted: float | None = None
    first: float | None = None
    last: float | None = None
    gaps: list = dataclasses.field(default_factory=list)
    preemptions: int = 0
    finished_at: float | None = None
    deferred_since: float | None = None
    prefilled: float | None = None
    waits: dict = dataclasses.field(default_factory=dict)

    @property
    def prefilling(self) -> bool:
        return self.computed < self.target

    @property
    def length(self) -> int:
        return self.prompt + self.out


# ---------------------------------------------------------------------------------------------
# The engine


@dataclasses.dataclass
class StepRecord:
    start: float
    duration: float
    n_dec: int
    prefill_tokens: int
    recompute_tokens: int
    chunks: int
    new_prompts: int
    emitted: int = 0


class Engine:
    """Steps back to back; each step is decided while the step before it runs."""

    def __init__(self, shape: Shape, policy: Policy | None = None, seed: int = 0):
        self.policy = p = policy or Policy()
        if p.alpha_scale != 1.0:
            shape = dataclasses.replace(shape, alpha=shape.alpha * p.alpha_scale)
        if p.dec_extrapolation is not None:
            shape = dataclasses.replace(shape, dec_extrapolation=p.dec_extrapolation)
        self.shape = shape
        self.rng = random.Random(seed)
        self.max_seqs = p.max_seqs or shape.max_seqs
        self.threshold = shape.threshold
        self.budget = shape.budget + (self.max_seqs - shape.max_seqs) * shape.rows
        self.interval = p.retention_interval or shape.retention_interval
        pool = int(shape.pool_blocks * p.pool_scale) + p.extra_pool_blocks
        self.cache = KVCache(pool, shape.checkpoint_slabs, p.eviction)
        self.engram_s = 0.0 if p.engram_prefetch else shape.engram_first_s
        self.short_e = shape.short_e - (0.005 if p.engram_prefetch else 0.0)
        self.now = 0.0
        self.events: list = []
        self.seq = 0
        self.waiting: collections.deque = collections.deque()
        self.running: list = []
        self.requests: list = []
        self.steps: list = []
        self.in_flight = None
        self.decided = None
        self.idle_pending = False
        self.next_rid = 0
        self.preemptions = 0
        self.accept_cache: dict = {}
        # No prefill progress while one could run (a running prefill, or a waiting request
        # and a free slot) for this long, or work left with nothing scheduled: a livelock.
        self.last_decision = 0.0
        self.wait_reason = "other"
        self.livelock_s = 300.0
        self.no_progress_since = None
        self.livelocked = False
        self.interleave_remaining = 0

    # -- events ---------------------------------------------------------------------------

    def at(self, time: float, kind: str, payload) -> None:
        self.seq += 1
        heapq.heappush(self.events, (time, self.seq, kind, payload))

    def submit(self, cls: str, sent: float, prompt: int, max_out: int, lineage: Lineage,
               on_finish=None, meta: dict | None = None) -> Request:
        self.next_rid += 1
        request = Request(self.next_rid, cls, sent, prompt, max_out, lineage, on_finish, meta or {})
        request.target = prompt
        request.arrival = sent + self.shape.fe0 + self.shape.fe1 * prompt
        self.requests.append(request)
        self.at(request.arrival, "arrive", request)
        return request

    def call_at(self, time: float, fn) -> None:
        self.at(time, "call", fn)

    def run(self, until: float) -> None:
        while self.events and not self.livelocked:
            time, _, kind, payload = self.events[0]
            if time > until:
                break
            heapq.heappop(self.events)
            self.now = time
            if kind == "arrive":
                payload.status = "waiting"
                self.waiting.append(payload)
                if self.in_flight is None and not self.idle_pending:
                    self.idle_pending = True
                    self.at(time + self.shape.host_s, "idle", None)
            elif kind == "call":
                payload(self, time)
            elif kind == "idle":
                self.idle_pending = False
                if self.in_flight is None:
                    plan = self.schedule()
                    if plan is not None:
                        self.launch(plan, time)
            elif kind == "decide":
                self.decided = self.schedule()
            elif kind == "end":
                self.finish_step(time)

    def launch(self, plan: dict, start: float) -> None:
        record = StepRecord(start, plan["cost"], plan["n_dec"], plan["prefill"], plan["recompute"],
                            len(plan["chunks"]), plan["new"])
        self.steps.append(record)
        self.in_flight = (start, plan, record)
        self.at(start + self.shape.decision_frac * plan["cost"], "decide", None)
        self.at(start + plan["cost"], "end", None)

    def finish_step(self, time: float) -> None:
        _, plan, record = self.in_flight
        self.in_flight = None
        record.emitted = self.apply_outputs(plan, time)
        nxt, self.decided = self.decided, None
        if nxt is None and self.waiting:
            nxt = self.schedule()
        if nxt is not None:
            self.launch(nxt, time)

    # -- outputs ----------------------------------------------------------------------------

    def tokens_this_step(self, q: float) -> int:
        accepted = 0
        while accepted < self.shape.drafts and self.rng.random() < q:
            accepted += 1
        return 1 + accepted

    def accept_rate(self, n_dec: int) -> float:
        rate = self.accept_cache.get(n_dec)
        if rate is None:
            rate = self.accept_cache[n_dec] = self.shape.accept_rate(n_dec)
        return rate

    def apply_outputs(self, plan: dict, time: float) -> int:
        hold = self.policy.sink_hold
        emitted = 0
        q = self.accept_rate(plan["n_dec"])
        for request, kind, tokens, x0, epoch in plan["entries"]:
            if request.status != "running" or request.epoch != epoch:
                continue  # finished or preempted since the step was decided
            if kind == "prefill":
                if x0 + tokens >= request.target and request.out == 0:
                    request.out = 1  # the prompt's last row samples the first token
                    request.prefilled = time
                    if not hold or request.max_out == 1:
                        emitted += self.emit(request, 1, time)
                    if request.out >= request.max_out:
                        self.complete(request, time)
            else:
                new = min(self.tokens_this_step(q), request.max_out - request.out)
                if new <= 0:
                    continue
                request.out += new
                if request.out >= request.max_out:
                    emitted += self.emit(request, request.out - request.emitted, time)
                    self.complete(request, time)
                else:
                    emitted += self.emit(request, request.out - request.emitted - (1 if hold else 0), time)
        return emitted

    def emit(self, request: Request, count: int, time: float) -> int:
        if count <= 0:
            return 0
        if request.first is None:
            request.first = time
        else:
            request.gaps.append(time - request.last)
        request.last = time
        request.emitted += count
        return count

    def complete(self, request: Request, time: float) -> None:
        request.status = "finished"
        request.finished_at = time
        self.running.remove(request)
        self.free_request(request)
        if request.on_finish is not None:
            request.on_finish(self, request, time)

    # -- KV accounting ----------------------------------------------------------------------

    def new_blocks(self, request: Request, tokens: int) -> int:
        """Blocks the request must allocate to hold its first `tokens` tokens."""
        need = (tokens + BLOCK - 1) // BLOCK
        full = tokens // BLOCK
        count = 0
        for j in range(request.keyed, full):
            if not self.cache.contains(("b", request.lineage.owner_of_block(j), j)):
                count += 1
        if need > full and not request.partial:
            count += 1
        if request.partial and full > request.keyed:
            count -= 1  # the private tail becomes a keyed block
        return max(count, 0)

    def grow(self, request: Request, tokens: int) -> None:
        cache = self.cache
        full = tokens // BLOCK
        while request.keyed < full:
            if request.partial:
                request.partial = 0
                cache.private -= 1
            j = request.keyed
            cache.acquire(("b", request.lineage.owner_of_block(j), j))
            request.keyed += 1
        if tokens % BLOCK and not request.partial:
            request.partial = 1
            cache.private += 1

    def free_request(self, request: Request) -> None:
        cache = self.cache
        if request.partial:
            cache.private -= 1
            request.partial = 0
            if request.status == "finished":
                j = request.keyed
                written = request.prompt + request.out - 1 - j * BLOCK
                cache.keep_partial(("p", request.lineage.owner_of_block(j), j), min(written, BLOCK - 1))
        if request.window:
            cache.private -= self.shape.window_blocks
            request.window = False
        if request.fill:
            cache.private -= request.fill
            request.fill = 0
        anchored: dict = collections.defaultdict(list)
        if cache.eviction == "prefix":
            for key in request.checkpoints:
                anchored[min((key[2] - 1) // BLOCK, max(request.keyed - 1, 0))].append(key)
        for j in range(request.keyed - 1, -1, -1):  # tail first
            cache.release(("b", request.lineage.owner_of_block(j), j))
            for key in anchored.pop(j, ()):
                cache.settle(key)
        for keys in anchored.values():
            for key in keys:
                cache.settle(key)
        request.keyed = 0
        request.checkpoints = []

    def preempt(self, request: Request) -> None:
        self.running.remove(request)
        self.free_request(request)
        self.preemptions += 1
        request.preemptions += 1
        request.epoch += 1
        request.status = "waiting"
        request.computed = 0
        request.target = request.prompt + request.out
        request.lookup_version = -1
        self.waiting.appendleft(request)

    # -- scheduling -------------------------------------------------------------------------

    def chunk_threshold(self, n_dec: int) -> int:
        tiers = self.policy.chunk_tiers
        if not tiers:
            return self.threshold
        chosen = tiers[0][1]
        for at_least, threshold in tiers:
            if n_dec >= at_least:
                chosen = threshold
        return chosen

    def aligned_chunk(self, computed: int, remaining: int, threshold: int) -> int:
        """vLLM's `_aligned_prefill_chunk`, or the policy's grid."""
        grid = self.policy.grid if self.policy.chunk_tiers else threshold
        end = (computed + threshold) // grid * grid
        if end <= computed:
            end = (computed // grid + 1) * grid
        return min(end - computed, remaining)

    def lookup(self, request: Request) -> None:
        """The prefix-cache hit of a waiting request, redone only after evictions."""
        if request.lookup_version == self.cache.version:
            return
        hit, key = self.cache.hit(request.lineage, request.target, self.shape.replay_tokens)
        request.lookup_hit, request.lookup_key = hit, key
        request.lookup_version = self.cache.version
        request.computed = hit

    def remaining(self, request: Request) -> int:
        return request.target - request.computed

    def schedule(self) -> dict | None:
        shape, policy, cache = self.shape, self.policy, self.cache
        now = self.now
        # TTFT decomposition: the time since the last decision, for every request still
        # waiting, charged to the reason it waited then.
        elapsed = now - self.last_decision
        if elapsed > 0:
            for request in self.waiting:
                if request.arrival < now:
                    span = min(elapsed, now - request.arrival)
                    request.waits[self.wait_reason] = request.waits.get(self.wait_reason, 0.0) + span
        self.last_decision = now
        runnable = sum(1 for r in self.running if not r.prefilling)
        threshold = self.chunk_threshold(runnable)
        budget = self.budget
        if policy.budget_alone and runnable == 0:
            budget = max(budget, policy.budget_alone)
        entries: list = []
        chunks: list = []
        stats = {"n_dec": 0, "new": 0, "prefill": 0, "recompute": 0, "engram": False}
        scheduled: set = set()
        preempted = False
        in_flight = {e[0].rid for e in self.in_flight[1]["entries"]} if self.in_flight else set()

        def preempt_for(request: Request, need: int) -> bool:
            nonlocal preempted
            while not cache.make_room(need):
                victim = self.running[-1]
                if victim.rid in scheduled:
                    self.unschedule(victim, entries, chunks)
                self.preempt(victim)
                preempted = True
                if victim is request:
                    return False
            return True

        def take_prefill(request: Request, admit: bool, budget_left: int) -> int:
            """Schedule the request's next aligned chunk if it fits whole; tokens used, or
            0 if it does not fit, -1 if the pool cannot hold it."""
            if admit:
                self.lookup(request)
            remaining = request.target - request.computed
            chunk = self.aligned_chunk(request.computed, remaining, threshold)
            if chunk <= 0 or chunk > budget_left:
                return 0
            x0 = request.computed
            continues = x0 + chunk < request.target
            fill = round(shape.fill_blocks * min(chunk, threshold) / self.threshold) if continues else 0
            if admit:
                self.grow(request, x0 - x0 % BLOCK)  # pin the hit's blocks before making room
                if request.lookup_key is not None:
                    cache.hold_checkpoint(request.lookup_key)
                    if cache.eviction == "prefix":
                        request.checkpoints.append(request.lookup_key)
                need = self.new_blocks(request, x0 + chunk) + shape.window_blocks + fill
                if not cache.make_room(need):
                    self.free_request(request)
                    request.lookup_version = -1
                    return -1
                request.window = True
                cache.private += shape.window_blocks
                request.admitted = now
                ideal, _ = cache.hit(request.lineage, request.target, shape.replay_tokens, ideal=True)
                if request.preemptions == 0:
                    request.hit_tokens, request.ideal_hit = x0, ideal
            else:
                need = self.new_blocks(request, x0 + chunk) + max(fill - request.fill, 0)
                if not preempt_for(request, need):
                    return -1
            if fill > request.fill:
                cache.private += fill - request.fill
                request.fill = fill
            if x0 == request.hit_tokens and request.preemptions == 0 and chunk >= 1024:
                stats["engram"] = True
            self.grow(request, x0 + chunk)
            interval = self.interval
            for position in range((x0 // interval + 1) * interval, x0 + chunk + 1, interval):
                key = cache.add_checkpoint(request.lineage.owner_of_position(position), position)
                request.checkpoints.append(key)
            end = x0 + chunk
            if end >= request.target and request.out == 0 and end % interval:
                key = cache.add_checkpoint(request.lineage.owner_of_position(end), end)
                request.checkpoints.append(key)
            request.computed = end
            if not continues and request.fill:
                cache.private -= request.fill
                request.fill = 0
            entries.append((request, "prefill", chunk, x0, request.epoch))
            chunks.append((chunk, x0))
            scheduled.add(request.rid)
            if x0 == request.hit_tokens:
                stats["new"] += 1
            if request.preemptions:
                recompute = chunk
            else:
                fresh = max(min(end, request.prompt) - max(x0, request.ideal_hit), 0)
                recompute = chunk - fresh
            request.recompute += recompute
            stats["recompute"] += recompute
            stats["prefill"] += chunk
            request.deferred_since = None
            return chunk

        def take_decode(request: Request, budget_left: int) -> int:
            rows = min(shape.rows, budget_left)
            if rows <= 0:
                return 0
            if request.out + (1 if request.rid in in_flight else 0) >= request.max_out:
                return 0  # the step in flight samples its last token (vLLM's async check)
            tokens = request.length + rows
            if not preempt_for(request, self.new_blocks(request, tokens)):
                return -1
            self.grow(request, tokens)
            entries.append((request, "decode", rows, request.length, request.epoch))
            scheduled.add(request.rid)
            stats["n_dec"] += 1
            return rows

        def admit(request: Request, budget_left: int) -> int:
            used = take_prefill(request, True, budget_left)
            if used > 0:
                self.waiting.remove(request)
                request.status = "running"
                self.running.append(request)
            return used

        held_first = any(not r.prefilling and r.first is None and r.out > 0 for r in self.running)

        # Decode rows: production's threshold leaves room for every decoding request.
        for request in list(self.running):
            if budget <= 0:
                break
            if request.status == "running" and not request.prefilling:
                used = take_decode(request, budget)
                budget -= max(used, 0)

        def running_prefills() -> list:
            return [r for r in self.running if r.status == "running" and r.prefilling and r.rid not in scheduled]

        def held_long(request: Request) -> bool:
            cap = policy.long_wait_cap_s
            return cap is not None and request.deferred_since is not None and now - request.deferred_since >= cap

        inflight_prefills = running_prefills()
        if not inflight_prefills:
            self.interleave_remaining = 0
        interleave_first_due = policy.sink_hold and any(
            not r.prefilling and r.first is None and (r.out > 0 or r.rid in in_flight)
            for r in self.running
        )
        interleave_step = (
            policy.interleave_decode_steps > 0
            and stats["n_dec"] > 0
            and bool(inflight_prefills)
            and (self.interleave_remaining > 0 or interleave_first_due)
        )
        if interleave_step and self.interleave_remaining > 0:
            self.interleave_remaining -= 1

        order = policy.order
        if interleave_step:
            # Defer running multi-chunk prefills via defer_prefills so active decodes run
            # as a fast graphed decode step, while still admitting any waiting short prefill.
            waiting = list(self.waiting)[: policy.lookup_window] if not preempted else []
            cut = policy.bypass_tokens or threshold
            slots = self.max_seqs - len(self.running)
            for request in waiting:
                if budget <= 0 or slots <= 0:
                    break
                self.lookup(request)
                if self.remaining(request) <= cut:
                    used = admit(request, budget)
                    if used > 0:
                        budget -= used
                        slots -= 1
        elif order == "fcfs":
            for request in running_prefills():
                if budget <= 0:
                    break
                budget -= max(take_prefill(request, False, budget), 0)
            if not preempted:
                while self.waiting and budget > 0 and len(self.running) < self.max_seqs:
                    used = admit(self.waiting[0], budget)
                    if used <= 0:
                        break
                    budget -= used
        elif order in ("bypass", "spf"):
            waiting = list(self.waiting)[: policy.lookup_window] if not preempted else []
            for request in waiting:
                self.lookup(request)
            candidates = running_prefills() + waiting
            cut = policy.bypass_tokens or threshold
            if order == "spf":
                candidates.sort(key=lambda r: (not held_long(r), self.remaining(r), r.sent))
            else:
                shorts = [r for r in candidates if self.remaining(r) <= cut or held_long(r)]
                longs = [r for r in candidates if not (self.remaining(r) <= cut or held_long(r))]
                if policy.first_token_priority and held_first:
                    for request in longs:
                        if request.deferred_since is None:
                            request.deferred_since = now
                    longs = []
                candidates = shorts + longs
            slots = self.max_seqs - len(self.running)
            for request in candidates:
                if budget <= 0:
                    break
                fresh = request.status == "waiting"
                if fresh and slots <= 0:
                    continue
                used = admit(request, budget) if fresh else take_prefill(request, False, budget)
                if used > 0:
                    budget -= used
                    slots -= 1 if fresh else 0
                elif self.remaining(request) > cut and request.deferred_since is None:
                    request.deferred_since = now
        elif order == "rr2":
            # vLLM's PrefillInterleaveStep (prefill_interleave.py) with two lanes, round-robin:
            # lanes go to the prefills scheduled longest ago; each lane's budget is the step's
            # remaining budget over the lanes not yet served. With aligned chunks a lane whose
            # whole chunk exceeds its share is skipped (running pass) or stops the waiting pass,
            # and keeps its lane. `lanes_whole_chunks` is vLLM's fix (f05748ead): an aligned
            # chunk is never capped at its share; it takes the step's budget when that holds it.
            started = running_prefills()
            open_ = len(self.running) < self.max_seqs and not preempted
            ranked = sorted(started + (list(self.waiting)[: policy.lookup_window] if open_ else []),
                            key=lambda r: (r.meta.get("last_step", -1), r.sent, r.rid))
            selected: list = []
            unavailable: set = set()
            cursor = [0]

            def replenish() -> None:
                while len(selected) < 2 and cursor[0] < len(ranked):
                    request = ranked[cursor[0]]
                    cursor[0] += 1
                    if request.rid not in unavailable:
                        selected.append(request)

            def release(request: Request) -> None:
                unavailable.add(request.rid)
                if request in selected:
                    selected.remove(request)
                replenish()

            def share() -> int:
                fanout = sum(1 for r in selected if r.rid not in scheduled and r.rid not in unavailable)
                return -(-budget // max(fanout, 1))

            def serve(request: Request, fresh: bool) -> int:
                nonlocal budget
                limit = budget if policy.lanes_whole_chunks else min(budget, share())
                used = admit(request, limit) if fresh else take_prefill(request, False, limit)
                if used > 0:
                    budget -= used
                    request.meta["last_step"] = len(self.steps)
                elif used < 0:
                    release(request)
                return used

            replenish()
            for request in started:
                if budget > 0 and request in selected:
                    serve(request, False)
            while budget > 0 and len(self.running) < self.max_seqs and not preempted:
                lanes = [r for r in selected if r.status == "waiting" and r.rid not in unavailable]
                if not lanes:
                    break
                if serve(lanes[0], True) == 0:
                    break  # "No partial chunk this step."
            for request in started:
                if budget > 0 and request in selected and request.rid not in scheduled:
                    serve(request, False)
        else:
            raise ValueError(order)

        if (
            policy.interleave_decode_steps > 0
            and not interleave_step
            and any(r.prefilling and r.rid in scheduled for r in self.running)
        ):
            self.interleave_remaining = policy.interleave_decode_steps

        if len(self.running) >= self.max_seqs:
            self.wait_reason = "slots"
        elif any(c >= 1024 for c, _ in chunks):
            self.wait_reason = "behind_prefill"
        else:
            self.wait_reason = "other"
        pending = any(r.prefilling for r in self.running) or (self.waiting and len(self.running) < self.max_seqs)
        if pending and not chunks:
            if self.no_progress_since is None:
                self.no_progress_since = now
            elif now - self.no_progress_since > self.livelock_s:
                self.livelocked = True
        else:
            self.no_progress_since = None
        if not entries:
            return None
        prompt = sum(c for c, _ in chunks)
        cost = shape.step_cost(stats["n_dec"], chunks, short_e=self.short_e,
                               engram_s=self.engram_s if stats["engram"] else 0.0,
                               overhead_cut=policy.overhead_cut_s)
        return {"entries": entries, "chunks": chunks, "n_dec": stats["n_dec"], "prefill": prompt,
                "recompute": stats["recompute"], "new": stats["new"], "cost": cost}

    def unschedule(self, victim: Request, entries: list, chunks: list) -> None:
        for index in range(len(entries) - 1, -1, -1):
            if entries[index][0] is victim:
                _, kind, tokens, x0, _ = entries.pop(index)
                if kind == "prefill":
                    chunks.remove((tokens, x0))


# ---------------------------------------------------------------------------------------------
# Results


def percentile(values: list, q: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def class_summary(requests: list) -> dict:
    done = [r for r in requests if r.first is not None]
    ttft = [r.first - r.sent for r in done]
    gaps = [g for r in done for g in r.gaps]
    tpot = [(r.last - r.first) / (r.emitted - 1) for r in done if r.finished_at and r.emitted > 1]
    prompt = sum(r.prompt for r in requests) or 1
    queued = [r.admitted - r.arrival for r in requests if r.admitted is not None]
    return {
        "requests": len(requests), "unfinished": sum(1 for r in requests if r.finished_at is None),
        "ttft_mean": sum(ttft) / len(ttft) if ttft else None,
        "ttft_p50": percentile(ttft, 0.5), "ttft_p90": percentile(ttft, 0.9), "ttft_p99": percentile(ttft, 0.99),
        "itl_p50": percentile(gaps, 0.5), "itl_p90": percentile(gaps, 0.9), "itl_p99": percentile(gaps, 0.99),
        "itl_max": max(gaps) if gaps else None,
        "tpot_p50": percentile(tpot, 0.5), "tpot_p99": percentile(tpot, 0.99),
        "prompt_p50": percentile([r.prompt for r in requests], 0.5),
        "hit_share": sum(r.hit_tokens for r in requests) / prompt,
        "ideal_hit_share": sum(r.ideal_hit for r in requests) / prompt,
        "queue_p50": percentile(queued, 0.5), "queue_p99": percentile(queued, 0.99),
        "preempted": sum(1 for r in requests if r.preemptions),
    }


def summarize(engine: Engine, start: float, end: float) -> dict:
    """Requests sent in [start, end); steps started in it."""
    window = end - start
    requests = [r for r in engine.requests if start <= r.sent < end]
    by_cls: dict = collections.defaultdict(list)
    for r in requests:
        by_cls[r.cls].append(r)
    steps = [s for s in engine.steps if start <= s.start < end]
    busy = sum(min(s.start + s.duration, end) - s.start for s in steps)
    prefill = sum(s.prefill_tokens for s in steps)
    recompute = sum(s.recompute_tokens for s in steps)
    long_steps = [s for s in steps if s.prefill_tokens >= 1024]
    return {
        "window_s": window, "steps": len(steps), "busy": busy / window,
        "output_tok_s": sum(s.emitted for s in steps) / window,
        "prefill_tok_s": (prefill - recompute) / window,
        "recompute_tok_s": recompute / window,
        "long_step_time_share": sum(s.duration for s in long_steps) / window,
        "step_p50": percentile([s.duration for s in steps], 0.5),
        "step_p99": percentile([s.duration for s in steps], 0.99),
        "decode_rows_per_step": sum(s.n_dec for s in steps) / len(steps) if steps else 0.0,
        "preemptions": engine.preemptions,
        "evicted_blocks": engine.cache.evicted_blocks,
        "evicted_checkpoints": engine.cache.evicted_checkpoints,
        "livelocked": engine.livelocked or (bool(engine.waiting or engine.running) and not engine.events),
        "all": class_summary(requests),
        "classes": {cls: class_summary(rs) for cls, rs in sorted(by_cls.items())},
    }
