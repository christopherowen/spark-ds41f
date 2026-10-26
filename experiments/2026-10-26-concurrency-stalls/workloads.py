"""Request mixes for stallsim: the release suite's loads, the stall step and its short-request
probe, AgentX-like agent sessions, the 2026-10-23 soak, and cold documents.

The agent sessions are a model, not the AgentX corpus (#574 has not run, and the corpus is not
on the laptop). Their parameters are set to reproduce what #574 publishes about the corpus:
- a median request input of about 142K tokens, about half of main-agent turns over 256K;
- main-agent turns 89.6% pure appends, 6.6% divergent, 3.9% new contexts;
- 24% of subagent turns return to an older context;
- an unbounded cache would hit about 98% of prompt tokens;
- each lane starts 25-75% into its session with a primer that carries its whole context.
calibrate.py prints the generated statistics beside those targets.
"""

from __future__ import annotations

import dataclasses
import math
import random

from stallsim import Engine, Lineage


def lognormal(rng: random.Random, median: float, sigma: float, low: float = 1, high: float = 1e12) -> int:
    return int(min(max(median * math.exp(sigma * rng.gauss(0.0, 1.0)), low), high))


# ---------------------------------------------------------------------------------------------
# Chat


def chat_poisson(engine: Engine, rng: random.Random, rate: float, start: float, end: float, *,
                 prompt=lambda rng: 20, out=lambda rng: 256, cls: str = "chat") -> None:
    """Independent requests (fresh cache salts) arriving as a Poisson process."""
    t = start + rng.expovariate(rate)
    while t < end:
        engine.submit(cls, t, prompt(rng), out(rng), Lineage())
        t += rng.expovariate(rate)


def closed_loop(engine: Engine, rng: random.Random, users: int, start: float, end: float, *,
                prompt=lambda rng: 20, out=lambda rng: 256, think=lambda rng: 0.0,
                cls: str = "chat", stagger: float = 0.0) -> None:
    """`users` loops: each sends its next request when the last one finishes."""

    def send(engine: Engine, time: float) -> None:
        if time < end:
            engine.submit(cls, time, prompt(rng), out(rng), Lineage(), on_finish=done)

    def done(engine: Engine, request, time: float) -> None:
        engine.call_at(time + 0.002 + think(rng), send)

    for user in range(users):
        engine.call_at(start + user * stagger, send)


def chat_mix_prompt(rng: random.Random) -> int:
    """Interactive chat: mostly short questions, some pasted text."""
    if rng.random() < 0.75:
        return lognormal(rng, 60, 0.8, 8, 2000)
    return lognormal(rng, 2500, 0.9, 500, 30000)


def chat_mix_out(rng: random.Random) -> int:
    return lognormal(rng, 350, 0.8, 16, 4096)


# ---------------------------------------------------------------------------------------------
# Documents


def documents(engine: Engine, rng: random.Random, rate: float, start: float, end: float, *,
              low: int = 32768, high: int = 524288, out: int = 300, cls: str = "doc") -> None:
    """Cold documents of log-uniform size arriving as a Poisson process."""
    t = start + rng.expovariate(rate)
    while t < end:
        size = int(math.exp(rng.uniform(math.log(low), math.log(high))))
        engine.submit(cls, t, size, out, Lineage())
        t += rng.expovariate(rate)


def stall_probe(engine: Engine, streams: int, size: int, *, warmup: float = 20.0,
                probes: tuple = (), probe_prompt: int = 20, probe_out: int = 64,
                stream_prompt: int = 20, stream_out: int = 16384) -> dict:
    """servebench stall: `streams` decode, then a cold document of `size` tokens (one output
    token); short probes are sent `probes` seconds after the document."""
    for i in range(streams):
        engine.submit("stream", 0.001 * i, stream_prompt, stream_out, Lineage())
    doc = engine.submit("doc", warmup, size, 1, Lineage())
    sent = [engine.submit("probe", warmup + offset, probe_prompt, probe_out, Lineage()) for offset in probes]
    return {"doc": doc, "probes": sent}


# ---------------------------------------------------------------------------------------------
# Agents


@dataclasses.dataclass
class AgentParams:
    system: int = 18000            # shared system prompt and tool definitions
    sub_shared: int = 12000        # the part a subagent shares with its tree
    final_median: float = 480000   # a session's last context
    final_sigma: float = 0.35
    final_cap: int = 990000
    append_median: float = 500     # a main turn's new input (tool results, user): mean ~1.3K
    append_sigma: float = 1.4
    append_cap: int = 60000
    out_median: float = 180
    out_sigma: float = 1.2
    out_cap: int = 8000
    think_median: float = 2.5
    think_sigma: float = 1.3
    think_cap: float = 300.0
    diverge: float = 0.066
    new_context: float = 0.039
    diverge_back_median: float = 3000
    new_context_median: float = 12000
    spawn: float = 0.035           # a main turn starts subagents
    spawn_mean_extra: float = 1.5  # subagents per spawn: 1 + Poisson(this), at most 10
    sub_task_median: float = 20000
    sub_turns_mean: float = 8.0
    sub_append_median: float = 1800
    sub_append_sigma: float = 1.1
    sub_out_median: float = 220
    sub_return: float = 0.24       # a subagent turn returns to an older context
    sub_think_median: float = 1.2
    first_session_start: tuple = (0.25, 0.75)


class AgentLane:
    """One AgentX lane: a session tree (main agent and subagents) after another, closed loop."""

    def __init__(self, engine: Engine, rng: random.Random, params: AgentParams, start: float, end: float,
                 cls: str = "agent"):
        self.engine, self.rng, self.p, self.end, self.cls = engine, rng, params, end, cls
        self.stats = {"turns": 0, "kinds": {"append": 0, "diverge": 0, "new": 0, "primer": 0}}
        engine.call_at(start, lambda e, t: self.new_session(t, first=True))

    def new_session(self, time: float, first: bool = False) -> None:
        p, rng = self.p, self.rng
        self.root = Lineage()  # a new play: cache-busted
        self.main = Lineage(self.root, p.system)
        self.final = lognormal(rng, p.final_median, p.final_sigma, 60000, p.final_cap)
        if first:
            share = rng.uniform(*p.first_session_start)
            self.context = int(p.system + share * (self.final - p.system))
            self.prompt_end = self.context
            self.stats["kinds"]["primer"] += 1
            self.engine.submit(self.cls + "-primer", time, self.context, 1, self.main,
                               on_finish=lambda e, r, t: self.next_main(t))
            return
        self.context = p.system
        self.prompt_end = p.system
        self.next_main(time, pending_input=lognormal(rng, 3000, 0.8, 200, 20000))

    def next_main(self, time: float, pending_input: int = 0) -> None:
        if time >= self.end:
            return
        p, rng = self.p, self.rng
        if self.context >= self.final:
            self.new_session(time + 1.0)
            return
        roll = rng.random()
        new = pending_input + lognormal(rng, p.append_median, p.append_sigma, 20, p.append_cap)
        if roll < p.new_context:
            # A side call with its own small context; the main agent then returns to its own.
            self.stats["turns"] += 1
            self.stats["kinds"]["new"] += 1
            side = Lineage(self.root, p.system)
            prompt = p.system + lognormal(rng, p.new_context_median, 0.6, 2000, 120000)
            out = lognormal(rng, p.out_median, p.out_sigma, 8, p.out_cap)
            think = min(lognormal(rng, p.think_median, p.think_sigma, 0, 1e9), p.think_cap)
            self.engine.call_at(time + think, lambda e, t: e.submit(
                self.cls + "-side", t, prompt, out, side,
                on_finish=lambda e2, r, t2: self.next_main(t2, pending_input=pending_input)))
            return
        if roll < p.new_context + p.diverge:
            back = min(lognormal(rng, p.diverge_back_median, 1.0, 64, 200000), self.prompt_end - p.system)
            branch = max(self.prompt_end - back, p.system)
            self.main = Lineage(self.main, branch)
            prompt = branch + new
            kind = "diverge"
        else:
            prompt = self.context + new
            kind = "append"
        prompt = min(prompt, 1048576 - 9000)
        self.stats["turns"] += 1
        self.stats["kinds"][kind] += 1
        out = lognormal(rng, p.out_median, p.out_sigma, 8, p.out_cap)
        think = min(lognormal(rng, p.think_median, p.think_sigma, 0, 1e9) * 1.0, p.think_cap)
        self.engine.call_at(time + think, lambda e, t: self.send_main(t, prompt, out))

    def send_main(self, time: float, prompt: int, out: int) -> None:
        if time >= self.end:
            return
        self.prompt_end = prompt

        def done(engine: Engine, request, t: float) -> None:
            self.context = request.prompt + request.out
            if self.rng.random() < self.p.spawn:
                self.spawn(t)
            else:
                self.next_main(t)

        self.engine.submit(self.cls, time, prompt, out, self.main, on_finish=done)

    def spawn(self, time: float) -> None:
        p, rng = self.p, self.rng
        count = min(10, 1 + poisson(rng, p.spawn_mean_extra))
        state = {"left": count, "results": 0}

        def sub_done(result: int, t: float) -> None:
            state["left"] -= 1
            state["results"] += result
            if state["left"] == 0:
                self.next_main(t, pending_input=state["results"])

        for _ in range(count):
            SubAgent(self, time + rng.uniform(0.0, 0.5), sub_done)


class SubAgent:
    def __init__(self, lane: AgentLane, time: float, done):
        self.lane, self.done = lane, done
        p, rng = lane.p, lane.rng
        self.lineage = Lineage(lane.root, p.sub_shared)
        self.context = p.sub_shared + lognormal(rng, p.sub_task_median, 0.5, 1000, 60000)
        self.history = [self.context]
        self.turns = 1 + geometric(rng, 1.0 / p.sub_turns_mean)
        self.last_out = 0
        lane.engine.call_at(time, lambda e, t: self.send(t, self.context))

    def send(self, time: float, prompt: int) -> None:
        lane, p, rng = self.lane, self.lane.p, self.lane.rng
        if time >= lane.end:
            return
        out = lognormal(rng, p.sub_out_median, 1.1, 8, 6000)
        lane.stats["turns"] += 1

        def finished(engine: Engine, request, t: float) -> None:
            self.last_out = request.out
            self.context = request.prompt + request.out
            self.turns -= 1
            if self.turns <= 0:
                self.done(self.last_out, t)
                return
            think = min(lognormal(rng, p.sub_think_median, 1.0, 0, 1e9), p.think_cap)
            if rng.random() < p.sub_return and len(self.history) > 2:
                branch = self.history[-3]
                self.lineage = Lineage(self.lineage, branch)
                nxt = branch + lognormal(rng, p.sub_append_median, p.sub_append_sigma, 20, 60000)
            else:
                nxt = self.context + lognormal(rng, p.sub_append_median, p.sub_append_sigma, 20, 60000)
            nxt = min(nxt, 1048576 - 7000)
            self.history.append(nxt)
            lane.engine.call_at(t + think, lambda e, tt: self.send(tt, nxt))

        lane.engine.submit(lane.cls + "-sub", time, prompt, out, self.lineage, on_finish=finished)


def poisson(rng: random.Random, mean: float) -> int:
    limit, k, p = math.exp(-mean), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def geometric(rng: random.Random, p: float) -> int:
    k = 0
    while rng.random() > p:
        k += 1
    return k


def agentx(engine: Engine, rng: random.Random, lanes: int, start: float, end: float,
           params: AgentParams | None = None, stagger: float = 2.0) -> list:
    params = params or AgentParams()
    return [AgentLane(engine, random.Random(rng.random()), params, start + i * stagger, end) for i in range(lanes)]


# ---------------------------------------------------------------------------------------------
# The 2026-10-23 soak (experiments/2026-10-23-next-release/memory/soak.py)


def soak_agents(engine: Engine, rng: random.Random, sessions: int, start: float, end: float, *,
                low: int = 16000, high: int = 48000, limit: int = 960000, out: int = 200,
                think: float = 1.0, cls: str = "soak") -> None:
    """Sessions that resend their whole history every turn, growing by a 16-48K tool result
    until they pass 960K tokens, then start over (a new, cold conversation)."""

    def turn(state: dict, time: float) -> None:
        if time >= end:
            return
        if state["context"] >= limit:
            state["lineage"] = Lineage()
            state["context"] = 0
        prompt = min(state["context"] + rng.randint(low, high), 1048576 - out - 64)

        def done(engine: Engine, request, t: float) -> None:
            state["context"] = request.prompt + request.out
            engine.call_at(t + think, lambda e, tt: turn(state, tt))

        engine.submit(cls, time, prompt, out, state["lineage"], on_finish=done)

    for i in range(sessions):
        state = {"lineage": Lineage(), "context": rng.randint(0, limit // 2)}
        engine.call_at(start + i * 3.0, lambda e, t, s=state: turn(s, t))
