"""Copy drafting variants walked over token streams: the steps a hybrid drafter would take.

A stream is one conversation's tokens in order (every turn's prompt is the stream so far), with
an output mask and turn ends. For every output position ``p`` (the token ``S[p]`` a step would
predict first), the proposal families are computed with numpy, then walked turn by turn:

- **exact** (per request): the latest earlier occurrence of the longest of the levels
  ``m, 2m, 4m, ...`` (at most ``MAX_LEVEL``) that ends the context at ``p``. The proposal is the
  tokens that followed it, limited to tokens that exist when it is made (``p - source``).
- **both**: the exact proposal where there is one, the global chained table elsewhere.
- **chained** (per request or global): a table maps each ``n``-gram to the token that followed
  its latest occurrence, and a draft extends one token at a time through it, so consecutive
  draft tokens may come from different places. ``global`` shares the table across every stream
  walked before (streams in chronological order); ``request`` keeps it per stream. A draft
  stops where the table has no entry; past its first wrong token it is assumed to run on to
  the maximum (its rows are verified either way). A table entry is taken as the latest
  occurrence before the predicted position, which can sit inside the draft itself on periodic
  text; an older occurrence then predicts the same token.

Walk policies, for a proposal family and a maximum draft ``k``:

- ``full``: every proposal is verified at ``min(k, its length)`` draft rows;
- ``grow``: a run of proposals starts at ``GROW_START`` draft rows and doubles while every row
  is accepted, up to ``k``; a rejected row restarts it.
- ``-nolag`` (either policy): the step's rows are laid out from its own proposal, the bound of
  what the one-step-stale allocation below costs.

The engine lays out a step's rows before the previous step's tokens are known (asynchronous
scheduling), so a step's allocation comes from the state one step earlier, assuming that step
is accepted in full. Allocated rows the real proposal does not use are dead rows (padding the
routed MoE skips). After a DSpark step the allocation is DSpark's 1 + 5 rows, so a copy's first
suffix step verifies at most 5 drafts.

Where no proposal is taken, DSpark drafts. Its tokens a step follow the copy probe (2026-10-23):
inside an exact copy (the exact ``m = 8`` continuation is right) ``min(5, run + 1)``, elsewhere
the release suite's 2.25 (steps of 2, 2, 2, 3 tokens).

Each step is recorded as ``(kind, allocated rows, live rows, tokens)`` in a histogram; the cost
model (cost.py) prices the rows.
"""

from collections import Counter

import numpy as np

DSPARK_DRAFTS = 5
DSPARK_COPY_TOKENS = 5
DSPARK_PROSE_PATTERN = (2, 2, 2, 3)  # 2.25 tokens a step, the release suite's single stream
MAX_LEVEL = 256
GROW_START = 8
COPY_M = 8  # what counts as copied text for DSpark's rate
KINDS = ("dspark_prose", "dspark_copy", "suffix")

_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)
_C3 = np.uint64(0x2545F4914F6CDD1D)


def _mix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * _C1
    x = x ^ (x >> np.uint64(27))
    x = x * _C2
    return x ^ (x >> np.uint64(31))


def window_hashes(S: np.ndarray, lengths) -> dict[int, np.ndarray]:
    """``h[L][e]`` hashes the window ``S[e-L:e]`` (0 where ``e < L``), for each length asked.
    Power-of-two windows double from single tokens; other lengths join them, longest first."""
    n = len(S)
    with np.errstate(over="ignore"):
        pow2 = {1: np.concatenate(([np.uint64(0)], _mix(S.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))))}
        top = max(lengths)
        size = 1
        while size * 2 <= top:
            h = pow2[size]
            joined = np.zeros(n + 1, dtype=np.uint64)
            joined[2 * size :] = _mix(h[size:-size] * _C3 + _mix(h[2 * size :] ^ np.uint64(2 * size)))
            pow2[size * 2] = joined
            size *= 2
        out = {}
        for length in lengths:
            parts = [1 << b for b in range(length.bit_length() - 1, -1, -1) if length >> b & 1]
            acc = None
            span = 0
            for part in parts:  # the window's leftmost part first
                h = np.zeros(n + 1, dtype=np.uint64)
                # the part covering S[e - length + span : e - length + span + part]
                shift = length - span - part
                h[length:] = pow2[part][length - shift : n + 1 - shift]
                acc = h if acc is None else _mix(acc * _C3 + _mix(h ^ np.uint64(length)))
                span += part
            acc[:length] = 0
            out[length] = acc
    return out


def prev_occurrence(h: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``prev[e]`` = the largest ``e' < e`` with ``h[e'] == h[e]`` (both valid), else -1."""
    idx = np.nonzero(valid)[0]
    order = np.argsort(h[idx], kind="stable")
    sh, si = h[idx][order], idx[order]
    prev = np.full(len(h), -1, dtype=np.int64)
    same = sh[1:] == sh[:-1]
    prev[si[1:][same]] = si[:-1][same]
    return prev


class Stream:
    """One stream's tokens and its per-position proposals for the variants asked."""

    def __init__(self, S: np.ndarray, turns: list[tuple[int, int]], ms=(8, 16, 24),
                 global_pred: dict[int, np.ndarray] | None = None):
        self.S, self.turns = S, turns
        n = len(S)
        self.turn_end = np.zeros(n, dtype=np.int64)
        for a, b in turns:
            self.turn_end[a:b] = b
        levels = sorted({m << s for m in ms for s in range(8) if m << s <= MAX_LEVEL} | {COPY_M})
        hashes = window_hashes(S, levels)
        positions = np.arange(n + 1)
        # prev[L][e]: the latest earlier end of the same L-window; the continuation starts there
        prev = {L: prev_occurrence(hashes[L], positions >= L) for L in levels}
        self.exact_source = {}
        for m in sorted(set(ms) | {COPY_M}):
            source = np.full(n, -1, dtype=np.int64)
            for L in sorted((x for x in levels if x >= m and (x // m) & (x // m - 1) == 0 and x % m == 0),
                            reverse=True):
                p = prev[L][:n]  # window ending at p (exclusive), i.e. the context before S[p]
                take = (source < 0) & (p >= 0)
                source[take] = p[take]
            self.exact_source[m] = source
        # chained: the token after the latest occurrence of the n-gram before each position,
        # within this stream ("request") or across every stream walked before ("global")
        self.chained = {"request": {}, "global": {}}
        for m in ms:
            p = prev[m][:n]
            pred = np.where(p >= 0, S[np.clip(p, 0, n - 1)], -1)
            self.chained["request"][m] = (pred >= 0, pred == S)
            if global_pred is not None:
                g = global_pred[m]
                self.chained["global"][m] = (g >= 0, g == S)

    def exact_accept(self, m: int, p: int, d: int) -> int:
        """Accepted drafts of the exact proposal at ``p`` verified at ``d`` rows."""
        source, S, end = self.exact_source[m][p], self.S, self.turn_end[p]
        a = 0
        while a < d and p + a < end and S[source + a] == S[p + a]:
            a += 1
        return a

    def exact_length(self, m: int, p: int) -> int:
        """Tokens the exact proposal at ``p`` can offer (0: none)."""
        source = self.exact_source[m][p]
        return int(p - source) if source >= 0 else 0

    def copy_run(self, p: int, cap: int) -> int:
        """Consecutive correct exact (m = 8) continuation tokens from ``p``, at most ``cap``."""
        source = self.exact_source[COPY_M][p]
        if source < 0:
            return 0
        S, end = self.S, self.turn_end[p]
        a = 0
        while a < cap and p + a < end and source + a < p and S[source + a] == S[p + a]:
            a += 1
        return a

    def chained_draft(self, table: str, m: int, p: int, k: int) -> tuple[int, int]:
        """(offered, accepted) of the chained draft at ``p`` with at most ``k`` drafts."""
        exists, right = self.chained[table][m]
        end = self.turn_end[p]
        a = 0
        while a < k and p + a < end and exists[p + a] and right[p + a]:
            a += 1
        if a == k or p + a >= end:
            return a, a
        if not exists[p + a]:  # the table ran out on the accepted path
            return a, a
        return k, a  # wrong token at p + a; the draft runs on off the path


def walk(stream: Stream, family: str, m: int, k: int, policy: str, hist: Counter) -> None:
    """Steps of one variant over every turn of ``stream``, into ``hist``:
    ``(kind, allocated rows, live rows, tokens) -> count``. ``family`` is ``exact``,
    ``request`` or ``global`` (the last two chained), ``both`` (exact, else the global table),
    or ``dspark``: DSpark alone, the baseline walked by the same rules."""
    lag = not policy.endswith("-nolag")
    policy = policy.removesuffix("-nolag")
    for a, b in stream.turns:
        p = a
        prose_i = 0
        alloc = 1 + DSPARK_DRAFTS  # the rows this step was laid out with
        grow = GROW_START
        while p < b:
            if family == "dspark":
                offered = 0
            elif family == "exact" or family == "both":
                length = stream.exact_length(m, p)
                offered = min(k, length)
                source = "exact"
                if not offered and family == "both":
                    offered, acc = stream.chained_draft("global", m, p, k)
                    source = "global"
            else:
                offered, acc = stream.chained_draft(family, m, p, k)
                source = family
            if offered > 0:
                want = offered if policy == "full" else min(offered, grow)
                if not lag:
                    alloc = 1 + max(DSPARK_DRAFTS, want)
                drafts = min(want, alloc - 1)
                accepted = stream.exact_accept(m, p, drafts) if source == "exact" else min(acc, drafts)
                tokens = min(accepted + 1, b - p)
                hist[("suffix", alloc, 1 + drafts, tokens)] += 1
                # The next step is laid out now, before this one is verified: as if every
                # draft is accepted, the same copy continues at the same cap (doubled by grow).
                cap = k if policy == "full" else min(k, 2 * grow)
                predicted = min(cap, length) if source == "exact" else cap
                if policy == "grow":
                    grow = min(k, 2 * grow) if accepted == drafts else GROW_START
                next_alloc = 1 + max(DSPARK_DRAFTS, predicted)
            else:
                if not lag:
                    alloc = 1 + DSPARK_DRAFTS
                run = stream.copy_run(p, DSPARK_COPY_TOKENS)
                if run >= 1:
                    tokens = min(DSPARK_COPY_TOKENS, run + 1, b - p)
                    hist[("dspark_copy", alloc, 1 + DSPARK_DRAFTS, tokens)] += 1
                else:
                    tokens = min(DSPARK_PROSE_PATTERN[prose_i & 3], b - p)
                    prose_i += 1
                    hist[("dspark_prose", alloc, 0, tokens)] += 1  # its rows are the suite's
                grow = GROW_START
                next_alloc = 1 + DSPARK_DRAFTS
            alloc = next_alloc
            p += tokens


def variants(ms=(8, 16, 24), ks=(16, 32, 64), policies=("full", "grow", "full-nolag", "grow-nolag"),
             families=("exact", "request", "global", "both")):
    """Every (family, m, k, policy) to walk, DSpark alone first; ``request`` and ``global`` are
    the chained table."""
    return [("dspark", 0, 0, "full")] + [(f, m, k, pol) for f in families for m in ms for k in ks
                                         for pol in policies]


def global_predictions(streams: list[np.ndarray], ms=(8, 16, 24)) -> dict[int, list[np.ndarray]]:
    """The chained table shared by every stream: for each stream position, the token after the
    latest earlier occurrence of its preceding n-gram in this stream or any stream before it
    (streams in the order given), -1 where there is none. Windows across a stream boundary
    never match."""
    lengths = np.array([len(s) for s in streams], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lengths)))
    S = np.concatenate(streams)
    owner_start = np.repeat(starts[:-1], lengths)
    ends = np.arange(len(S) + 1)
    out = {}
    for m in ms:
        h = window_hashes(S, [m])[m]
        valid = np.zeros(len(S) + 1, dtype=bool)
        # window S[e-m:e] lies inside one stream: e - m >= the start of the stream holding e - 1
        valid[1:] = (ends[1:] - m) >= owner_start
        prev = prev_occurrence(h, valid)[: len(S)]
        pred = np.where(prev >= 0, S[np.clip(prev, 0, len(S) - 1)], -1)
        del h, valid, prev
        out[m] = [pred[starts[i] : starts[i + 1]] for i in range(len(streams))]
    return out


