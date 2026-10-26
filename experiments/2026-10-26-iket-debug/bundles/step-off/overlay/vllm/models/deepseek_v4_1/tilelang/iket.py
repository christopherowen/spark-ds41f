# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IKET debug mode: where each TileLang kernel loads, computes and waits.

``VLLM_DS41_IKET=<directory>`` turns it on for a process; the GPU worker calls
``enable`` first thing in ``init_device``, before any kernel compiles.
From then on every TileLang kernel the process compiles carries IKET events
(TileLang's ``tilelang.tools.cuda.iket``, NVIDIA's in-kernel event tracing): the
phase pass below reads each kernel's generated CUDA and wraps its phases in
warp-local IKET ranges. The kernels' own sources are unchanged, and so is every
kernel while the variable is unset: then nothing here is imported and no TileLang
callback is registered.

When on:

- TileLang's kernel and binary caches are off and its cache directory is
  ``<directory>/tilelang-cache``, so an instrumented kernel is never read from or
  written to the deployment's cache. Kernels still go through declare, prepare and
  freeze; preparation compiles every one of them.
- ``<directory>/sources`` holds each instrumented module's CUDA source and its
  sites (``<kernel>.<hash>.cu`` and ``.json``: each range's name, phase, source
  lines and first statement), which the trace's range names refer to.
- ``VLLM_DS41_IKET_CTAS``, ``VLLM_DS41_IKET_ITERS`` and ``VLLM_DS41_IKET_SKIP``
  (optional) sample CTAs and loop iterations and leave named kernels out, which an
  engine-wide trace needs: run-iket's buffer is one allocation for the process, and
  every instrumented launch reserves the run's largest warp record for each warp.
- Traces need NVIDIA's profiler: run the process under ``run-iket profile`` (CuTe
  DSL's ``nvidia-cutlass-dsl-libs-base``), which patches the events into trace
  stores. Outside it an event's placeholder store writes shared address 0x20, in
  the 1 KiB per-block reservation below the dynamic shared base, and the kernels
  compute what they compute without IKET, more slowly.

Phases. The pass parses each ``__global__`` body into statements and blocks and
gives each statement a phase from the TileLang intrinsic it calls (``_PATTERNS``)
or from the global buffers it reads or writes: ``load`` (TMA, cp.async, a global
read), ``wait`` (cp.async wait, mbarrier wait, block sync, PDL grid dependency,
sparknet's flag wait, fences, a spin on a global flag), ``compute`` (MMA and
ldmatrix, warp reductions), ``store`` (a global write, TMA store, atomics,
sparknet's staging) and ``signal`` (sparknet's doorbell, release stores and tail).
A block's phase is its statements' phase when they share one. Consecutive siblings
of one phase become one range, the phaseless statements between them included; a
block whose statements mix phases is entered instead. Every wait statement is a
range of its own. A range never contains ``return``, ``break``, ``continue`` or
``goto``, so it closes on every warp path. Names are ``<phase>.<kind>`` (just
``<phase>`` over several kinds), with ``#<n>`` for later sites of a name; at most 30
names a module, as IKET keeps the event id in the five low bits of a 32 ns
timestamp and id 31 ends a range. A module with more names folds the ``#<n>``
sites into their first name, and traces only its first 30.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

MAX_EVENTS = 30
MAX_NAME_BYTES = 32
RANGE_END = 31
_CUDA_POSTPROC = "tilelang_callback_cuda_postproc"
# ptxas emits a block barrier as BAR.SYNC.DEFER_BLOCKING: the warp goes on issuing
# until an instruction touches memory, so a range end's %globaltimer read would run
# before the barrier releases the warp, and the wait would land after the range
# (sparknet's flag wait, peers 20 us late: warps closed the barrier's range at 1 us
# and stalled in the next event's store to 24 us). A volatile shared load (IKET's
# own slot) first holds the timer read until the barrier is through.
_SETTLE = (
    '#define TL_IKET_SETTLE() asm volatile("{\\n\\t.reg .b32 a, s;\\n\\t'
    "mov.u32 a, 0x20;\\n\\t"  # ptxas takes no immediate shared address
    'ld.volatile.shared.u32 s, [a];\\n\\t}" ::: "memory")\n'
)

# (phase, kind, patterns): the first kind a statement matches names its phase.
_PATTERNS = [
    ("wait", "pdl", [r"\bcudaGridDependencySynchronize\s*\("]),
    ("wait", "cp_async", [r"\btl::cp_async_wait\s*<", r"\bcp\.async\.(bulk\.)?wait_"]),
    ("wait", "tma_store", [r"\btl::tma_store_wait\s*<"]),
    ("wait", "mbarrier", [r"\.wait\s*\(", r"\bmbarrier_wait\w*\s*\(",
                          r"mbarrier\.\w+_wait"]),
    ("wait", "sync", [r"\b__syncthreads\s*\(", r"\btl::__sync_thread_partial\s*\(",
                      r"\bbar\.sync\b", r"\bbarrier\.sync\b", r"\.sync\s*\(\s*\)"]),
    ("wait", "flags", [r"\broce_wait_\w+\s*\("]),
    ("wait", "fence", [r"\broce_fence_\w+\s*\(", r"\b__threadfence\w*\s*\(",
                       r"\bfence\.(sc|acq_rel)\b"]),
    ("load", "tma", [r"\btl::tma_load\w*", r"\.expect_transaction\s*\(",
                     r"\barrive_and_expect_tx\b"]),
    ("load", "cp_async", [r"\btl::cp_async_\w+", r"\bcp\.async\b"]),
    ("compute", "mma", [r"\btl::\w*mma\w*", r"\bmma\.sync\b",
                        r"\btl::ptx_ldmatrix\w*", r"\btl::gemm_\w+", r"\bwmma::"]),
    ("compute", "reduce", [r"\btl::AllReduce\b", r"\btl::warp_reduce\w*",
                           r"\b__shfl_\w+", r"\broce_reduce_\w+"]),
    ("store", "tma", [r"\btl::tma_store\w*"]),
    ("store", "stage", [r"\broce_copy_\w+\s*\("]),
    ("signal", "proxy", [r"\broce_doorbell\s*\(", r"\broce_tail\s*\(",
                         r"\broce_st_\w+\s*\("]),
    ("store", "atomic", [r"\batomic\w+\s*\(", r"\btl::Atomic\w+\s*\("]),
]  # fmt: skip
_COMPILED = [
    (phase, kind, re.compile("|".join(alts))) for phase, kind, alts in _PATTERNS
]
_JUMP = re.compile(r"(?:return|break|continue|goto)\b")
_KEYWORDS = ("for", "while", "if", "switch", "do")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_ATTRIBUTE = re.compile(r"__\w+__\s*")
# A preprocessor line other than a pragma: a range could split its #if and #endif.
_CONDITIONAL = re.compile(r"^[ \t]*#(?!\s*pragma\b)", re.M)


@dataclass
class _Node:
    """A statement (``body`` None) or a block (``for``/``while``/``if``/scope, with
    ``orelse`` for an ``else`` part). ``start``..``end`` are whole source lines, the
    node's ``#pragma`` lines included; ``alone`` says no other node shares them."""

    start: int
    end: int
    text: str
    alone: bool
    body: list[_Node] | None = None
    orelse: list[_Node] | None = None
    phase: str | None = None  # None: no phase of its own; "mixed": several
    kinds: set[str] = field(default_factory=set)
    jumps: bool = False


class _Parser:
    """Statements and blocks of a ``__global__`` body, in TVM's CUDA layout (one
    statement per line, every control statement braced)."""

    def __init__(self, source: str) -> None:
        self.s = source

    def skip_space(self, i: int) -> int:
        s = self.s
        while i < len(s):
            if s[i].isspace():
                i += 1
            elif s.startswith("//", i):
                i = self.line_end(i)
            elif s.startswith("/*", i):
                i = s.index("*/", i) + 2
            else:
                break
        return i

    def literal_end(self, i: int) -> int:
        s, quote = self.s, self.s[i]
        i += 1
        while s[i] != quote:
            i += 2 if s[i] == "\\" else 1
        return i + 1

    def scan(self, i: int, stop: str | None) -> int:
        """End of the bracketed group opening at ``i`` (``stop`` None), or of the
        statement starting at ``i``, after its top-level ``stop``."""
        s, depth = self.s, 0
        while True:
            c = s[i]
            if c in "\"'":
                i = self.literal_end(i)
                continue
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0 and stop is None:
                    return i + 1
            elif c == stop and depth == 0:
                return i + 1
            i += 1

    def line_start(self, i: int) -> int:
        return self.s.rfind("\n", 0, i) + 1

    def line_end(self, i: int) -> int:
        end = self.s.find("\n", i)
        return len(self.s) if end < 0 else end + 1

    def blank(self, a: int, b: int) -> bool:
        return not self.s[a:b].strip()

    def block(self, i: int) -> tuple[list[_Node], int]:
        """The nodes of the braced block opening at ``i``, and the end of its ``}``."""
        nodes, i, pragma = [], i + 1, None
        while True:
            i = self.skip_space(i)
            if self.s[i] == "}":
                return nodes, i + 1
            if self.s[i] == "#":  # #pragma unroll goes with the next statement
                pragma = self.line_start(i) if pragma is None else pragma
                i = self.line_end(i)
                continue
            node, i = self.node(i)
            first = self.line_start(node.start)
            node.alone = node.alone and self.blank(first, node.start)
            node.start = pragma if pragma is not None else first
            pragma = None
            nodes.append(node)

    def node(self, i: int) -> tuple[_Node, int]:
        """The node starting at ``i``, and the end of its last character."""
        s = self.s
        if s[i] == "{":
            body, j = self.block(i)
            return self.close(_Node(i, 0, "{", True, body), j), j
        word = _IDENTIFIER.match(s, i)
        keyword = word.group(0) if word else ""
        if keyword not in _KEYWORDS:
            j = self.scan(i, ";")
            return self.close(_Node(i, 0, s[i:j], True), j), j
        j = self.skip_space(word.end())
        if keyword == "do":
            body, j = self.block(j)
            j = self.scan(self.skip_space(j), ";")
            return self.close(_Node(i, 0, s[i:j], True, body), j), j
        header_end = self.scan(j, None)
        j = self.skip_space(header_end)
        node = _Node(i, 0, s[i:header_end], True)
        if s[j] == "{":
            node.body, j = self.block(j)
        else:
            child, j = self.node(j)
            node.body, child.alone = [child], False
        k = self.skip_space(j)
        if (
            keyword == "if"
            and s.startswith("else", k)
            and not _IDENTIFIER.match(s, k + 4)
        ):
            k = self.skip_space(k + 4)
            if s[k] == "{":
                node.orelse, j = self.block(k)
            else:  # else if: the inner if starts mid-line
                child, j = self.node(k)
                node.orelse, child.alone = [child], False
        return self.close(node, j), j

    def close(self, node: _Node, j: int) -> _Node:
        node.end = self.line_end(j - 1)
        node.alone = node.alone and self.blank(j, node.end)
        return node


def _assignment(text: str) -> int:
    """Offset of the top-level assignment operator in a statement, or -1."""
    depth = 0
    for i, c in enumerate(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "=" and depth == 0:
            before, after = text[i - 1 : i], text[i + 1 : i + 2]
            if after == "=" or before in ("=", "!"):
                continue
            if before in ("<", ">") and text[i - 2 : i - 1] != before:
                continue  # <= and >=, but not <<= and >>=
            return i
    return -1


def _classify(node: _Node, params: frozenset[str]) -> None:
    """Set ``node``'s phase, kinds and jumps, bottom up."""
    if node.body is None:
        node.jumps = bool(_JUMP.match(node.text))
        for phase, kind, pattern in _COMPILED:
            if pattern.search(node.text):
                node.phase, node.kinds = phase, {kind}
                return
        at = _assignment(node.text)
        if at >= 0 and set(_IDENTIFIER.findall(node.text[:at])) & params:
            node.phase, node.kinds = "store", {"global"}
        elif set(_IDENTIFIER.findall(node.text)) & params:
            node.phase, node.kinds = "load", {"global"}
        return
    phases = set()
    for child in node.body + (node.orelse or []):
        _classify(child, params)
        node.jumps |= child.jumps
        if child.phase is not None:
            phases.add(child.phase)
            node.kinds |= child.kinds
    if set(_IDENTIFIER.findall(node.text)) & params:
        if node.text.startswith("while"):
            phases.add("wait")  # a spin on a global flag
            node.kinds.add("spin")
        else:  # a condition or bound read from global memory
            phases.add("load")
            node.kinds.add("global")
    node.phase = None if not phases else phases.pop() if len(phases) == 1 else "mixed"


@dataclass
class _Range:
    phase: str
    kinds: set[str]
    first: _Node
    last: _Node
    loops: tuple[str, ...] = ()  # the headers of the for loops around the range
    kernel: str = ""
    base: str = ""
    name: str = ""


def _ranges(nodes: list[_Node], loops: tuple[str, ...] = ()) -> list[_Range]:
    """Ranges over a statement list: runs of one phase, each wait alone. A block
    that mixes phases, jumps, shares a line or waits (other than a spin loop) is
    entered instead."""
    found: list[_Range] = []
    run: _Range | None = None
    for node in nodes:
        if node.phase is None and not node.jumps:
            continue  # phaseless: joins a run that continues past it
        # A wait is its statement alone; a block is a wait only as a spin loop.
        waits = (
            node.phase == "wait" and node.body is not None and "spin" not in node.kinds
        )
        if node.jumps or node.phase == "mixed" or not node.alone or waits:
            run = None
            inner = (*loops, node.text) if node.text.startswith("for") else loops
            for children in (node.body, node.orelse):
                if children and node.phase is not None:
                    found.extend(_ranges(children, inner))
            continue
        if node.phase == "wait":
            found.append(_Range("wait", set(node.kinds), node, node, loops))
            run = None
        elif run is not None and run.phase == node.phase:
            run.last = node
            run.kinds |= node.kinds
        else:
            run = _Range(node.phase, set(node.kinds), node, node, loops)
            found.append(run)
    return found


_FOR = re.compile(
    r"for\s*\(\s*(?:[\w:]+\s+)+(\w+)\s*=\s*([^;]+);\s*\1\s*<\s*([^;]+);\s*\+\+\1\s*\)"
)


def _iterations(header: str, iters: int) -> str | None:
    """True in at most ``iters`` iterations of a TVM ``for`` loop (``for (int v =
    start; v < bound; ++v)``), each in the middle of its share of the loop, so one
    sample is the middle iteration, not the pipeline's fill; None for another header."""
    loop = _FOR.match(header)
    if loop is None:
        return None
    v, start, bound = (part.strip() for part in loop.groups())
    stride = f"max(1, (int)((({bound}) - ({start}) + {iters - 1}) / {iters}))"
    return f"((int)(({v}) - ({start})) % {stride} == {stride} / 2)"


def _kernels(code: str) -> Iterator[tuple[str, frozenset[str], int, int]]:
    """(name, pointer parameters, body start, body end) of each ``__global__``
    definition."""
    parser = _Parser(code)
    for match in re.finditer(r"__global__\s+void\s+", code):
        i = match.end()
        while attribute := _ATTRIBUTE.match(code, i):  # __launch_bounds__(...)
            i = attribute.end()
            if code[i] == "(":
                i = parser.skip_space(parser.scan(i, None))
        name = _IDENTIFIER.match(code, i)
        i = parser.skip_space(name.end()) if name else i
        if name is None or code[i] != "(":
            continue
        params_end = parser.scan(i, None)
        j = parser.skip_space(params_end)
        if code[j] != "{":  # a declaration
            continue
        pointers = frozenset(
            _IDENTIFIER.findall(param)[-1]
            for param in code[i + 1 : params_end - 1].split(",")
            if "*" in param and _IDENTIFIER.findall(param)
        )
        yield name.group(0), pointers, j, parser.scan(j, None)


def annotate(
    code: str, token: Callable[[str, int], str], ctas: int = 0, iters: int = 0
) -> tuple[str, list[dict]]:
    """``code`` with each phase of each kernel in an IKET range, and its sites.

    ``token(name, range_id)`` returns the IKET metadata token of a range named
    ``name``. With ``ctas`` > 0 only about that many CTAs of a launch, evenly
    spread, record their ranges (run-iket still records every warp's lifetime):
    a kernel with many CTAs and loop-carried ranges otherwise writes millions of
    records a launch. With ``iters`` > 0 a range inside a ``for`` loop records in
    at most that many of its iterations, evenly spread, and only lane 0 of a warp
    records (where lanes diverge each group of them would record): run-iket
    reserves the run's largest warp record for every warp of every instrumented
    launch, so one long or divergent kernel sizes the whole run's buffer. A module with
    no ``__global__`` definition comes back unchanged; a kernel with a
    preprocessor conditional in its body is left as it is."""
    found: list[_Range] = []
    sites: list[dict] = []
    bodies: dict[str, int] = {}
    for kernel, params, start, end in _kernels(code):
        if _CONDITIONAL.search(code, start, end):
            sites.append(
                {"kernel": kernel, "skipped": "a preprocessor conditional in its body"}
            )
            continue
        bodies[kernel] = start
        nodes, _ = _Parser(code).block(start)
        for node in nodes:
            _classify(node, params)
        for item in _ranges(nodes):
            kind = next(iter(item.kinds)) if len(item.kinds) == 1 else ""
            item.kernel, item.base = (
                kernel,
                f"{item.phase}.{kind}" if kind else item.phase,
            )
            found.append(item)
    counts: dict[str, int] = {}
    for item in found:
        counts[item.base] = counts.get(item.base, 0) + 1
        item.name = (
            item.base if counts[item.base] == 1 else f"{item.base}#{counts[item.base]}"
        )
    if len({item.name for item in found}) > MAX_EVENTS:
        for item in found:
            item.name = item.base
    traced = list(dict.fromkeys(item.name for item in found))[:MAX_EVENTS]
    inserts: list[tuple[int, int, str]] = []
    for kernel in dict.fromkeys(item.kernel for item in found if item.name in traced):
        sample = ""
        if ctas:
            # Uniform in a CTA, so a range opens and closes in every warp or in none.
            sample += (
                "  const bool __tl_iket_cta = (((blockIdx.z * gridDim.y + blockIdx.y)"
                " * gridDim.x + blockIdx.x) % max(1u, gridDim.x * gridDim.y * gridDim.z"
                f" / {ctas}u)) == 0u;\n"
            )
        if iters:
            # A warp records an event each time a group of its lanes runs it; where its
            # lanes diverge that is many records for one statement. Lane 0 alone keeps
            # a warp's record within the sites' bound.
            sample += (
                "  unsigned __tl_iket_laneid;\n"
                '  asm volatile("mov.u32 %0, %%laneid;" : "=r"(__tl_iket_laneid));\n'
                "  const bool __tl_iket_lane0 = __tl_iket_laneid == 0u;\n"
            )
        if sample:
            inserts.append((code.index("\n", bodies[kernel]) + 1, 0, sample))
    for item in found:
        # Each condition holds through a range: the CTA's, lane 0's, and its loops'
        # iterations (a loop variable does not change inside one iteration).
        conditions = ["__tl_iket_cta"] if ctas else []
        if iters:
            conditions.append("__tl_iket_lane0")
        unsampled = []
        for header in item.loops if iters else ():
            condition = _iterations(header, iters)
            if condition is None:
                unsampled.append(" ".join(header.split()))
            else:
                conditions.append(condition)
        guard = f"if ({' && '.join(conditions)}) " if conditions else ""
        sites.append(
            {
                "kernel": item.kernel,
                "range": item.name,
                "phase": item.phase,
                "kinds": sorted(item.kinds),
                "first_line": code.count("\n", 0, item.first.start) + 1,
                "last_line": code.count("\n", 0, item.last.end - 1) + 1,
                "statement": " ".join(item.first.text.split())[:160],
                "loops": len(item.loops),
                **({"unsampled_loops": unsampled} if unsampled else {}),
                "traced": item.name in traced,
            }
        )
        if item.name not in traced:
            continue
        indent = re.match(r"[ \t]*", code[item.first.start :]).group(0)
        range_id = zlib.crc32(item.name.encode()) & 0xFFFFFFFF or 1
        inserts.append(
            (
                item.first.start,
                1,
                f'{indent}{guard}TL_IKET_EVENT(1, "{token(item.name, range_id)}");\n',
            )
        )
        # Settled: a barrier inside the range (a block sync, TileLang's AllReduce)
        # holds the range's end, not the next statement's.
        end = (
            f"{indent}{guard}TL_IKET_SETTLE();\n"
            f"{indent}{guard}TL_IKET_EVENT({RANGE_END});\n"
        )
        inserts.append((item.last.end, 0, end))
    # The events a warp can record, with the loops sampled: each range opens and closes
    # in at most ``iters`` iterations of each loop around it. None: no bound.
    bounds: dict[str, int | None] = {}
    for site in sites:
        if not site.get("traced"):
            continue
        count = None if site["loops"] and not iters else 2 * iters ** site["loops"]
        if site.get("unsampled_loops"):
            count = None
        previous = bounds.get(site["kernel"], 0)
        bounds[site["kernel"]] = None if None in (previous, count) else previous + count
    for site in sites:
        if "range" in site:
            site["events_per_warp_bound"] = bounds.get(site["kernel"])
    out, last = [], 0
    # At one offset a range's end goes before the next range's start.
    for offset, _, text in sorted(inserts, key=lambda insert: insert[:2]):
        out += [code[last:offset], text]
        last = offset
    annotated = "".join(out) + code[last:]
    if "TL_IKET_SETTLE();" in annotated:
        annotated = _SETTLE + annotated
    return annotated, sites


_lock = threading.Lock()
_directory: Path | None = None
# (CTAs a launch, iterations a loop, kernels left as they are); 0 and "": all, none
_sampling: tuple[int, int, str] = (0, 0, "")


def directory() -> Path | None:
    """The IKET output directory, or None while the mode is off."""
    return _directory


def enable(
    output: str | os.PathLike[str], ctas: int = 0, iters: int = 0, skip: str = ""
) -> Path:
    """Turn the IKET mode on for this process, before it compiles any TileLang
    kernel; ``ctas`` and ``iters`` > 0 sample CTAs and loop iterations
    (``annotate``), and kernels whose name ``skip`` matches (a regex) compile
    uninstrumented. Idempotent for the same directory and settings."""
    global _directory, _sampling
    root = Path(output).expanduser().resolve()
    sampling = (int(ctas), int(iters), skip or "")
    if skip:
        re.compile(skip)
    with _lock:
        if _directory is not None:
            if (root, sampling) != (_directory, _sampling):
                raise RuntimeError(
                    f"IKET mode is already on with {_directory}, sampling {_sampling}; "
                    f"not {root}, {sampling}"
                )
            return _directory
        import tilelang.cache
        import tvm_ffi
        from tilelang.env import disable_cache

        compiled = sum(
            len(cache._memory_cache) for cache in tilelang.cache._dispatch_map.values()
        )
        if compiled:
            # Their calls would keep launching the uninstrumented kernels.
            raise RuntimeError(
                "IKET mode must be on before TileLang compiles; "
                f"{compiled} kernels are compiled"
            )
        if tvm_ffi.get_global_func(_CUDA_POSTPROC, allow_missing=True) is not None:
            raise RuntimeError(f"{_CUDA_POSTPROC} is already registered")
        (root / "sources").mkdir(parents=True, exist_ok=True)
        # TileLang reads its cache directory at each use; with its caches off nothing
        # is read or written there either.
        os.environ["TILELANG_CACHE_DIR"] = str(root / "tilelang-cache")
        disable_cache()
        tvm_ffi.register_global_func(_CUDA_POSTPROC, f=_postproc)
        _directory, _sampling = root, sampling
    return root


def _token(name: str, range_id: int) -> str:
    from tilelang.tools.cuda.iket.metadata import Event, encode_event

    if len(name.encode()) > MAX_NAME_BYTES:
        raise ValueError(f"IKET range name over {MAX_NAME_BYTES} bytes: {name!r}")
    return encode_event(Event(name=name, event_id=1, kind="range", range_id=range_id))


def _postproc(code: str, target) -> str:
    from tilelang.tools.cuda.iket.codegen import inject_iket_cuda

    digest = hashlib.sha256(code.encode()).hexdigest()[:12]
    ctas, iters, skip = _sampling
    kernel = next((name for name, *_ in _kernels(code)), "module")
    if skip and re.search(skip, kernel):
        return code  # run-iket still records its warps' lifetimes
    annotated, sites = annotate(code, _token, ctas, iters)
    instrumented = inject_iket_cuda(annotated, target, runtime_payloads=False)
    stem = f"{kernel}.{digest}"
    (_directory / "sources" / f"{stem}.cu").write_text(instrumented)
    manifest = {"kernel": kernel, "ctas": ctas, "iters": iters, "sites": sites}
    (_directory / "sources" / f"{stem}.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    return instrumented
