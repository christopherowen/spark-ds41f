"""A DS4.1 decode step from run-iket records, as a streamed Perfetto trace.

usage: iket_perfetto.py [DECODED_JSON] OUT.pftrace [--step K] [--layer L] [--title TEXT]
                        [--max-traced-warps N] [--base-iket BASE.json | --base-nsys BASE.sqlite]
                        [--slice SLICE.json ...] [--allow-mismatch]
                        [--rank NODE=DECODED.json ...]

Reads a run-iket ``iket.decoded_results.json`` in streaming passes. ``Records`` indexes the
records, then the step's records and the warp locations they use are decoded. Perfetto
``TracePacket``s are written as they are made: interned names, slices as begin and end
events, counters as values. Nothing holds the input file whole. Open the output in
ui.perfetto.dev. For a file too large for a browser tab, run
``trace_processor_shell --httpd OUT.pftrace`` on the Mac and open ui.perfetto.dev, which
attaches to it.

Modes:
- **Single run** (``iket_perfetto.py DECODED.json OUT.pftrace``):
  Renders the step's launches, model hierarchy, occupancy/wait counters, and per-SM/CTA/warp
  phases from a single IKET run.
- **With ``--base-iket BASE.json``** (recommended low-overhead base):
  Uses a ``run-iket profile --warp-trace`` (``VLLM_DS41_IKET`` off) ``iket.decoded_results.json``
  as the timeline of record. Every kernel (TileLang, ATen, Triton, NCCL, sparknet) has exact
  warp lifetimes and SM/CTA placement without ``nsys``. Instrumented launches from the primary
  trace (and any ``--slice`` runs) are paired by ``(source, kernel, grid, block)`` ordinal and
  rescaled into the base launch's span. Uninstrumented kernels keep their base warp lifetimes
  and SM/CTA residency.
- **With ``--base-nsys BASE.sqlite``**:
  Uses an ``nsys export --type sqlite`` file of the same step without IKET as the base timeline.
- **With ``--slice SLICE.json [SLICE.json ...]``**:
  Stitches multiple family-sliced IKET runs (each recorded with ``VLLM_DS41_IKET_SKIP``
  excluding other families) onto the primary/base timeline. Verifies that each slice's decode
  step has the same ``(kernel, grid, block)`` launch sequence as the base step (refusing on
  mismatch unless ``--allow-mismatch`` is passed), remaps string and location tables, and
  rescales each slice's instrumented launches into the base launch's span.
- **With ``--rank NODE=DECODED.json [NODE=DECODED.json ...]``**:
  Aligns multi-rank traces (e.g. ``dgx1=r0.json dgx2=r1.json dgx3=r2.json dgx4=r3.json``) into
  one Perfetto trace. Fits per-rank clock offset and linear drift by ordinary least squares on
  collective rendezvous completion timestamps (``oneshot_allreduce*``, ``oneshot_allgather*``,
  ``ncclDevKernel*``), decomposing each collective into arrival jitter wait (waiting for the
  last-arriving rank) vs. fabric transport time.
"""

import argparse
import difflib
import json
import math
import mmap
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

MARKER = "iket_step_marker_kernel"
STEP_KERNEL = "prepare_pos_seq_lens_kernel"
PREFILL_KERNEL = "prepare_prefill_inputs_kernel"
HEAD = "vocab_packed_projection_kernel"
COMBINE = "moe_combine_kernel"
COLLECTIVE_PREFIXES = (
    "oneshot_allreduce",
    "oneshot_allgather",
    "ncclDevKernel",
    "cross_device_reduce",
)
# Above this instrumentation overhead (IKET duration / base duration - 1), a kernel's phase
# proportions are flagged as distorted: tiny kernels run about 3x under IKET.
OVERHEAD_LIMIT = 0.5
DECODER = json.JSONDecoder()

# ---------------------------------------------------------------- protobuf


def varint(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def f_varint(field, n):
    return varint(field << 3) + varint(n)


def f_bytes(field, data):
    return varint(field << 3 | 2) + varint(len(data)) + data


def f_str(field, text):
    return f_bytes(field, text.encode())


def f_double(field, x):
    return varint(field << 3 | 1) + struct.pack("<d", x)


class Writer:
    """Perfetto TracePackets, written one by one on one sequence, with interned event
    names, categories and debug-annotation names."""

    SEQ = 7
    SLICE_BEGIN, SLICE_END, INSTANT, COUNTER = 1, 2, 3, 4

    def __init__(self, path):
        self.out = open(path, "wb")
        self.first = True
        self.interned = {1: {}, 2: {}, 3: {}}  # categories, event names, annotation names
        self.uuid = 1 << 20
        self.packets = 0

    def _packet(self, body):
        flags = 3 if self.first else 2  # incremental state cleared; needs incremental state
        self.first = False
        data = body + f_varint(10, self.SEQ) + f_varint(13, flags)
        self.out.write(f_bytes(1, data))
        self.packets += 1

    def _intern(self, table, text, interned):
        ids = self.interned[table]
        iid = ids.get(text)
        if iid is None:
            iid = ids[text] = len(ids) + 1
            interned.append(f_bytes(table, f_varint(1, iid) + f_str(2, text)))
        return iid

    def track(self, name, parent=None, pid=None, counter=False, rank=None, explicit=False):
        self.uuid += 1
        body = f_varint(1, self.uuid) + f_str(2, name)
        if parent is not None:
            body += f_varint(5, parent)
        if pid is not None:
            body += f_bytes(3, f_varint(1, pid) + f_str(6, name))
        if counter:
            body += f_bytes(8, b"")
        if explicit:
            body += f_varint(11, 3)  # child_ordering: EXPLICIT
        if rank is not None:
            body += f_varint(12, rank)  # sibling_order_rank
        self._packet(f_bytes(60, body))
        return self.uuid

    def event(self, ts, track, kind, name=None, category=None, args=None, value=None):
        interned = []
        body = f_varint(9, kind) + f_varint(11, track)
        if name is not None:
            body += f_varint(10, self._intern(2, name, interned))
        if category is not None:
            body += f_varint(3, self._intern(1, category, interned))
        for key, val in (args or {}).items():
            ann = f_varint(1, self._intern(3, key, interned))
            if isinstance(val, bool):
                ann += f_varint(2, int(val))
            elif isinstance(val, int):
                ann += f_varint(4, val)
            elif isinstance(val, float):
                ann += f_double(5, val)
            else:
                ann += f_str(6, str(val))
            body += f_bytes(4, ann)
        if value is not None:
            body += f_double(44, float(value))
        packet = f_varint(8, int(ts)) + f_bytes(11, body)
        if interned:
            packet += f_bytes(12, b"".join(interned))
        self._packet(packet)

    def slice(self, track, start, end, name, category=None, args=None):
        self.event(start, track, self.SLICE_BEGIN, name, category, args)
        self.event(max(end, start), track, self.SLICE_END)

    def instant(self, track, ts, name, args=None):
        self.event(ts, track, self.INSTANT, name, None, args)

    def counter(self, track, ts, value):
        self.event(ts, track, self.COUNTER, value=value)

    def close(self):
        self.out.close()


class Lanes:
    """Child tracks of ``parent`` for spans that overlap: each span takes the first lane
    free at its start (spans must come in start order)."""

    def __init__(self, writer, parent, prefix):
        self.writer, self.parent, self.prefix = writer, parent, prefix
        self.ends, self.tracks = [], []

    def place(self, start, end):
        for i, last in enumerate(self.ends):
            if last <= start:
                self.ends[i] = end
                return self.tracks[i], i
        self.tracks.append(self.writer.track(f"{self.prefix} {len(self.tracks)}", parent=self.parent,
                                             rank=len(self.tracks)))
        self.ends.append(end)
        return self.tracks[-1], len(self.tracks) - 1


# ---------------------------------------------------------------- the records


class Records:
    """Streaming access to a run-iket ``iket.decoded_results.json``: one JSON object whose
    ``graphLaunches`` (launch lists by graph exec) and ``launches`` hold launch records,
    ``locationTable`` each warp location and ``stringTable`` the range names. Its text is
    ASCII, so latin-1 decoding keeps byte offsets."""

    SPACE = " \n\r\t"

    def __init__(self, path, chunk=64 << 20):
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.chunk = chunk
        self.keys = {key: self.map.find(f'"{key}":'.encode())
                     for key in ("graphLaunches", "launches", "locationTable", "stringTable")}
        missing = [key for key, pos in self.keys.items() if pos < 0]
        if missing:
            raise SystemExit(f"{path}: no {', '.join(missing)}")

    def after(self, key):
        return self.keys[key] + len(f'"{key}":')

    def value_at(self, pos, size=1 << 20):
        """The JSON value starting at byte ``pos`` and the byte just after it."""
        pos = self.skip_space(pos)
        while True:
            text = self.map[pos : pos + size].decode("latin-1")
            try:
                value, end = DECODER.raw_decode(text)
                return value, pos + end
            except json.JSONDecodeError:
                if pos + size >= len(self.map):
                    raise
                size *= 4

    def skip_space(self, pos):
        while self.map[pos : pos + 1] in (b" ", b"\n", b"\r", b"\t"):
            pos += 1
        return pos

    def array(self, pos):
        """Each element of the JSON array at byte ``pos``: (offset, value). Elements are
        decoded from large windows, one C call each; after the last, ``self.end`` is the
        byte after the closing bracket."""
        pos = self.skip_space(pos)
        assert self.map[pos : pos + 1] == b"[", self.map[pos : pos + 40]
        self.base, self.text, self.at = pos + 1, self.map[pos + 1 : pos + 1 + self.chunk].decode("latin-1"), 0
        if self.next_token() == "]":
            self.end = self.base + self.at + 1
            return
        while True:
            try:
                value, stop = DECODER.raw_decode(self.text, self.at)
            except json.JSONDecodeError:
                # The element crosses the window's end: a window that starts at it.
                if self.base + len(self.text) >= len(self.map):
                    raise
                value, stop_byte = self.value_at(self.base + self.at)
                offset = self.base + self.at
                self.base, self.text, self.at = stop_byte, self.map[stop_byte : stop_byte + self.chunk].decode("latin-1"), 0
                yield offset, value
            else:
                yield self.base + self.at, value
                self.at = stop
            token = self.next_token()
            if token == "]":
                self.end = self.base + self.at + 1
                return
            assert token == ",", self.text[self.at : self.at + 40]
            self.at += 1
            self.next_token()

    def next_token(self):
        """The next non-space character, moving the window on when it runs out."""
        while True:
            while self.at < len(self.text) and self.text[self.at] in self.SPACE:
                self.at += 1
            if self.at < len(self.text):
                return self.text[self.at]
            start = self.base + self.at
            if start >= len(self.map):
                raise ValueError("unexpected end of file")
            self.base, self.text, self.at = start, self.map[start : start + self.chunk].decode("latin-1"), 0

    def launch_records(self):
        """(offset, record) for every launch record: every graph exec's, then ``launches``."""
        pos = self.skip_space(self.after("graphLaunches"))
        assert self.map[pos : pos + 1] == b"{"
        pos = self.skip_space(pos + 1)
        while self.map[pos : pos + 1] != b"}":
            _, pos = self.value_at(pos, 4096)  # the graph exec's key
            pos = self.skip_space(pos)
            assert self.map[pos : pos + 1] == b":"
            yield from self.array(pos + 1)
            pos = self.skip_space(self.end)
            if self.map[pos : pos + 1] == b",":
                pos = self.skip_space(pos + 1)
        yield from self.array(self.after("launches"))

    def index(self):
        """Every launch record with warps: (start, end, kernel, offset), by start."""
        out = []
        for offset, record in self.launch_records():
            lives = record.get("warpLifetimes") or []
            if lives and "kernelName" in record:
                out.append((min(w["startTs"] for w in lives), max(w["endTs"] for w in lives),
                            record["kernelName"], offset))
        out.sort()
        return out

    def record(self, offset):
        return self.value_at(offset)[0]

    def strings(self):
        return self.value_at(self.after("stringTable"))[0]

    def locations(self, wanted):
        """The location-table entries whose index is in ``wanted``."""
        out = {}
        last = max(wanted) if wanted else -1
        for i, (_, entry) in enumerate(self.array(self.after("locationTable"))):
            if i in wanted:
                out[i] = entry
            if i >= last:
                break
        return out


# ---------------------------------------------------------------- the model


SUBLAYERS = [
    ("mHC", ("mhc_",)),
    ("norm and cast", ("norm_forward", "per_token_cast")),
    ("attention inputs (rope, KV cache, metadata)", ("rope_", "cache_writer", "metadata_")),
    ("indexer and compressor", ("indexer_", "compress")),
    ("sparse MLA", ("sparse_mla",)),
    ("attention output (WO)", ("gemm_fp8w_bf16_partials", "gemm_splitk_reduce_cast")),
    ("collectives (allreduce / allgather)", COLLECTIVE_PREFIXES),
    ("MoE routing", ("router_partials", "moe_dispatch", "routing_masks", "moe_topk_gate")),
    ("routed experts", ("moe_grouped", "moe_swiglu_cast")),
    ("shared expert", ("activation_swiglu",)),
    ("MoE combine", ("moe_combine",)),
    ("head", ("vocab_packed_projection", "vocab_embedding")),
    ("L2 prefetch (side stream)", ("l2_prefetch",)),
    ("sampler, verification, sink rejection",
     ("sink_", "verify", "resample", "insert", "greedy", "gumbel", "partial_argmax", "finish_argmax",
      "logits_rows", "local_stats", "token_", "topk_", "topp_", "flatten_sampled", "dspark_", "bias_",
      "bad_words", "bincount", "penalties", "temperature", "min_p", "prompt_ids", "grammar", "post_update",
      "ced_copy_rows")),
    ("dense GEMMs", ("gemm_",)),
    ("inputs", ("prepare_", "combine_sampled", "gather_", "compute_slot", "ced_", "fill_", "padding_rows",
                "scatter_rows", "clear_rows", "apply_write", "assign_draft", "adaptive_offsets", "context_inputs",
                "expand_idx", "zero_packs", "iket_")),
]


def sublayer(name):
    for label, prefixes in SUBLAYERS:
        if name.startswith(prefixes):
            return label
    return "other"


def is_collective(name):
    return name.startswith(COLLECTIVE_PREFIXES)


WAIT_REASONS = [("cp.async", "cp_async"), ("mbarrier (TMA)", "mbarrier"), ("block barrier", "sync"),
                ("PDL dependency", "pdl"), ("flag or spin (sparknet)", ("flag", "spin", "peer")),
                ("fence", "fence")]


def wait_reason(range_name):
    """The wait reason of a ``wait.*`` range, or None for other phases."""
    if not range_name.startswith("wait"):
        return None
    what = range_name.split(".", 1)[1] if "." in range_name else ""
    what = what.split("#")[0].split("*")[0]
    for label, keys in WAIT_REASONS:
        if what.startswith(keys if isinstance(keys, tuple) else (keys,)):
            return label
    return "other wait"


def decode_steps(index):
    marks = [i for i, item in enumerate(index) if item[2] == MARKER]
    if len(marks) < 2:
        raise SystemExit(f"{MARKER}: {len(marks)} launches, need two")
    steps, current = [], None
    for item in index[marks[0] + 1 : marks[-1]]:
        if item[2] == STEP_KERNEL:
            current = []
            steps.append(current)
        if current is not None:
            current.append(item)
    return [s for s in steps if not any(item[2] == PREFILL_KERNEL for item in s)]


def segments(launches):
    """Runs of launches from one graph exec (or eager), named by what they hold."""
    out = []
    for launch in launches:
        if out and out[-1]["source"] == launch["source"]:
            out[-1]["items"].append(launch)
        else:
            out.append({"source": launch["source"], "items": [launch]})
    for seg in out:
        names = Counter(l["name"] for l in seg["items"])
        combines = names[COMBINE]
        if seg["source"] == "eager":
            seg["name"] = ("eager: the target's verification head, sampler, sink rejection" if names[HEAD]
                           else "eager: the step's inputs" if names[STEP_KERNEL] else "eager launches")
        elif combines >= 6:
            seg["name"] = f"target forward ({seg['source']}): {combines} layers"
        elif combines:
            seg["name"] = f"drafter ({seg['source']}): {combines} layers" + (" and its head" if names[HEAD] else "")
        else:
            seg["name"] = f"graph replay ({seg['source']}): {len(seg['items'])} launches"
        seg["kind"] = ("target" if combines >= 6 else "drafter" if combines else "eager"
                       if seg["source"] == "eager" else "graph")
    return out


def assign_layers(launches):
    """Each launch's layer: target layers T0.., drafter layers D0.., cut after each MoE
    combine; launches outside a layer get the segment's name."""
    for seg in segments(launches):
        layer, prefix = 0, {"target": "T", "drafter": "D"}.get(seg["kind"])
        for launch in seg["items"]:
            launch["segment"] = seg["name"]
            launch["segment_kind"] = seg["kind"]
            launch["layer"] = f"{prefix}{layer}" if prefix else None
            if prefix and launch["name"] == COMBINE:
                layer += 1
        if prefix:
            # Launches after the segment's last combine (the drafter's head, the final mHC)
            # belong to no layer.
            last = max((i for i, l in enumerate(seg["items"]) if l["name"] == COMBINE), default=-1)
            for launch in seg["items"][last + 1 :]:
                launch["layer"] = None


# ---------------------------------------------------------------- reading


def _read_raw_step(path, index):
    """Read decode step ``index`` from a run-iket ``iket.decoded_results.json`` without
    rebasing or layer filtering."""
    records = Records(path)
    print(f"{path}: indexing", file=sys.stderr, flush=True)
    items = []
    for source, offset, record in source_records(records):
        lives = record.get("warpLifetimes") or []
        if lives and "kernelName" in record:
            items.append((min(w["startTs"] for w in lives), max(w["endTs"] for w in lives),
                          record["kernelName"], offset, source))
    items.sort()
    steps = decode_steps(items)
    if not steps:
        raise SystemExit(f"{path}: no decode step")
    index = len(steps) // 2 if index is None else min(index, len(steps) - 1)
    launches = []
    for ordinal, (start, end, name, offset, source) in enumerate(steps[index]):
        record = records.record(offset)
        launches.append({
            "ordinal": ordinal, "name": name, "source": source, "start": start, "end": end,
            "grid": (record["gridDimX"], record["gridDimY"], record["gridDimZ"]),
            "block": (record["blockDimX"], record["blockDimY"], record["blockDimZ"]),
            "lives": [(w["locIdx"], w["startTs"], w["endTs"]) for w in record["warpLifetimes"]],
            "ranges": [(r["warpLocIdxs"][0], r["startTs"], r["endTs"], r["rangeNameIdx"])
                       for r in record.get("ranges", [])],
        })
    assign_layers(launches)
    wanted = {loc for l in launches for loc, _, _ in l["lives"]}
    locations = {i: (e["smId"], tuple(e["ctaId"]), e["warpId"])
                 for i, e in records.locations(wanted).items()}
    return launches, records.strings(), locations, index, len(steps), len(items)


def load_step(path, index, layer, base=None, slice_paths=None, allow_mismatch=False):
    launches, names, locations, index, step_count, total_items = _read_raw_step(path, index)
    slice_stats = None
    if slice_paths:
        launches, names, locations, slice_stats = stitch_slices(
            launches, names, locations, slice_paths, index, allow_mismatch=allow_mismatch
        )
    rebased = None
    if base is not None:
        launches, rebased = rebase(launches, base, locations)
    if slice_stats is not None:
        if rebased is None:
            rebased = {}
        rebased["slices"] = slice_stats
    if layer is not None:
        launches = [l for l in launches if l["layer"] == layer]
        if not launches:
            raise SystemExit(f"no launch in layer {layer}")
    wanted = {loc for l in launches for loc, _, _ in l["lives"]}
    print(f"{path}: {total_items} launches, {step_count} decode steps; step {index}: "
          f"{len(launches)} launches, {len(wanted)} warp locations", file=sys.stderr, flush=True)
    return launches, names, locations, index, step_count, rebased


def source_records(records):
    """(source, offset, record): graph launches tagged with their graph exec, then the
    eager ones."""
    pos = records.skip_space(records.after("graphLaunches"))
    pos = records.skip_space(pos + 1)
    while records.map[pos : pos + 1] != b"}":
        key, pos = records.value_at(pos, 4096)
        pos = records.skip_space(pos)
        exec_id = key.split(":")[0].replace("graph_exec_", "graph ")
        for offset, record in records.array(pos + 1):
            yield exec_id, offset, record
        pos = records.skip_space(records.end)
        if records.map[pos : pos + 1] == b",":
            pos = records.skip_space(pos + 1)
    for offset, record in records.array(records.after("launches")):
        yield "eager", offset, record


# ---------------------------------------------------------------- base timelines & slice stitching


def nsys_base(path, index):
    """Decode step ``index`` of an nsys export (an IKET-off run): every kernel with its
    stream, grid and graph membership, and the step's span."""
    db = sqlite3.connect(path)
    kernels = [
        {"start": s, "end": e, "name": n, "stream": st, "graph": g is not None,
         "grid": (gx, gy, gz), "block": (bx, by, bz)}
        for s, e, n, st, g, gx, gy, gz, bx, by, bz in db.execute(
            "SELECT k.start, k.end, s.value, k.streamId, k.graphNodeId, k.gridX, k.gridY, k.gridZ,"
            " k.blockX, k.blockY, k.blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL k"
            " JOIN StringIds s ON s.id = k.shortName ORDER BY k.start")
    ]
    steps = decode_steps([(k["start"], k["end"], k["name"], i) for i, k in enumerate(kernels)])
    if not steps:
        raise SystemExit(f"{path}: no decode step")
    index = len(steps) // 2 if index is None else min(index, len(steps) - 1)
    step = [kernels[item[3]] for item in steps[index]]
    end = (kernels[steps[index + 1][0][3]]["start"] if index + 1 < len(steps)
           else max(k["end"] for k in step))
    return {"path": str(path), "kind": "nsys", "kernels": step, "begin": step[0]["start"],
            "end": end, "index": index, "count": len(steps)}


def iket_base(path, index):
    """Decode step ``index`` of a ``run-iket profile --warp-trace`` (``VLLM_DS41_IKET`` off)
    ``iket.decoded_results.json``: every kernel (TileLang, ATen, Triton, NCCL, sparknet)
    with its exact warp lifetimes, SM/CTA locations, and source (graph exec or eager)."""
    records = Records(path)
    items = []
    for source, offset, record in source_records(records):
        lives = record.get("warpLifetimes") or []
        if lives and "kernelName" in record:
            items.append((min(w["startTs"] for w in lives), max(w["endTs"] for w in lives),
                          record["kernelName"], offset, source))
    items.sort()
    steps = decode_steps(items)
    if not steps:
        raise SystemExit(f"{path}: no decode step")
    index = len(steps) // 2 if index is None else min(index, len(steps) - 1)
    step = []
    wanted = set()
    for start, end, name, offset, source in steps[index]:
        record = records.record(offset)
        lives = [(w["locIdx"], w["startTs"], w["endTs"]) for w in record["warpLifetimes"]]
        wanted.update(loc for loc, _, _ in lives)
        step.append({
            "start": start, "end": end, "name": name, "source": source,
            "stream": record.get("streamId", source), "graph": source != "eager",
            "grid": (record["gridDimX"], record["gridDimY"], record["gridDimZ"]),
            "block": (record["blockDimX"], record["blockDimY"], record["blockDimZ"]),
            "lives": lives,
        })
    end = (steps[index + 1][0][0] if index + 1 < len(steps)
           else max(k["end"] for k in step))
    locations = {i: (e["smId"], tuple(e["ctaId"]), e["warpId"])
                 for i, e in records.locations(wanted).items()}
    return {"path": str(path), "kind": "iket-warp-trace", "kernels": step,
            "locations": locations, "begin": step[0]["start"], "end": end,
            "index": index, "count": len(steps)}


def _pair_launches(launches, candidates):
    """Pair ``launches`` with ``candidates`` by (source, name, grid, block) ordinal when
    both carry ``source``, then by (name, grid, block) ordinal, then by kernel name."""
    pairs = []
    left_i = list(sorted(launches, key=lambda l: l["start"]))
    left_b = list(sorted(candidates, key=lambda k: k["start"]))

    for key_fn in (
        lambda x: (x.get("source"), x["name"], tuple(x["grid"]), tuple(x["block"]))
        if "source" in x else None,
        lambda x: (x["name"], tuple(x["grid"]), tuple(x["block"])),
        lambda x: x["name"],
    ):
        by_i, by_b = defaultdict(list), defaultdict(list)
        rem_i, rem_b = [], []
        for l in left_i:
            k = key_fn(l)
            if k is None:
                rem_i.append(l)
            else:
                by_i[k].append(l)
        for b in left_b:
            k = key_fn(b)
            if k is None:
                rem_b.append(b)
            else:
                by_b[k].append(b)
        for k, items in by_i.items():
            other = by_b.pop(k, [])
            if len(other) == len(items):
                pairs.extend(zip(items, other))
            else:
                rem_i.extend(items)
                rem_b.extend(other)
        for other in by_b.values():
            rem_b.extend(other)
        left_i, left_b = rem_i, rem_b
        if not left_i or not left_b:
            break

    unpaired = Counter(l["name"] for l in left_i)
    unmatched_base = Counter(b["name"] for b in left_b)
    return pairs, unpaired, unmatched_base


def rebase(launches, base, locations=None):
    """Pairs each IKET launch with the base run's launch at the same ordinal, then moves its
    warps, CTAs and ranges into the base launch's span: t -> base start + (t - IKET start)
    x (base duration / IKET duration). When ``base`` is an ``iket-warp-trace`` run and a
    launch has no ranges (e.g. skipped by ``VLLM_DS41_IKET_SKIP``), its warp lifetimes and
    SM/CTA locations come directly from the base run."""
    names = {l["name"] for l in launches}
    tl = [k for k in base["kernels"] if k["name"] in names]
    pairs, unpaired, unmatched_base = _pair_launches(launches, tl)

    base_loc_map = {}
    if locations is not None and "locations" in base:
        next_loc = (max(locations) + 1) if locations else 0
        for b_loc, entry in base["locations"].items():
            base_loc_map[b_loc] = next_loc
            locations[next_loc] = entry
            next_loc += 1

    kept, grids = [], 0
    for l, k in pairs:
        i0, i1, b0, b1 = l["start"], l["end"], k["start"], k["end"]
        scale = (b1 - b0) / (i1 - i0) if i1 > i0 else 1.0
        move = lambda t, i0=i0, b0=b0, scale=scale: int(b0 + (t - i0) * scale)
        if not l["ranges"] and "lives" in k and base_loc_map:
            l["lives"] = [(base_loc_map[loc], s, e) for loc, s, e in k["lives"]]
            l.update(start=b0, end=b1, real_us=round((b1 - b0) / 1000, 3),
                     instrumented_us=round((b1 - b0) / 1000, 3), scale=1.0,
                     overhead=0.0, base_stream=k["stream"],
                     grid_match=tuple(k["grid"]) == tuple(l["grid"]), base_grid=k["grid"])
        else:
            l["lives"] = [(loc, move(s), move(e)) for loc, s, e in l["lives"]]
            l["ranges"] = [(loc, move(s), move(e), n) for loc, s, e, n in l["ranges"]]
            l.update(start=b0, end=b1, real_us=round((b1 - b0) / 1000, 3),
                     instrumented_us=round((i1 - i0) / 1000, 3), scale=round(scale, 4),
                     overhead=(i1 - i0) / (b1 - b0) - 1 if b1 > b0 else 0.0, base_stream=k["stream"],
                     grid_match=tuple(k["grid"]) == tuple(l["grid"]), base_grid=k["grid"])
        k["paired"] = l
        grids += not l["grid_match"]
        kept.append(l)
    kept.sort(key=lambda l: l["ordinal"])
    stats = {"base": base["path"], "base kind": base.get("kind", "nsys"),
             "base step": f"{base['index']} of {base['count']}",
             "paired launches": len(kept), "IKET launches without a base launch": dict(unpaired),
             "base TileLang launches without an IKET launch": dict(unmatched_base),
             "paired launches whose grids differ": grids,
             "base kernels (all, TileLang or not)": len(base["kernels"])}
    return kept, stats


def stitch_slices(base_launches, names, locations, slice_paths, step_index, allow_mismatch=False):
    """Stitch instrumented launches from ``slice_paths`` onto ``base_launches``.

    Each slice is a family-sliced IKET run of the same step. Unless ``allow_mismatch`` is
    True, the multiset of ``(name, grid, block)`` launches in the slice's decode step must
    match ``base_launches`` (or be a subset when only a subset of kernels ran). For each
    slice launch that recorded ``ranges``, its string and location indices are merged into
    ``names`` and ``locations``, and its ``lives`` and ``ranges`` are rescaled into the
    corresponding base launch's span."""
    name_to_idx = {s: i for i, s in enumerate(names)}
    next_loc = (max(locations) + 1) if locations else 0
    base_sig = [f"{l['name']} {l['grid']} {l['block']}" for l in base_launches]
    base_counts = Counter(base_sig)
    slice_reports = []

    for spath in slice_paths:
        s_launches, s_names, s_locations, _, _, _ = _read_raw_step(spath, step_index)
        s_sig = [f"{l['name']} {l['grid']} {l['block']}" for l in s_launches]
        s_counts = Counter(s_sig)
        if s_counts != base_counts and not allow_mismatch:
            diff = "\n".join(difflib.unified_diff(
                base_sig, s_sig, fromfile="base", tofile=str(spath), lineterm="", n=2
            ))
            raise SystemExit(
                f"{spath}: step launch sequence differs from base timeline "
                f"({len(s_launches)} vs {len(base_launches)} launches); pass --allow-mismatch to override:\n{diff[:2000]}"
            )
        str_map = {}
        for idx, text in enumerate(s_names):
            if text not in name_to_idx:
                name_to_idx[text] = len(names)
                names.append(text)
            str_map[idx] = name_to_idx[text]
        loc_map = {}
        for loc_idx, entry in s_locations.items():
            loc_map[loc_idx] = next_loc
            locations[next_loc] = entry
            next_loc += 1

        pairs, unpaired, _ = _pair_launches(s_launches, base_launches)
        stitched = 0
        for sl, bl in pairs:
            if not sl["ranges"]:
                continue
            i0, i1, b0, b1 = sl["start"], sl["end"], bl["start"], bl["end"]
            scale = (b1 - b0) / (i1 - i0) if i1 > i0 else 1.0
            move = lambda t, i0=i0, b0=b0, scale=scale: int(b0 + (t - i0) * scale)
            bl["lives"] = [(loc_map.get(loc, loc), move(s), move(e)) for loc, s, e in sl["lives"]]
            bl["ranges"] = [(loc_map.get(loc, loc), move(s), move(e), str_map[n])
                            for loc, s, e, n in sl["ranges"]]
            bl["slice_source"] = str(spath)
            if "real_us" not in bl:
                bl.update(
                    real_us=round((b1 - b0) / 1000, 3),
                    instrumented_us=round((i1 - i0) / 1000, 3),
                    scale=round(scale, 4),
                    overhead=(i1 - i0) / (b1 - b0) - 1 if b1 > b0 else 0.0,
                    base_stream=bl["source"],
                    grid_match=tuple(bl["grid"]) == tuple(sl["grid"]),
                    base_grid=bl["grid"],
                )
            stitched += 1
        slice_reports.append({
            "slice": str(spath),
            "instrumented_launches_stitched": stitched,
            "unpaired": dict(unpaired),
        })
    return base_launches, names, locations, slice_reports


# ---------------------------------------------------------------- multi-rank alignment


def _collective_sync_ts(launch, names):
    """The synchronization timestamp of a collective launch: the end of its peer/flag/spin
    wait range if instrumented, else the launch's end timestamp."""
    peer_ends = []
    for _, _, re_, name_idx in launch["ranges"]:
        rname = names[name_idx] if name_idx < len(names) else ""
        if wait_reason(rname) == "flag or spin (sparknet)":
            peer_ends.append(re_)
    return max(peer_ends) if peer_ends else launch["end"]


def _fit_affine_clock(ref_ts, rank_ts):
    """Fit t_ref = alpha + beta * (t_rank - t_rank[0]) from collective sync points.
    Returns (t0, alpha, beta, rms_us, max_us)."""
    if not ref_ts or not rank_ts or len(ref_ts) != len(rank_ts):
        return 0, 0.0, 1.0, 0.0, 0.0
    t0 = rank_ts[0]
    n = len(ref_ts)
    if n == 1:
        return t0, float(ref_ts[0]), 1.0, 0.0, 0.0
    xs = [float(t - t0) for t in rank_ts]
    ys = [float(y) for y in ref_ts]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        beta = 1.0
    else:
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        beta = sxy / sxx
        # Guard against pathological drift (>5% clock skew) on degenerate traces.
        if abs(beta - 1.0) > 0.05:
            beta = 1.0
    alpha = my - beta * mx
    residuals_us = [abs((alpha + beta * x) - y) / 1000.0 for x, y in zip(xs, ys)]
    rms_us = math.sqrt(sum(r * r for r in residuals_us) / n)
    max_us = max(residuals_us)
    return t0, alpha, beta, rms_us, max_us


def align_ranks(rank_specs, step_index, layer):
    """Load and clock-align multiple rank traces specified as ``NODE=PATH``.

    Rank 0 is the reference clock. Collective launches (``oneshot_allreduce*``,
    ``oneshot_allgather*``, ``ncclDevKernel*``) provide the hardware rendezvous points
    across ranks. Each rank's timestamps are mapped into rank 0's clock frame via ordinary
    least squares (offset + linear drift)."""
    loaded = []
    for spec in rank_specs:
        if "=" not in spec:
            raise SystemExit(f"--rank expects NODE=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        launches, names, locations, idx, count, _ = _read_raw_step(path, step_index)
        collectives = [l for l in launches if is_collective(l["name"])]
        sync_points = [_collective_sync_ts(l, names) for l in collectives]
        loaded.append({
            "label": label, "path": path, "launches": launches, "names": names,
            "locations": locations, "step": idx, "count": count,
            "collectives": collectives, "sync_points": sync_points,
        })

    ref = loaded[0]
    ref_sync = ref["sync_points"]
    alignment_report = []

    for r_idx, rank_data in enumerate(loaded):
        if r_idx == 0:
            t0, alpha, beta, rms_us, max_us = (
                (ref_sync[0] if ref_sync else rank_data["launches"][0]["start"]),
                float(ref_sync[0] if ref_sync else rank_data["launches"][0]["start"]),
                1.0, 0.0, 0.0,
            )
        elif ref_sync and len(rank_data["sync_points"]) == len(ref_sync):
            t0, alpha, beta, rms_us, max_us = _fit_affine_clock(ref_sync, rank_data["sync_points"])
        else:
            t0 = rank_data["launches"][0]["start"]
            alpha = float(ref["launches"][0]["start"])
            beta, rms_us, max_us = 1.0, 0.0, 0.0

        move = lambda t, t0=t0, alpha=alpha, beta=beta: int(round(alpha + beta * (t - t0)))
        for l in rank_data["launches"]:
            l["start"] = move(l["start"])
            l["end"] = move(l["end"])
            l["lives"] = [(loc, move(s), move(e)) for loc, s, e in l["lives"]]
            l["ranges"] = [(loc, move(s), move(e), n) for loc, s, e, n in l["ranges"]]

        drift_ppm = (beta - 1.0) * 1e6
        offset_us = (alpha - t0) / 1000.0
        rank_data["clock_fit"] = {
            "rank": r_idx, "node": rank_data["label"],
            "collectives_matched": len(rank_data["sync_points"]),
            "offset_us": round(offset_us, 3), "drift_ppm": round(drift_ppm, 3),
            "residual_rms_us": round(rms_us, 3), "residual_max_us": round(max_us, 3),
        }
        alignment_report.append(rank_data["clock_fit"])

    # Compute per-collective arrival spread, late rank, and per-rank jitter wait vs fabric time.
    collective_events = []
    n_coll = min(len(r["collectives"]) for r in loaded)
    for c_idx in range(n_coll):
        items = [(r["label"], r["collectives"][c_idx]) for r in loaded]
        arrivals = {label: l["start"] for label, l in items}
        exits = {label: l["end"] for label, l in items}
        late_label = max(arrivals, key=lambda k: arrivals[k])
        late_arrival = arrivals[late_label]
        first_arrival = min(arrivals.values())
        last_exit = max(exits.values())
        spread_us = round((late_arrival - first_arrival) / 1000.0, 3)
        fabric_us = round(max(0, last_exit - late_arrival) / 1000.0, 3)
        for label, l in items:
            l["collective_ordinal"] = c_idx
            l["late_rank"] = late_label
            l["jitter_wait_us"] = round((late_arrival - l["start"]) / 1000.0, 3)
            l["fabric_us"] = round(max(0, l["end"] - late_arrival) / 1000.0, 3)
            l["arrival_spread_us"] = spread_us
        collective_events.append({
            "ordinal": c_idx, "name": items[0][1]["name"],
            "layer": items[0][1].get("layer") or "-",
            "start": first_arrival, "late_arrival": late_arrival, "end": last_exit,
            "late_rank": late_label, "arrival_spread_us": spread_us, "fabric_us": fabric_us,
            "arrivals": arrivals,
        })

    if layer is not None:
        for r in loaded:
            r["launches"] = [l for l in r["launches"] if l["layer"] == layer]
            if not r["launches"]:
                raise SystemExit(f"{r['label']}: no launch in layer {layer}")
        collective_events = [c for c in collective_events if c["layer"] == layer]

    return loaded, alignment_report, collective_events


# ---------------------------------------------------------------- writing


def sweep(intervals):
    """(time, value) change points of a sum of (start, end, weight) intervals."""
    deltas = defaultdict(float)
    for start, end, weight in intervals:
        deltas[start] += weight
        deltas[end] -= weight
    value = 0.0
    for t in sorted(deltas):
        value += deltas[t]
        yield t, value


def _write_gpu_tracks(w, pid, title, scope, launches, names, locations, args,
                      base=None, rebased=None, begin=None, end=None, clock_fit=None):
    """Render one GPU process hierarchy (1 model, 2 kernel launches, 3 counters, 4 SMs,
    and optional 5 base streams) into ``w``."""
    if begin is None:
        begin = min(l["start"] for l in launches)
    if end is None:
        end = max(l["end"] for l in launches)
    gpu = w.track(f"{title}: {scope}", pid=pid, explicit=True)
    model = w.track("1 model", parent=gpu, rank=0, explicit=True)
    t_step = w.track("1 step", parent=model, rank=0)
    t_seg = w.track("2 segment: draft / verify", parent=model, rank=1)
    t_layer = w.track("3 layer", parent=model, rank=2)
    sub_parent = w.track("4 sublayer", parent=model, rank=3, explicit=True)
    launch_parent = w.track("2 kernel launches", parent=gpu, rank=1, explicit=True)
    counters = w.track("3 counters", parent=gpu, rank=2, explicit=True)
    sms_parent = w.track("4 SMs", parent=gpu, rank=3, explicit=True)
    info = {
        "covers": "6 target layers, the drafter's 3 and the head",
        "times": "IKET's own (%globaltimer), including its instrumentation",
        "step": scope,
    }
    if clock_fit is not None:
        info.update({f"clock: {k}": v for k, v in clock_fit.items()})
    if rebased is not None:
        if base is not None:
            info["times"] = (
                f"base timeline from {base['path']} ({base.get('kind', 'nsys')}); "
                "instrumented launches rescaled into their base launch spans"
            )
            info["overhead flag"] = f"instrumented / real - 1 above {OVERHEAD_LIMIT:.0%}: phase proportions distorted"
        info.update({f"base: {k}": (json.dumps(v) if isinstance(v, (dict, list)) else v)
                     for k, v in rebased.items()})
        flagged = [l for l in launches if l.get("overhead", 0) > OVERHEAD_LIMIT]
        info["launches over the overhead limit"] = len(flagged)
    w.slice(t_step, begin, end, scope, "step", info)
    w.instant(t_step, begin, "trace info", info)

    # model: segments, layers, sublayers.
    seg_runs = []
    for l in sorted(launches, key=lambda l: l["ordinal"]):
        if seg_runs and seg_runs[-1][0] == l["segment"]:
            seg_runs[-1][1].append(l)
        else:
            seg_runs.append((l["segment"], [l]))
    for name, items in seg_runs:
        w.slice(t_seg, min(l["start"] for l in items), max(l["end"] for l in items), name,
                "segment", {"launches": len(items)})
    by_layer = defaultdict(list)
    for l in launches:
        if l["layer"]:
            by_layer[l["layer"]].append(l)
    layer_lanes = Lanes(w, t_layer, "layer lane")
    for name, items in sorted(by_layer.items(), key=lambda kv: min(l["start"] for l in kv[1])):
        s, e = min(l["start"] for l in items), max(l["end"] for l in items)
        track, _ = layer_lanes.place(s, e)
        w.slice(track, s, e, f"layer {name}", "layer", {"launches": len(items)})
    runs = []
    for l in launches:
        label = sublayer(l["name"])
        if runs and runs[-1]["label"] == label and runs[-1]["layer"] == l["layer"]:
            runs[-1]["items"].append(l)
        else:
            runs.append({"label": label, "layer": l["layer"], "items": [l]})
    spans = sorted(((min(x["start"] for x in r["items"]), max(x["end"] for x in r["items"]), r) for r in runs),
                   key=lambda s: s[0])
    sub_lanes = Lanes(w, sub_parent, "sublayer lane")
    for s, e, r in spans:
        track, _ = sub_lanes.place(s, e)
        w.slice(track, s, e, r["label"], "sublayer", {"layer": r["layer"] or "-", "launches": len(r["items"])})

    # kernel launches.
    launch_lanes = Lanes(w, launch_parent, "launch lane")
    head_launches = [l for l in launches if l["name"] == HEAD]
    for l in sorted(launches, key=lambda l: l["start"]):
        track, _ = launch_lanes.place(l["start"], l["end"])
        note = ("the target's verification head: the launch the engine grid's dummy step does not run"
                if l["name"] == HEAD and l["source"] == "eager"
                else "the drafter's head" if l["name"] == HEAD else "")
        args_ = {"grid": str(l["grid"]), "block": str(l["block"]), "launch #": l["ordinal"],
                 "source": l["source"], "layer": l["layer"] or "-", "sublayer": sublayer(l["name"]),
                 "warps": len(l["lives"]), "traced warps": len({r[0] for r in l["ranges"]})}
        if note:
            args_["note"] = note
        if "slice_source" in l:
            args_["slice_source"] = l["slice_source"]
        if "late_rank" in l:
            args_.update({
                "late_rank": l["late_rank"],
                "jitter_wait_us": l["jitter_wait_us"],
                "fabric_us": l["fabric_us"],
                "arrival_spread_us": l["arrival_spread_us"],
            })
        if "real_us" in l:
            args_.update({"real_us": l["real_us"], "instrumented_us": l["instrumented_us"], "scale": l["scale"],
                          "overhead %": round(100 * l["overhead"], 1), "base stream": str(l["base_stream"])})
            if l["overhead"] > OVERHEAD_LIMIT:
                args_["distorted"] = f"overhead above {OVERHEAD_LIMIT:.0%}: phase proportions unreliable"
            if not l["grid_match"]:
                args_["grid mismatch"] = f"base grid {l['base_grid']}: a different launch shape"
        w.slice(track, l["start"], l["end"], l["name"].replace("_kernel", ""), sublayer(l["name"]), args_)
    for l in head_launches:
        w.instant(t_seg, l["start"], "head 1: the target's verification head (not in the grid's dummy step)"
                  if l["source"] == "eager" else "head 2: the drafter's head (in its graph)")
    by_layer_order = defaultdict(list)
    for l in launches:
        by_layer_order[(l["source"], l["layer"])].append(l)
    for items in by_layer_order.values():
        for i, x in enumerate(items):
            if x["name"] == COMBINE:
                w.instant(t_layer, x["end"], "all-reduce point: MoE output")
            if x["name"] == "gemm_splitk_reduce_cast_kernel":
                nxt = next((y for y in items[i + 1 :] if y["name"] == "gemm_mxfp8_kernel"), None)
                if nxt is not None:
                    w.instant(t_layer, nxt["end"], "all-reduce point: attention output")

    # The base run's streams: every kernel, TileLang or not, as it ran without IKET.
    if base is not None:
        streams_parent = w.track(f"5 streams (base {base.get('kind', 'nsys')} run)", parent=gpu, rank=4, explicit=True)
        lo, hi = begin, end
        by_stream = defaultdict(list)
        for k in base["kernels"]:
            if k["end"] > lo and k["start"] < hi:
                by_stream[k["stream"]].append(k)
        order = sorted(by_stream, key=lambda st: min(k["start"] for k in by_stream[st]))
        for rank, stream in enumerate(order):
            items = by_stream[stream]
            top = ", ".join(n.replace("_kernel", "") for n, _ in Counter(k["name"] for k in items).most_common(3))
            kind = "graph nodes" if sum(k["graph"] for k in items) * 2 > len(items) else "eager"
            lanes = Lanes(w, w.track(f"stream {stream} ({kind}): {top}", parent=streams_parent, rank=rank),
                          "lane")
            for k in items:
                track, _ = lanes.place(k["start"], k["end"])
                args_ = {"real_us": round((k["end"] - k["start"]) / 1000, 3), "grid": str(k["grid"]),
                         "in a CUDA graph": k["graph"],
                         "TileLang (IKET)": "paired" in k}
                if "paired" in k:
                    args_["instrumented_us"] = k["paired"]["instrumented_us"]
                w.slice(track, k["start"], k["end"], k["name"].replace("_kernel", ""), "base kernel", args_)

    # SMs: CTA slots and traced warps.
    ctas = []
    for l in launches:
        group = defaultdict(list)
        for loc, s, e in l["lives"]:
            sm, cta, warp = locations.get(loc, (-1, (-1, -1, -1), -1))
            group[(sm, cta)].append((s, e, warp, loc))
        traced = {r[0] for r in l["ranges"]}
        for (sm, cta), warps in group.items():
            ctas.append((min(s for s, _, _, _ in warps), max(e for _, e, _, _ in warps), sm, cta, l,
                         [x for x in warps if x[3] in traced], len(warps)))
    ctas.sort(key=lambda c: c[0])
    sm_tracks, slot_lanes, warp_tracks = {}, {}, {}
    n_sm = max((c[2] for c in ctas), default=-1) + 1
    for sm in range(n_sm):
        sm_tracks[sm] = w.track(f"SM {sm:02d}", parent=sms_parent, rank=sm, explicit=True)
        slot_lanes[sm] = Lanes(w, sm_tracks[sm], "slot")
    traced_kept = 0
    wait_intervals = defaultdict(list)
    alive = []
    for start, stop, sm, cta, l, traced, _ in ctas:
        if sm < 0:
            continue
        track, slot = slot_lanes[sm].place(start, stop)
        w.slice(track, start, stop, l["name"].replace("_kernel", ""), "cta",
                {"CTA": str(cta), "launch #": l["ordinal"], "layer": l["layer"] or "-",
                 "sublayer": sublayer(l["name"]), "traced warps": len(traced)})
        if args.max_traced_warps:
            traced = sorted(traced, key=lambda x: x[2])[: args.max_traced_warps]
        for s, e, warp, loc in traced:
            key = (sm, slot, warp)
            if key not in warp_tracks:
                warp_tracks[key] = w.track(f"warp {warp:02d}", parent=track, rank=warp)
            wt = warp_tracks[key]
            w.slice(wt, s, e, f"warp {warp}", "warp", {"CTA": str(cta), "kernel": l["name"]})
            alive.append((s, e, 1.0))
            last = None
            for loc2, rs, re_, name_idx in sorted((r for r in l["ranges"] if r[0] == loc), key=lambda r: r[1]):
                if last is not None and rs < last:
                    rs = last
                rs, re_ = max(rs, s), min(re_, e)
                if re_ < rs:
                    continue
                name = names[name_idx] if name_idx < len(names) else str(name_idx)
                phase = name.split(".")[0].split("#")[0].split("*")[0]
                w.slice(wt, rs, re_, name, phase, {"us": round((re_ - rs) / 1000, 3)})
                reason = wait_reason(name)
                if reason:
                    wait_intervals[reason].append((rs, re_, 1.0))
                last = re_
            traced_kept += 1

    # counters.
    c_ctas = w.track("CTAs resident", parent=counters, counter=True, rank=0)
    for t, v in sweep((c[0], c[1], 1.0) for c in ctas):
        w.counter(c_ctas, t, v)
    c_warps = w.track("warps resident (each CTA's warps over its lifetime)", parent=counters, counter=True, rank=1)
    for t, v in sweep((c[0], c[1], float(c[6])) for c in ctas):
        w.counter(c_warps, t, v)
    alive_points = list(sweep(alive))
    for rank, (reason, intervals) in enumerate(sorted(wait_intervals.items()), start=2):
        track = w.track(f"traced warps waiting: {reason} (share)", parent=counters, counter=True, rank=rank)
        waits = list(sweep(intervals))
        merged = sorted({t for t, _ in waits} | {t for t, _ in alive_points})
        wi = ai = 0
        wv = av = 0.0
        for t in merged:
            while wi < len(waits) and waits[wi][0] <= t:
                wv = waits[wi][1]
                wi += 1
            while ai < len(alive_points) and alive_points[ai][0] <= t:
                av = alive_points[ai][1]
                ai += 1
            w.counter(track, t, wv / av if av else 0.0)

    return {"launches": len(launches), "CTAs": len(ctas), "SMs": n_sm,
            "traced warps": traced_kept, "begin": begin, "end": end}


def write_multirank(args):
    loaded, alignment_report, collective_events = align_ranks(args.rank, args.step, args.layer)
    w = Writer(args.out)
    scope = (f"layer {args.layer}" if args.layer
             else f"decode step {loaded[0]['step']} of {loaded[0]['count']}")
    coll_root = w.track(
        f"0 collective alignment ({len(loaded)} ranks): {scope}", pid=1, explicit=True
    )
    t_window = w.track("1 collective span (first arrival -> exit)", parent=coll_root, rank=0)
    t_jitter = w.track("2 arrival jitter (first arrival -> late peer arrival)", parent=coll_root, rank=1)
    t_fabric = w.track("3 fabric transport (late peer arrival -> exit)", parent=coll_root, rank=2)
    for ev in collective_events:
        c_args = {
            "collective #": ev["ordinal"], "layer": ev["layer"],
            "late_rank": ev["late_rank"],
            "arrival_spread_us": ev["arrival_spread_us"],
            "fabric_us": ev["fabric_us"],
        }
        w.slice(t_window, ev["start"], ev["end"],
                f"{ev['name'].replace('_kernel', '')} (late: {ev['late_rank']})", "collective", c_args)
        if ev["late_arrival"] > ev["start"]:
            w.slice(t_jitter, ev["start"], ev["late_arrival"],
                    f"wait for {ev['late_rank']} ({ev['arrival_spread_us']} us)", "jitter", c_args)
        if ev["end"] > ev["late_arrival"]:
            w.slice(t_fabric, ev["late_arrival"], ev["end"],
                    f"fabric ({ev['fabric_us']} us)", "fabric", c_args)
        w.instant(t_window, ev["late_arrival"], f"late peer {ev['late_rank']} arrives", c_args)

    total_launches = 0
    total_ctas = 0
    total_warps = 0
    max_sms = 0
    global_begin = min(l["start"] for r in loaded for l in r["launches"])
    global_end = max(l["end"] for r in loaded for l in r["launches"])
    for r_idx, r in enumerate(loaded):
        stats = _write_gpu_tracks(
            w, pid=r_idx + 2, title=f"GPU rank {r_idx} ({r['label']})", scope=scope,
            launches=r["launches"], names=r["names"], locations=r["locations"],
            args=args, begin=global_begin, end=global_end, clock_fit=r["clock_fit"],
        )
        total_launches += stats["launches"]
        total_ctas += stats["CTAs"]
        total_warps += stats["traced warps"]
        max_sms = max(max_sms, stats["SMs"])

    w.close()
    size = Path(args.out).stat().st_size
    summary = {
        "out": str(args.out), "bytes": size, "packets": w.packets,
        "ranks": len(loaded), "collectives_aligned": len(collective_events),
        "rank_alignment": alignment_report,
        "launches": total_launches, "CTAs": total_ctas, "SMs": max_sms,
        "traced warps": total_warps, "scope": scope,
        "span us": round((global_end - global_begin) / 1000, 1),
    }
    print(summary)
    return summary


def write(args):
    if getattr(args, "rank", None):
        return write_multirank(args)
    if not args.records:
        raise SystemExit("records path is required unless --rank is provided")
    if getattr(args, "base_iket", None) and getattr(args, "base_nsys", None):
        raise SystemExit("--base-iket and --base-nsys are mutually exclusive")
    base = (iket_base(args.base_iket, args.step) if getattr(args, "base_iket", None)
            else nsys_base(args.base_nsys, args.step) if getattr(args, "base_nsys", None)
            else None)
    launches, names, locations, index, count, rebased = load_step(
        args.records, args.step, args.layer, base,
        slice_paths=getattr(args, "slice", None),
        allow_mismatch=getattr(args, "allow_mismatch", False),
    )
    w = Writer(args.out)
    begin = min(l["start"] for l in launches)
    end = max(l["end"] for l in launches)
    if base is not None and args.layer is None:
        begin, end = base["begin"], base["end"]
    scope = f"layer {args.layer}" if args.layer else f"decode step {index} of {count}"
    stats = _write_gpu_tracks(
        w, pid=1, title=f"GPU 0: {args.title or 'harness'}", scope=scope,
        launches=launches, names=names, locations=locations, args=args,
        base=base, rebased=rebased, begin=begin, end=end,
    )
    w.close()
    size = Path(args.out).stat().st_size
    summary = {"out": str(args.out), "bytes": size, "packets": w.packets,
               "launches": stats["launches"], "CTAs": stats["CTAs"], "SMs": stats["SMs"],
               "traced warps": stats["traced warps"], "scope": scope,
               "span us": round((end - begin) / 1000, 1)}
    if rebased is not None:
        summary["rebased"] = rebased
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="?", help="primary run-iket decoded JSON (or omit when using --rank)")
    parser.add_argument("out", help="output .pftrace path")
    parser.add_argument("--step", type=int, help="decode step index (default: the middle one)")
    parser.add_argument("--layer", help="one layer only: T0..T5 (target), D0..D2 (drafter)")
    parser.add_argument("--title", help="names the GPU process")
    parser.add_argument("--max-traced-warps", type=int, default=0,
                        help="traced warps kept a CTA (0: all)")
    parser.add_argument("--base-iket",
                        help="base timeline from a run-iket --warp-trace (IKET-off) decoded JSON; "
                             "all kernels keep exact warp lifetimes and instrumented launches rescale into them")
    parser.add_argument("--base-nsys",
                        help="base timeline from an nsys SQLite export of the same step without IKET; "
                             "IKET launches are rescaled into its kernels' spans")
    parser.add_argument("--slice", nargs="*", default=[],
                        help="additional family-sliced IKET decoded JSON files to stitch onto the base timeline")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="allow stitching --slice runs whose step launch sequence differs from the base")
    parser.add_argument("--rank", nargs="*", default=[],
                        help="multi-rank traces as NODE=DECODED.json (aligns clocks via collective rendezvous)")
    write(parser.parse_args())


if __name__ == "__main__":
    main()
