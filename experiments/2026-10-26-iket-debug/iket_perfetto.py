"""A DS4.1 decode step from run-iket records, as a streamed Perfetto trace.

usage: iket_perfetto.py DECODED_JSON OUT.pftrace [--step K] [--layer L] [--title TEXT]
                        [--max-traced-warps N]

Reads a run-iket ``iket.decoded_results.json`` in streaming passes. ``Records`` indexes the
records, then the step's records and the warp locations they use are decoded. Perfetto ``TracePacket``s are written as they are made: interned names, slices
as begin and end events, counters as values. Nothing holds the input file whole. Open
the output in ui.perfetto.dev. For a file too large for a browser tab, run
``trace_processor_shell --httpd OUT.pftrace`` on the Mac and open ui.perfetto.dev, which
attaches to it.

What the trace covers:
- **The step:** what IKET recorded, which is the TileLang kernels only. ATen and Triton
  kernels, copies and collectives that are not TileLang are absent.
- **The harness** (``harness/iket_step.py``): six target layers, the drafter's three and
  the head, at one rank's shapes on one GPU. Not the model's 43 layers.
- **Times** are IKET's own (``%globaltimer``), so they include its instrumentation: GEMMs
  +26-40%, tiny kernels about 3x.
- **With ``--base-nsys``**, the timeline is the base run's: an nsys export of the same step
  without IKET. This is the interim stand-in for a ``run-iket --warp-trace`` base run.
  - Each IKET launch pairs with the base launch at the same ordinal: its place among
    the step's launches of one kernel, grid and block, else of one kernel name.
  - The IKET launch's CTAs, warps and phases are rescaled into the base launch's span.
    Kernel spans and overlap are the base's.
  - Each launch carries ``real_us`` (the base's duration), ``instrumented_us`` (IKET's
    span) and the scale.
  - It is flagged when the overhead tops ``OVERHEAD_LIMIT`` (50%) or the paired grids
    differ.
  - The base's own kernels, ATen and Triton included, get a "5 streams" group.

Tracks, in one GPU process:
- **model: step, segment, layer, sublayer.**
  - Segments are graph replays and eager launches: target, drafter and verification.
  - Layers are cut at each MoE combine.
  - Sublayers come from kernel names (mHC, attention front, indexer and compressor,
    sparse MLA, attention output, MoE routing, routed experts, shared expert, combine,
    head, sampler, inputs, L2 prefetch).
  - Overlapping spans go to lanes.
- **kernel launches:** each launch from its first warp's start to its last warp's end, in
  lanes where launches overlap (graph branches, the L2 prefetch side stream).
- **counters:**
  - CTAs resident;
  - warps resident (each CTA's warps over its lifetime);
  - the share of traced warps waiting, by reason: cp.async (weight and activation
    loads), mbarrier (TMA), block barrier, PDL grid dependency, flag or spin
    (sparknet), other.
- **SM 00..NN:** each CTA's lifetime on the SM it ran on, in slots where CTAs overlap.
  Each slot's child tracks are its traced warps (``warp w``). Their phases (``load``,
  ``wait``, ``compute``, ``store``, ``signal``) are named ranges nested in the warp's
  lifetime.
- **markers:**
  - the step's start and end;
  - the target's verification head (eager: the launch the engine grid's dummy step
    does not run) and the drafter's head (in its graph);
  - each layer's all-reduce points, after the attention output projection and after
    the MoE combine. One GPU runs no all-reduce; the markers show where a TP rank would
    wait.

Sampling (the run's):
- ``VLLM_DS41_IKET_CTAS=4`` traces four CTAs a launch. Every warp's lifetime is
  recorded, so CTA slices and counters cover all CTAs; phases cover the traced ones.
- ``VLLM_DS41_IKET_ITERS=1`` traces each loop's middle iteration, and its range name
  carries ``*N``, the factor to the whole loop.
- ``--max-traced-warps`` keeps at most N traced warps a launch, the first warp of each
  CTA first (0: all).
"""

import argparse
import difflib
import json
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


def load_step(path, index, layer, base=None):
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
        raise SystemExit("no decode step")
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
    rebased = None
    if base is not None:
        launches, rebased = rebase(launches, base)
    if layer is not None:
        launches = [l for l in launches if l["layer"] == layer]
        if not launches:
            raise SystemExit(f"no launch in layer {layer}")
    wanted = {loc for l in launches for loc, _, _ in l["lives"]}
    print(f"{path}: {len(items)} launches, {len(steps)} decode steps; step {index}: "
          f"{len(launches)} launches, {len(wanted)} warp locations", file=sys.stderr, flush=True)
    locations = {i: (e["smId"], tuple(e["ctaId"]), e["warpId"]) for i, e in records.locations(wanted).items()}
    return launches, records.strings(), locations, index, len(steps), rebased


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


# ---------------------------------------------------------------- an nsys base timeline


def nsys_base(path, index):
    """Decode step ``index`` of an nsys export (an IKET-off run): every kernel with its
    stream, grid and graph membership, and the step's span. The interim stand-in for a
    ``run-iket --warp-trace`` base run."""
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
    return {"path": str(path), "kernels": step, "begin": step[0]["start"], "end": end,
            "index": index, "count": len(steps)}


def rebase(launches, base):
    """Pairs each IKET launch with the base run's launch at the same ordinal, then moves its
    warps, CTAs and ranges into the base launch's span: t -> base start + (t - IKET start)
    x (base duration / IKET duration). Unpaired IKET launches leave the timeline; the
    counts say how many.

    The ordinal is a launch's place among the step's launches of the same kernel, grid and
    block, in start order. Graph branches run concurrently, so the order between different
    kernels can change from run to run, while one kernel shape's launches keep theirs.
    Shapes whose launch counts differ between the runs then pair by kernel name alone,
    if those counts match; otherwise they stay unpaired."""
    names = {l["name"] for l in launches}
    tl = [k for k in base["kernels"] if k["name"] in names]
    pairs = []
    left_iket, left_base = [], []
    by_shape_i, by_shape_b = defaultdict(list), defaultdict(list)
    for l in sorted(launches, key=lambda l: l["start"]):
        by_shape_i[(l["name"], tuple(l["grid"]), tuple(l["block"]))].append(l)
    for k in tl:
        by_shape_b[(k["name"], tuple(k["grid"]), tuple(k["block"]))].append(k)
    for key, items in by_shape_i.items():
        other = by_shape_b.pop(key, [])
        if len(other) == len(items):
            pairs += list(zip(items, other))
        else:
            left_iket += items
            left_base += other
    for items in by_shape_b.values():
        left_base += items
    by_name_i, by_name_b = defaultdict(list), defaultdict(list)
    for l in left_iket:
        by_name_i[l["name"]].append(l)
    for k in left_base:
        by_name_b[k["name"]].append(k)
    unpaired, unmatched_base = Counter(), Counter()
    for name, items in by_name_i.items():
        other = sorted(by_name_b.pop(name, []), key=lambda k: k["start"])
        items = sorted(items, key=lambda l: l["start"])
        if len(other) == len(items):
            pairs += list(zip(items, other))
        else:
            unpaired[name] += len(items)
            unmatched_base[name] += len(other)
    for name, items in by_name_b.items():
        unmatched_base[name] += len(items)
    kept, grids = [], 0
    for l, k in pairs:
        i0, i1, b0, b1 = l["start"], l["end"], k["start"], k["end"]
        scale = (b1 - b0) / (i1 - i0) if i1 > i0 else 1.0
        move = lambda t, i0=i0, b0=b0, scale=scale: int(b0 + (t - i0) * scale)
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
    stats = {"base": base["path"], "base step": f"{base['index']} of {base['count']}",
             "paired launches": len(kept), "IKET launches without a base launch": dict(unpaired),
             "base TileLang launches without an IKET launch": dict(unmatched_base),
             "paired launches whose grids differ": grids,
             "base kernels (all, TileLang or not)": len(base["kernels"])}
    return kept, stats


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


def write(args):
    base = nsys_base(args.base_nsys, args.step) if args.base_nsys else None
    launches, names, locations, index, count, rebased = load_step(args.records, args.step, args.layer, base)
    w = Writer(args.out)
    begin = min(l["start"] for l in launches)
    end = max(l["end"] for l in launches)
    if base is not None and args.layer is None:
        begin, end = base["begin"], base["end"]
    scope = f"layer {args.layer}" if args.layer else f"decode step {index} of {count}"
    gpu = w.track(f"GPU 0: {args.title or 'harness'}: {scope}", pid=1, explicit=True)
    model = w.track("1 model", parent=gpu, rank=0, explicit=True)
    t_step = w.track("1 step", parent=model, rank=0)
    t_seg = w.track("2 segment: draft / verify", parent=model, rank=1)
    t_layer = w.track("3 layer", parent=model, rank=2)
    sub_parent = w.track("4 sublayer", parent=model, rank=3, explicit=True)
    launch_parent = w.track("2 kernel launches", parent=gpu, rank=1, explicit=True)
    counters = w.track("3 counters", parent=gpu, rank=2, explicit=True)
    sms_parent = w.track("4 SMs", parent=gpu, rank=3, explicit=True)
    info = {
        "covers": "the harness: 6 target layers, the drafter's 3 and the head at one rank's shapes "
                  "(not the model's 43 layers); TileLang kernels only (ATen, Triton absent)",
        "times": "IKET's own (%globaltimer), including its instrumentation",
        "sampling": "4 traced CTAs a launch (VLLM_DS41_IKET_CTAS=4), loops at their middle iteration "
                    "(*N: the factor to the whole loop); every warp's lifetime is recorded",
        "records": str(args.records), "step": scope,
    }
    if rebased is not None:
        info["times"] = (f"INTERIM: kernel spans and overlap from {base['path']} (an nsys export of an "
                         "IKET-off run, standing in for a run-iket --warp-trace base run); each IKET "
                         "launch's CTAs, warps and phases are rescaled into its base launch's span")
        info["overhead flag"] = f"instrumented / real - 1 above {OVERHEAD_LIMIT:.0%}: phase proportions distorted"
        info.update({f"base: {k}": (json.dumps(v) if isinstance(v, dict) else v) for k, v in rebased.items()})
        flagged = [l for l in launches if l.get("overhead", 0) > OVERHEAD_LIMIT]
        info["launches over the overhead limit"] = len(flagged)
    w.slice(t_step, begin, end, scope, "step", info)
    w.instant(t_step, begin, "trace info", info)

    # model: segments (named from the whole step, before a --layer filter), layers, sublayers.
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
        if "real_us" in l:
            args_.update({"real_us": l["real_us"], "instrumented_us": l["instrumented_us"], "scale": l["scale"],
                          "overhead %": round(100 * l["overhead"], 1), "base stream": l["base_stream"]})
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
        names_seq = [x["name"] for x in items]
        for i, x in enumerate(items):
            if x["name"] == COMBINE:
                w.instant(t_layer, x["end"], "all-reduce point: MoE output (a TP rank waits here; none runs on one GPU)")
            if x["name"] == "gemm_splitk_reduce_cast_kernel":
                nxt = next((y for y in items[i + 1 :] if y["name"] == "gemm_mxfp8_kernel"), None)
                if nxt is not None:
                    w.instant(t_layer, nxt["end"],
                              "all-reduce point: attention output (a TP rank waits here; none runs on one GPU)")

    # The base run's streams: every kernel, TileLang or not, as it ran without IKET.
    if base is not None:
        streams_parent = w.track("5 streams (base run)", parent=gpu, rank=4, explicit=True)
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
    w.close()
    size = Path(args.out).stat().st_size
    summary = {"out": str(args.out), "bytes": size, "packets": w.packets, "launches": len(launches),
               "CTAs": len(ctas), "SMs": n_sm, "traced warps": traced_kept, "scope": scope,
               "span us": round((end - begin) / 1000, 1)}
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("records")
    parser.add_argument("out")
    parser.add_argument("--step", type=int, help="decode step index (default: the middle one)")
    parser.add_argument("--layer", help="one layer only: T0..T5 (target), D0..D2 (drafter)")
    parser.add_argument("--title", help="names the GPU process")
    parser.add_argument("--max-traced-warps", type=int, default=0,
                        help="traced warps kept a CTA (0: all)")
    parser.add_argument("--base-nsys",
                        help="interim base timeline: an nsys export of the same step without IKET; "
                             "IKET launches are rescaled into its kernels' spans")
    write(parser.parse_args())


if __name__ == "__main__":
    main()
