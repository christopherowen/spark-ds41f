#!/usr/bin/env python3
"""Which shared-memory loads can still be in flight at each barrier arrive?

usage: sass_pending.py file.sass [...]   (cuobjdump -sass output, sm_90+ encoding)

Decodes each instruction's scoreboard control bits (write barrier, wait mask)
and runs a forward dataflow over the control-flow graph: a generic-proxy shared
load (LDS*, LDSM*) is pending from its issue until an instruction waits on its
write scoreboard (wait mask bit, or DEPBAR.LE SBk, 0). At every mbarrier arrive
(SYNCS.ARRIVE*), CTA barrier (BAR.SYNC / BAR.ARV / BAR.RED) and async-proxy
shared write issue (UTMALDG, UBLKCP), prints the shared loads that may still be
pending there, and whether a MEMBAR / FENCE.VIEW.ASYNC.S lies on the path since
the load (a fence completes the loads before the arrive, so they are reported as
fenced, not pending).
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]{16}) \*/")
HI = re.compile(r"^\s+/\* (0x[0-9a-f]{16}) \*/")
FUNC = re.compile(r"^\s+Function : (\S+)")


def parse(path):
    funcs, cur, pending = {}, None, None
    for line in open(path):
        m = FUNC.match(line)
        if m:
            cur = funcs.setdefault(m.group(1), [])
            continue
        m = LINE.match(line)
        if m and cur is not None:
            pending = [int(m.group(1), 16), m.group(2), int(m.group(3), 16), None]
            cur.append(pending)
            continue
        m = HI.match(line)
        if m and pending is not None and pending[3] is None:
            pending[3] = int(m.group(1), 16)
    return funcs


def ctrl(hi):
    return {
        "wait": (hi >> 52) & 0x3F,
        "rd": (hi >> 49) & 0x7,
        "wr": (hi >> 46) & 0x7,
    }


def opcode(text):
    t = re.sub(r"^@!?U?P[T0-9]+\s+", "", text)
    return t.split()[0], t


def analyze(name, insts):
    addr_index = {a: i for i, (a, *_ ) in enumerate(insts)}
    n = len(insts)
    succ = [[] for _ in range(n)]
    for i, (a, text, lo, hi) in enumerate(insts):
        op, body = opcode(text)
        predicated = text.startswith("@")
        tgt = re.search(r"0x([0-9a-f]+)\s*$", body)
        if op.startswith("BRA") or op.startswith("BRX"):
            if tgt and int(tgt.group(1), 16) in addr_index:
                succ[i].append(addr_index[int(tgt.group(1), 16)])
            # conditional (predicate or BRA.U UP / BRA.DIV) falls through too
            if predicated or "," in body or op.startswith("BRX"):
                if i + 1 < n:
                    succ[i].append(i + 1)
        elif op in ("EXIT", "RET.REL.NODEC", "BREAK.RELIABLE") and not predicated:
            pass
        else:
            if op.startswith("CALL") and tgt and int(tgt.group(1), 16) in addr_index:
                succ[i].append(addr_index[int(tgt.group(1), 16)])
            if i + 1 < n:
                succ[i].append(i + 1)
    # state: frozenset of (load_index, sb, fenced, earlier) where `earlier` is
    # the frozenset of shared loads pending when this one issued. Shared loads
    # complete in order (ptxas relies on it: it gives a scoreboard only to the
    # last load of a batch), so waiting on a load's scoreboard also completes
    # every shared load issued before it. sb 7 = no scoreboard of its own.
    state_in = [None] * n
    state_in[0] = frozenset()
    work = [0]
    reports = defaultdict(set)
    while work:
        i = work.pop()
        s = set(state_in[i])
        a, text, lo, hi = insts[i]
        op, body = opcode(text)
        c = ctrl(hi if hi is not None else 0)
        done_sbs = set(b for b in range(6) if (c["wait"] >> b) & 1)
        if op.startswith("DEPBAR.LE"):
            m = re.search(r"SB(\d),\s*(?:0x0|0)\b", body)
            if m:
                done_sbs.add(int(m.group(1)))
        if done_sbs:
            done = {x for x in s if x[1] in done_sbs}
            covered = set()
            for x in done:
                covered |= x[3]
            s = {x for x in s if x not in done and x[0] not in covered}
        if op.startswith("MEMBAR") or op.startswith("FENCE.VIEW.ASYNC"):
            s = {(l, sb, True, e) for (l, sb, f, e) in s}
        if (op.startswith("SYNCS.ARRIVE") or op.startswith("BAR.SYNC") or op.startswith("BAR.ARV")
                or op.startswith("BAR.RED") or op.startswith("UTMALDG") or op.startswith("UBLKCP")):
            for x in s:
                reports[i].add(x[:3])
        if op.startswith("LDS"):
            s.add((i, c["wr"], False, frozenset(x[0] for x in s)))
        fs = frozenset(s)
        for j in succ[i]:
            if state_in[j] is None:
                state_in[j] = fs
                work.append(j)
            elif not fs <= state_in[j]:
                state_in[j] = state_in[j] | fs
                work.append(j)
    return reports


VERBOSE = False


def main():
    global VERBOSE
    args = sys.argv[1:]
    if args and args[0] == "-v":
        VERBOSE, args = True, args[1:]
    for path in args:
        for name, insts in parse(path).items():
            reports = analyze(name, insts)
            kinds = defaultdict(int)
            for i, (a, text, lo, hi) in enumerate(insts):
                op, _ = opcode(text)
                for k in ("SYNCS.ARRIVE", "BAR.SYNC", "BAR.ARV", "UTMALDG", "UBLKCP", "LDSM", "LDS", "FENCE.VIEW.ASYNC"):
                    if op.startswith(k):
                        kinds[k] += 1
                        break
            print(f"== {path} :: {name[:90]}  ({len(insts)} insts; "
                  + ", ".join(f"{k} {v}" for k, v in kinds.items()) + ")")
            for i in sorted(reports):
                live = [x for x in reports[i] if not x[2]]
                fenced = [x for x in reports[i] if x[2]]
                if not live and not fenced:
                    continue
                a, text, lo, hi = insts[i]
                tag = "PENDING" if live else "fenced"
                loads = sorted({insts[l][0] for (l, sb, f) in live})
                print(f"  {tag:8s} /*{a:04x}*/ {text.strip()[:70]:70s} "
                      f"pending={len(loads)} fenced={len({x[0] for x in fenced})} "
                      f"loads={','.join(f'{x:04x}' for x in loads[:8])}"
                      + (" ..." if len(loads) > 8 else ""))
                if VERBOSE:
                    for l in sorted({x[0] for x in live}):
                        print(f"             load /*{insts[l][0]:04x}*/ {insts[l][1].strip()}")


if __name__ == "__main__":
    main()
