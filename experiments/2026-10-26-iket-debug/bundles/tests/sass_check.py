"""Where ptxas puts each IKET event's timer read, relative to the barriers around it.

usage: sass_check.py SOURCES_DIR OUT_DIR REGEX [REGEX ...]

Compiles the IKET mode's instrumented sources (``<kernel>.<hash>.cu``) whose kernel
matches a REGEX to cubins with TileLang's flags for these kernels, disassembles them
with ``$NVDISASM`` and, for every event (a ``PMTRIG``), finds the ``%globaltimer`` read
that stamps it. An event whose timer read comes before a barrier (``BAR``, ``WARPSYNC``,
an mbarrier ``SYNCS``) that comes before the event was stamped before that barrier
released its warp. ptxas keeps program order there; the hardware does not wait:
a ``BAR.SYNC.DEFER_BLOCKING`` holds the warp only at its next memory access, so for
each block barrier it also reports whether a memory instruction comes before the
next timer read (the IKET mode's ``TL_IKET_SETTLE``). Prints, per kernel, the events,
the hoisted ones and the unsettled barriers, and writes each ``.sass``.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tilelang.contrib import nvcc
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

TIMER = re.compile(r"SR_GLOBALTIMER")
EVENT = re.compile(r"\bPMTRIG\b")
BARRIER = re.compile(r"\bBAR\.|\bWARPSYNC\b|\bSYNCS\.|\bBSYNC\b")
MEMORY = re.compile(r"^(@!?U?P\w+\s+)?(LDS|STS|LDG|STG|LD|ST|ATOM|ATOMS|RED|LDGSTS|LDSM)\b")


def instructions(sass):
    out = []
    for line in sass.splitlines():
        match = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
        if match:
            out.append((int(match.group(1), 16), match.group(2).strip()))
    return out


def main():
    sources, out, patterns = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    options = ["-std=c++20", "-I" + TILELANG_TEMPLATE_PATH, "-I" + CUTLASS_INCLUDE_DIR,
               "--ptxas-options=--register-usage-level=10"]
    report = {}
    for path in sorted(sources.glob("*.cu")):
        kernel = path.name.split(".")[0]
        if not any(re.search(p, kernel) for p in patterns):
            continue
        cubin = nvcc.compile_cuda(path.read_text(), "cubin", ["-arch=sm_121a"], options=options)
        cubin_path = out / f"{path.stem}.cubin"
        cubin_path.write_bytes(cubin)
        sass = subprocess.run([os.environ.get("NVDISASM", "nvdisasm"), str(cubin_path)],
                              text=True, capture_output=True, check=True).stdout
        (out / f"{path.stem}.sass").write_text(sass)
        code = instructions(sass)
        events, hoisted = 0, []
        for i, (_, text) in enumerate(code):
            if not EVENT.search(text):
                continue
            events += 1
            timer = next((j for j in range(i - 1, -1, -1) if TIMER.search(code[j][1])), None)
            if timer is None:
                continue
            between = [code[j] for j in range(timer + 1, i) if BARRIER.search(code[j][1])]
            if between:
                hoisted.append({"event": f"{code[i][0]:#06x}", "timer": f"{code[timer][0]:#06x}",
                                "barriers": [f"{a:#06x} {t}" for a, t in between]})
        # BAR.SYNC.DEFER_BLOCKING releases the warp's issue at its next memory access:
        # the first timer read after a block barrier should come after one.
        settled, unsettled = 0, []
        for i, (address, text) in enumerate(code):
            if not text.startswith("BAR.SYNC"):
                continue
            for j in range(i + 1, len(code)):
                if MEMORY.search(code[j][1]):
                    settled += 1
                    break
                if TIMER.search(code[j][1]):
                    unsettled.append(f"{address:#06x} -> {code[j][0]:#06x}")
                    break
        report[path.stem] = {"events": events, "hoisted": hoisted, "barriers_settled": settled,
                             "barriers_unsettled": unsettled}
        print(f"{path.stem}: {events} events, {len(hoisted)} stamped before a barrier they follow; "
              f"block barriers: {settled} meet memory first, {len(unsettled)} a timer read first "
              f"{unsettled[:6]}")
        for item in hoisted[:12]:
            print("   ", json.dumps(item))
    (out / "sass_check.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
