#!/usr/bin/env python3
"""Re-pair run.sh's commands.txt: each run appends its header at start and its footer
(end_utc, loadavg_after, exit_code, output) at exit, so runs that overlapped (one
waiting on the GPU lock while another ran) interleave. Footers name their output file,
which identifies their run. Prints one well-formed block per run, in start order."""
import sys

lines = open(sys.argv[1]).read().splitlines()
headers, order, footers = {}, [], {}
cur, buf = None, []
pending_footer = []
for ln in lines:
    if ln.startswith("### "):
        cur = ln[4:].strip()
        order.append(cur)
        headers[cur] = [ln]
        continue
    if ln.startswith(("end_utc:", "loadavg_after:", "exit_code:")):
        pending_footer.append(ln)
        continue
    if ln.startswith("output:"):
        label = ln.split(":", 1)[1].strip()[: -len(".txt")]
        footers[label] = pending_footer + [ln]
        pending_footer = []
        continue
    if ln.strip() == "":
        continue
    if cur is not None and not pending_footer:
        headers[cur].append(ln)
for label in order:
    print("\n".join(headers[label]))
    print("\n".join(footers.get(label, ["(no footer: run did not finish)"])))
    print()
