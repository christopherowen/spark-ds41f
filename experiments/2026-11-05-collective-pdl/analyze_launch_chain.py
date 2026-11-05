"""Report all placement comparisons from six passing, source-qualified runs.

Intervals use the two independent process/round means (Student t, df=1), not
pooled graph replays. They are pointwise exploratory intervals, not a serving
verdict or a multiple-comparison guarantee. Every measured cell is retained.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

T95_DF1 = 12.706204736432095


def interval(values):
    if len(values) != 2:
        raise ValueError("exactly two independent process rounds are required")
    center = statistics.mean(values)
    half = T95_DF1 * statistics.stdev(values) / math.sqrt(2)
    return {"mean": center, "ci95": [center-half, center+half], "rounds": values}


def read_run(path):
    receipt = json.loads(path.read_text())
    if receipt.get("passed") is not True or receipt.get("mode") != "benchmark":
        raise ValueError(f"not a passing benchmark: {path}")
    if not receipt.get("gates_receipt_sha256"):
        raise ValueError(f"missing source qualification: {path}")
    jobs = receipt["jobs"]
    if len(jobs) != 1 or jobs[0]["kind"] != "benchmark":
        raise ValueError(f"unexpected job set: {path}")
    if len(jobs[0]["outcomes"]) != 3: raise ValueError(f"expected three rank outcomes: {path}")
    ranks = {}
    for row in jobs[0]["outcomes"]:
        log = path.parent/row["log"]
        if row["exit"] or row["problems"] or hashlib.sha256(log.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"failed or changed rank log: {log}")
        data = row["benchmark_result"]
        if data["passed"] is not True or data["source_commit"] != receipt["source_commit"]:
            raise ValueError(f"rank identity differs: {log}")
        ranks[data["rank"]] = data
    if set(ranks) != {0,1,2}:
        raise ValueError(f"missing or duplicated rank: {path}")
    cells = {}
    for row in ranks[0]["cases"]:
        key = (row["rows"],row["operation"],row["background_mib"])
        samples = row["rank_samples_us"]
        if not row["exact"] or len(samples) != 3 or any(len(values) != 40 for values in samples):
            raise ValueError(f"incomplete cell: {path}: {key}")
        if any(not math.isfinite(x) or x <= 0 for values in samples for x in values):
            raise ValueError(f"invalid timing: {path}: {key}")
        for rank in (1,2):
            peers = [value for value in ranks[rank]["cases"] if (value["rows"],value["operation"],value["background_mib"]) == key]
            if len(peers) != 1 or peers[0] != row:
                raise ValueError(f"ranks disagree on gathered cell: {path}: {key}")
        if key in cells: raise ValueError(f"duplicate cell: {path}: {key}")
        cells[key] = statistics.mean(max(values[i] for values in samples) for i in range(40))
    expected = {(r,o,b) for r in (1,6,9,16,24,48,96) for o in ("all_reduce","all_gather","mixed") for b in (0,4,32)}
    if set(cells) != expected: raise ValueError(f"incomplete matrix: {path}")
    identity = {key:receipt[key] for key in ("image","map_sha256","harness_commit","benchmark_sha256")}
    return receipt, identity, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts",type=Path,nargs=6)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    runs, identities, sources, receipts = {},[],{},[]
    for path in args.receipts:
        receipt,identity,cells = read_run(path)
        key = (receipt["phase"],receipt["process_run"])
        if key in runs: raise ValueError(f"duplicate process run: {key}")
        runs[key]=cells; identities.append(identity)
        source=(receipt["source_commit"],receipt["source_tree"])
        if key[0] in sources and sources[key[0]] != source: raise ValueError("source changed between rounds")
        sources[key[0]]=source
        receipts.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "phase":key[0],"process_run":key[1],"memory":receipt["memory"],"thermal":receipt["thermal"]})
    if set(runs) != {(phase,run) for phase in ("entry","staged","arrived") for run in (1,2)}:
        raise ValueError("entry/staged/arrived each require both process rounds")
    if any(value != identities[0] for value in identities[1:]): raise ValueError("image/map/harness changed")
    rows=[]
    for key in sorted(runs["entry",1]):
        row={"rows":key[0],"operation":key[1],"background_mib":key[2],"latency_us":{},"versus_entry":{}}
        reference=[runs["entry",run][key] for run in (1,2)]
        for phase in ("entry","staged","arrived"):
            values=[runs[phase,run][key] for run in (1,2)]
            row["latency_us"][phase]=interval(values)
            if phase != "entry":
                delta=interval([value-base for value,base in zip(values,reference)])
                log_ratio=interval([math.log(value/base) for value,base in zip(values,reference)])
                row["versus_entry"][phase]={"delta_us":delta,"delta_percent":{
                    "mean":100*math.expm1(log_ratio["mean"]),
                    "ci95":[100*math.expm1(bound) for bound in log_ratio["ci95"]]},
                    "both_rounds_faster":all(value<base for value,base in zip(values,reference))}
        rows.append(row)
    result={"scope":"synthetic only; no released-build or serving comparison", "identity":identities[0],
            "sources":sources,"receipts":receipts,"method":"Pointwise Student-t 95% intervals on two process-round means/differences, df=1. Percent changes use log ratios. No multiplicity correction or automatic winner. Lower latency is better.","cases":rows}
    with args.out.open("x") as handle: json.dump(result,handle,indent=2);handle.write("\n")


if __name__ == "__main__": main()
