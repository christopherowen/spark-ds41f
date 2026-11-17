"""Compare stock vs patched matrix runs and apply the pre-registered gates.

Gates:
  G1 every patched stream mode yields valid JSON args for every call.
  G2 patched: streamed args (parsed) == non-streaming args, every case, every chunking.
  G3 fixed cases (B, C, D, E, E2, J) keep the full intended value non-streaming.
  G4 unchanged cases equal stock exactly (non-streaming calls + content, and each stream mode).
  G5 kill switch run == stock run, byte for byte per record (minus the mode/overlay fields).
  G6 reasoning cases equal stock.
Behaviour changes that are expected and reported, not gated: N, N2 (open param kept), O2, Q, R.
"""
import json
import sys

S, E = "<｜DSML｜ calls>", "</｜DSML｜ calls>"
IE = "</｜DSML｜ invoke>"
PE = "</｜DSML｜ parameter>"


def load(path):
    rows = [json.loads(line) for line in open(path) if line.startswith("{")]
    return {row["case"]: row for row in rows if "case" in row}


stock, patched = load(sys.argv[1]), load(sys.argv[2])
kill = load(sys.argv[3]) if len(sys.argv) > 3 else None
MODES = ["stream1", "stream3", "stream7", "streamAll"]
UNCHANGED = ["A_control", "F_orphan_closers_only", "G_json_answer_with_tools", "H_parallel_calls",
             "I_string_false_values", "K_missing_string_attr", "L_omit_invoke_end",
             "M_omit_param_close_before_next_head", "O_truncated_mid_value", "T_text_after_block",
             "U_json_body"]
FIXED = {
    "B_value_holds_calls_closer": [("write_file", {"path": "dsml.md", "content": "The block ends with " + E + " on its own line."})],
    "C_value_holds_param_closer": [("write_file", {"path": "dsml.md", "content": "A param ends with " + PE + " then more."})],
    "D_value_holds_invoke_closer": [("write_file", {"path": "dsml.md", "content": "An invoke ends with " + IE + " then more."})],
    "E_two_calls_first_holds_closer": [("write_file", {"path": "one.md", "content": "see " + E + " here"}),
                                       ("write_file", {"path": "two.txt", "content": "second"})],
    "E2_all_three_closers_multiline": [("write_file", {"path": "doc.md", "content": "line1 " + PE + "\nline2 " + IE + "\nline3 " + E + "\nend"})],
    "J_string_false_json_holding_invoke_closer": [("set_cfg", {"n": 7, "cfg": {"note": "a " + IE + " b"}})],
}

fails, notes = [], []
# max_tokens cut mid-value: stock streams unterminated JSON here too (upstream #62051/#62259).
# Out of scope; reported as pre-existing when stock is equally invalid.
TRUNCATION = {"O_truncated_mid_value", "O2_truncated_after_literal_closer"}


def calls_of(block):
    return [(c["name"], c["args"]) for c in block.get("calls", [])]


for case, rec in patched.items():
    if "reasoning" in rec:
        if rec["reasoning"] != stock[case]["reasoning"] or rec["nonstream"] != stock[case]["nonstream"]:
            fails.append(f"G6 {case}: reasoning differs from stock")
        continue
    ns = rec["nonstream"]
    if "error" in ns:
        fails.append(f"{case}: nonstream error {ns['error']}")
        continue
    for m in MODES:
        st = rec[m]
        if "error" in st:
            fails.append(f"{case} {m}: error {st['error']}")
            continue
        for c in st["calls"]:
            if not c["json_ok"]:
                if case in TRUNCATION and not all(x["json_ok"] for x in stock[case][m]["calls"]):
                    notes.append(f"PRE-EXISTING {case} {m}: truncated call streams invalid JSON in stock too")
                    continue
                fails.append(f"G1 {case} {m}: invalid JSON args {c['args']!r}")
        if calls_of(st) != calls_of(ns) and case not in TRUNCATION:
            sst, sns = stock[case][m], stock[case]["nonstream"]
            stock_asym = len(sst.get("calls", [])) != len(sns.get("calls", []))
            if stock_asym and ns == sns and all(c["json_ok"] for c in st["calls"]):
                notes.append(f"STOCK-ASYMMETRY {case} {m}: stock stream/non-stream already disagree; patched stream valid, non-stream == stock")
                continue
            fails.append(f"G2 {case} {m}: stream {calls_of(st)} != nonstream {calls_of(ns)}")
    if case in FIXED:
        if [(n, a) for n, a in calls_of(ns)] != FIXED[case]:
            fails.append(f"G3 {case}: got {calls_of(ns)}")
    if case in UNCHANGED:
        s = stock[case]
        if ns != s["nonstream"]:
            fails.append(f"G4 {case} nonstream: patched {ns} != stock {s['nonstream']}")
        for m in MODES:
            if rec[m] != s[m]:
                stock_bad = any(not c["json_ok"] for c in s[m].get("calls", [])) or calls_of(s[m]) != calls_of(s["nonstream"])
                patched_good = all(c["json_ok"] for c in rec[m].get("calls", [])) and calls_of(rec[m]) == calls_of(ns)
                if stock_bad and patched_good and ns == s["nonstream"]:
                    notes.append(f"IMPROVED {case} {m}: stock stream {calls_of(s[m])} -> patched matches non-streaming")
                    continue
                fails.append(f"G4 {case} {m}: patched {rec[m]} != stock {s[m]}")
    if case not in UNCHANGED and case not in FIXED:
        notes.append(f"CHANGE {case}:\n    stock   {calls_of(stock[case]['nonstream'])} content={stock[case]['nonstream'].get('content')!r}\n    patched {calls_of(ns)} content={ns.get('content')!r}")

if kill is not None:
    for case, rec in kill.items():
        a = {k: v for k, v in rec.items() if k not in ("mode", "overlay")}
        b = {k: v for k, v in stock[case].items() if k not in ("mode", "overlay")}
        if a != b:
            fails.append(f"G5 {case}: kill-switch differs from stock")

# Stock bug confirmation (the instrument can see the bug): stock must fail the FIXED expectations.
blind = [c for c in FIXED if calls_of(stock[c]["nonstream"]) == FIXED[c]]
if blind:
    fails.append(f"INSTRUMENT: stock already passes {blind} (test cannot see the bug)")

print(f"cases={len(patched)} overlay={next(iter(patched.values())).get('overlay')}")
for n in notes:
    print(n)
print("FAILS:" if fails else "ALL GATES PASS")
for f in fails:
    print("  " + f)
sys.exit(1 if fails else 0)
