"""Where #986's modeled step-time savings come from: its screen's step_latency_ms split into
its terms (TP1 dense FP8 table, TP1 sparse MLA table, indexer per row, and the hand-set
'routed MoE + drafter + allreduce' term 15,600 + 420/210/440 us a row), for the regimes it
claims, from its own boot-1/2 profiles. Also the same row change priced by the tp4-ring
engine grid (candidate-47 / no986, four boots: per-row slope 17..64 and the >64 step)."""
import json, sys
prof = [json.load(open(p)) for p in sys.argv[1:3]]
def avg(key, *ks):
    vals = []
    for p in prof:
        d = p[key]
        for k in ks: d = d[str(k)]
        vals.append(d)
    return sum(vals) / len(vals)
def bucket(r):
    rs = sorted(int(k) for k in prof[0]["dense_us"]); return next((x for x in rs if x >= r), rs[-1])
def moe(r):
    b1 = min(r, 16); b2 = max(0, min(r - 16, 48)); b3 = max(0, r - 64)
    return 15600 + 420 * b1 + 210 * b2 + 440 * b3
def ctxb(L):
    ls = sorted(int(k) for k in prof[0]["idx_us"]); return next((x for x in ls if x >= L), ls[-1])
def terms(rows, L, reqrows):
    rb = bucket(rows)
    idx = sum(avg("idx_us", ctxb(L), rb) / rb * rr for rr in reqrows)
    return {"dense_us": avg("dense_us", rb), "mla_us": avg("mla_us", rb), "idx_us": idx, "moe_drafter_us": moe(rows)}
cases = {
    "c16_second_wave_saturation_2k": (16, 2048, 80, 64),
    "c8_med_accept_32k": (8, 32768, 40, 32),
    "c8_med_accept_128k": (8, 131072, 40, 32),
    "c1_copy_burst_plateau_2k": (1, 2048, 20, 16),
}
out = {}
for name, (n, L, rb_, rc_) in cases.items():
    tb = terms(rb_, L, [rb_ / n] * n); tc = terms(rc_, L, [rc_ / n] * n)
    d = {k: (tb[k] - tc[k]) / 1000 for k in tb}
    tot = sum(d.values())
    out[name] = {"rows": [rb_, rc_], "saved_ms_by_term": {k: round(v, 3) for k, v in d.items()}, "saved_ms": round(tot, 3),
                 "share_by_term": {k: round(v / tot, 3) for k, v in d.items()}}
    print(name, out[name])
# dense table monotonicity
dense = {int(k): avg("dense_us", k) for k in prof[0]["dense_us"]}
out["dense_us_table_mean"] = {str(k): round(v, 1) for k, v in sorted(dense.items())}
out["dense_non_monotone"] = [f"R={a}: {dense[a]:.0f} us > R={b}: {dense[b]:.0f} us" for a, b in ((1, 16), (17, 32), (40, 64), (80, 96)) if dense[a] > dense[b]]
print(out["dense_non_monotone"])
json.dump(out, open(sys.argv[3], "w"), indent=1)

# The same 40 -> 32 trim at 2K, priced by the screen's own latency model (its cost model keeps 40 there).
conf = [0.78, 0.72, 0.71, 0.71, 0.73]
surv = []; s = 1.0
for c in conf:
    s *= c; surv.append(s)
tok40 = 8 + 8 * sum(surv[:4]); tok32 = 8 + 8 * sum(surv[:3])
def latency(rows, L, n=8):
    t = terms(rows, L, [rows / n] * n); return sum(t.values()) / 1000
r = {"tokens_40_rows": tok40, "tokens_32_rows": tok32}
for L in (2048, 32768, 131072):
    a, b = latency(40, L), latency(32, L)
    r[str(L)] = {"ms_40": round(a, 3), "ms_32": round(b, 3), "tok_per_s_change_pct_trim_to_32": round(100 * ((tok32 / b) / (tok40 / a) - 1), 2)}
out["c8_med_accept_same_trim_by_context"] = r
print(r)
json.dump(out, open(sys.argv[3], "w"), indent=1)
