import json, sys
r = json.load(open(sys.argv[1]))
print("turns", r["turns"], "blocks", r["blocks"], "valid positions", r["positions_valid"], "test blocks", r["test_blocks"])
print("replay agreement %.4f | head input: %s | sanity AUC %s" % (r["replay_agreement"], r["z_used"], {k: round(v, 3) for k, v in r["sanity_auc_conditional"].items()}))
print("accept by position (given earlier accepted):", [round(a, 3) for a in r["raw_head"]["accept_by_pos"]])
print("raw head mean prob by position:            ", [round(a, 3) for a in r["raw_head"]["mean_prob_by_pos"]])
print("accepted-run histogram 0..5 (test):", [round(a, 3) for a in r["run_hist_test"]], "| all drafts accepted: %.3f" % r["all_drafts_accepted_share_test"])
print("accepted per step at full depth %.3f of %.3f drafted" % (r["test"]["accepted_per_block_at_full_depth"], r["test"]["drafted_rows_per_block"]))
print("--- discrimination on the held-out half")
for k, v in r["discrimination_test"].items():
    print(f"  {k:14s} AUC {v['auc']:.4f}  logloss {v['logloss']:.4f}")
print("--- throughput gain over the serving form, % [95% CI]; rows verified / wasted per step")
names = [k for k in r["policy_test"]["rho=0.11"] if k not in ("no_drafts",)]
print("  %-20s" % "rho" + "".join("%22s" % k.split("=")[1] for k in r["policy_test"]))
for n in names:
    cells = []
    for rho, o in r["policy_test"].items():
        v = o[n]
        if "gain_vs_serving_pct" in v:
            cells.append("%+6.2f [%+5.2f,%+5.2f]" % (v["gain_vs_serving_pct"], *v["gain_ci95_pct"]))
        else:
            cells.append("%+6.2f" % (100 * (v["throughput"] / o["serving"]["throughput"] - 1)))
    print("  %-20s" % n + "".join("%22s" % c for c in cells))
for rho, o in r["policy_test"].items():
    s = o["serving"]
    print("  %s serving: tau %.3f tokens/step %.3f rows %.3f wasted %.3f | best scalar (z+draft): rows %.3f wasted %.3f | oracle rows %.3f" % (
        rho, s["tau"], s["tokens_per_step"], s["rows_per_step"], s["wasted_rows_per_step"],
        o["z+draft"]["rows_per_step"], o["z+draft"]["wasted_rows_per_step"], o["oracle"]["rows_per_step"]))
