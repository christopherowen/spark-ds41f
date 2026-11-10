import json,sys
for p in sys.argv[1:]:
    d=json.load(open(p))
    print("#####", p.split('/')[-1], "seed", d["seed"], "steps", d["steps"], "elapsed", round(d["elapsed_s"],1))
    for c,r in d["concurrency"].items():
        print("==", c, "anchor", {k:(round(v,2) if isinstance(v,float) else v) for k,v in r["anchor"].items() if k!='overhead_note'}, "prefill", round(r["share_prefill_steps"],4))
        for nm,a in r["arms"].items():
            p_=a["prediction"]
            print("  %-20s tau %.4f b %.3f rows %.3f tok %.4f diff %.2f%% | tok %+.3f%% gpu %+.4fms (%+.3f%%) thrGPU %+.3f%% thrA %s B %+.3f%% interp %s" % (
                nm, a["effective_dead_rows_tau"], a["draft_budget_mean"], a["rows_mean"], a["tokens_mean"], 100*a["steps_budget_differs_from_c54@tp3"],
                p_["tokens_per_step_pct"], p_["grid_gpu_step_ms_delta"], p_["grid_gpu_step_pct"], p_["throughput_grid_gpu_only_pct"],
                ("%+.3f%% %s" % (p_["throughput_model_A_pct"], [round(x,3) for x in p_["throughput_model_A_pct_bootstrap95"]])) if "throughput_model_A_pct" in p_ else "-",
                p_["throughput_model_B_tp4_regression_pct"], ("%+.3f%%" % p_["interp_gpu_step_pct"]) if "interp_gpu_step_pct" in p_ else "-"))
        a=r["arms"]["c54@tp3"]; print("   c54 cov", [round(x,3) for x in a["coverage"]], "surv", [round(x,3) for x in a["survival"]])
    print("tp4", json.dumps(d["tp4_bit_for_bit"]))
