import json, sys
d = json.load(open(sys.argv[1]))
print(d["arm"], d["world"], d.get("seconds"))
for s in d["scenarios"]:
    r = s["ranks"][0] if s["ranks"] else {}
    print(s["scenario"], s["step"], s["layer"], s["T"], "err=%s" % s["error"], "conv=%s" % s["converged"],
          "agree=%s/%s/%s" % (s.get("ranks_agree_on_output"), s.get("ranks_agree_on_topk"), s.get("ranks_agree_on_caches")),
          "out=%s" % r.get("out"), "gathers=%s" % [g[0][1] if len(g[0]) > 1 else g[0][0] for g in r.get("gathers", [])])
