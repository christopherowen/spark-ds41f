import collections,json,pathlib,re
home=pathlib.Path.home();base=home/"tl-logs/lossless-memory/recovery";analysis=json.loads((base/"scale-analysis-attempt2/performance.json").read_text());snap=base/"scales-round4-snapshot/dgx1/projects/spark-ds41f";result={"boots":{},"verdicts":{},"weight_load_lines":{}}
for label,boot in analysis["boots"].items():
 output={"startup_memory":{n:r["min_available_gib"] for n,r in boot["startup_memory"].items()},"suites":{}}
 for name,value in boot.items():
  if name=="startup_memory":continue
  steps={}
  for key,row in value["steps"].items():
   if isinstance(row,dict) and "summary" in row:
    s=row["summary"];steps[key]={"tps":s["output_tps"]["mean"],"ci95_pct":s["output_tps"].get("ci95_pct"),"ttft_p50_s":s["latency"]["ttft_s"]["p50"],"ttft_p99_s":s["latency"]["ttft_s"]["p99"],"tpot_p50_ms":s["latency"]["tpot_ms"]["p50"],"tpot_p99_ms":s["latency"]["tpot_ms"]["p99"],"accepted_per_draft":s["server"]["accepted_per_draft"]["mean"]}
  output["suites"][name]={"steps":steps,"memory":{n:r["min_available_gib"] for n,r in value["conditions"]["memory"].items()}}
 result["boots"][label]=output
for name,value in analysis["comparisons"].items():
 rows={}
 for view in value["all_pair_and_floor_views"]:
  for row in view["rows"]:
   key=row["step"]+" / "+row["metric"];rows.setdefault(key,[]).append(row)
 result["verdicts"][name]={k:{"changes_pct":[round(r["change_pct"],4) for r in rs],"raw_verdicts":dict(collections.Counter(r["plain_verdict"] for r in rs)),"eligible_views":sum(r["eligible_performance_claim"] for r in rs),"margin_range_pct":[min(r["margin_pct"] for r in rs),max(r["margin_pct"] for r in rs)]} for k,rs in rows.items()}
for node in ['dgx1','dgx2','dgx3']:
 folder=base/"scales-round4-snapshot"/node/"projects/spark-ds41f/results/private/lossless-memory-scales-round4/native-engine-logs"
 for path in sorted(folder.glob('*startup*')):
  lines=[line for line in path.read_text().splitlines() if re.search(r'Model loading took|GPU memory at startup|Available KV cache memory|weight.*GiB|weights.*GiB',line,re.I)]
  result["weight_load_lines"][node+'/'+path.name]=lines
(base/"scale-analysis-attempt2/core-summary.json").write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
