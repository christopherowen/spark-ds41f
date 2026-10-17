#!/usr/bin/env node
// sparkDash 1.8.8's decode and prefill benches (github.com/MiaAI-Lab/sparkDash, Apache-2.0,
// commit 754f40a7, "Release 1.8.8") driven headless, with the settings MiaAI-Lab's
// DeepSeek-v4.1-Flash-DGX-Sparks release reports (docs/tp4.md): greedy (temperature 0, top_p 1,
// thinking off), 256 new tokens, prose / code / structured / json at 1, 2, 4, 8 and 16 streams,
// then cold prefill at 4K-262K, two passes. The benches' own code runs unchanged; this file only
// starts their job managers and waits for them.
//
// usage: node sparkdash_bench.mjs SPARKDASH_CHECKOUT OUT_DIR HOST PORT MODEL [MAX_CONTEXT]
import fs from "fs";
import path from "path";

const [checkout, out, host, port, model, maxContext = "262144"] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
// The managers read these at import: their history files land in OUT_DIR.
process.env.BENCH_HISTORY_PATH = path.join(out, "decode-history.json");
process.env.BENCH_ACTIVE_PATH = path.join(out, "decode-active.json");
process.env.PREFILL_BENCH_HISTORY_PATH = path.join(out, "prefill-history.json");
process.env.PREFILL_BENCH_ACTIVE_PATH = path.join(out, "prefill-active.json");
const { decodeBenchManager } = await import(path.join(checkout, "server/collectors/DecodeBench.js"));
const { prefillBenchManager } = await import(path.join(checkout, "server/collectors/PrefillBench.js"));

const target = { sparkId: "spark-ds41f", lanIp: host, port: Number(port), modelId: model };
const TYPES = ["prose", "code", "structured", "json"];
const STREAMS = [1, 2, 4, 8, 16];
const SIZES = [4096, 16384, 32768, 65536, 131072, 262144].filter((s) => s <= Number(maxContext));

async function finish(manager, started) {
  for (;;) {
    const job = manager.getJob(started.benchId);
    if (job && job.status !== "running") return job;
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

const report = { sparkdash: "754f40a7 (1.8.8)", target: { host, port: Number(port), model }, decode: {}, prefill: [] };
for (const type of TYPES) {
  const job = await finish(decodeBenchManager, decodeBenchManager.start(
    { ...target, concurrencies: STREAMS, maxTokens: 256, promptType: type }));
  report.decode[type] = job;
  const cells = job.results.map((w) => `c${w.concurrency} ${w.aggregateDecodeTps} (${w.meanDecodeTps})`);
  console.log(`decode ${type} [${job.status}]: ${cells.join(" | ")}`);
}
for (let pass = 1; pass <= 2; pass++) {
  const job = await finish(prefillBenchManager, prefillBenchManager.start({ ...target, contextSizes: SIZES }));
  report.prefill.push(job);
  const cells = job.results.map((r) => `${r.targetTokens ?? r.size}: ${Math.round(r.prefillTps ?? r.tps ?? 0)} tok/s, TTFT ${Math.round(r.ttftMs)} ms`);
  console.log(`prefill pass ${pass} [${job.status}]: ${cells.join(" | ")}`);
}
fs.writeFileSync(path.join(out, "sparkdash-report.json"), JSON.stringify(report, null, 2));
