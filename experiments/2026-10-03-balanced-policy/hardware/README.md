# Native evidence

Archive SHA-256: `5261e0af3cd48fcb9f9dd902a0e5b5b769f1ed12893cadc93dfd163fd528a6ca`. The archive contains 709 native receipts and
support files from `.work/balanced-policy/`. `files.sha256.json` fingerprints
every member relative to the extracted `balanced-policy/` directory.

It includes all 16 transport runs on all four ranks, per-case TX/RX vport and
RoCEnante payload counters, channel plans, telemetry, source tests, image build
and distribution receipts, four serving boots and their benchmark reports,
the failed ABI-loader build, the failed LAN client attempt, final live doctor,
and the idle-return/window-release receipt. Failed attempts are retained.

The first control has startup logs and full benchmark reports; end-of-run
container logs were captured for the other three boots. All four serving boots
used the same image digest, published deployment commit and pinned cost table.
The shared deployment checkouts remained at `cfa6e5c`; qualification checkouts
were clean at `6b6861b`. The window was released with every node idle and each
fan-control service active.

```sh
mkdir -p /tmp/balanced-evidence
tar -xzf experiments/2026-10-03-balanced-policy/hardware/runs.tar.gz -C /tmp/balanced-evidence
python3 experiments/2026-10-03-balanced-policy/analyze.py \
  /tmp/balanced-evidence/balanced-policy selected-control-final selected-final
python3 experiments/2026-10-03-balanced-policy/summarize-serving.py \
  /tmp/balanced-evidence/balanced-policy
```

The initial 11-arm summary is `stage1-results.json`, the adaptive comparison is
`adaptive-results.json`, the handoff comparison is `handoff-results.json`, and
the final same-image comparison is `final-transport-results.json`. Exact
per-node launch commands are retained under `probes/<run>/`. Profiles without
a matching run directory were prepared but not measured; do not treat them as
evidence. GPU numerical checks and scalar/odd-size fallbacks are in each
rank's final JSON record. Physical-port counters are shared views and are not
added as separate cable bandwidth.
