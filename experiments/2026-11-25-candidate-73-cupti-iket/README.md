# `candidate-73` Fresh-Reboot CUPTI & IKET Analysis

- **Base commit**: `24950b3eca` (frozen `candidate-73` tag on `main`)
- **Image**: `spark-ds41f:candidate-73` (`sha256:cfef8ba79ad807de00ff04a31a779759c9f94fb3f5292fee4e422aa26bb96669`)
- **Goal**: Profile freshly rebooted `tp4-ring` (`dgx1..dgx4`, `O13 >= 210` 512 MiB blocks on every node) with CUPTI (`torch.profiler`) and TileLang IKET to measure fresh-reboot decode throughput vs `spark-ds41f-1120` and inspect per-kernel hotspots (`c1`, `1x3`, `4x24`, `8x24`, `8x48`, `16x96`) for any regressions or actionable improvements prior to full rebooted switchless release benches.
