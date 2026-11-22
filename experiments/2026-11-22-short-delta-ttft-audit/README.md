# Short-Delta TTFT Breakdown & Concurrent Co-Admission Audit

Date: 2026-11-22.
Base: `spark-ds41f:candidate-65` (`tp3-triangle`, `16` seats, `--enable-per-request-metrics`).

## Goal

Measure the exact live millisecond breakdown of cached short-delta requests (`client_ttft_ms`, `server_queue_ms`, `server_ttft_ms`, `host_overhead_ms`, `cached_tokens`) across:

1. **Scenario A**: Solo `62K` cached prefix + uncached delta ladder `{128, 256, 512, 1024, 2048}` tokens.
2. **Scenario B**: Concurrent waves of `4` and `8` `62K`-cached turns × `~350` uncached delta tokens (`0 ms` simultaneous vs `25 ms` staggered arrival).
3. **Scenario C**: `1K` uncached delta straddling a `6,048`-token `align_prefill_chunks` boundary vs staying inside one chunk.
4. **Scenario D**: Short `512`-token cached delta arriving while `4` decode streams are active.

## CPU Pre-Audit (`dgx4`, `spark-ds41f:candidate-64`, zero GPU impact)

- `apply_chat_template` on `144K` non-repetitive tokens: `1.2 ms`.
- Rust BPE `tokenizer.encode()` on `144K` non-repetitive tokens: `38.4 ms` (`~26 ms` at `100K`).
- `sha256` prefix-cache block hashing (`100K` tokens, `390` × `256`-token blocks): `1.30 ms`.
- Conclusion: host tokenization + block hashing is `< 30–40 ms` at `100K` tokens, ruling them out as the primary source of the historical `~0.9 s` (`2026-09-28`) live TTFT.

## Status

Ready to run [audit_short_delta.py](audit_short_delta.py) on `tp3-triangle` once `spark-ds41f:candidate-65` finishes building and distributes to `dgx1..dgx3`.
