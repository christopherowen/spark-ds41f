# r5p and r6r on the same eight-stream tests

r5p's TP3 promotion (2026-10-05) reported 173 tok/s (prose) and 192 tok/s (code) at
eight streams from `bin/spark3 bench`, which sends one prompt to every stream. The
release suites since 2026-10-15 judge distinct prompts; r6r's gate read 143.5 tok/s at
eight distinct streams. The two numbers measure different things. This run puts both
builds through both tests in one window, r6r first and last:

- the old bench's decode suite as r5p ran it (prose and code, 1, 2, 4 and 8 streams,
  three samples, 256 tokens);
- eight and four distinct prompts (`scripts/distinct_streams.py`).

`r5p-tp3.json` is the configuration r5p's TP3 benchmark booted
(`experiments/2026-10-05-tp3-benchmark/b12x.json`) with only the repository's renamed
paths changed: the cache mount and the deployment checkout.

## Results (2026-10-16, 17:19-17:41 UTC, TP3, APST tuned)

Distinct prompts, 256 tokens, four samples (tok/s, ±95%):

| | r6r | r5p | r6r again |
| --- | ---: | ---: | ---: |
| eight streams | 141.8 ±0.8% | 129.8 ±2.2% | 140.2 ±0.9% |
| four streams | 108.3 ±1.1% | 99.2 ±3.8% | 108.3 ±1.1% |

The old bench's decode suite (the same prompt on every stream; aggregate tok/s at 1, 2,
4 and 8 streams), with r5p's 2026-10-05 promotion run for reference (same prompt set,
`cc7f29400eff`):

| | 1 | 2 | 4 | 8 | distinct outputs at 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose, r5p 2026-10-05 | 50.4 | 75.9 | 118.8 | 173.3 | 24 |
| prose, r5p | 49.3 | 72.6 | 115.4 | 166.0 | 24 |
| prose, r6r | 60.6 | 90.8 | 171.2 | 247.7 | 1 |
| code, r5p 2026-10-05 | 60.1 | 90.8 | 136.3 | 192.3 | 24 |
| code, r5p | 61.9 | 85.2 | 132.4 | 191.9 | 24 |
| code, r6r | 68.7 | 101.2 | 180.7 | 299.2 | 1 |

- On distinct prompts r6r decodes 8-9% more than r5p at four and eight streams; on the
  old bench's single stream, 23% (prose) and 11% (code) more.
- r5p reproduces its promotion numbers today.
- The old bench's multi-stream points do not compare builds: r6r is deterministic, so its
  eight streams produce one output and run in lockstep, while r5p's diverge (24 distinct
  outputs). Its 173 tok/s at eight streams was never comparable with the distinct-prompt
  143.5.
