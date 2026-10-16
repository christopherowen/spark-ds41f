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
