# CED row-selection kernels

Base: r6x f3b58ed9d. First port: per-request integer kept-row limits and compaction starts, positions, replay floors and counts. Limits 128 and INT32_MAX reproduce the old 128/full-row interface. The module is not wired into serving until exactness and per-call timing pass.

Source module 304bd84a1. Tests cover empty and padded batches, zero/short/long requests, prefix floors and mixed keep-full flags at 1, 8, 16 and 32 requests.
