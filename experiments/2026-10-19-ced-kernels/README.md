# CED row-selection kernels

Base: r6x f3b58ed9d. First port: per-request integer kept-row limits and compaction starts, positions, replay floors and counts. Limits 128 and INT32_MAX reproduce the old 128/full-row interface. The module is not wired into serving until exactness and per-call timing pass.

Source module 304bd84a1. Tests cover empty and padded batches, zero/short/long requests, prefix floors and mixed keep-full flags at 1, 8, 16 and 32 requests.

First GPU attempt: the request kernel did not compile because TileLang inferred incompatible fragment layouts for a prefix entry and the total. No serving path was changed. Candidate 9772676cb moves the small prefix arrays to shared memory; the replay, index-map and row-copy gates follow. Keep the original failure output under results/private/lab/postlaunch/ced-requests on DGX4.
