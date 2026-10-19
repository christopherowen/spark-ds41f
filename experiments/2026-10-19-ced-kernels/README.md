# CED row-selection kernels

Base: r6x f3b58ed9d. First port: per-request integer kept-row limits and compaction starts, positions, replay floors and counts. Limits 128 and INT32_MAX reproduce the old 128/full-row interface. The module is not wired into serving until exactness and per-call timing pass.

Source module 304bd84a1. Tests cover empty and padded batches, zero/short/long requests, prefix floors and mixed keep-full flags at 1, 8, 16 and 32 requests.

First GPU attempt: the request kernel did not compile because TileLang inferred incompatible fragment layouts for a prefix entry and the total. No serving path was changed. Candidate 9772676cb moves the small prefix arrays to shared memory; the replay, index-map and row-copy gates follow. Keep the original failure output under results/private/lab/postlaunch/ced-requests on DGX4.

Second attempt: 30 request/index exactness cases pass. The generic row copier fails during specialization because a symbolic loop variable indexed a Python layout tuple; candidate 85e2e4a37 constructs the strided address at specialization time. Request metadata with the shared-memory scan is 10–24% slower per call across the measured shapes (15 alternating blocks, 95% intervals exclude zero); candidate 05f74aca2 uses a warp fragment scan and consistent lane indexing. Neither implementation is connected to serving.

Third attempt: all 40 metadata/row-copy cases pass. TileLang's fragment scan still lowers through shared memory and retains the timing regression. Candidate 614873756 uses explicit lane shuffles, with no shared-memory scan, and extends correctness coverage across two warp-sized request batches. The expanded timing gate measures actual live versus maximum request counts, index maps and gather/scatter rows (30 alternating blocks, no statistically supported per-call regression allowed).
