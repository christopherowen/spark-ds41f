# CED and draft prefix replay

The current r6x source makes both decoder and drafter SWA groups private: `SwaCache.get_kv_cache_spec` sets `prefix_cache_enabled=False` and `prefill_replay_window=128`. The hybrid coordinator omits those groups from prefix lookup, and the manager caps the hit at prompt length minus at least 128 replay tokens. A same-request continuation retains its private written tail.

The premise that DSpark context pages are shared on a prefix hit does not hold for this release. The existing regression checks fresh private pages, 128-row replay, and short prompt boundaries. Run those tests on the unmodified image before proposing another draft-window change. The CED skip still needs its own unwritten-row and sampled acceptance gates.
