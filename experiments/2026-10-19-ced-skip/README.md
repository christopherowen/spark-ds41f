# CED nonfinal row skip: CPU preparation

Candidate source 54c4fb11a. Startup requires TileLang, aligned chunked prefill, DSpark and a threshold covering the replay reach (199 tokens at k=5). Every multimodal request is ineligible. A skipped chunk records its promised next start and row count. A changed or shortened successor raises before metadata or GPU work; no replay-floor adjustment or environment switch remains.

This proposal is based on the pre-port CED call sites and is not wired into serving. It must be reconciled with the qualified TileLang ports before GPU/full-model gates. The simulation loads the actual candidate functions through AST and fingerprints target dependencies against no skip. Valid schedules have no target differences or unwritten target reads; 4,981 shortened promised successors fail before staging. Arbitrarily shortened unpromised chunks are a separate fault case and are not claimed deterministic.

GPU tests, teacher-forced identity, multi-request numerical identity, sampled prefix-hit acceptance and release prefill remain required. No promotion.
