# PR #1012 rework, round 1: two release tests broken by stale fixtures

**Miss.** In candidate-47's release tests (dgx4, image sha256:52c7019c0c5a), two tests failed:

- `tests/models/test_deepseek_v4_1_lsa.py::test_lsa_opt_in_32k_to_1m_routing_and_metadata_wiring`: no attribute `compressor`.
- `tests/v1/worker/test_workspace.py::test_v41_tilelang_attention_reserves_scratch_in_its_model_lane`: no attribute `indexer`.

**Cause.** 0767 adds `overlap_specs = self._decode_overlap_workspace_specs(self.DECODE_CHUNK)` to `_build_tilelang_plans` (`vllm/models/deepseek_v4_1/tilelang/attention.py:329` in the integrated tree). That function (`attention.py:1094-1111`) reads `self.indexer`, `self.compressor` and the indexer's `wk`, `wq_b` and `weights_proj`.

Both tests build the layer by hand (`__new__` plus `nn.Module.__init__`) and then call `_build_tilelang_plans()`:

- the workspace test never sets `indexer`;
- the LSA test sets an indexer without `compressor`, and its fake indexer lacks the three projections.

**Why the tests, not the code.** Every real `DeepseekV4Attention` sets `compressor` (`attention.py:607`, a compressor only on KV sources) and `indexer` (618/631). The indexer always has `wq_b` and `weights_proj`, and has `wk` exactly when it owns the keys (line 411, `owns_k=self.is_kv_source`).

docs/testing.md treats hand-built objects that miss attributes the code now reads as a fixture fix. Both tests come from our own series (0506 and 0741). `getattr` defaults in serving code would only hide incomplete fakes: in the LSA test, real KV-source layers (2 and 20) would take the no-compressor branch.

Counterpoint: 0747 took the code-side route in the same function (`getattr(self, "n_local_heads", 1)`).

**Change.** Tests only, 16 lines; the assertions are unchanged.

- LSA test: the layer gets `compressor = SimpleNamespace()`, and its indexer gets `wk`, `wq_b` and `weights_proj` set to `None`.
- Workspace test: the index layer gets a compressor and an indexer with those projections; the window-only drafter gets neither.
- The workspace test's exact sizes still hold: the 512 B overlap reservation sits under the 2 MiB index scratch.
- The original author and date are kept.
- `git am` of 0762..0766, then the corrected 0767, then the original 0768 and 0769 applies cleanly; `git diff --check` is clean.

**Evidence (sp8).** The vLLM tree is copied out of the 1107 image, candidate-47's eight pure-Python patches are applied, and the result is bind-mounted over the 1107 image. No build. Raw outputs are under `sp8/`.

- On candidate-47 the two tests fail, matching dgx4. With the fix they pass.
- The seven files (the two above plus #1012's own five): 155 passed and 2 failed on candidate-47; 157 passed on both fix boots.
- The bench on both boots, TP4 and TP3: bit-exact in eager and graph replay for every family at M=1..48. The 8-indexed-layer step delta is negative in every cell: −32 to −76 µs warm, −36 to −62 µs cold-L2.

**Caveat.** #1012's bench rebuilds the two schedules from kernel modules and never imports the patched attention or compressor modules. It shows the overlap's potential, not the patched forward path. #1012's serving effect is judged from candidate-47's engine decode grid.

The root cause and change are the rework sub-agent's (round 1); the coordinator wrote up the verdict from the collected outputs.
