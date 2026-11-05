# CED boundary local normalization and compact carry exchange

Status: draft experiment; no GPU execution, serving run, promotion or throughput
claim. Source review: https://github.com/christopherowen/vllm/pull/4.

## Identity and one variable

Deployment base: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
The frozen candidate-36 stack has 558 patches on vLLM
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`, producing head
`6d4e49d1cdbb6513819e47145e0e52fdab327b6b`, tree
`cc0fdf582d0bddff71c6a853e237bc9e87587832`. The single appended patch produces
`2b320049a5df8e62e1b3ee70f53304d6d55a15e3`, tree
`0247c83187c14ec7d9b00d72104c8e852963b1e4`. See `identity.json`, `source.json`,
`upstreams.lock.json` and `vllm/series` for full identities. Later candidate-36
or candidate-38 revisions are different builds and are not evidence for this tree.

The intended variable is the CED boundary's carried-row communication. Before,
all ranks gather the full hidden state, four-stream residual and three mix tensors,
then normalize every row and compact. After, each rank normalizes its local rows,
gathers the normalized input required for full global KV construction, and
exchanges only replay rows of the residual and mixes. Integer ownership selection
chooses the owning rank's slot; it introduces no floating-point reduction. Dense
selections retain the original full-carry path when replicated compact slots would
cost at least as many rows. The decoder leaves SP after this boundary.

Native checkpoint weights, projection arithmetic, rank-order sums, sampling,
attention backend, Engram storage and transport policy are unchanged. Global KV
still sees every encoder row. CED's existing replay selection remains authoritative:
non-final chunks can keep one dummy row and final prompt replay keeps its bounded
window. The new kernel is declared before startup freezes the kernel cache.

`payload-model.json` counts gathered tensor bytes per rank, not physical wire bytes
or elapsed time. At 16,384 rows and TP4 the estimate falls from 801.5 MiB to 160.2 MiB
with one replay row, or 180.0 MiB with 128. At 6,144 rows and TP3 the reduction is
about 75–80%. This saving occurs once per chunk at the CED boundary; it is not an
estimate of whole-model speedup. No extra replicated weights are introduced. A further local global-KV projection
variant is under investigation; this draft is not asserted to be the strongest
possible implementation before that comparison and hardware qualification.

## Validation and all attempts

Work ran in `/home/chris/spark-lab/ced-local-boundary` on Spaceport sp8. Source
preparation used one CPU and a 4 GiB scope; CPU test/compile containers used one CPU,
4 GiB, no swap growth, one worker and no GPU device or network. The Mac only
coordinated small reads and edits after the recovery instruction.

- A local interrupted preparation and temporary source edit produced no complete
  validation receipt before recovery; the temporary edit was lost. It was
  reconstructed in the isolated remote source branch. No performance run occurred.
- Initial CPU container entry `python` failed because the image uses `python3`.
  Importing source without the CPU adapter then failed because `tilelang_utils`
  intentionally exposes no TileLang backend on CPU. `cpu.py` supplies the installed
  TileLang definitions for declaration-only tests; it never executes a GPU kernel.
- `cpu-1.log`: 121 passed, 1 GPU-only test skipped. A startup declaration regression
  test was then added. `cpu-2.log`: 122 passed, 1 skipped, 10.51 s. These use the real
  model/layer forwards with mathematical stand-ins and simulated collectives; they
  do not certify native GPU arithmetic or distributed hardware behavior.
- The final CPU process peaked at 1,262,240 KiB RSS (1.204 GiB). Host MemAvailable
  was 115.58 GiB before and 115.17 GiB after. These are endpoint observations, not
  a continuous host-memory minimum. No serving-memory measurement exists.
- Initial lint reported new line length/import issues; those were corrected.
  All new-code/test lint passes with the source project's Ruff 0.14.0. Three
  inherited diagnostics in model/SP files remain untouched (unused import and two
  long lines). Whitespace checks pass.
- `fresh-base.log` replays all 558 base patches in a clean checkout and reproduces
  the recorded base head/tree. An inherited whitespace warning is retained.
  `fresh-candidate.log` applies the new patch, reproduces its head/tree and passes
  the full series whitespace check.
- Offline kernel parsing passed. Compilation attempts 1–3 used an obsolete target
  string, omitted the target context, then named a nonexistent context helper.
  Those harness errors are retained. `kernel-compile-4.log` compiles the actual
  kernel for SM121 through the installed TileLang/CUDA toolchain without a GPU.
  Peak process RSS was 1,207,936 KiB. GPU execution and graph replay remain pending.
- Doctor first rejected Spaceport's switched map for a triangle profile. The
  example triangle map passed default doctor but failed the archived arm's explicit
  master address. The final read-only site-map copy from dgx1 passes candidate
  doctor. These ignored maps are not deployment artifacts.
- Build input preparation first rejected an inherited patch count of 558; the new
  source manifest correctly records 559. Complete inputs were then prepared in a
  separate directory from verified cached inputs. `build-inputs-2.log` records a
  passing full `build check`. No image build or GPU smoke has run.
- The TP3 boot/repeat plan passes `lab.py --dry-run` (13 steps). The TP4 plan
  initially resolved an old 1025 reference through its archived base; both specs
  now pin the 1030 suite paths explicitly. TP4 planning refuses until its site
  ring map is available. Nothing was queued
  or applied. Existing dgx1/sp8 holds and requests were respected, not overwritten.

Raw successful and failed receipts are in `receipts/`; no unfavorable measurement
has been removed. No serving measurements exist for this candidate.

## Reproduce the completed checks

From the experiment's deployment checkout, with a clean candidate source checkout
at the commit above (commands run on the remote CPU host):

```sh
experiments/2026-11-05-ced-local-boundary/cpu.sh /path/to/vllm
bin/spark doctor
bin/spark --cluster-config experiments/2026-11-05-ced-local-boundary/tp3.json doctor
bin/spark --cluster-config experiments/2026-11-05-ced-local-boundary/tp3.json build prepare
bin/spark --cluster-config experiments/2026-11-05-ced-local-boundary/tp3.json build check
git diff --check
scripts/lab.py run experiments/2026-11-05-ced-local-boundary/qualify-tp3.json --dry-run
```

`cpu.sh` pins its dependency image by digest. For offline device compilation, use
that same CPU-only container and mounts, with `/check/compile.py` as the entry script.
`verify_source.py --cache /path/to/vllm --destination /new/checkout` performs
fresh source verification. It starts at the pinned upstream base, applies every path
in `vllm/series` with `git am --committer-date-is-author-date` and the repository's
fixed patch committer, then compares the resulting head and tree to `source.json`.

## Hardware qualification plan

Read dgx1's hold first. GPU-only checks need a free reservation and two clean
container boots of the published source: run the CED row-map/graph tests, full CED
kernel suite and native mHC rank-order tests, including nonfinite payloads and
frozen-cache startup. Compilation alone does not satisfy this gate.

The coordinated serving run requires explicit owner authorization, a published
clean deployment revision accepted by the lab guard, one built OCI digest on all
ranks, and the selected site's qualified cabling/map. Never bypass the publish or
shape guard. Resolve the restore profile against the live service immediately
before the window: the current hold names candidate-36/tp3-dev.json, so the lab
default must not silently restore a different release. Validate that profile with
`doctor --live` and pass it as `--production-config` to the authorized window. Build once, run the image smoke/label checks, read the sync/start
plan's live/shape lines, and use only `bin/spark cluster` for node operations.

`qualify-tp3.json` and `qualify-tp4.json` measure this candidate on a boot and repeat
boot against the immutable 1030 reference suites. They do not boot or remeasure
the released build. TP4 needs its own four-node ring window; it cannot run inside
the current three-node triangle window. The reference includes earlier differences
between 1030 and frozen candidate-36, so a release delta alone cannot attribute all
movement to this patch. Keep that limitation explicit alongside per-boundary timing.

Both release suites cover 32K, 131K, 262K, 524K and 1M source-text prefill, model
quality, closed-loop decode, mixed long-prefill/short-request TTFT and telemetry;
startup memory is collected on both boots. Compare confidence intervals and all
failures, startup/steady memory, temperature/clocks, cold prefix-cache salts, and
short-request TTFT before accepting the change. Add the long-needle quality run
with fixed corpus, needle positions and expected answers to that authorized window;
no long-needle result is claimed here. Retain prompt bytes/token IDs and digests
alongside future raw receipts before interpreting cross-run workload identity.

The serving checkpoint is DeepSeek-V4.1-Flash-1017 at
`1bae0e1088993b9b82de08f59de5be5a34e42170`. TP3 uses 6,144 rows per chunk; TP4 uses
16,384. `recorded-prefill.json` extracts the existing reference means and 95%
intervals. Its metric is prompt tokens divided by request-to-first-content time,
with `max_tokens=1`, temperature zero and a fresh prefix-cache salt. Those old
receipts do not preserve exact prompt bytes/token IDs and do not prove a cold disk
Engram cache. They must not be relabeled as new measurements or an exact-byte match.

Conclusion: the communication change is implemented, CPU-tested, device-compiled
and reproducible from the pinned series. Native GPU equality, distributed behavior,
TTFT/memory effects and throughput benefit are still unmeasured; do not promote.
