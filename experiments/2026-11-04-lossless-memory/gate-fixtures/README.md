# Frozen-runner qualification fixtures

The first expert-scale full gate hit inherited test-fixture errors. Its
component gate passed 118 tests on every rank; the full run is retained and
does not qualify serving. Runtime worker/runner/profiling files and the three
affected original test files are byte-identical to the frozen promoted base.

The repository already carries candidate-38 corrections 0685–0687 for these
fixtures. They supply the fields/methods the real runner/worker owns, verify
one profiling block per group plus its null block, and model the free-memory
input for the native-accounting unit test. The real CUDA allocation, graph
allocation/release and accounting code still run. The accounting assertion is
exact; host-wide free-memory drift is measured by serving memory observations
rather than asserted as a deterministic unit-test input.

`fixes.patch` applies to fresh canonical scale and Engram source worktrees.
`manifest.json` pins the original and corrected test sources and proves the
affected runtime files unchanged. Test containers mount only these three
corrected test files. Serving containers, image IDs, runtime code, all selected
132 catalog files and exclusions remain unchanged. No tolerance waiver is added. The original
bundles and plans stay alongside the distinct `full-gate-fixtures` jobs; no
failed observation is overwritten.

After the original window restores production, the controller can reuse its
exact image and passed native component gate with `--reuse-kernel-from`. It
requires the same per-rank image identities/layers, original successful build
receipt, native verdict hash, every tracked bundle input and mounted external
source against its previous published commit. Non-Git bundle files are refused. It refuses prior serving boots,
a failed component gate, or existing output for the replacement full gate.
Every selected test must then pass in the new full gate before either serving
boot can start. Reusing both completed gates remains a separate option.
