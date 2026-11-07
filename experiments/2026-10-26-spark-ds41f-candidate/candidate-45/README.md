# Candidate 45: explicit compiler preparation rework

These are draft source/build profiles on validated, default-enabled C40. The
intended dependency variable is the corrected TileLang compiler packet in
`experiments/2026-11-05-compiler-prepared-rework`: the ten unchanged authored
patches, one focused explicit-preparation correction and a source-bound package
VERSION. C40 vLLM/kernel source and every non-TileLang source record are retained.

The compiler refresh is maintenance and retained-feature preservation. No
speedup is required or claimed. Correctness and no material incremental
core/serving regression versus both saved C40 boots are required. The explicit
four-image request surface retains the recorded qualification surface; the
global default of 67 is untouched. The development profiles add only the
existing engine-grid RPC option. TP4 source configuration grants no TP4 hardware
or release qualification.

Recheck and record the best validated default-enabled runtime base before the
actual image freeze. C46 is presently unmeasured; it is not a source-preparation
dependency. A later validated base change needs explicit source/profile review,
not silent inheritance. No deployment execution commit is frozen by this draft.

The intended tag is `spark-ds41f:candidate-45`; the actual digest is unknown. No
image, native gate, serving boot or performance point exists for C45. These
profiles are not launch ready: the root coordinator must adapt and review new
controls, complete the listed source checks remotely, publish one exact clean
execution commit and use normal shared scheduling. The historical C42 gate
runner is not a runnable C45 control. See the source packet's
`control-readiness.json` for source-bound cache first-use, installed/loaded-code
identity and unchanged full qualification requirements.

C42's failed native image and all errors remain preserved. C44 stays withdrawn
and unmeasured. Neither is resumed, retagged, rebuilt or remeasured by this draft.
