# Explicit functional evidence before timing

Base: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
This is a CPU-only validation change; no runtime configuration, image or upstream
patch changes. The intended variable is whether declared collector requirements
are checked before a functional phase can advance to timing.

The shared lab previously used collector process exits without interpreting
functional verdicts. A collector can validly exit zero while reporting an
observational difference. A consumer requiring a particular verdict must declare
and enforce it. The new contract distinguishes requirements from observations,
rejects incomplete evidence and preserves the runner's existing restore behavior.

C43 adopts explicit contracts for its three greedy gates and sampled isolated
repeat, plus complete private sampled recordings. Its sampled batch differences
remain observations. C40's historical audit already recorded these differences
as a limitation and required only the isolated repeat; it did not claim sampled
batch invariance. C40/C41/C42 specs and all published evidence remain unchanged.
Other collectors still use their existing exit/audit contracts; this does not
claim that every functional property is covered by the new declarations.

The separate verification-budget attempt in [PR 917](https://github.com/christopherowen/spark-ds41f/pull/917)
failed its then-declared sampled batch requirement and restored C36 before release.
It remains failed, with no timed result or candidate qualification. This change
neither diagnoses a runtime sampling bug nor retroactively passes that attempt.

## CPU proof and validation

`run_cpu_proof.py` runs real short-lived collector subprocesses against the base
lab and the new lab, with cluster/window/timing actions mocked. Identical inputs
exercise false, missing and malformed required evidence, a failed required
isolated repeat, observed batch differences and a valid required verdict. It
records whether the timing step was reached and whether failure restoration was
requested. A request/response fixture also exercises all 24 sampled records,
separate token/logprob differences and manifest tamper detection in the unit tests.

Validation runs on Spaceport8, in a clean published checkout, under the shared CPU
locks with one core and a 2 GiB limit. Exact commands, all exits and memory
observations will be retained under `runs/`; no GPU operation is part of this work.

## Future row-cost qualification contract

Before another hardware turn, a new plan must declare the intended requirements
and workload. Greedy token/text equality and native logprob equality remain
required. Sampled isolated repeats require identical request controls; batch
trajectory differences are observations unless the full native RNG inputs and
step grouping can be shown identical. HTTP request seeds alone cannot establish
that condition. Private per-case recordings preserve diagnosis material but do
not expose native logit positions or speculative steps.

Sampling quality still needs the existing release/quality gates and appropriate
distributional or teacher-forced checks, alongside the source/math tests. An
observed batch difference is neither a sampling-quality failure by itself nor a
quality pass. Timed work and a repeat boot can begin only after the predeclared
requirements pass. Compare against recorded reference runs; no reference rerun
or new hardware window is authorized by this CPU change.
