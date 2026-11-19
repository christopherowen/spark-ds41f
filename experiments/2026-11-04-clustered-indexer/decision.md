# Reject the tested grouping for production

The integrated four-boot Spaceport comparison now answers the request-speed
question. Against the same unreleased C36 image and scratch-read guard with
search off, median API request times increase1.8–3.8% for the six256-token
prose/code decode points,14.5% for nominal32K source prefill and34.4% for
nominal131K source prefill. Actual prefill inputs contain28095–30739 and
116664–121877 tokens. All564 requests have matching model inputs, output digests,
token counts and finish reasons across arms. Every boot passes5/5 quality checks.

See `runs/serving-window-02/comparison.md` and `comparison.json`. Two boot medians
per arm give descriptive Student-t intervals (df1); normality is unverified.
Several small decode effects span parity; both prefill effects remain penalties.
The existing Spaceport4KiB kernel and stock fan control are held constant.
No released build was measured again, and no release-relative speedup is claimed.

The first pilot completed one LSA arm and is retained in `serving-window-01`.
It yielded after its current job to repair a benchmark-method risk: adaptive
sample counts changed the RNG consumed before source-prefill generation. The
corrected matrix uses four fixed samples and hashes every model input before
native timing. This is a retained pilot/method correction, not a candidate
correctness failure or an omitted unfavorable run.

Earlier isolated evidence also remains: twelve actual C36 operands replay exactly
twice but retain every visible key, and the tested signed-head/smaller-group/
semantic-cluster/affine screens find no useful pruning. The serving adapter passes
two20-case graph processes, a frozen-declaration process and all three dedicated
sanitizers. Its initial flat-scratch view error and correction are retained.

The original C37 service is restored on every Spaceport rank. Independent checks
verify image/settings, removal of search mounts/environment, unchanged original
checkouts and absence of all native reservations. No production configuration,
image tag or immutable baseline was promoted. The independent top-k guard merged
in PR878. The experimental search remains draft research.

Do not spend full release qualification on this losing candidate. A different
search needs a measured real-workload benefit first. Full teacher-forced
distributions, mixed traffic and admission qualification are not claimed.
The current gate is generated-token equivalence, as defined in `acceptance.md`.
Selected keys and intermediate values may differ. Validation and proof work stay
outside the timed implementation; changed generated tokens still need independent
evidence of superior quality before acceptance.
