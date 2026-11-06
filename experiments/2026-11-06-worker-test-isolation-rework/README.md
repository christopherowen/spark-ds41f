# Worker-test lazy-reference isolation rework

Candidate46 is reserved for a focused correction of C43's release-test failure.
This packet is a **reviewed patch draft, not a prepared or qualified source**.
CPU fixture proof, canonical replay, source head/tree, image, native gates and
both metric boots remain pending safe remote capacity. No C43 gate or image is
rerun or resumed. Its original failed verdict remains unchanged.

C43 built successfully. The native gate recorded8103 passes,360 skips and one
failure before top-k initialization, functional or serving work. The failure is
`test_gpu_verified_logprob_token_ids.py::test_tilelang_worker_modules_hold_no_live_triton_reference_kernels`:
`logprob._topk_log_softmax_kernel` is unexpectedly a real module global.

The relevant test and runtime files are byte-identical in C40 and C43. Adding
C43's two frontend regression files shifts the ten-file runner groups: the
logprob-softmax producer test moves from C40 chunk011 into C43 chunk012 before
the lazy-global consumer. The producer patches a lazily exposed reference using
`monkeypatch.setattr`; cleanup can restore the resolved reference as a real
attribute. Runtime `logprob.__getattr__` itself never caches that attribute.
This is source evidence of a fixture-isolation mechanism, not yet a native
reproduction or proof that the full corrected gate passes.

A read-only scan of all78 worker test files found one setter matching the
consumer's complete lazy-reference map, at producer line449. Other kernel and
dynamic setters inspected address real TileLang helpers or ordinary globals.
No other matching lazy-reference setter was found; the invariant's remaining
checks are preserved rather than waived.

The draft changes only the producer fixture: dictionary-based poisoning avoids
lazy attribute resolution during setup/teardown; an explicit context removes
the poison before a post-context laziness assertion. Every original numerical,
routing, empty-shape and launch-count assertion remains. The consumer invariant,
manifest138 files, chunk size, runtime code and C43's two frontend patches stay
unchanged. It adds no tolerance, deselection, fallback, kernel or serving policy.
The existing parameterized producer becomes the focused cleanup regression.

The first inspection helper did not handle the runner's tuple constant
assignment; its KeyError is retained. Two subsequent small metadata probes
also encountered a vanished hold and an incorrectly assumed series path; these
are inspection setup errors, not native or candidate failures. No state changed.

A queued `ced-local-boundary` request explicitly asks for DGX4 kernel work next.
Only bounded read-only source inspection and small patch drafting proceeded;
no replay, environment install, Docker container, CPU tests, GPU or serving
operation ran. Further work must use a safely coordinated remote CPU slot,
one CPU and at most8GiB address space, with the host CPU-work lock. The Mac is
limited to coordination, small reads/edits and concise results.

The corrected draft now passes read-only `git apply --check` against exact C43
and both the proposed file and witness parse as Python3.12 on DGX4. The first
hunk-only draft omitted trailing file context; that formatting error, original
diff and corrected compatibility check remain preserved. No source was applied.

Both new frontend files passed their actual C43 gate cases (10 and20). C40's
recorded producer cases passed in chunk011 and consumer passed in012; C43's
producer cases passed before the consumer failed within012. These are old native
records read as evidence, never rerun measurements.

Pytest-version uncertainty is explicit: [8.4.x implementation](https://docs.pytest.org/en/8.4.x/_modules/_pytest/monkeypatch.html)
resolves then restores an attribute, while the [current implementation](https://docs.pytest.org/en/stable/_modules/_pytest/monkeypatch.html)
adds dictionary-state handling. The actual image's pytest version/source hash
and the drafted fixture witness must still be captured before calling native
causality proven. No pytest dependency was installed on the host.
