# Variance margin for shortlist selection

Base commit `9ca41a512afa197fd68b133e26bf0ee7d66b0caa`, C36 image
`sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713`.
One intended variable: add a cluster-variation margin to coarse selection while
retaining original page grouping, budgets and native fine scoring.

This is an **untimed, out-of-band proxy**, with no serving implementation or
speed claim. It reuses the separately collected native operands after all-row
native replay. A zero-margin arm must reproduce the original selected keys and
source handoff/counts exactly. Every candidate retains the same native fine
scorer. Input hashes, every coefficient, per-query recall and all failures stay
in the receipt; tensors remain private on the node.

The score is `centroid_score + beta * RMS_distance * sum_h(abs(w_h)*norm(q_h))`.
RMS distance is measured over each balanced 64-key cluster around its stored
centroid. It is a heuristic, not a maximum-error bound. The predeclared grid is
`0, 0.01, 0.05, 0.2, 1`. A single coefficient maximizes mean native-selection
recall across the two fitting documents: dense top512 keys and source handoff
blocks, with each document/interface weighted equally. Ties choose the smaller
coefficient. Only zero and the chosen coefficient are then evaluated on the
separate document. That document was already inspected for the PCA probe, so
this is a reused diagnostic, not fresh statistical validation. Benchmark
questions, answers and model generations supply no fitting inputs.

The Torch calculations here are for deciding whether a runtime change is worth
building. A future serving implementation would have to maintain required
statistics on cache writes and include their full cost in request timing. No
reference, grading or duplicate scoring would enter timed serving. No candidate
is qualified by this proxy alone.

The screen completes at runtime commit
`ff00c355de3173400435db95488b493161261c91` in `runs/screen-01`. The zero-margin
control exactly reproduces original keys and source handoffs/counts on every
capture. Fitting selects beta0.01: its aggregate fitting recall is 75.50% versus
75.30% at zero. Larger margins degrade it, down to 40.94% at beta1; every result
is retained.

On the reused diagnostic document, dense top512 recall changes by +0.59, +0.13
and +0.25 percentage points in layers2/8/14. Source top512 key recall improves
by +0.32 points, and source handoff-block recall by +0.28. The full per-query
values and comparison are retained. Adjacent queries do not increase the
independent document count beyond one.

Conclusion: this small teacher-agreement gain establishes neither superior
final answers nor request performance. The margin remains an offline probe; it
does not enter timed serving. No model-quality ranking or promotion is claimed.

The standalone Torch allocator peaks at 376,239,104 allocated bytes and
648,019,968 reserved bytes. Those statistics exclude full-model memory
qualification. The lease is released at13:09:22 UTC, with no containers remaining.
Independent verification confirms our lease is absent and the checkout is clean;
by that check the other work has resumed under its own distinct lease. Both
serving clusters remain untouched by the standalone screen.
