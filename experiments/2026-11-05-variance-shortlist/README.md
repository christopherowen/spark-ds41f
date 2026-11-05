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
