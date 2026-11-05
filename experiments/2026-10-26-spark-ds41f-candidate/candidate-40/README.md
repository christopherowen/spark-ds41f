# Candidate 40: top-k scratch-read guard

Base: candidate-39 default CopyDrafter, measured in
`experiments/2026-11-04-copy-default-screening`. One runtime variable: guard
inactive top-k scratch reads with the lazy outer condition from #878. No draft
policy, model, weights, topology, compiler or hardware setting changes.

The fresh full canonical replay reproduced candidate-39 exactly before applying
the guard. Source identity and all qualification receipts belong to
`experiments/2026-11-04-topk-live-qualification`. This is not a release promotion
or TP4 hardware authorization. Build once, gate the actual image without source
overlays, then measure fresh TP3 boots and restore the tracked candidate-36
profile through the normal shared window protocol.
