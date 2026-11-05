# Retain for review

The sampler patch passes its isolated correctness and CUDA launch-inventory gates
on both final boots. All timings, including regressions, remain in results.json.

No serving deployment or promotion. Qualification here is a pinned-source
GB10 kernel/unit screen. Full image, model serving and relevant multi-node
transport qualification remain separate gates.

The accepted performance correction and strict two-boot gates are in
`../2026-11-05-sampler-copy-performance/`. This historical result is retained.
