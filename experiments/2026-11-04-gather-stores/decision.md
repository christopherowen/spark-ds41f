# Retain for review

The gather patch passes its isolated correctness and CUDA launch-inventory gates
on both final boots. All timings, including regressions, remain in results.json.

No serving deployment or promotion. Qualification here is a pinned-source
GB10 kernel/unit screen. Full image, model serving and relevant multi-node
transport qualification remain separate gates.

The final correction is in `../2026-11-05-gather-tail-performance/`. It passes
combined correctness twice; tiny-tail performance remains unresolved. This PR
stays in draft pending its direct kernel-duration and full inventory checks.
