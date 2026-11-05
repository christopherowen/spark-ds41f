# Decision

Retain the focused source change as a draft for serving qualification. Two fresh
Spaceport boots establish a component gain with exact integer metadata, and the
selected placement is faster than the eager, direct-kernel and separate-graph
alternatives tested. The static-consumer lower bound leaves approximately
1.55 us of residual refresh cost at 8 requests/8K; further fusion has limited
headroom in this fixture.

The isolated gain is mostly hidden by preceding GPU work: with about 775 us
queued first, the measured saving is 2.940 us (95% CI 2.578–3.308) on boot 1 and
2.663 us (2.418–2.905) on boot 2. This is 0.376% and 0.340% of that synthetic
sequence. Neither the isolated 97.6–97.7% component reduction nor the overlap
percentage is a serving-throughput prediction. See [micro-results.md](micro-results.md).

No serving run, source-complete image build, target sampling/rank validation or
promotion has occurred. Keep launch disabled and both PRs draft. Full serving
qualification remains necessary before claiming a user-visible performance gain.
The owner explicitly allowed the isolated Spaceport screen; it respected the
local GPU/CPU reservations, yielded between boots, and released its hold. The
serving cluster was not changed.
