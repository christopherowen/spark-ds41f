# TileLang decode kernels against B12X

Goal: before the TileLang family becomes the default, match or beat B12X on
every kernel it owns in the TP4 decode step.

The 1M-recipe decode profiles (rank 0, median six-row step) left these behind
B12X per call: the attention Q-B (24.4 against 21.2 µs), fused Q-A/KV (20.3
against 18.2), indexer Q-B (45.4 against 21.8), the DSpark 6400-wide
projection (196 against 177) and the sparse MLA decode kernel (24.4 against
19.7, though B12X adds a 6.2 µs page-mapping kernel). Everything else the
family owns was already ahead.

Work in progress; results follow.
