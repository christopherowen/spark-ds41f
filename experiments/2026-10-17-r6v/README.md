# r6v: r6u with the row all-to-all over a four-node ring

r6v is r6u ([experiments/2026-10-17-r6u](../2026-10-17-r6u/README.md)) with vLLM 0180.

## Why

r6u could not boot on the four-node ring (2026-10-17, three attempts): Engram's
sequence-parallel exchange (`all_to_all_rows`, vLLM 0150, since r6p) sends every peer its
block directly, and on a switchless ring ranks two apart share no cable. NCCL's queue pair to
the opposite rank timed out in memory profiling (`ibv_modify_qp ... INIT -> RTR`, 110, local
10.23.1.2 to remote 10.14.1.4). r6s, which brought the exchange, was released on the triangle
only; r6n, from before it, boots on the ring.

## Change

- **vLLM 0180**: at four ranks `PyNcclCommunicator.all_to_all_rows` sends its neighbours their
  blocks and each half of the opposite rank's block through one neighbour, which forwards it
  in a second group, as the rank-order reduce-scatter's exchange does. The four links carry
  equal bytes; blocks keep their real lengths and their bits. Other rank counts keep the direct
  exchange, so TP3 runs r6u's code.

## Runs

1. `build.json`: the image on dgx2, copied over the ring's links, and the release tests.
2. `gate-tp4-1017.json`: temperature-0 and greedy determinism, 16, 8 and 4 distinct streams.
3. `release-tp4-1017.json`: the `release` suite, r6v then r6n (on its production ring curves),
   both on 1017. Owner: only the benching promotion and the release announcement need.
