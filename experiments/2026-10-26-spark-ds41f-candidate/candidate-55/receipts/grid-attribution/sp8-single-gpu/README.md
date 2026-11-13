# Single-GPU screen on sp8, 2026-11-12

Copied from `/tmp/c55-fusions` and `/tmp/iket-diag` on sp8, where a second line of work on candidate-55's grid ran
them (17:55Z to 18:13Z). One boot an arm on one GPU, dummy weights at a TP4 rank's shapes, 60 replays a shape; the
repeat of `c55_l2cap` did not run and `c55_pr1081` did not boot.

- [run_all_arms.sh](run_all_arms.sh), [run_single_var_screen.py](run_single_var_screen.py): the runner. It needs
  `iket_step.py` and the kernel caches under `/tmp/iket-diag` on sp8, which are not copied.
- [results/](results): each arm's median and mean forward, drafter and step time per shape.
- `cupti-c54-l2cap-summary.json`, `cupti-c55-l2cap-summary.json`: the per-replay times of the two CUPTI-traced
  boots (20 replays). The traces themselves, 14 MB a shape, stay on sp8 under `/tmp/iket-diag/cupti-c5{4,5}-l2cap`.

Read in [grid-attribution](../../../grid-attribution/README.md#single-gpu-screen-on-sp8-one-boot-an-arm).
