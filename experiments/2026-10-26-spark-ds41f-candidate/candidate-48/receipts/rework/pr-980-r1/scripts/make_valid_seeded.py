"""Write screen_zero_broadcast_sync_rowsfix_valid_seeded.py: the harness-fixed screen
(rowsfix_valid) with --sampling-seed, handed to SamplingStates as the configured seed
(ModelConfig.seed) when given, so the candidate modes run the seed-keyed path."""
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


sub(
    "        states = SamplingStates(num_reqs, vocab_size)\n",
    "        states = (\n"
    "            SamplingStates(num_reqs, vocab_size)\n"
    "            if args.sampling_seed is None\n"
    "            else SamplingStates(num_reqs, vocab_size, seed=args.sampling_seed)\n"
    "        )\n",
)
sub(
    '    parser.add_argument("--bench-iters", type=int, default=1000)\n',
    '    parser.add_argument("--bench-iters", type=int, default=1000)\n'
    '    parser.add_argument("--sampling-seed", type=int, default=None)\n',
)
sub(
    '            "mode": mode,\n            "steps": steps_verify,\n',
    '            "mode": mode,\n            "sampling_seed": args.sampling_seed,\n            "steps": steps_verify,\n',
)
open(dst_path, "w").write(s)
