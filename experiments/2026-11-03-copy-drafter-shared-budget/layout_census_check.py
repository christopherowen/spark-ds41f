"""Check of overlay/minimal-crc's layout CRC (the minimal-m8-nodead arm's)."""
from types import SimpleNamespace

from vllm.v1.worker.gpu import model_runner as mr

lines = []
mr.logger.warning = lambda fmt, *args: lines.append(fmt % args)


def runner(budgets):
    r = SimpleNamespace(adaptive_verification=SimpleNamespace(_batch_budget=None))
    for eligible, toks in budgets:
        r.adaptive_verification._batch_budget = (eligible, {}, 0)
        mr.GPUModelRunner._layout_census(r, list(eligible), toks)
    return r


steps = [({"a": 5, "b": 16}, 18), ({"a": 8}, 9)] * 512
one, two = runner(steps), runner(steps)
assert one._layout_crc == two._layout_crc and one._layout_reads == 1024
other = runner([({"a": 5, "b": 15}, 18), ({"a": 8}, 9)] + steps[2:])
assert other._layout_crc != one._layout_crc
assert len(lines) == 3 and lines[0].startswith("COPY-LIMITS reads=1024 layout=1 crc="), lines
print(lines[0], "| layout census OK")
