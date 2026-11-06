"""CPU-only checks of the actual prototype's generation/publication methods.

CUDA-dependent construction is replaced with small fakes. These tests cannot
establish device correctness; that is the separate GPU gate.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Base:
    def append_block_ids(self, req_index, new_block_ids, overwrite):
        self.events.append(("stage", req_index, overwrite))

    def apply_staged_writes(self):
        self.events.append("apply_ids_and_counts")


class Epochs:
    def __init__(self, events):
        self.np = [0, 0]
        self.events = events

    def copy_to_uva(self):
        self.events.append(("publish_epochs", self.np.copy()))


def prototype_class():
    module = ast.parse(Path(__file__).with_name("incremental.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ("append_block_ids", "apply_staged_writes")]
    namespace = {"BlockTables": Base}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "incremental.py", "exec"), namespace)
    return namespace[cls.name]


class HostContract(unittest.TestCase):
    def make(self):
        obj = prototype_class()()
        obj.events = []
        obj.epochs = Epochs(obj.events)
        obj._epochs_dirty = False
        return obj

    def test_append_does_not_publish_a_replacement_generation(self):
        obj = self.make()
        obj.append_block_ids(0, ([1, 2],), False)
        obj.apply_staged_writes()
        self.assertEqual(obj.epochs.np, [0, 0])
        self.assertEqual(obj.events, [("stage", 0, False), "apply_ids_and_counts"])

    def test_same_count_and_empty_replacements_advance_generation(self):
        obj = self.make()
        for values in ([1], [2], []):
            obj.append_block_ids(1, (values,), True)
        self.assertEqual(obj.epochs.np, [0, 3])
        obj.apply_staged_writes()
        self.assertEqual(obj.events[-2:], ["apply_ids_and_counts", ("publish_epochs", [0, 3])])
        obj.apply_staged_writes()
        self.assertEqual(sum(isinstance(x, tuple) and x[0] == "publish_epochs" for x in obj.events), 1)

    def test_generation_exhaustion_fails_before_staging(self):
        obj = self.make()
        obj.epochs.np[0] = (1 << 63) - 1
        with self.assertRaises(OverflowError):
            obj.append_block_ids(0, ([3],), True)
        self.assertEqual(obj.events, [])


if __name__ == "__main__":
    unittest.main()
