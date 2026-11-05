"""Diagnostic CPU call origins for selected CUDA ATen operators, no data reads."""
import json
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class GPUCallOrigins(TorchDispatchMode):
    def __init__(self, path):
        super().__init__()
        self.handle = path.open('w', buffering=1)
        self.seen = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name.startswith(('aten.index', 'aten.cat.', 'aten.stack.', 'aten._to_copy.',
                            'aten.clamp', 'aten.arange.', 'aten.fill_', 'aten.cumsum.')):
            tensors = []

            def visit(value):
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        visit(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        visit(item)

            visit(args)
            visit(kwargs or {})
            if any(t.is_cuda for t in tensors):
                stack = traceback.format_stack(limit=24)[:-1]
                meta = [dict(shape=list(t.shape), strides=list(t.stride()),
                             dtype=str(t.dtype), device=str(t.device)) for t in tensors]
                record = dict(op=name, tensors=meta, stack=stack)
                key = json.dumps(record, sort_keys=True)
                if key not in self.seen:
                    assert len(self.seen) < 16384, 'unexpectedly many call origins'
                    self.seen.add(key)
                    self.handle.write(json.dumps(record) + '\n')
        return func(*args, **(kwargs or {}))
