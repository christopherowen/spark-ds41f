"""Production TileLang kernels recompiled with ordinary loads.

TileLang declares a kernel's buffer parameters ``__restrict__`` (and the read-only ones
``const``), which lets nvcc route their loads through the read-only cache path
(``ld.global.nc``, ``LDG.E.CONSTANT``). Triton's loads are ordinary global loads. A
kernel recompiled here has every buffer it never stores to listed in
``tl.non_restrict_params`` (what ``T.annotate_restrict_buffers`` sets), so those loads
are ordinary ones; the arithmetic and the stores are unchanged.

``Plain(module, pass_configs).active()`` makes the module's ``_kernel(factory, *key)``
return the recompiled kernels while it is entered (the V2 runner's and the sampling
family's wrappers launch through it).
"""
import contextlib

import tilelang
import tvm

_TIR = getattr(tvm, "tirx", None) or tvm.tir


def plain(prim):
    """``prim`` with its never-stored buffer parameters not ``__restrict__``."""
    stored = set()

    def visit(node):
        if isinstance(node, _TIR.BufferStore):
            stored.add(node.buffer.data)

    _TIR.stmt_functor.post_order_visit(prim.body, visit)
    inputs = [buf.data for buf in prim.buffer_map.values() if buf.data not in stored]
    return prim.with_attr("tl.non_restrict_params", inputs)


def compile_plain(factory, key, pass_configs):
    config = {getattr(k, "value", k): v for k, v in pass_configs.items()}
    return tilelang.compile(plain(factory.get_tir(*key)), pass_configs=config)


class Plain:
    """``module.<attribute>`` replaced while ``active()``: ``_kernel(factory, *key)``,
    or, with ``factory``, a one-factory resolver called as ``attribute(*key)`` (the
    lookback's ``_bound``)."""

    def __init__(self, module, pass_configs, attribute="_kernel", factory=None):
        self.module, self.pass_configs, self.attribute = module, pass_configs, attribute
        self.factory = factory
        self.cache = {}

    def kernel(self, *args):
        factory, key = (self.factory, args) if self.factory else (args[0], args[1:])
        slot = (id(factory), key)
        if slot not in self.cache:
            self.cache[slot] = compile_plain(factory, key, self.pass_configs)
        return self.cache[slot]

    @contextlib.contextmanager
    def active(self):
        saved = getattr(self.module, self.attribute)
        setattr(self.module, self.attribute, self.kernel)
        try:
            yield
        finally:
            setattr(self.module, self.attribute, saved)
