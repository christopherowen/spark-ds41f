"""pytest plugin: tblib where the serving image lacks it.

tests/conftest.py imports tblib and calls pickling_support.install(), which lets an
exception raised in a worker process carry its traceback when pickled back to the test.
The serving image does not install test extras; without the stand-in the conftest cannot
load at all. With it, worker failures still raise, carrying their message only.
"""
import sys
import types

try:
    import tblib  # noqa: F401
except ImportError:
    tblib = types.ModuleType("tblib")
    pickling_support = types.ModuleType("tblib.pickling_support")
    pickling_support.install = lambda *args, **kwargs: None
    tblib.pickling_support = pickling_support
    sys.modules["tblib"] = tblib
    sys.modules["tblib.pickling_support"] = pickling_support
