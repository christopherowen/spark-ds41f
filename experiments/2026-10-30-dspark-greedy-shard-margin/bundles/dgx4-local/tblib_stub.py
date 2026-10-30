"""pytest plugin: tblib where the serving image lacks it."""
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
