import sys
import types

tblib = types.ModuleType("tblib")
pickling_support = types.ModuleType("tblib.pickling_support")
pickling_support.install = lambda *args, **kwargs: None
tblib.pickling_support = pickling_support
sys.modules["tblib"] = tblib
sys.modules["tblib.pickling_support"] = pickling_support
