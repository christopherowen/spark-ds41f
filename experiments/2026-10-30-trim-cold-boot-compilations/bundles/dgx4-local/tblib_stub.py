import sys
import types

tblib = types.ModuleType("tblib")
tblib.pickling_support = types.SimpleNamespace(install=lambda *a, **kw: None)
sys.modules.setdefault("tblib", tblib)
sys.modules.setdefault("tblib.pickling_support", tblib.pickling_support)
