"""Stand-in for tblib in place of #984 /tmp/stub_pkgs (not committed by the PR): tests/conftest.py
imports tblib.pickling_support and calls install(); the serving image lacks tblib. Same behaviour
as candidate-53 bundles/release-tests/tblib_stub.py: install() does nothing."""
from . import pickling_support  # noqa: F401
