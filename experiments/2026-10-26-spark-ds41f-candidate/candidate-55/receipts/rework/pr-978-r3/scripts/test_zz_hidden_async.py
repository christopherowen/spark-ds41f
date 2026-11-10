# Dev-only wrapper (not part of the patch): tests/v1/engine/test_admission_control.py's
# async test, which the release environment collects but cannot run (no async plugin),
# run with asyncio.run so that a change reading attributes its request stand-ins lack
# shows up.
import asyncio

from tests.v1.engine import test_admission_control as admission


def test_concurrent_single_request_admission_respects_limit_run():
    asyncio.run(admission.test_concurrent_single_request_admission_respects_limit())
