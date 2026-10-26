"""pytest plugin: while tests whose node id contains one of LOGSTATE_MATCH's parts set up,
run and tear down, print the stack of every call that sets the root logger, or a handler
attached to it, to ERROR or above (and of logging.disable), once per call site."""

import logging
import os
import traceback

import pytest

MATCH = [part for part in os.environ.get("LOGSTATE_MATCH", "").split(",") if part]
_seen = set()
_active = False
_logger_set = logging.Logger.setLevel
_handler_set = logging.Handler.setLevel
_disable = logging.disable


def _report(what, level):
    stack = traceback.extract_stack()[:-2]
    site = (stack[-1].filename, stack[-1].lineno) if stack else None
    if site in _seen:
        return
    _seen.add(site)
    # The whole stack: which import pulls the library in.
    frames = [frame for frame in stack if "importlib._bootstrap" not in frame.filename]
    print(f"\nTRAP {what} -> {level}:\n" + "".join(traceback.format_list(frames)), flush=True)


def _logger_setlevel(self, level):
    if _active and self is logging.getLogger() and logging._checkLevel(level) >= logging.ERROR:
        _report("root logger setLevel", level)
    return _logger_set(self, level)


def _handler_setlevel(self, level):
    if _active and self in logging.getLogger().handlers and logging._checkLevel(level) >= logging.ERROR:
        _report(f"root handler {type(self).__name__} setLevel", level)
    return _handler_set(self, level)


def _logging_disable(level=logging.CRITICAL):
    if _active:
        _report("logging.disable", level)
    return _disable(level)


logging.Logger.setLevel = _logger_setlevel
logging.Handler.setLevel = _handler_setlevel
logging.disable = _logging_disable


def _matches(item) -> bool:
    return any(part in item.nodeid for part in MATCH)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global _active
    _active = _matches(item)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    if _matches(item):
        root = logging.getLogger()
        print(f"\nTRAP after {item.nodeid}: root level {root.level}, handler levels "
              f"{[h.level for h in root.handlers]}", flush=True)
