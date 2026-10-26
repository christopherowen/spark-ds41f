"""pytest plugin: a test that needs a resource these nodes do not have skips, naming it.

The release tests run offline, with only the deployed model's weights, in the serving image.
A test whose failure is exactly one of these is reported as skipped with the missing resource
as its reason, so a file mixing such tests with runnable ones can still gate on the rest:

- a Hugging Face model or config that is not cached (offline lookups, ``ModelConfig``'s
  "Invalid repository ID or local directory" for a hub id);
- ray, which the serving image does not install.

Any other failure, including any other ``ModelConfig`` validation error, stays a failure.
"""
import re

import pytest

_OFFLINE = re.compile(
    r"Invalid repository ID or local directory specified: '([^']+)'"
    r"|LocalEntryNotFoundError|OfflineModeIsEnabled"
    r"|couldn't connect to 'https://huggingface\.co'"
)


def missing_resource(text: str) -> str | None:
    if "No module named 'ray'" in text:
        return "ray (not in the serving image)"
    match = _OFFLINE.search(text)
    if match:
        return f"Hugging Face {match.group(1) or 'download'} (offline)"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed and call.excinfo is not None:
        resource = missing_resource(f"{call.excinfo.typename}: {call.excinfo.value}")
        if resource is not None:
            report.outcome = "skipped"
            report.longrepr = (str(item.path), item.location[1] or 0, f"needs {resource}")
