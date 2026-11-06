"""Validate the exact expected old-contract failures in pytest's JUnit format."""
import xml.etree.ElementTree as ET


def verify_contract_before(path, returncode):
    cases = list(ET.parse(path).iter('testcase'))
    if returncode != 1 or len(cases) != 4:
        raise RuntimeError('expected exactly four failing old-contract tests')
    for case in cases:
        failure = case.find('failure')
        if not case.get('name', '').startswith('test_grouped_gemm_emits_requested_prefill_swizzle['):
            raise RuntimeError('unexpected old-contract test identity')
        if (failure is None or case.find('error') is not None
                or case.find('skipped') is not None):
            raise RuntimeError('old contract must fail assertions, not setup/skip')
        # pytest normally emits message and text, without a type attribute.
        # Check the failure message itself, not just nearby displayed source.
        if ('rasterization2DRow<3>' not in failure.get('message', '')
                or 'AssertionError' not in (failure.text or '')):
            raise RuntimeError('failure does not reproduce the ignored-swizzle contract')
    return len(cases)
