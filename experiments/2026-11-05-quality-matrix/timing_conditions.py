"""Out-of-band timing qualification; no serving or model calls."""
def fan_curve_ready(records, expected_nodes):
    expected=set(expected_nodes)
    if not expected or len(records)!=len(expected):return False
    if {r.get('node') for r in records}!=expected:return False
    return all(r.get('service')=='active' or
               (r.get('service')=='inactive' and r.get('started') is True)
               for r in records)
