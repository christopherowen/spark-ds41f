#!/usr/bin/env python3
"""Run a release-test manifest inside the image and keep every test's outcome.

usage: release_test_runner.py MANIFEST.json OUT_DIR

Files run in chunks of CHUNK per pytest process (one interpreter start per chunk), each with
CHUNK_TIMEOUT. A chunk that times out, crashes or leaves no JUnit report is rerun file by
file with FILE_TIMEOUT, so a hang or a crash costs only its own file. OUT_DIR gets
results.jsonl (one line per test file: counts, seconds, failures) and the JUnit reports.
stdout ends with FAILED lines and a pytest-style summary, which the lab's exit verdict reads;
the exit status is 0 only when every selected test passed or was skipped.
"""
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

CHUNK, CHUNK_TIMEOUT, FILE_TIMEOUT = 10, 1800, 600


def pytest(paths, out: Path, tag: str, timeout: int) -> tuple[object, Path, float, str]:
    report = out / f"{tag}.xml"
    report.unlink(missing_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider",
               "-p", "tblib_stub", "--continue-on-collection-errors",
               "-o", "junit_family=xunit1", "-o", "faulthandler_timeout=300",
               "--junitxml", str(report), "--basetemp", f"/c/pytest-tmp-release/{tag}", *paths]
    started = time.time()
    try:
        process = subprocess.run(command, timeout=timeout, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
        code, tail = process.returncode, process.stdout[-3000:]
    except subprocess.TimeoutExpired as expired:
        output = expired.stdout or b""
        code = "timeout"
        tail = (output.decode(errors="replace") if isinstance(output, bytes) else output)[-3000:]
    return code, report, time.time() - started, tail


def outcomes(report: Path) -> dict[str, list[dict]]:
    """Per test file, each test case: id, outcome, seconds, message."""
    by_file: dict[str, list[dict]] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        path = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
        outcome, message = "passed", ""
        for kind in ("failure", "error", "skipped"):
            node = case.find(kind)
            if node is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[kind]
                message = (node.get("message") or "").splitlines()[0][:300] if node.get("message") else ""
                break
        by_file.setdefault(path, []).append({
            "id": f"{path}::{case.get('name')}", "outcome": outcome,
            "seconds": float(case.get("time") or 0), "message": message})
    return by_file


def summarize(path: str, cases: list[dict], seconds: float, status: str = "") -> dict:
    counts = {k: sum(c["outcome"] == k for c in cases) for k in ("passed", "failed", "error", "skipped")}
    return {"file": path, "status": status or ("ok" if not counts["failed"] + counts["error"] else "fail"),
            "seconds": round(seconds, 1), **counts,
            "failures": [{"id": c["id"], "outcome": c["outcome"], "message": c["message"]}
                         for c in cases if c["outcome"] in ("failed", "error")]}


def main(manifest_path: str, out_dir: str) -> int:
    files = sorted(json.loads(Path(manifest_path).read_text())["files"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for start in range(0, len(files), CHUNK):
        chunk = [f for f in files[start:start + CHUNK] if Path(f).exists()]
        results += [summarize(f, [], 0, "missing") for f in files[start:start + CHUNK] if not Path(f).exists()]
        if not chunk:
            continue
        code, report, seconds, tail = pytest(chunk, out, f"chunk{start // CHUNK:03d}", CHUNK_TIMEOUT)
        if code in (0, 1, 5) and report.exists():
            found = outcomes(report)
            for f in chunk:
                cases = found.get(f, [])
                results.append(summarize(f, cases, sum(c["seconds"] for c in cases),
                                         "" if cases else "no tests"))
            print(f"chunk {start // CHUNK:03d}: exit {code}, {seconds:.0f} s", flush=True)
            continue
        print(f"chunk {start // CHUNK:03d}: exit {code} after {seconds:.0f} s; rerunning file by file",
              flush=True)
        for index, f in enumerate(chunk):
            code, report, seconds, tail = pytest([f], out, f"chunk{start // CHUNK:03d}-{index:02d}",
                                                 FILE_TIMEOUT)
            if code in (0, 1, 5) and report.exists():
                cases = outcomes(report).get(f, [])
                results.append(summarize(f, cases, seconds, "" if cases else "no tests"))
            else:
                status = "timeout" if code == "timeout" else f"crash (exit {code})"
                results.append({**summarize(f, [], seconds, status),
                                "tail": tail.splitlines()[-15:]})
            print(f"  {f}: {results[-1]['status']} ({seconds:.0f} s)", flush=True)
    with (out / "results.jsonl").open("w") as handle:
        for record in results:
            handle.write(json.dumps(record) + "\n")
    totals = {k: sum(r[k] for r in results) for k in ("passed", "failed", "error", "skipped")}
    broken = [r for r in results if r["status"] not in ("ok", "no tests")]
    print(f"\n{len(results)} files, {sum(r['seconds'] for r in results):.0f} s of tests; "
          f"{len(broken)} files not ok: " + ", ".join(
              f"{s}: {sum(r['status'] == s for r in broken)}" for s in sorted({r['status'] for r in broken})))
    for record in results:
        for failure in record["failures"]:
            print(f"{'FAILED' if failure['outcome'] == 'failed' else 'ERROR'} {failure['id']} - "
                  f"{failure['message'][:200]}")
        if record["status"] in ("timeout", "missing") or record["status"].startswith("crash"):
            print(f"ERROR {record['file']} - {record['status']}")
    # As pytest's summary: zero counts are left out (the lab's verdict reads "N failed").
    print(", ".join(f"{totals[k]} {name}" for k, name in (
        ("failed", "failed"), ("passed", "passed"), ("skipped", "skipped"), ("error", "errors"))
        if totals[k]) or "no tests ran")
    return 0 if not broken else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
