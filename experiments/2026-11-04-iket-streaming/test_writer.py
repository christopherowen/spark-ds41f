"""The sink must preserve order across wraparound, backpressure, and I/O failure."""
import ctypes
import importlib.util
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('stream_export', HERE / 'stream/export.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    so = tmp_path_factory.mktemp('native') / 'writer.so'
    subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-pthread',
                    str(HERE / 'stream/writer.cpp'), '-lz', '-o', str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.stream_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64,
                                 ctypes.c_uint64, ctypes.c_uint64]
    lib.stream_start.restype = ctypes.c_void_p
    lib.stream_publish.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.stream_finish.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                                  ctypes.POINTER(ctypes.c_uint64)]
    return lib


def run(lib, root, delay=0, fail=0, producers=1):
    capacity = 8
    backing = ctypes.create_string_buffer(64 + capacity * 64 + 63)
    address = (ctypes.addressof(backing) + 63) & ~63
    header = (ctypes.c_uint64 * 8).from_address(address)
    header[0] = capacity
    for i in range(capacity):
        ctypes.c_uint64.from_address(address + 64 + i * 64).value = i
    handle = lib.stream_start(address, str(root).encode(), 4096, delay, fail)
    assert handle
    def publish(worker):
        for i in range(1000):
            record = (ctypes.c_uint64 * 7)(worker * 1000 + i, 1, 2, 3, 0, 0, 0)
            code = lib.stream_publish(address, ctypes.byref(record))
            if code:
                return code
        return 0
    with ThreadPoolExecutor(max_workers=producers) as pool:
        codes = list(pool.map(publish, range(producers)))
    written, chunks = ctypes.c_uint64(), ctypes.c_uint64()
    status = lib.stream_finish(handle, ctypes.byref(written), ctypes.byref(chunks))
    return codes, status, written.value, header[4]


def test_concurrent_wraparound_has_no_lost_or_repeated_records(library, tmp_path):
    codes, status, written, _ = run(library, tmp_path, producers=4)
    assert codes == [0] * 4 and status == 0 and written == 4000
    rows = list(export.records(tmp_path))
    assert len(rows) == 4000
    assert sorted(r[1] for r in rows) == list(range(4000))


def test_slow_storage_applies_backpressure_without_dropping(library, tmp_path):
    codes, status, written, waits = run(library, tmp_path, delay=2000)
    assert codes == [0] and status == 0 and written == 1000 and waits > 0
    assert len(list(export.records(tmp_path))) == 1000


def test_disk_full_wakes_blocked_producers_and_retains_complete_chunks(library, tmp_path):
    codes, status, written, _ = run(library, tmp_path, fail=100)
    assert codes[0] == status == 28
    assert written <= 100
    assert list(tmp_path.glob('*.partial'))
    assert len(list(export.records(tmp_path))) <= written
