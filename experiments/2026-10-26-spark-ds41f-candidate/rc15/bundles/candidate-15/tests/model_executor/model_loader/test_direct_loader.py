# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The direct loader (--load-format direct) without a GPU.

- Descriptor planning: the descriptors the session records for tensor-parallel slices,
  padded and strided destinations, expert slices, byte casts and BF16 to FP32
  expansion, executed on host memory with _batch.c's semantics, equal Torch's own
  copies; the scratch reads of a dtype conversion cover the view exactly; views the
  descriptors cannot express are refused.
- Checkpoint metadata: the session's safetensors header checks, its name, prefix,
  index and skip selection, and its coalesced metadata reads, with the native helper's
  O_DIRECT reads replaced by plain preads.
- The loader: its registration and options, its refusal of a device without host page
  tables, and the file-backed descriptors it yields for tensors the model routes as
  files (DS4.1's Engram tables), which never reach the session's reader.

The session's own copies (the bounce ring, CUDA destinations) are in
test_direct_loader_cuda.py, which needs a GPU.
"""

import ctypes
import json
import os
import typing

import numpy as np
import pytest
import regex as re
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor import model_loader
from vllm.model_executor.model_loader import direct_loader, weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.direct_io import (
    DirectWeightSession,
    scratch_chunks,
    view_offset,
    view_records,
)
from vllm.model_executor.model_loader.direct_io._checkpoint import (
    EXPAND_BF16,
    HOST_COPY,
)
from vllm.model_executor.weight_transfer import get_file_tensor_source


class _PreadNative:
    """The helper's header and metadata reads as plain preads, counted."""

    def __init__(self):
        self.reads = 0

    def direct_bytes(self, reader, fd, offset, nbytes):
        self.reads += 1
        data = os.pread(fd, nbytes, offset)
        if len(data) != nbytes:
            raise RuntimeError("short read")
        return data


@pytest.fixture
def session(monkeypatch):
    """A session for metadata and routing only: no CUDA, no native helper, and files
    opened without O_DIRECT (so any filesystem serves)."""
    monkeypatch.setattr(os, "O_DIRECT", 0, raising=False)
    session = DirectWeightSession.__new__(DirectWeightSession)
    session.native = _PreadNative()
    session.reader = None
    session.device = 0
    session.files = []
    session.sources = {}
    session.metadata_bytes = 0
    yield session
    for fd in session.files:
        os.close(fd)


def _execute_on_host(records, backing):
    """_batch.c's descriptors on host memory: each row of a read copies `bytes` from the
    file (BF16 expansion writes each value as FP32 with the same bits above 16 zero
    bits); a host copy copies `bytes` from the address `offset`. Every write must stay
    in ``backing``, as the helper checks against the CUDA allocation."""
    low, high = backing.data_ptr(), backing.data_ptr() + backing.nbytes
    for (
        fd,
        offset,
        nbytes,
        pointer,
        transform,
        rows,
        source_stride,
        dest_stride,
    ) in records:
        for row in range(rows):
            if transform == HOST_COPY:
                data = ctypes.string_at(offset, nbytes)
            else:
                data = os.pread(fd, nbytes, offset + row * source_stride)
                assert len(data) == nbytes
            if transform == EXPAND_BF16:
                data = (np.frombuffer(data, "<u2").astype("<u4") << 16).tobytes()
            target = pointer + row * dest_stride
            assert low <= target and target + len(data) <= high
            ctypes.memmove(target, data, len(data))


def _bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8)


def _source(session, path, name):
    return dict(session.weights([path]))[name]


def _entry(session, source):
    return session.sources[source.untyped_storage()._cdata][1]


def _routed(session, source, destination):
    entry = _entry(session, source)
    expand = source.dtype == torch.bfloat16 and destination.dtype == torch.float32
    return view_records(
        entry, view_offset(entry, source, destination), source, destination, expand
    )


def _checkpoint(tmp_path, tensors, name="model.safetensors"):
    path = tmp_path / name
    save_file(tensors, path)
    with safe_open(path, framework="pt") as archive:
        real = {key: archive.get_tensor(key) for key in archive.keys()}  # noqa: SIM118
    return path, real


# Each case: the checkpoint tensor, the view a weight loader takes, and the
# destination's backing, dtype and the destination view within it.
VIEWS = {
    "tp column shard": (
        lambda: torch.arange(2 * 16 * 32, dtype=torch.float32).reshape(2, 16, 32),
        lambda w: w[:, :, 16:],
        lambda: torch.full((2, 16, 16), -7.0),
        lambda d: d,
    ),
    "tp column shard, padded rows": (
        lambda: torch.arange(2 * 16 * 32, dtype=torch.float32).reshape(2, 16, 32),
        lambda w: w[:, :, :16],
        lambda: torch.full((2, 16, 18), -7.0),
        lambda d: d[:, :, 1:-1],
    ),
    "tp row shard": (
        lambda: torch.arange(64 * 24, dtype=torch.int32).reshape(64, 24),
        lambda w: w[16:48],
        lambda: torch.full((40, 24), -7, dtype=torch.int32),
        lambda d: d[4:36],
    ),
    "expert slice": (
        lambda: torch.arange(8 * 12 * 6, dtype=torch.int16).reshape(8, 12, 6),
        lambda w: w[2:6, 4:10],
        lambda: torch.full((4, 6, 6), -7, dtype=torch.int16),
        lambda d: d,
    ),
    "transposed destination": (
        lambda: torch.arange(12 * 20, dtype=torch.float32).reshape(12, 20),
        lambda w: w,
        lambda: torch.full((20, 12), -7.0),
        lambda d: d.t(),
    ),
    "bf16 expanded to fp32": (
        lambda: (
            torch.arange(-3000, 3000, 3, dtype=torch.float32)
            .reshape(40, 50)
            .to(torch.bfloat16)
        ),
        lambda w: w[:, 25:],
        lambda: torch.full((40, 27), -7.0),
        lambda d: d[:, 1:-1],
    ),
    "int8 bytes into uint8": (
        lambda: (
            torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(16, 16)
        ),
        lambda w: w[:, 8:],
        lambda: torch.zeros((16, 8), dtype=torch.uint8),
        lambda d: d,
    ),
    "e8m0 exponents viewed as bytes": (
        lambda: (
            torch.arange(256, dtype=torch.uint8)
            .reshape(16, 16)
            .view(torch.float8_e8m0fnu)
        ),
        lambda w: w.view(torch.uint8)[:, 4:12],
        lambda: torch.zeros((16, 8), dtype=torch.uint8),
        lambda d: d,
    ),
    "scalar": (
        lambda: torch.tensor(3.5),
        lambda w: w,
        lambda: torch.zeros(()),
        lambda d: d,
    ),
}


@pytest.mark.parametrize("case", list(VIEWS))
def test_descriptors_copy_what_torch_copies(tmp_path, session, case):
    make, view, make_backing, place = VIEWS[case]
    path, real = _checkpoint(tmp_path, {"w": make()})
    backing = make_backing()
    destination = place(backing)
    expected = backing.clone()
    place(expected).copy_(view(real["w"]))
    records = _routed(session, view(_source(session, path, "w")), destination)
    _execute_on_host(records, backing)
    assert torch.equal(_bytes(backing), _bytes(expected)), case


def test_contiguous_rows_make_one_strided_descriptor_per_outer_index(tmp_path, session):
    path, _ = _checkpoint(
        tmp_path, {"w": torch.zeros((3, 64, 128), dtype=torch.bfloat16)}
    )
    source = _source(session, path, "w")[:, :, 64:]
    destination = torch.empty((3, 64, 64), dtype=torch.bfloat16)
    records = _routed(session, source, destination)
    entry = _entry(session, source)
    assert [r[1] for r in records] == [
        entry.offset + 64 * 2 + i * 64 * 128 * 2 for i in range(3)
    ]
    assert {r[2:3] + r[4:] for r in records} == {(64 * 2, 0, 64, 128 * 2, 64 * 2)}


def test_whole_tensors_are_one_contiguous_row(tmp_path, session):
    path, _ = _checkpoint(tmp_path, {"w": torch.zeros((5, 7, 9))})
    source = _source(session, path, "w")
    (record,) = _routed(session, source, torch.empty((5, 7, 9)))
    assert record[1:3] == (_entry(session, source).offset, 5 * 7 * 9 * 4)
    assert record[4:] == (0, 1, 0, 0)


@pytest.mark.parametrize("capacity", [1, 7, 64, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64, torch.float16])
def test_scratch_reads_cover_a_conversion_exactly(tmp_path, session, capacity, dtype):
    values = torch.arange(1000, dtype=torch.float32) / 7 - 50
    path, real = _checkpoint(tmp_path, {"w": values})
    source = _source(session, path, "w")[37:911]
    entry = _entry(session, source)
    destination = torch.full((874,), -7.0, dtype=dtype)
    offset = view_offset(entry, source, destination)
    scratch = torch.empty(capacity, dtype=source.dtype)
    covered = 0
    for start, count, position, nbytes in scratch_chunks(
        entry, offset, source, capacity
    ):
        assert start == covered and 0 < count <= capacity
        assert nbytes == count * 4
        ctypes.memmove(scratch.data_ptr(), os.pread(entry.fd, nbytes, position), nbytes)
        destination[start : start + count].copy_(scratch[:count])
        covered += count
    assert covered == source.numel()
    assert torch.equal(destination, real["w"][37:911].to(dtype))


def test_views_the_descriptors_cannot_express_are_refused(tmp_path, session):
    path, _ = _checkpoint(
        tmp_path, {"w": torch.zeros((8, 8), dtype=torch.bfloat16), "s": torch.ones(4)}
    )
    sources = dict(session.weights([path]))
    w = sources["w"]
    destination = torch.empty((8, 8), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="unsupported data transformation"):
        session(destination, w + 1)
    with pytest.raises(NotImplementedError, match="explicit transform for w"):
        view_offset(_entry(session, w), w.view(torch.uint8), destination)
    with pytest.raises(NotImplementedError, match="explicit transform for w"):
        view_offset(_entry(session, w), w[:4], destination)
    beyond = torch.as_strided(sources["s"], (4,), (2,))
    with pytest.raises(ValueError, match="source view exceeds checkpoint range: s"):
        view_offset(_entry(session, beyond), beyond, torch.empty(4))
    with pytest.raises(ValueError, match="session CUDA device"):
        session(destination, w)
    w.add_(1)  # an in-place op on the meta view: its values are no longer the file's
    with pytest.raises(NotImplementedError, match="explicit transform for w"):
        session(destination, w)


def test_weights_select_by_name_prefix_index_and_skip(tmp_path, session):
    path, _ = _checkpoint(
        tmp_path,
        {
            "mtp.b": torch.ones(2),
            "mtp.a": torch.ones(3),
            "mtp.elsewhere": torch.ones(4),
            "mtp.skipped": torch.ones(5),
            "model.c": torch.ones(6),
        },
    )
    (tmp_path / "other.safetensors").write_bytes(b"")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.a": "model.safetensors",
                    "mtp.b": "model.safetensors",
                    "mtp.skipped": "model.safetensors",
                    "mtp.elsewhere": "other.safetensors",
                    "model.c": "model.safetensors",
                }
            }
        )
    )
    loaded = list(
        session.weights(
            [path],
            prefixes=("mtp.",),
            prefix="draft.",
            index_path=index,
            skip=lambda name: name == "mtp.skipped",
        )
    )
    assert [name for name, _ in loaded] == ["draft.mtp.a", "draft.mtp.b"]
    for (_, tensor), shape in zip(loaded, ((3,), (2,)), strict=True):
        assert tensor.device.type == "meta" and tensor.shape == shape
    assert sorted(entry.name for _, entry in session.sources.values()) == [
        "mtp.a",
        "mtp.b",
    ]


def test_declared_metadata_reads_coalesce_and_keep_their_values(tmp_path, session):
    values = {f"scale_{i:04d}": torch.tensor(float(i)) for i in range(2048)}
    values["layer_multipliers"] = torch.arange(6, dtype=torch.float32)
    values["empty"] = torch.zeros((0, 4))
    path, _ = _checkpoint(tmp_path, values)
    loaded = dict(session.weights([path], needs_values=lambda entry: True))
    assert session.native.reads < 2 + 10  # the header's two reads, then the values
    for name, expected in values.items():
        assert loaded[name].device.type == "cpu"
        assert torch.equal(loaded[name], expected), name
    assert not session.sources
    session.metadata_bytes = 64 << 20
    with pytest.raises(ValueError, match="64 MiB session budget"):
        dict(session.weights([path], needs_values=lambda entry: True))


def _raw_checkpoint(path, header, payload_bytes, length=None):
    encoded = json.dumps(header).encode()
    head = (len(encoded) if length is None else length).to_bytes(8, "little")
    path.write_bytes(head + encoded + bytes(payload_bytes))
    return path


@pytest.mark.parametrize(
    "header,payload,length,error",
    [
        (
            {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
            8,
            1 << 40,
            "invalid safetensors header length",
        ),
        (
            {"a": {"dtype": "F4", "shape": [2], "data_offsets": [0, 1]}},
            1,
            None,
            "invalid tensor metadata: a",
        ),
        (
            {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
            4,
            None,
            "invalid tensor file range: a",
        ),
        (
            {"a": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]}},
            16,
            None,
            "overlap or hole in safetensors payload: a",
        ),
        (
            {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
            12,
            None,
            "payload does not cover file",
        ),
        ({"__metadata__": {"format": 1}}, 0, None, "invalid safetensors metadata"),
        ([1, 2], 0, None, "header must be an object"),
    ],
)
def test_invalid_headers_are_refused(tmp_path, session, header, payload, length, error):
    path = _raw_checkpoint(tmp_path / "bad.safetensors", header, payload, length)
    with pytest.raises(ValueError, match=error):
        session.manifest(path)


def test_duplicate_header_keys_are_refused(tmp_path, session):
    entry = '{"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}'
    encoded = f'{{"a": {entry}, "a": {entry}}}'.encode()
    path = tmp_path / "duplicate.safetensors"
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(4))
    with pytest.raises(ValueError, match="duplicate safetensors key: a"):
        session.manifest(path)


def test_direct_is_a_registered_load_format():
    assert "direct" in typing.get_args(model_loader.LoadFormats)
    config = LoadConfig(load_format="direct")
    loader = model_loader.get_model_loader(config)
    assert isinstance(loader, direct_loader.DirectModelLoader)
    assert loader.io_threads == 8
    assert loader.weight_arena is True
    assert loader.order13_sponge is True
    assert loader.load_config.load_format == "safetensors"
    assert config.load_format == "direct"
    custom = LoadConfig(
        load_format="direct",
        model_loader_extra_config={
            "io_threads": 3,
            "weight_arena": False,
            "order13_sponge": False,
        },
    )
    custom_loader = model_loader.get_model_loader(custom)
    assert custom_loader.io_threads == 3
    assert custom_loader.weight_arena is False
    assert custom_loader.order13_sponge is False


def test_parse_buddyinfo_order13_counts_sub_and_full_512mib_blocks():
    buddyinfo = (
        "Node 0, zone      DMA      0 0 0 0 0 0 0 0 0 0 0 0 2 3\n"
        "Node 0, zone   Normal      4 0 0 0 0 1 0 0 0 0 0 0 1 190\n"
        "ignored line\n"
    )
    sub_bytes, order13_bytes = direct_loader.parse_buddyinfo_order13(
        buddyinfo, page_bytes=65536
    )
    expected_sub = (
        4 * 65536
        + 1 * (65536 << 5)
        + 3 * (65536 << 12)
    )
    expected_order13 = 193 * (65536 << 13)
    assert sub_bytes == expected_sub
    assert order13_bytes == expected_order13
    assert direct_loader.parse_buddyinfo_order13(buddyinfo, page_bytes=4096) == (0, 0)


@pytest.mark.parametrize(
    "options,strategy,error",
    [
        ({"enable_multithread_load": True}, None, "in one thread"),
        ({}, "eager", "no safetensors read strategy"),
        ({"read_mode": "gds"}, None, "Unexpected extra config keys"),
    ],
)
def test_options_the_direct_loader_does_not_take(options, strategy, error):
    config = LoadConfig(
        load_format="direct",
        model_loader_extra_config=options,
        safetensors_load_strategy=strategy,
    )
    with pytest.raises(ValueError, match=error):
        direct_loader.DirectModelLoader(config)


def _no_session(*args, **kwargs):
    raise AssertionError("the loader opened a session")


def test_devices_without_host_page_tables_are_refused(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(direct_loader, "DirectWeightSession", _no_session)
    monkeypatch.setattr(
        direct_loader, "capabilities", lambda device: {"host_page_tables": 0}
    )
    loader = direct_loader.DirectModelLoader(LoadConfig(load_format="direct"))
    config = SimpleNamespace(device_config=SimpleNamespace(device="cuda:1"))
    with pytest.raises(RuntimeError, match="CUDA device 1 does not; it has no GPU"):
        loader.load_model(config, SimpleNamespace())
    cpu = SimpleNamespace(device_config=SimpleNamespace(device="cpu"))
    with pytest.raises(ValueError, match="requires a CUDA device"):
        loader.load_model(cpu, SimpleNamespace())


def test_weights_are_read_only_inside_load_model(tmp_path):
    loader = direct_loader.DirectModelLoader(LoadConfig(load_format="direct"))
    source = DefaultModelLoader.Source(str(tmp_path), revision=None)
    with pytest.raises(RuntimeError, match="only in load_model"):
        next(loader._get_weights_iterator(source))


def test_draft_weights_follow_the_index_and_never_open_other_shards(tmp_path, session):
    save_file({"mtp.weight": torch.arange(16)}, tmp_path / "draft.safetensors")
    save_file({"mtp.bias": torch.arange(4) + 100}, tmp_path / "bias.safetensors")
    (tmp_path / "target.safetensors").write_bytes(b"must not be opened")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.weight": "draft.safetensors",
                    "mtp.bias": "bias.safetensors",
                    "model.weight": "target.safetensors",
                }
            }
        )
    )
    loader = direct_loader.DirectModelLoader(LoadConfig(load_format="direct"))
    loader._session = session
    source = DefaultModelLoader.Source(
        str(tmp_path), revision=None, prefix="draft.", weight_name_prefixes=("mtp.",)
    )
    loaded = dict(loader._get_weights_iterator(source))
    assert set(loaded) == {"draft.mtp.weight", "draft.mtp.bias"}
    assert sorted(entry.name for _, entry in session.sources.values()) == [
        "mtp.bias",
        "mtp.weight",
    ]
    assert loader.counter_before_loading_weights > 0


ENGRAM = re.compile(r"layers\.\d+\.engram\.embed\.(?:weight|scale)")


def test_file_backed_tables_are_descriptors_and_never_reach_the_reader(
    tmp_path, session
):
    """DS4.1's filter on a mixed shard, a table-only shard and a shard without tables:
    tables become file-source descriptors, the rest routes through the session, all
    in name order per file, with the index and the expert filter applied to both."""
    mixed = {
        "layers.1.engram.embed.weight": torch.arange(64, dtype=torch.uint8).reshape(
            8, 8
        ),
        "layers.1.engram.embed.scale": torch.ones((8, 1), dtype=torch.uint8),
        "layers.1.engram.layer_multipliers": torch.tensor([0.5, 2.0]),
        "layers.1.attn.wq.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
        "layers.1.ffn.experts.1.w1.weight": torch.zeros((2, 2)),
        "layers.1.ffn.experts.0.w1.weight": torch.zeros((2, 2)),
        "layers.1.stale.weight": torch.zeros(3),
    }
    tables = {
        "layers.2.engram.embed.weight": torch.zeros((4, 8), dtype=torch.uint8),
        "layers.2.engram.embed.scale": torch.zeros((4, 1), dtype=torch.uint8),
    }
    plain = {
        "layers.0.norm.weight": torch.ones(4),
        "layers.1.stale.weight": torch.ones(3),
    }
    for name, tensors in (("a", mixed), ("b", tables), ("c", plain)):
        save_file(tensors, tmp_path / f"{name}.safetensors")
    weight_map = {name: "a.safetensors" for name in mixed}
    weight_map.update({name: "b.safetensors" for name in tables})
    weight_map.update({name: "c.safetensors" for name in plain})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    loader = direct_loader.DirectModelLoader(LoadConfig(load_format="direct"))
    loader._session = session
    loader.local_expert_ids = {0}
    opened = []
    manifest = session.manifest

    def recorded(path):
        opened.append(os.path.basename(path))
        return manifest(path)

    session.manifest = recorded
    source = DefaultModelLoader.Source(
        str(tmp_path),
        revision=None,
        prefix="model.",
        file_weight_filter=lambda name: ENGRAM.fullmatch(name) is not None,
    )
    paths = [str(tmp_path / f"{name}.safetensors") for name in "abc"]
    loaded = list(loader._file_backed_weights_iterator(paths, source, index))
    assert [name for name, _ in loaded] == [
        "model.layers.1.attn.wq.weight",
        "model.layers.1.engram.embed.scale",
        "model.layers.1.engram.embed.weight",
        "model.layers.1.engram.layer_multipliers",
        "model.layers.1.ffn.experts.0.w1.weight",
        "model.layers.2.engram.embed.scale",
        "model.layers.2.engram.embed.weight",
        "model.layers.0.norm.weight",
        "model.layers.1.stale.weight",
    ]
    assert opened == ["a.safetensors", "c.safetensors"]  # b holds only tables
    loaded = dict(loaded)
    for name in (*mixed, *tables):
        if ENGRAM.fullmatch(name) is None:
            continue
        path = tmp_path / ("a.safetensors" if name in mixed else "b.safetensors")
        expected = weight_utils.safetensors_file_sources(str(path))[name]
        assert get_file_tensor_source(loaded[f"model.{name}"]) == expected
        assert loaded[f"model.{name}"].device.type == "meta"
    assert {entry.name for _, entry in session.sources.values()} == {
        "layers.1.attn.wq.weight",
        "layers.1.ffn.experts.0.w1.weight",
        "layers.0.norm.weight",
        "layers.1.stale.weight",
    }
    assert torch.equal(
        loaded["model.layers.1.engram.layer_multipliers"], torch.tensor([0.5, 2.0])
    )
    # The index maps the stale name to c, so a's copy is never routed.
    stale = _entry(session, loaded["model.layers.1.stale.weight"])
    assert os.fstat(stale.fd).st_ino == os.stat(tmp_path / "c.safetensors").st_ino


def _fake_build_tools(root, *, liburing=True):
    """A compiler that writes its -o file and logs its arguments, pkg-config, and a
    CUDA_HOME with the headers the build checks for."""
    tools = root / "bin"
    tools.mkdir()
    (tools / "cc").write_text(
        "#!/bin/sh\n"
        '[ "$1" = --version ] && { echo "fake cc 1"; exit 0; }\n'
        'echo "$@" >> "$FAKE_CC_LOG"\n'
        'while [ "$#" -gt 1 ]; do [ "$1" = -o ] && echo built > "$2"; shift; done\n'
    )
    (tools / "pkg-config").write_text(
        "#!/bin/sh\n"
        f"[ {int(liburing)} = 1 ] || exit 1\n"
        'case "$1" in --cflags) echo -I/fake/uring;; --libs) echo -luring;;'
        " --modversion) echo 2.5;; esac\n"
    )
    for tool in tools.iterdir():
        tool.chmod(0o755)
    cuda = root / "cuda"
    (cuda / "include").mkdir(parents=True)
    for header in ("cuda_runtime_api.h", "driver_types.h"):
        (cuda / "include" / header).write_text(f"/* {header} */\n")
    return tools, cuda


def test_the_helper_builds_once_into_the_xdg_cache(tmp_path, monkeypatch):
    from vllm.model_executor.model_loader.direct_io import _native

    tools, cuda = _fake_build_tools(tmp_path)
    log = tmp_path / "cc.log"
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("FAKE_CC_LOG", str(log))
    monkeypatch.delenv("CC", raising=False)
    target = _native._build()
    assert target.parent.parent == tmp_path / "cache" / "vllm" / "direct_io"
    assert re.fullmatch(r"[0-9a-f]{24}", target.parent.name)
    assert target.name.startswith("_direct_io_storage") and target.is_file()
    (command,) = log.read_text().splitlines()
    arguments = command.split()
    assert str(_native._SOURCE) in arguments and "-Werror" in arguments
    assert arguments[-5:-2] == ["-lcudart", "-lcuda", "-luring"]
    assert "-I/fake/uring" in arguments and f"-I{cuda / 'include'}" in arguments
    assert _native._build() == target
    assert len(log.read_text().splitlines()) == 1


def test_the_helper_refuses_to_build_without_liburing(tmp_path, monkeypatch):
    from vllm.model_executor.model_loader.direct_io import _native

    tools, cuda = _fake_build_tools(tmp_path, liburing=False)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("CC", raising=False)
    with pytest.raises(RuntimeError, match="needs liburing and pkg-config"):
        _native._build()
    monkeypatch.setenv("CC", str(tmp_path / "no-such-compiler"))
    with pytest.raises(RuntimeError, match="needs a C99 compiler"):
        _native._build()
