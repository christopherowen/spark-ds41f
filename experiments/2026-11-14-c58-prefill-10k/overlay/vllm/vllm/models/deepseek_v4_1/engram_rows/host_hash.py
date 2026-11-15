# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram n-gram hash on the host, for the tokens whose ids the host already has.

A step's Engram lookups split by when their row IDs can exist. A prompt token's id
is on the host when the runner prepares the step, a step before the GPU runs it, so
its rows can be read while the previous step computes; a token the GPU samples (a
decode token or a draft) is known only to the GPU when the step starts. This module
hashes the first kind exactly as ``tilelang/engram_hash.py`` does on the GPU (the
same int64 arithmetic: wrapping multiplies, XOR, TileLang's floor remainder), so the
reader can be handed those rows early and the GPU's ids for them are never needed.

``UNKNOWN`` marks a lookup whose id only the GPU will know; ``-1`` is a known "no
row" (a padding token past the step's live tokens), as the GPU writes it.
"""

import os
from pathlib import Path

import numpy as np

HEADS = 24
ALL_COLUMNS = (1 << HEADS) - 1
LAGS = 4
PAD_TOKEN = 2
IMAGE_SENTINEL = 129264
UNKNOWN = np.iinfo(np.int64).min


def _parse_cache_kib(raw: str) -> int:
    text = raw.strip().upper()
    if not text:
        return 0
    if text.endswith("K"):
        return int(text[:-1])
    if text.endswith("M"):
        return int(text[:-1]) * 1024
    if text.endswith("G"):
        return int(text[:-1]) * 1024 * 1024
    return max(1, int(text) // 1024)


def max_l2_cpus(
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
    allowed: set[int] | None = None,
) -> list[int]:
    """Return the online CPUs with the largest private L2 cache (`cache/index2/size`).

    On GB10's 20-core Grace CPU, `cpu0..4` and `cpu10..14` are Cortex-A725 cores with a
    512 KiB L2, while `cpu5..9` and `cpu15..19` are Cortex-X925 cores with a 2048 KiB L2.
    """
    root = Path(sysfs_root)
    if allowed is None and hasattr(os, "sched_getaffinity"):
        try:
            allowed = set(os.sched_getaffinity(0))
        except OSError:
            allowed = None
    best_kib = 0
    best_cpus: list[int] = []
    if not root.is_dir():
        return sorted(allowed) if allowed else []
    for entry in root.iterdir():
        name = entry.name
        if not (name.startswith("cpu") and name[3:].isdigit()):
            continue
        cpu = int(name[3:])
        if allowed is not None and cpu not in allowed:
            continue
        size_path = entry / "cache" / "index2" / "size"
        try:
            kib = _parse_cache_kib(size_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if kib > best_kib:
            best_kib = kib
            best_cpus = [cpu]
        elif kib == best_kib and kib > 0:
            best_cpus.append(cpu)
    best_cpus.sort()
    if best_cpus:
        return best_cpus
    return sorted(allowed) if allowed else []


def pin_to_max_l2_cpus(
    pid: int = 0,
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
) -> list[int]:
    """Pin thread/process `pid` to the allowed CPUs with the largest L2 cache."""
    if os.environ.get("SPARK_DS41F_ENGRAM_CPU_AFFINITY") == "0":
        return []
    cpus = max_l2_cpus(sysfs_root=sysfs_root)
    if cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(pid, set(cpus))
        except OSError:
            return []
    return cpus


class HostEngramHash:
    """The hash geometry and tokenizer map of a model's Engram layers, on the host."""

    def __init__(self, token_map, multipliers, primes, offsets):
        token_map_arr = np.asarray(token_map)
        if (
            token_map_arr.size == 0
            or (
                int(token_map_arr.min()) >= np.iinfo(np.int32).min
                and int(token_map_arr.max()) <= np.iinfo(np.int32).max
            )
        ):
            self.token_map = np.asarray(token_map_arr, dtype=np.int32)
        else:
            self.token_map = np.asarray(token_map_arr, dtype=np.int64)
        self.multipliers = np.asarray(multipliers, dtype=np.int64)  # [layers, 4]
        self.primes = np.asarray(primes, dtype=np.int64)  # [layers, 24]
        self.layers = self.multipliers.shape[0]
        if self.multipliers.shape != (self.layers, LAGS) or self.primes.shape != (
            self.layers,
            HEADS,
        ):
            raise ValueError(
                "Engram hash geometry must be [layers, 4] multipliers and "
                "[layers, 24] primes"
            )
        self.offsets = np.broadcast_to(
            np.asarray(offsets, dtype=np.int64), (self.layers, HEADS)
        )
        self.pad = self.token_map[PAD_TOKEN]
        self.vocab = self.token_map.shape[0]
        self._lags = np.arange(LAGS, dtype=np.int64)[None, :]
        self._multipliers_u64 = np.ascontiguousarray(
            self.multipliers[:, None, :], dtype=np.uint64
        )
        self._primes_3x8 = np.ascontiguousarray(
            self.primes.reshape(self.layers, 1, 3, 8)
        )
        self._offsets_3x8 = np.ascontiguousarray(
            self.offsets.reshape(self.layers, 1, 3, 8)
        )
        self._column_plans: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        if os.environ.get("SPARK_DS41F_ENGRAM_HOST_HASH_AFFINITY", "1") != "0":
            pin_to_max_l2_cpus(0)

    def _compress_compact(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.int64)
        valid = (raw >= 0) & (raw < self.vocab) & (raw != IMAGE_SENTINEL)
        missing = self.token_map.dtype.type(-1)
        return np.where(valid, self.token_map[np.where(valid, raw, 0)], missing)

    def _column_plan(
        self, columns: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        plan = self._column_plans.get(columns)
        if plan is None:
            if not 0 < columns <= ALL_COLUMNS:
                raise ValueError(
                    f"Engram host hash columns must be a nonzero 24-bit mask: {columns:#x}"
                )
            cols = np.array(
                [c for c in range(HEADS) if (columns >> c) & 1], dtype=np.intp
            )
            orders = cols // 8
            primes_sub = np.ascontiguousarray(self.primes[:, cols][:, None, :])
            offsets_sub = np.ascontiguousarray(self.offsets[:, cols][:, None, :])
            plan = (cols, orders, primes_sub, offsets_sub)
            self._column_plans[columns] = plan
        return plan

    def compress(self, raw: np.ndarray) -> np.ndarray:
        """Compressed ids of raw token ids; -1 for a missing token (negative), an
        image token or one outside the tokenizer map."""
        return self._compress_compact(raw).astype(np.int64, copy=False)

    def _hash_positions_into(
        self,
        planes: np.ndarray,
        first_row: int,
        tokens: np.ndarray,
        positions: np.ndarray,
        columns: int = ALL_COLUMNS,
        layer: int | None = None,
    ) -> None:
        positions = np.asarray(positions, dtype=np.int64)
        n = positions.shape[0]
        lags = positions[:, None] - self._lags  # [T, 4]
        raw = np.where(lags >= 0, tokens[np.maximum(lags, 0)], -1)
        sources = self._compress_compact(raw)
        # From the first missing source on, every lag takes the pad id.
        blocked = np.logical_or.accumulate(sources == -1, axis=1)
        values_u64 = np.where(blocked, self.pad, sources).astype(np.uint64)
        if layer is None:
            layer_slice = slice(None)
            layer_count = self.layers
        else:
            layer_idx = int(layer)
            layer_slice = slice(layer_idx, layer_idx + 1)
            layer_count = 1
        # value * multiplier in wrapping int64, as the GPU computes it.
        products = (
            values_u64[None, :, :] * self._multipliers_u64[layer_slice]
        ).view(np.int64)  # [layer_count, T, 4]
        mixed2 = products[..., 0] ^ products[..., 1]
        mixed3 = mixed2 ^ products[..., 2]
        mixed4 = mixed3 ^ products[..., 3]
        # Heads 0-7 hash 2-grams, 8-15 3-grams, 16-23 4-grams.
        # The floor remainder (the sign of the divisor), as TileLang's int64 %. The
        # model's bounded multipliers keep ``mixed`` non-negative, where it equals
        # C's truncating one.
        if columns == ALL_COLUMNS:
            mixed_orders = np.stack((mixed2, mixed3, mixed4), axis=-1)[
                ..., None
            ]  # [layer_count, T, 3, 1]
            planes[:, first_row : first_row + n, :] = (
                np.remainder(mixed_orders, self._primes_3x8[layer_slice])
                + self._offsets_3x8[layer_slice]
            ).reshape(layer_count, n, HEADS)
        else:
            cols, orders, primes_sub, offsets_sub = self._column_plan(columns)
            mixed_orders = np.stack((mixed2, mixed3, mixed4), axis=-1)
            mixed_sub = mixed_orders[:, :, orders]
            planes[:, first_row : first_row + n, :] = -1
            planes[:, first_row : first_row + n, cols] = (
                np.remainder(mixed_sub, primes_sub[layer_slice])
                + offsets_sub[layer_slice]
            )

    def hash_positions(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        *,
        columns: int = ALL_COLUMNS,
    ) -> np.ndarray:
        """Row ids ``[len(positions), layers, 24]`` of the tokens at ``positions``
        of one request whose token ids up to the last position are ``tokens`` (prompt
        order; the GPU's query tokens and committed history are the same ids).
        Returned as a view over a layer-major ``[layers, len(positions), 24]`` buffer
        so each layer's slice ``[:, layer]`` is C-contiguous."""
        n = len(positions)
        planes = np.empty((self.layers, n, HEADS), dtype=np.int64)
        if n:
            self._hash_positions_into(planes, 0, tokens, positions, int(columns))
        return planes.transpose(1, 0, 2)

    def step_ids(
        self,
        num_tokens: int,
        requests,
        *,
        columns: int = ALL_COLUMNS,
        layer: int | None = None,
    ) -> np.ndarray:
        """A step's row ids ``[num_tokens, layers, 24]`` (or ``[num_tokens, 24]``
        when ``layer`` is given): ``requests`` yields, per request in the step's
        token order, ``(first_row, rows, tokens, positions)``: the request's first
        row and row count in the step, its token ids, and the positions of its
        leading rows whose ids the host has (the rest of its rows are ``UNKNOWN``).
        Rows past the last request's rows are padding, -1 as the GPU writes them.
        Returned as a view over a layer-major ``[layers, num_tokens, 24]`` buffer so
        each layer's slice ``[:, layer]`` is C-contiguous."""
        columns = int(columns)
        if columns != ALL_COLUMNS:
            self._column_plan(columns)
        if layer is not None:
            layer_idx = int(layer)
            if not 0 <= layer_idx < self.layers:
                raise ValueError(f"Engram layer index {layer_idx} out of range [0, {self.layers})")
            planes = np.full((1, num_tokens, HEADS), UNKNOWN, dtype=np.int64)
            out = planes[0]
        else:
            layer_idx = None
            planes = np.full((self.layers, num_tokens, HEADS), UNKNOWN, dtype=np.int64)
            out = planes.transpose(1, 0, 2)
        custom_hash = "hash_positions" in self.__dict__
        live = 0
        for first_row, rows, tokens, positions in requests:
            live = max(live, first_row + rows)
            if len(positions):
                if custom_hash and columns == ALL_COLUMNS:
                    hashed = self.hash_positions(tokens, positions)
                    out[first_row : first_row + len(positions)] = (
                        hashed[:, layer_idx] if layer_idx is not None else hashed
                    )
                else:
                    self._hash_positions_into(
                        planes, first_row, tokens, positions, columns, layer=layer_idx
                    )
        planes[:, live:, :] = -1
        return out


def placed_by_host(requests: list, drafts: np.ndarray | None) -> list:
    """``requests`` (``(first_row, rows, tokens, positions)`` in batch order) with the
    host's positions dropped from every request with drafts (``drafts``, per request).

    Adaptive verification gives verifying requests row counts the GPU chooses, but
    keeps them at the front of the batch with the host's total, so a request after
    them starts at the same row on the host and the GPU, and one among them does
    not. A request still prefilling can carry drafts (the scheduler pads a prompt
    with one token left to the verification width with placeholder drafts), so its
    rows stay ``UNKNOWN`` for the GPU to hash. Verifying requests after one without
    drafts would leave no row placed by the host, so the GPU hashes the whole step."""
    if drafts is None:
        return requests
    verifying = np.flatnonzero(drafts)
    front = len(verifying)
    if front and verifying[-1] >= front:
        front = len(requests)
    empty = np.empty(0, dtype=np.int64)
    return [
        (first, rows, tokens, empty if i < front else positions)
        for i, (first, rows, tokens, positions) in enumerate(requests)
    ]


__all__ = [
    "ALL_COLUMNS",
    "HEADS",
    "UNKNOWN",
    "HostEngramHash",
    "max_l2_cpus",
    "pin_to_max_l2_cpus",
    "placed_by_host",
]
