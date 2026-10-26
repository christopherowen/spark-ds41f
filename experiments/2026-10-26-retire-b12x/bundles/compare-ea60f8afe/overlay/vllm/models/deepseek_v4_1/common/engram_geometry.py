# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Engram: the compressed-token map, the hash geometry and the
per-layer capacities.

From B12X (local-inference-lab/b12x at 236ddff04a5f, Apache-2.0):
``b12x.sequence.engram.geometry`` (``build_compressed_token_map``, ``Geometry``,
``build_geometry``), ``b12x.sequence.engram._impl.Caps`` and the 64-bit prime test of
``b12x.sequence.ple_hash.reference``, which follow DeepSeek's reference
(inference/engram.py). The same tokenizer map, primes, offsets and multipliers, and the
same capacity checks.
"""

from dataclasses import dataclass

import numpy as np
import torch


def is_prime_64(value: int) -> bool:
    """Deterministically test primality over the unsigned 64-bit range."""
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    odd = value - 1
    shifts = 0
    while odd % 2 == 0:
        odd //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, odd, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Each token id's compressed id, and the compressed vocabulary's size. Uses raw
    backend decoding, keeping malformed UTF-8 token spellings."""
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    keys: dict[str, int] = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "�" in text
            else normalizer.normalize_str(text) or text
        )
        if key not in keys:
            keys[key] = len(keys)
        lookup.append(keys[key])
    return lookup, len(keys)


@dataclass(frozen=True)
class Geometry:
    """Host-only immutable geometry; all device geometry uses signed int64."""

    layer_ids: tuple[int, ...]
    primes: tuple[tuple[int, ...], ...]
    offsets: tuple[tuple[int, ...], ...]
    multipliers: tuple[tuple[int, ...], ...]
    num_embeddings: tuple[int, ...]
    compressed_vocab_size: int


def build_geometry(
    *,
    layer_ids=(1, 14),
    base_table_size=16_000_000,
    compressed_vocab_size=99_092,
) -> Geometry:
    """Generate 2/3/4-gram, eight-head geometry with globally unreused primes."""
    layer_ids = tuple(layer_ids)
    if (
        not layer_ids
        or len(set(layer_ids)) != len(layer_ids)
        or any(i < 0 for i in layer_ids)
    ):
        raise ValueError("layer_ids must be unique nonnegative actual layer IDs")
    if base_table_size < 2 or not 0 < compressed_vocab_size < (1 << 62):
        raise ValueError("invalid table or compressed vocabulary size")
    seen: set[int] = set()
    primes, offsets, multipliers, totals = [], [], [], []
    bound = max(1, (((1 << 63) - 1) // compressed_vocab_size) // 2)
    for layer in layer_ids:
        sizes = []
        for _ in range(3):
            current = base_table_size - 1
            for _ in range(8):
                current += 1
                while current in seen or not is_prime_64(current):
                    current += 1
                seen.add(current)
                sizes.append(current)
        cumulative, total = [], 0
        for size in sizes:
            cumulative.append(total)
            total += size
        if total * 256 > (1 << 63) - 1:
            raise ValueError("table extent exceeds signed int64 indexing")
        rng = np.random.Generator(np.random.PCG64(10007 * layer))
        values = rng.integers(0, bound, size=4, dtype=np.int64) * 2 + 1
        primes.append(tuple(sizes))
        offsets.append(tuple(cumulative))
        multipliers.append(tuple(int(v) for v in values))
        totals.append(total)
    return Geometry(
        layer_ids,
        tuple(primes),
        tuple(offsets),
        tuple(multipliers),
        tuple(totals),
        compressed_vocab_size,
    )


def _device(value) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


@dataclass(frozen=True, kw_only=True)
class EngramCaps:
    """One Engram layer's capacities on one TP rank."""

    device: torch.device | str
    max_tokens: int
    max_seqs: int
    max_requests: int
    vocab_size: int = 129280
    layer_id: int = 1
    tp_size: int = 8
    tp_rank: int = 0

    def __post_init__(self):
        object.__setattr__(self, "device", _device(self.device))
        positive = ("max_tokens", "max_seqs", "max_requests", "vocab_size", "tp_size")
        if self.device.type != "cuda" or any(
            type(getattr(self, name)) is not int or getattr(self, name) <= 0
            for name in positive
        ):
            raise ValueError("Engram requires positive CUDA capacities")
        if (
            not 0 <= self.tp_rank < self.tp_size
            or self.max_tokens >= 2**31
            or self.max_seqs >= 2**31
        ):
            raise ValueError("invalid Engram rank or int32 capacity")


__all__ = [
    "EngramCaps",
    "Geometry",
    "build_compressed_token_map",
    "build_geometry",
    "is_prime_64",
]
