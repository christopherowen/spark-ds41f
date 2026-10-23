# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering: a weightless control vector projected off the residual stream.

A GLP file (GGUF Layer Projection; github.com/msuiche/weightless, spec
docs/spec/GLP.md) holds one unit direction ``direction.N`` per layer N. Steering
applies

    h <- h - alpha * (h . d) d

at ``residual_stream_post_layer``: layer N's write folded into the
hyper-connection streams, before layer N + 1 reads Engram. Every stream is
projected, so their mean, the reduction the directions are derived on, is
projected too. The layer's write alone (``ffn_out_pre_residual``) is a different
site, and a file for it is refused.

Off unless WEIGHTLESS_STEER_PATH names a file; the model then has no steering
module and its forward is unchanged. The reader is fail-closed: a file this
runtime cannot apply exactly stops the boot rather than serving unsteered or
steering at another site.
"""

import argparse
import math
import re
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

HOOK_POINT = "residual_stream_post_layer"
SPEC_VERSION = 1
# A direction's norm may differ from 1 by float32 export rounding, no more.
UNIT_NORM_TOLERANCE = 1e-3

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_GGUF_F32 = 0
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_DIRECTION = re.compile(r"direction\.(\d+)")


class _Cursor:
    def __init__(self, path: str, data: bytes):
        self.path = path
        self.data = data
        self.offset = 0

    def bytes(self, size: int) -> bytes:
        if self.offset + size > len(self.data):
            raise ValueError(f"{self.path}: truncated GGUF at byte {self.offset}")
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        return raw

    def take(self, fmt: str):
        values = struct.unpack(fmt, self.bytes(struct.calcsize(fmt)))
        return values[0] if len(values) == 1 else values

    def string(self) -> str:
        try:
            return self.bytes(self.take("<Q")).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{self.path}: GGUF string is not UTF-8") from None

    def value(self, kind: int):
        if kind == _GGUF_STRING:
            return self.string()
        if kind == _GGUF_ARRAY:
            item, count = self.take("<IQ")
            if item == _GGUF_ARRAY:
                raise ValueError(f"{self.path}: nested GGUF arrays are not supported")
            return [self.value(item) for _ in range(count)]
        if kind not in _GGUF_SCALARS:
            raise ValueError(f"{self.path}: unknown GGUF value type {kind}")
        return self.take(_GGUF_SCALARS[kind])


def read_gguf(path: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """A GGUF v3 file's metadata and tensors, which must be 1-D float32.

    Anything else, and any byte past the last tensor's alignment, is refused.
    """
    data = Path(path).read_bytes()
    cursor = _Cursor(path, data)
    if cursor.bytes(4) != _GGUF_MAGIC:
        raise ValueError(f"{path}: not a GGUF file")
    version = cursor.take("<I")
    if version != _GGUF_VERSION:
        raise ValueError(f"{path}: GGUF version {version}, expected {_GGUF_VERSION}")
    tensor_count, key_count = cursor.take("<QQ")

    metadata: dict[str, object] = {}
    for _ in range(key_count):
        key = cursor.string()
        if key in metadata:
            raise ValueError(f"{path}: duplicate GGUF key {key!r}")
        metadata[key] = cursor.value(cursor.take("<I"))

    infos = []
    for _ in range(tensor_count):
        name = cursor.string()
        dims = cursor.take(f"<{cursor.take('<I')}Q")
        dims = (dims,) if isinstance(dims, int) else dims
        kind, offset = cursor.take("<IQ")
        if kind != _GGUF_F32:
            raise ValueError(f"{path}: tensor {name} is GGML type {kind}, not F32")
        if len(dims) != 1:
            raise ValueError(f"{path}: tensor {name} has shape {dims}, not 1-D")
        infos.append((name, dims[0], offset))

    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0:
        raise ValueError(f"{path}: general.alignment is {alignment!r}")
    base = -(-cursor.offset // alignment) * alignment
    tensors: dict[str, np.ndarray] = {}
    end = base
    for name, count, offset in infos:
        if name in tensors:
            raise ValueError(f"{path}: duplicate tensor {name}")
        start = base + offset
        if start % alignment or start + 4 * count > len(data):
            raise ValueError(f"{path}: tensor {name} lies outside the file")
        tensors[name] = np.frombuffer(data, "<f4", count, start).copy()
        end = max(end, start + 4 * count)
    if len(data) - end >= alignment:
        raise ValueError(
            f"{path}: {len(data) - end} unexpected bytes after the last tensor"
        )
    return metadata, tensors


def _layer_list(text: str, what: str) -> list[int]:
    try:
        return sorted(int(item) for item in text.split(",") if item.strip())
    except ValueError:
        raise ValueError(f"{what} is not a comma list of layer ids: {text!r}") from None


def read_glp(
    path: str, num_layers: int, hidden_size: int
) -> tuple[dict[int, np.ndarray], float | None]:
    """The file's unit directions by layer, and its ``glp.alpha_default``.

    Refuses whatever this reader cannot apply exactly (spec, Reader
    conformance): another mode, spec version, rank or hook point; a direction
    for layer 0, outside the model, of another width, not finite or not unit
    norm; a tensor that is not a direction; a layer list that disagrees with the
    tensor names.
    """
    metadata, tensors = read_gguf(path)
    mode = metadata.get("glp.mode")
    if mode != "project":
        raise ValueError(
            f"{path}: glp.mode is {mode!r}; this runtime projects ('project') "
            "and never adds a control vector (an absent mode means 'add')"
        )
    version = metadata.get("glp.spec_version")
    if type(version) is not int or version != SPEC_VERSION:
        raise ValueError(
            f"{path}: glp.spec_version is {version!r}; this reader implements "
            f"version {SPEC_VERSION} (rank 1, no alpha scaling keys)"
        )
    rank = metadata.get("glp.rank", 1)
    if type(rank) is not int or rank != 1:
        raise ValueError(f"{path}: glp.rank is {rank!r}; this reader applies rank 1")
    hook = metadata.get("glp.hook_point")
    if hook != HOOK_POINT:
        raise ValueError(
            f"{path}: glp.hook_point is {hook!r}; this runtime applies at "
            f"{HOOK_POINT} only"
        )
    derived_at = metadata.get("glp.derived_at", hook)
    if derived_at != hook:
        logger.warning(
            "%s: transferred vector, derived at %s and applied at %s",
            path,
            derived_at,
            hook,
        )
    alpha = metadata.get("glp.alpha_default")
    if alpha is not None and not (isinstance(alpha, float) and math.isfinite(alpha)):
        raise ValueError(f"{path}: glp.alpha_default is {alpha!r}")

    directions: dict[int, np.ndarray] = {}
    for name, values in tensors.items():
        match = _DIRECTION.fullmatch(name)
        if match is None:
            raise ValueError(f"{path}: tensor {name} is not a direction.<layer>")
        layer = int(match.group(1))
        if not 1 <= layer < num_layers:
            raise ValueError(
                f"{path}: {name} is outside layers 1..{num_layers - 1} "
                "(direction.0 cannot be expressed)"
            )
        if values.size != hidden_size:
            raise ValueError(
                f"{path}: {name} has width {values.size}, not {hidden_size}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} is not finite")
        norm = float(np.linalg.norm(values.astype(np.float64)))
        if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
            raise ValueError(f"{path}: {name} has norm {norm:.6f}, not 1")
        directions[layer] = (values.astype(np.float64) / norm).astype(np.float32)
    if not directions:
        raise ValueError(f"{path}: no direction tensors")
    declared = metadata.get("glp.layer_ids_zero_based")
    if declared is not None and _layer_list(str(declared), path) != sorted(directions):
        raise ValueError(
            f"{path}: glp.layer_ids_zero_based {declared!r} disagrees with the "
            f"direction tensors {sorted(directions)}; re-export the file"
        )
    logger.info(
        "GLP file %s: %d directions, layers %s, alpha_default %s, base %s@%s",
        path,
        len(directions),
        sorted(directions),
        alpha,
        metadata.get("general.base_model.0.name"),
        metadata.get("general.base_model.0.version"),
    )
    return directions, alpha


class GLPSteering(nn.Module):
    """Projects a layer's direction off its folded hyper-connection streams."""

    def __init__(self, directions: torch.Tensor, alpha: float):
        super().__init__()
        # One float32 row per model layer; unsteered layers are zero rows, so
        # the traced graph does not depend on which layers a file covers.
        self.register_buffer("directions", directions, persistent=False)
        # A tensor, not a Python float, which would be a graph constant.
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.float32), persistent=False
        )

    def forward(self, streams: torch.Tensor, layer: int) -> torch.Tensor:
        h = streams.float()
        d = self.directions[layer]
        coefficient = h @ d
        return (h - (self.alpha * coefficient).unsqueeze(-1) * d).to(streams.dtype)


def steer_post_layer(
    glp: GLPSteering | None,
    mhc: nn.Module,
    layer: int,
    hidden_states,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
):
    """Layer outputs for the next layer, steered when ``glp`` is set.

    Unsteered, the outputs pass through untouched, for the next layer's fused
    post/pre. Steered, the layer's write is folded into its streams here,
    projected, and passed on as the next layer's input with no residual: the
    layer then reads Engram and runs its pre on the folded streams, as an Engram
    layer does.
    """
    if glp is None:
        return hidden_states, residual, post_mix, res_mix
    return (
        glp(mhc.post(hidden_states, residual, post_mix, res_mix), layer),
        None,
        None,
        None,
    )


def glp_steering(num_layers: int, hidden_size: int) -> GLPSteering | None:
    """The steering WEIGHTLESS_STEER_PATH asks for, or None when it is unset.

    WEIGHTLESS_STEER_ALPHA overrides the file's alpha_default and
    WEIGHTLESS_STEER_LAYERS restricts steering to a comma list of the file's
    layers. Both are refused when malformed.
    """
    path = envs.WEIGHTLESS_STEER_PATH
    if path is None:
        return None
    directions, alpha = read_glp(path, num_layers, hidden_size)
    if envs.WEIGHTLESS_STEER_ALPHA is not None:
        try:
            alpha = float(envs.WEIGHTLESS_STEER_ALPHA)
        except ValueError:
            alpha = math.nan
        if not math.isfinite(alpha):
            raise ValueError(
                f"WEIGHTLESS_STEER_ALPHA is {envs.WEIGHTLESS_STEER_ALPHA!r}"
            )
    if alpha is None:
        raise ValueError(f"{path}: no glp.alpha_default and no WEIGHTLESS_STEER_ALPHA")
    layers = sorted(directions)
    if envs.WEIGHTLESS_STEER_LAYERS is not None:
        layers = _layer_list(envs.WEIGHTLESS_STEER_LAYERS, "WEIGHTLESS_STEER_LAYERS")
        missing = sorted(set(layers) - set(directions))
        if not layers or missing:
            raise ValueError(
                f"WEIGHTLESS_STEER_LAYERS {envs.WEIGHTLESS_STEER_LAYERS!r} names "
                f"layers without a direction in {path}: {missing}"
            )
    stack = torch.zeros((num_layers, hidden_size), dtype=torch.float32)
    for layer in layers:
        stack[layer] = torch.from_numpy(directions[layer])
    logger.info("GLP steering on: alpha %s, layers %s", alpha, layers)
    return GLPSteering(stack, alpha)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a GLP file against this runtime's reader without serving."
    )
    parser.add_argument("path")
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=5120)
    args = parser.parse_args()
    metadata, _ = read_gguf(args.path)
    directions, alpha = read_glp(args.path, args.num_layers, args.hidden_size)
    for key in sorted(metadata):
        print(f"{key} = {metadata[key]!r}")
    print(f"directions: {len(directions)}, layers {sorted(directions)}, alpha {alpha}")


if __name__ == "__main__":
    main()
