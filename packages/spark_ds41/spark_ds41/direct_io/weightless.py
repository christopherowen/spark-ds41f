# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests to the steered model name (WEIGHTLESS_STEER_MODEL).

One engine serves a model and, under a second served model name, the model
steered by a GLP file (``vllm/models/deepseek_v4_1/glp.py``). The OpenAI server
marks a request to the steered name in its sampling parameters' ``extra_args``;
the scheduler keys the request's cached prefix blocks by the mark, since steered
rows write different KV; the worker marks the request's rows for the steering
kernels. Unmarked requests are served as with steering off.
"""

import json
import math
from typing import Any

import vllm.envs as envs

EXTRA_ARGS_KEY = "weightless_steer"
# A block hash extra key no other kind of key can equal (cache salts and LoRA
# names are strings).
BLOCK_HASH_KEY = ("weightless_steer",)


def steered_model() -> str | None:
    """The served model name whose requests are steered, or None."""
    return envs.WEIGHTLESS_STEER_MODEL


def is_steered_model(model_name: str | None) -> bool:
    name = steered_model()
    return name is not None and model_name == name


def mark(extra_args: dict[str, Any] | None, steered: bool) -> dict[str, Any] | None:
    """``extra_args`` with the request's mark set, or cleared: only the server
    marks a request, never its caller."""
    extra = dict(extra_args or {})
    if steered:
        extra[EXTRA_ARGS_KEY] = True
    else:
        extra.pop(EXTRA_ARGS_KEY, None)
    return extra or None


def is_marked(sampling_params: Any) -> bool:
    """Whether ``sampling_params`` (or None) belong to a steered request."""
    extra = getattr(sampling_params, "extra_args", None)
    return bool(extra and extra.get(EXTRA_ARGS_KEY))


_ALLOWED_DEFAULTS = frozenset({"temperature", "top_p", "top_k", "min_p"})


def generation_defaults() -> dict[str, Any]:
    """Per-model sampling defaults for WEIGHTLESS_STEER_MODEL requests, or {}."""
    raw = envs.WEIGHTLESS_STEER_GENERATION_DEFAULTS
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"WEIGHTLESS_STEER_GENERATION_DEFAULTS is not valid JSON: {raw!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"WEIGHTLESS_STEER_GENERATION_DEFAULTS must be a JSON object, got {type(parsed).__name__}"
        )
    unknown = sorted(set(parsed) - _ALLOWED_DEFAULTS)
    if unknown:
        raise ValueError(
            f"WEIGHTLESS_STEER_GENERATION_DEFAULTS has unsupported keys {unknown}"
        )
    for key, val in parsed.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(
                f"WEIGHTLESS_STEER_GENERATION_DEFAULTS[{key!r}] must be a finite number, got {val!r}"
            )
        if key == "temperature" and val < 0:
            raise ValueError(
                f"WEIGHTLESS_STEER_GENERATION_DEFAULTS['temperature'] must be >= 0, got {val!r}"
            )
        if key == "top_p" and not (0 < val <= 1):
            raise ValueError(
                f"WEIGHTLESS_STEER_GENERATION_DEFAULTS['top_p'] must be in (0, 1], got {val!r}"
            )
        if key == "min_p" and not (0 <= val <= 1):
            raise ValueError(
                f"WEIGHTLESS_STEER_GENERATION_DEFAULTS['min_p'] must be in [0, 1], got {val!r}"
            )
    return parsed


def sampling_defaults(
    model_name: str | None, base_defaults: dict[str, Any]
) -> dict[str, Any]:
    """Return sampling defaults for ``model_name``: steered defaults merged over
    ``base_defaults`` for WEIGHTLESS_STEER_MODEL, otherwise ``base_defaults`` unchanged."""
    if not is_steered_model(model_name):
        return base_defaults
    overrides = generation_defaults()
    if not overrides:
        return base_defaults
    return {**base_defaults, **overrides}
