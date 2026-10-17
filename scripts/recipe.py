"""One recipe for every cluster shape.

``config/cluster.json`` holds everything the shapes share (image, kernels,
features, serving flags) once, and a ``shapes`` table with only what a shape
really changes:

- the cabling: the node map, the one-shot transport and the NCCL settings of
  sparknet's profile for that fabric;
- the sizing that follows from memory per rank: context, streams, batched
  tokens, KV cache bytes, prefix-cache retention, the display carve-out;
- the DSpark cost curves measured on that shape;
- ``benchmark_reference``: the release benchmark that qualified the recipe's
  image on that shape. A shape without one may not start in production.

A shape may ``extend`` another (the 4 KiB-page variant of TP3 changes the host
kernel and the sizing only). An experiment arm is an overlay: ``extends`` names
a configuration (the recipe at a shape, or another arm) and ``set``/``unset``
hold only what the arm changes, so arms cannot drift from what they compare to. ``models``
names each checkpoint the recipe serves (its snapshot and blob store) and ``model`` the one
mounted at /models and /blobs; ``generation_defaults`` is the sampling a request that sets none
gets (vLLM's ``--override-generation-config``). An arm may set either. The site names the shape it is cabled as in
``config/site.json``; ``--shape`` overrides it for planning.

``resolve`` returns the same concrete cluster configuration a per-shape file
used to be, so everything downstream reads one form. Derived values are not
written by hand: the tensor-parallel size is the node count, CUDA-graph
capture sizes run up to streams x (1 + drafts), and the long-prefill threshold
leaves every stream's decode and draft rows beside a whole chunk.
"""

from __future__ import annotations

import copy
import json
import string
from pathlib import Path

SITE_FILE = "config/site.json"
# Where every shape mounts the selected model's snapshot and blob store.
MODEL_MOUNTS = ("/models", "/blobs")
SIZING_KEYS = ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes",
               "prefix_cache_retention_interval")
# Keys a shape may set beside its sizing; anything else belongs to every shape.
SHAPE_KEYS = {"extends", "nodes", "nodes_config", "fabric", "sizing", "environment", "host",
              "benchmark_reference", "notes"}


def is_recipe(cluster: dict) -> bool:
    return "shapes" in cluster


def capture_sizes(max_size: int) -> list[int]:
    """The CUDA-graph capture ladder up to ``max_size``: 1-4, 6, 8, then every 4th to 32, then every 8th."""
    ladder = [1, 2, 3, 4, 6, 8, *range(12, 33, 4), *range(40, max_size + 1, 8)]
    return [size for size in ladder if size <= max_size]


def shape_entry(recipe: dict, name: str) -> dict:
    """The shape's own entry merged over the one it extends."""
    shapes = recipe["shapes"]
    if name not in shapes:
        raise SystemExit(f"unknown shape {name!r}; the recipe defines {', '.join(sorted(shapes))}")
    entry = shapes[name]
    unknown = set(entry) - SHAPE_KEYS
    if unknown:
        raise SystemExit(f"shape {name}: {', '.join(sorted(unknown))} belong to every shape, not one")
    if "extends" not in entry:
        return copy.deepcopy(entry)
    merged = shape_entry(recipe, entry["extends"])
    for key, value in entry.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    if "benchmark_reference" not in entry:
        # A variant is qualified by its own release benchmark, never through its base.
        merged.pop("benchmark_reference", None)
    return merged


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def model_mounts(cluster: dict) -> list[list[str]]:
    """The mounts of the selected model (``model``, one of ``models``): its snapshot at
    /models and its blob store at /blobs, both read-only."""
    name, models = cluster.get("model"), cluster.get("models", {})
    if name not in models:
        raise SystemExit(f"unknown model {name!r}; the recipe defines {', '.join(sorted(models))}")
    return [[models[name]["snapshot"], MODEL_MOUNTS[0], "ro"],
            [models[name]["blobs"], MODEL_MOUNTS[1], "ro"]]


def _with_model_mounts(cluster: dict) -> None:
    mounts = cluster["container"].get("mounts", [])
    cluster["container"]["mounts"] = model_mounts(cluster) + [
        mount for mount in mounts if mount[1] not in MODEL_MOUNTS]


def _with_generation_defaults(cluster: dict) -> None:
    """Render ``generation_defaults`` (the sampling a request that sets none gets) as
    vLLM's ``--override-generation-config``, beside ``--generation-config``."""
    args = list(cluster["serve_args"])
    if "--override-generation-config" in args:
        at = args.index("--override-generation-config")
        del args[at : at + 2]
    defaults = cluster.get("generation_defaults")
    if defaults:
        flag = ["--override-generation-config", json.dumps(defaults, separators=(",", ":"), sort_keys=True)]
        at = args.index("--generation-config") + 2 if "--generation-config" in args else len(args)
        args[at:at] = flag
    cluster["serve_args"] = args


def values(recipe: dict, shape: dict) -> dict[str, str]:
    """The placeholders of ``serve_args`` for one shape."""
    sizing = shape["sizing"]
    missing = [key for key in SIZING_KEYS if key not in sizing]
    if missing:
        raise SystemExit(f"shape sizing lacks {', '.join(missing)}")
    nodes, drafts = int(shape["nodes"]), int(recipe["drafts"])
    rows = int(sizing["max_num_seqs"]) * (1 + drafts)
    derived = {
        "tensor_parallel_size": nodes,
        "max_cudagraph_capture_size": rows,
        "cudagraph_capture_sizes": json.dumps(capture_sizes(rows), separators=(",", ":")),
        "long_prefill_token_threshold": int(sizing["max_num_batched_tokens"]) - rows,
        "drafts": drafts,
    }
    return {key: str(value) for key, value in {**sizing, **derived}.items()}


def resolve(recipe: dict, name: str) -> dict:
    """The concrete cluster configuration of shape ``name``."""
    shape = shape_entry(recipe, name)
    cluster = {key: copy.deepcopy(value) for key, value in recipe.items() if key not in ("shapes", "drafts")}
    cluster["schema_version"] = 1
    cluster["shape"] = name
    fill = values(recipe, shape)
    try:
        cluster["serve_args"] = [string.Template(arg).substitute(fill) for arg in recipe["serve_args"]]
    except KeyError as error:
        raise SystemExit(f"serve_args names ${{{error.args[0]}}}, which no shape value defines") from error
    cluster["environment"] = {**recipe["environment"], **shape.get("environment", {})}
    overlap = set(recipe["environment"]) & set(shape.get("environment", {}))
    if overlap:
        raise SystemExit(f"shape {name} repeats shared environment {', '.join(sorted(overlap))}")
    cluster["fabric"] = shape["fabric"]
    cluster["host"] = _merge(recipe["host"], shape.get("host", {}))
    if "nodes_config" in shape:
        cluster["nodes_config"] = shape["nodes_config"]
    if "benchmark_reference" in shape:
        cluster["benchmark_reference"] = shape["benchmark_reference"]
    if "models" in recipe:
        _with_model_mounts(cluster)
    if "generation_defaults" in recipe:
        _with_generation_defaults(cluster)
    return cluster


def is_overlay(cluster: dict) -> bool:
    return "extends" in cluster


def overlay(arm: dict, base: dict) -> dict:
    """An experiment arm: ``base`` (the recipe at a shape, or another arm, already
    resolved) with ``set`` merged in and the dotted paths of ``unset`` removed.

    An arm is not production: it carries neither the recipe's shape nor its
    qualifying benchmark, so starting it needs no qualification and a bench names
    its own reference.
    """
    unknown = set(arm) - {"extends", "set", "unset", "notes"}
    if unknown:
        raise SystemExit(f"an arm overlay holds extends, set, unset and notes; not {', '.join(sorted(unknown))}")
    cluster = _merge(base, arm.get("set", {}))
    if "model" in arm.get("set", {}):
        # The arm serves another of the recipe's models, whatever mounts it lists.
        _with_model_mounts(cluster)
    if "generation_defaults" in arm.get("set", {}):
        _with_generation_defaults(cluster)
    for key in ("shape", "benchmark_reference"):
        cluster.pop(key, None)
    for dotted in arm.get("unset", []):
        *parents, last = dotted.split(".")
        node = cluster
        for part in parents:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if not isinstance(node, dict) or last not in node:
            raise SystemExit(f"the arm unsets {dotted}, which its base does not set")
        del node[last]
    return cluster


def site_shape(root: Path) -> str | None:
    """The shape this site is cabled as (``config/site.json``), or None."""
    path = root / SITE_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["shape"]


def node_map_problems(shape: dict, nodes: dict) -> list[str]:
    """A node map that cannot serve this shape."""
    count = len(nodes["nodes"])
    if count != int(shape["nodes"]):
        return [f"the node map has {count} nodes and the shape needs {shape['nodes']}"]
    return []
