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
names each checkpoint the recipe serves (a snapshot and its blob store, or a hub cache and the
snapshot within it) and ``model`` the one mounted and served; ``generation_defaults`` is the sampling a request that sets none
gets (vLLM's ``--override-generation-config``). An arm may set either. ``glp``, when present,
serves a GLP-steered variant of the model beside it: its ``served_model_name`` and the
``file`` of a Hugging Face ``repo`` at a pinned ``revision`` (with its ``sha256``), rendered
as the second served name, vLLM's steering variables and a read-only mount of the file
from the node's hub cache. The site names the shape it is cabled as in
``config/site.json``; a site that names none gets the recipe's ``default_shape``,
and ``--shape`` overrides either for planning.

``resolve`` returns the same concrete cluster configuration a per-shape file
used to be, so everything downstream reads one form. Derived values are not
written by hand: the tensor-parallel size is the node count, CUDA-graph
capture sizes run up to streams x (1 + drafts), and the long-prefill threshold
leaves every stream's decode and draft rows beside a whole chunk.
"""

from __future__ import annotations

import copy
import json
import shlex
import string
from pathlib import Path

SITE_FILE = "config/site.json"
# Where the selected model is mounted: a snapshot at /models with its blob store at
# /blobs, or a whole Hugging Face hub cache at /hub (a snapshot whose blobs link on into
# a shared store resolves only inside the hub it was laid out in).
MODEL_MOUNTS = ("/models", "/blobs", "/hub")
# GLP steering (``glp``): where the file is mounted, the variables vLLM reads, and the hub
# cache on each node that ``hf download`` fills.
GLP_TARGET = "/glp/vector.gguf"
STEER_PATH, STEER_MODEL = "WEIGHTLESS_STEER_PATH", "WEIGHTLESS_STEER_MODEL"
GLP_HUB = "{home}/.cache/huggingface/hub"
GLP_KEYS = ("served_model_name", "repo", "revision", "file", "sha256")
SIZING_KEYS = ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes",
               "prefix_cache_retention_interval")
# Keys a shape may set beside its sizing; anything else belongs to every shape.
SHAPE_KEYS = {"extends", "nodes", "nodes_config", "fabric", "sizing", "environment", "host",
              "benchmark_reference", "notes"}


def is_recipe(cluster: dict) -> bool:
    return "shapes" in cluster


def capture_sizes(max_size: int) -> list[int]:
    """The CUDA-graph capture ladder up to ``max_size``: 1-4, 6, 8, then every 4th to 32, then
    every 8th, ending at ``max_size`` itself. vLLM refuses a ladder whose largest size is not
    ``--max-cudagraph-capture-size``, so a stream count whose rows fall between two steps
    (6 streams x 6 rows = 36) adds its own last step (#434)."""
    ladder = [1, 2, 3, 4, 6, 8, *range(12, 33, 4), *range(40, max_size + 1, 8)]
    return [*(size for size in ladder if size < max_size), max_size]


def _argument(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    at = args.index(flag) + 1
    return args[at] if at < len(args) else None


def capture_problems(cluster: dict) -> tuple[list[str], list[str]]:
    """Errors and warnings when a configuration's CUDA-graph capture sizes disagree with its
    sizing. The recipe derives both from ``sizing.max_num_seqs`` x (1 + drafts); a profile
    that writes its own ``serve_args`` can leave the ladder behind when it changes the
    stream count. vLLM refuses to start when the ladder does not end at
    ``--max-cudagraph-capture-size`` (an error); a ladder that ends below the rows a full
    decode step holds boots, but runs those steps without a CUDA graph (a warning). A
    ladder above the rows is not flagged: its larger sizes still serve mixed prefill
    steps as piecewise graphs."""
    args = cluster.get("serve_args", [])
    if "--enforce-eager" in args:
        return [], []
    try:
        compilation = json.loads(_argument(args, "--compilation-config") or "{}")
        speculative = json.loads(_argument(args, "--speculative-config") or "{}")
    except ValueError:
        return ["--compilation-config and --speculative-config must be JSON"], []
    sizes = compilation.get("cudagraph_capture_sizes")
    maximum = _argument(args, "--max-cudagraph-capture-size") or compilation.get("max_cudagraph_capture_size")
    streams = _argument(args, "--max-num-seqs")
    if not sizes or maximum is None or streams is None:
        return [], []
    drafts = int(speculative.get("num_speculative_tokens", 0)) if speculative else 0
    rows = int(streams) * (1 + drafts)
    rule = f"sizing.max_num_seqs ({streams}) x (1 + {drafts} drafts) = {rows} rows"
    largest, maximum = max(sizes), int(maximum)
    if largest != maximum:
        return [f"the CUDA-graph capture sizes end at {largest} and --max-cudagraph-capture-size is "
                f"{maximum}, which vLLM refuses at startup; both follow {rule}: a profile that writes "
                "its own serve_args changes all three together"], []
    if largest < rows:
        return [], [f"CUDA graphs are captured up to {largest} rows, below {rule}: "
                    "decode steps above it run without a CUDA graph"]
    return [], []


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


def _model_profile(cluster: dict) -> dict:
    name, models = cluster.get("model"), cluster.get("models", {})
    if name not in models:
        raise SystemExit(f"unknown model {name!r}; the recipe defines {', '.join(sorted(models))}")
    return models[name]


def model_mounts(cluster: dict) -> tuple[list[list[str]], str]:
    """The mounts of the selected model (``model``, one of ``models``) and the path vLLM
    serves it from. A profile names a ``snapshot`` and its ``blobs`` (mounted at /models
    and /blobs), or a ``hub`` cache root and the ``snapshot`` path within it (the root
    mounted at /hub, served from /hub/<snapshot>); every mount is read-only."""
    profile = _model_profile(cluster)
    if "hub" in profile:
        return [[profile["hub"], "/hub", "ro"]], f"/hub/{profile['snapshot']}"
    return ([[profile["snapshot"], "/models", "ro"], [profile["blobs"], "/blobs", "ro"]],
            "/models")


def model_snapshot(cluster: dict) -> str:
    """The selected model's snapshot directory on the host (``{home}`` unexpanded)."""
    profile = _model_profile(cluster)
    return f"{profile['hub']}/{profile['snapshot']}" if "hub" in profile else profile["snapshot"]


def _with_model_mounts(cluster: dict) -> None:
    mounts, path = model_mounts(cluster)
    cluster["container"]["mounts"] = mounts + [
        mount for mount in cluster["container"].get("mounts", []) if mount[1] not in MODEL_MOUNTS]
    if not cluster["serve_args"] or not cluster["serve_args"][0].startswith("/"):
        raise SystemExit("serve_args must start with the model path")
    cluster["serve_args"] = [path, *cluster["serve_args"][1:]]


def glp_file(cluster: dict) -> str | None:
    """The host path (``{home}`` unexpanded) of the ``glp`` block's file: its pinned
    revision in the node's Hugging Face hub cache, where ``hf download`` puts it."""
    glp = cluster.get("glp")
    if not glp:
        return None
    owner, name = glp["repo"].split("/", 1)
    return f"{GLP_HUB}/models--{owner}--{name}/snapshots/{glp['revision']}/{glp['file']}"


def glp_download(cluster: dict) -> str:
    glp = cluster["glp"]
    return shlex.join(["hf", "download", glp["repo"], glp["file"], "--revision", glp["revision"]])


def _glp_rendering(glp: dict) -> tuple[dict[str, str], list[str]]:
    environment = {STEER_PATH: GLP_TARGET, STEER_MODEL: glp["served_model_name"]}
    return environment, [glp_file({"glp": glp}), GLP_TARGET, "ro"]


def _with_glp(cluster: dict, previous: dict | None = None) -> None:
    """Render ``glp`` (the steered model name and the GLP file it is steered with) as the
    second ``--served-model-name``, vLLM's two ``WEIGHTLESS_STEER_*`` variables and a
    read-only mount of the file, after removing what ``previous`` rendered."""
    args, environment = cluster["serve_args"], cluster["environment"]
    mounts = cluster["container"].setdefault("mounts", [])
    if previous:
        if previous["served_model_name"] in args:
            args.remove(previous["served_model_name"])
        for key in (STEER_PATH, STEER_MODEL):
            environment.pop(key, None)
        mounts[:] = [mount for mount in mounts if mount[1] != GLP_TARGET]
    glp = cluster.get("glp")
    if not glp:
        return
    missing = [key for key in GLP_KEYS if not glp.get(key)]
    if missing:
        raise SystemExit(f"glp lacks {', '.join(missing)}")
    if STEER_PATH in environment or STEER_MODEL in environment or any(
            mount[1] == GLP_TARGET for mount in mounts):
        raise SystemExit(f"glp renders {STEER_PATH}, {STEER_MODEL} and the {GLP_TARGET} mount; "
                         "a profile does not set them itself")
    if "--served-model-name" not in args:
        raise SystemExit("glp needs --served-model-name in serve_args: the stock name it serves beside")
    at = args.index("--served-model-name") + 1
    while at < len(args) and not args[at].startswith("--"):
        at += 1
    args.insert(at, glp["served_model_name"])
    rendered, mount = _glp_rendering(glp)
    environment.update(rendered)
    mounts.append(mount)


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
    cluster = {key: copy.deepcopy(value) for key, value in recipe.items()
               if key not in ("shapes", "drafts", "default_shape")}
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
    if "glp" in recipe:
        _with_glp(cluster)
    return cluster


def is_overlay(cluster: dict) -> bool:
    return "extends" in cluster


def overlay(arm: dict, base: dict) -> dict:
    """An experiment arm: ``base`` (the recipe at a shape, or another arm, already
    resolved) with ``set`` merged in, the values of the ``serve_options`` it changes, and
    the dotted paths of ``unset`` removed.

    ``serve_options`` maps a flag that ``serve_args`` passes once, with a value, to the
    arm's value, so an arm that changes one serving option still follows every other
    argument of its base (``set`` replaces a list whole).

    An arm is not production: it carries neither the recipe's shape nor its
    qualifying benchmark, so starting it needs no qualification and a bench names
    its own reference.
    """
    unknown = set(arm) - {"extends", "set", "serve_options", "unset", "notes"}
    if unknown:
        raise SystemExit("an arm overlay holds extends, set, serve_options, unset and notes; not "
                         + ", ".join(sorted(unknown)))
    cluster = _merge(base, arm.get("set", {}))
    for flag, value in arm.get("serve_options", {}).items():
        args = cluster["serve_args"]
        if args.count(flag) != 1 or args.index(flag) + 1 == len(args) or args[args.index(flag) + 1].startswith("--"):
            raise SystemExit(f"the arm sets {flag}, which its base's serve_args do not pass once with a value")
        if not isinstance(value, str):
            raise SystemExit(f"the arm sets {flag} to {value!r}: serve_args values are strings")
        args[args.index(flag) + 1] = value
    sets = arm.get("set", {})
    if "models" in cluster and ("model" in sets or "mounts" in sets.get("container", {})):
        # The arm serves its model (another of the recipe's, or the one it extends)
        # whatever mounts it lists: a mount list replaces the base's, which held the
        # model's mounts.
        _with_model_mounts(cluster)
    if "generation_defaults" in arm.get("set", {}):
        _with_generation_defaults(cluster)
    for dotted in arm.get("unset", []):
        *parents, last = dotted.split(".")
        node = cluster
        for part in parents:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if not isinstance(node, dict) or last not in node:
            raise SystemExit(f"the arm unsets {dotted}, which its base does not set")
        del node[last]
    for key in ("shape", "benchmark_reference"):
        cluster.pop(key, None)
    if (cluster.get("fabric", {}).get("topology") == "switched"
            and base.get("fabric", {}).get("topology") != "switched"):
        explicit_env = sets.get("environment", {})
        for key in ("NCCL_SWITCHLESS_BIDIRECTIONAL", "NCCL_MIN_TRAFFIC_PER_CHANNEL",
                    "NCCL_THREAD_THRESHOLDS", "NCCL_ALGO", "NCCL_MIN_NCHANNELS",
                    "NCCL_ALGO_PLUGIN", "NCCL_TUNER_PLUGIN", "NCCL_GRAPH_FILE", "NCCL_TOPO_FILE"):
            if key not in explicit_env:
                cluster.get("environment", {}).pop(key, None)
    if base.get("glp") or cluster.get("glp"):
        # Rendered again whatever the arm changed: a serve_args or mount list it sets
        # replaces the base's, which held the rendering.
        _with_glp(cluster, previous=base.get("glp"))
    return cluster


def site_shape(root: Path) -> str | None:
    """The shape this site is cabled as (``config/site.json``), or None."""
    path = root / SITE_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["shape"]


def chosen_shape(recipe: dict, root: Path, override: str | None = None) -> str | None:
    """The shape to resolve ``recipe`` at: ``override`` (``--shape``), else the one this
    site is cabled as (``config/site.json``), else the recipe's ``default_shape``."""
    return override or site_shape(root) or recipe.get("default_shape")


def node_map_problems(shape: dict, nodes: dict) -> list[str]:
    """A node map that cannot serve this shape."""
    count = len(nodes["nodes"])
    if count != int(shape["nodes"]):
        return [f"the node map has {count} nodes and the shape needs {shape['nodes']}"]
    return []
