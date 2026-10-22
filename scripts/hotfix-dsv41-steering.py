#!/usr/bin/env python3
"""Hotfix: projective activation steering (GLP) for DeepSeek-V4.1-Flash in spark-ds41f.

Port of weightless's hotfix-dsv4-steering-projective.py
(https://github.com/msuiche/weightless, format spec docs/spec/GLP.md) via the
dsv41-feat port for tonyd2wild's recipe, to this repository's vLLM tree:
local-inference-lab/vllm integration/karmic-kraken-beta @ 19f2c20 plus the
promoted r6v patch series (172 patches, experiments/2026-10-17-r6v). Applies

    h <- h - alpha * (h . d_hat) d_hat

at the model loop's per-layer boundary, gated by env:

    WEIGHTLESS_STEER_PATH    .gguf (spec-conformant cvec) or .pt {layer: tensor}
    WEIGHTLESS_STEER_ALPHA   float; unset => the file's glp.alpha_default
                               (the published 1017 vector pins 0.75), else 1.0
    WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids
    WEIGHTLESS_STEER_HOOK    if set, must be residual_stream_post_layer

The published vector for this checkpoint is
msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75
(glp.hook_point == glp.derived_at == residual_stream_post_layer, 39 layers,
alpha_default 0.75, width 5120).

Hook site, and why the label matches it on this tree: the loop anchor's
``hidden_states`` is the layer's PENDING FFN WRITE, shape (tokens,
hidden_size) — the mHC fold of that write into the hyper-connection streams
is deferred to the next layer's fused post/pre call (the same return
convention as the 0731 tree and the dsv41-feat tree). The fold is linear in
the write, and the write is the only hidden-width post-layer surface the loop
materialises, so the post-layer projection lands here exactly; this is the
same anchor every published vLLM-lane DSV4 number was measured at. A file
declaring any other glp.hook_point fails closed. Vectors captured at a
different site than declared (glp.derived_at != glp.hook_point) are
transferred vectors: legal, warned loudly, never silent.

Engram note: layers 1 and 14 (0-based) inject Engram reads between the
previous layer's post and their own pre. Steering direction.N subtracts from
layer N's pending write only; the Engram additions are untouched.

DSpark note: the DSpark draft model is a separate class and is NOT steered.
The aux hidden states the draft reads are reconstructed after the steering
apply (same order as the 0731 lane), so the verifier runs steered while the
drafter's proposal stream sees the steered aux inputs at the aux layers
only. Spec decode stays on.

Differences from the dsv41-feat port:

- Anchors retargeted at the r6v tree: the layer call takes ``sp_rows`` after
  ``ced_indices``, __init__ ends at ``prepare_disk_engram`` (the disk-Engram
  reader), and ``current_platform`` is not imported (the device resolves
  through torch directly).
- Everything else is verbatim: fail-closed glp.* gates, embedded minimal
  GGUF v3 parser, dense zero-padded _steer_stack indexed by GLOBAL layer id
  (correct under PP; zero rows are a numeric no-op so the traced graph is
  identical for every layer set), alpha as a registered tensor buffer (a
  Python float would be baked into the torch.compile cache), alpha_default
  honoured from the file when the env is unset.

Failure semantics (fail-closed where it matters):

- Anchors not found at patch time: the build dies (a lane that asked for
  steering must not silently ship unsteered).
- WEIGHTLESS_STEER_PATH set at boot but the vector file missing/invalid/
  non-project/wrong hook: the engine __init__ raises before the model load.
- Runtime load failures with steering armed (direction width != hidden,
  no layers matched, direction layers out of range) re-raise: the boot
  dies rather than serving unsteered.

Patches the named model.py in place, ONLY when a GLP vector is configured
(WEIGHTLESS_STEER_PATH). With no vector configured, nothing is injected:
the model.py is left byte-identical to stock — zero extra kernels, zero
extra launches, same traced graph. Steering is a build-time configuration
in this repository, not a runtime toggle: an image built without it pays
nothing. (WEIGHTLESS_STEERING_MODEL_PY overrides the target path for
out-of-tree application; in this repository the patch ships as a git-am
patch in the promoted vLLM series instead — the steering apply enters the
series output only when the launch configures it.)
``--status`` reports state; ``--check`` validates the vector named by
WEIGHTLESS_STEER_PATH without touching model.py.
"""
import os
from pathlib import Path
import sys

# Overridable for dry-runs against a copy of model.py outside the container.
P = Path(os.environ.get(
    "WEIGHTLESS_STEERING_MODEL_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py",
))
MARK = "# [steering-hotfix] projective activation steering (dsv41 residual, karmic r6v)"

# ---------------------------------------------------------------------------
# Injected source: minimal GGUF reader + spec-conformant cvec loader.
# Kept as one constant so the hotfix's own --check mode executes the exact
# code that gets injected into model.py.
# ---------------------------------------------------------------------------
GGUF_SRC = r'''
def _read_gguf_cvec(path):
    """Minimal GGUF v3 reader for control vectors (F32 tensors only).

    The Anemll 0.1.1 image does not ship the gguf package, so the container
    is parsed directly. Only what the control-vector format needs: metadata
    scalars/strings and direction.<N> tensor payloads.
    """
    import struct

    import numpy as np

    _SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    _FM = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"GGUF":
        raise ValueError(f"{path}: bad magic, not a GGUF file")
    ver, n_tensors, n_kv = struct.unpack_from("<IQQ", data, 4)
    if ver != 3:
        raise ValueError(f"{path}: GGUF version {ver}, expected 3")
    off = 24

    def rd_string(o):
        (n,) = struct.unpack_from("<Q", data, o)
        o += 8
        return data[o:o + n].decode("utf-8"), o + n

    meta = {}
    for _ in range(n_kv):
        key, off = rd_string(off)
        (vtype,) = struct.unpack_from("<I", data, off)
        off += 4
        if vtype == 8:  # string
            val, off = rd_string(off)
        elif vtype == 9:  # array
            etype, cnt = struct.unpack_from("<IQ", data, off)
            off += 12
            if etype == 8:
                arr = []
                for _ in range(cnt):
                    s, off = rd_string(off)
                    arr.append(s)
                val = arr
            else:
                val = struct.unpack_from(f"<{cnt}{_FM[etype][1]}", data, off)
                off += _SZ[etype] * cnt
        else:
            (val,) = struct.unpack_from(_FM[vtype], data, off)
            off += _SZ[vtype]
        meta[key] = val

    infos = []
    for _ in range(n_tensors):
        name, off = rd_string(off)
        (nd,) = struct.unpack_from("<I", data, off)
        off += 4
        dims = struct.unpack_from(f"<{nd}Q", data, off)
        off += 8 * nd
        (tt,) = struct.unpack_from("<I", data, off)
        off += 4
        (toff,) = struct.unpack_from("<Q", data, off)
        off += 8
        infos.append((name, dims, tt, toff))

    align = meta.get("general.alignment", 32)
    base = (off + align - 1) // align * align
    tensors = {}
    for name, dims, tt, toff in infos:
        if tt != 0:
            raise ValueError(f"{path}: {name} is not F32 (ggml type {tt})")
        n = 1
        for d in dims:
            n *= d
        tensors[name] = np.frombuffer(
            data, dtype="<f4", count=n, offset=base + toff
        ).copy()
    return meta, tensors


def _load_gguf_control_vector(path: str) -> dict:
    """Load a projective control vector from GGUF into {layer_id: tensor}.

    Container and tensor convention follow llama.cpp: tensors named
    "direction.<N>", fp32, 1-D, N >= 1, and **N is the layer index**.

    That last point is easy to get wrong, and we did.
    common_control_vector_load_one() stores direction.N at data offset
    (N-1)*n_embd, which reads like "N-1 is the layer". But
    llama_adapter_cvec::apply() then fills tensors[il] from offset
    (il-1)*n_embd, so the two -1s cancel: tensors[il] holds direction.il,
    applied at graph layer il. Confirmed by measurement in
    tests/test-cvec-layer-map.cpp in github.com/msuiche/llama.cpp -- a
    direction placed in one slot moves logits only when the layer range is
    restricted to the matching index.

    An off-by-one here degrades quietly instead of failing: adjacent-layer
    refusal directions have cosine 0.83-0.91, so a stack shifted by one
    layer still produces plausible output. Layer 0 cannot be expressed.

    `glp.mode` is enforced, not advisory. llama.cpp ADDS a control
    vector; we PROJECT one out. The same file under the wrong operation
    produces no error, just wrong output -- an additive apply pushes every
    token along the refusal axis instead of removing the component. So an
    unrecognised mode is a hard failure rather than a fallback.
    """
    import numpy as np

    meta, tensors = _read_gguf_cvec(path)

    mode = meta.get("glp.mode")
    if mode is None:
        raise ValueError(
            f"{path}: no glp.mode. Refusing to guess: an additive "
            f"control vector and a projective one are different operations."
        )
    if mode != "project":
        raise ValueError(
            f"{path}: glp.mode={mode!r}, but this runtime only "
            f"implements projective ablation (h -= alpha*(h.d)d). "
            f"Refusing to apply."
        )

    hook = meta.get("glp.hook_point")
    if hook is not None and hook != "residual_stream_post_layer":
        raise ValueError(
            f"{path}: glp.hook_point={hook!r} does not match this hook "
            f"(residual_stream_post_layer — realised at the pending FFN "
            f"write on this tree; see the site note in the hotfix "
            f"docstring). Refusing to apply at the wrong site."
        )

    # The file's own dose: glp.alpha_default applies unless the env overrides
    # it. Non-finite is fatal, like every other malformed contract key.
    alpha_meta = meta.get("glp.alpha_default")
    if alpha_meta is not None:
        import math as _math

        if not _math.isfinite(float(alpha_meta)):
            raise ValueError(
                f"{path}: glp.alpha_default must be finite, not {alpha_meta!r}"
            )
        global _GLP_FILE_ALPHA
        _GLP_FILE_ALPHA = float(alpha_meta)

    # A transferred vector (captured at one site, calibrated for another) is
    # legal but must not be silent: derived_at != hook_point means the
    # direction was estimated on a different distribution than the one it is
    # about to edit. Warning, never a refusal (docs/spec/GLP.md); alpha_default belongs
    # to the apply site.
    derived_at = meta.get("glp.derived_at")
    if derived_at and hook and derived_at != hook:
        logger.warning(
            "%s: transferred vector -- glp.derived_at=%r, apply hook %r",
            path, derived_at, hook,
        )

    logger.info(
        "weightless GLP vector: mode=%s spec_version=%s base_model=%s rev=%s",
        mode,
        meta.get("glp.spec_version", "?"),
        meta.get("general.base_model.0.name") or meta.get("glp.base_model"),
        str(meta.get("general.base_model.0.version")
            or meta.get("glp.base_revision") or "?")[:12],
    )

    out = {}
    for name, arr in tensors.items():
        dot = name.find(".")
        if dot < 0 or name[:dot] != "direction":
            continue
        try:
            idx = int(name[dot + 1:])
        except ValueError:
            raise ValueError(
                f"{path}: malformed tensor name {name!r} — 'direction.' "
                f"must be followed by an integer layer id"
            ) from None
        if idx < 1:
            raise ValueError(
                f"{path}: {name} is invalid; direction.0 is rejected "
                f"upstream and layer 0 cannot be expressed in this "
                f"container"
            )
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        out[idx] = torch.from_numpy(arr.copy())  # N is the layer, no offset
    if not out:
        raise ValueError(f"{path}: no direction.<N> tensors found")
    widths = {v.numel() for v in out.values()}
    if len(widths) != 1:
        raise ValueError(
            f"{path}: inconsistent n_embd across directions: {widths}"
        )

    # Cross-check the tensor names against the informational layer list.
    # These are written from the same source, so a disagreement means the
    # file was produced by a broken exporter -- which is exactly what
    # happened here once: an exporter wrote direction.11..39 for directions
    # derived at layers 10..38 while this field still read 10..38. The
    # names are what execute, so the mismatch shifted the whole stack one
    # layer, and a one-layer shift degrades rather than fails (adjacent-
    # layer cosine 0.83-0.91). The file carried the evidence and nothing
    # looked at it. Now something does.
    declared = meta.get("glp.layer_ids_zero_based")
    if declared:
        try:
            want = sorted(int(x) for x in declared.split(",") if x.strip())
        except ValueError:
            raise ValueError(
                f"{path}: glp.layer_ids_zero_based is not a comma list of "
                f"integers: {declared!r}"
            ) from None
        if want and want != sorted(out):
            raise ValueError(
                f"{path}: glp.layer_ids_zero_based declares layers "
                f"{want[0]}..{want[-1]} ({len(want)} entries) but the "
                f"direction tensors resolve to {sorted(out)[0]}.."
                f"{sorted(out)[-1]} ({len(out)} entries). The tensor names "
                f"are what get applied, so this file would steer the wrong "
                f"layers. Re-export it."
            )
    logger.info(
        "weightless GLP vector: %d directions, n_embd=%d, layers %s",
        len(out), next(iter(widths)), sorted(out),
    )
    return out
'''

# Module-level block injected after the ubatching import.
MODULE_BLOCK = (
    "\n"
    "from vllm.logger import init_logger\n"
    "\n"
    "logger = init_logger(__name__)\n"
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Projective activation steering (DSpark). [steering-hotfix]\n"
    "#\n"
    "# h <- h - alpha * (h . d_hat) d_hat on the residual stream at chosen layers.\n"
    "# Activation space rather than weight space on purpose: on this checkpoint the\n"
    "# fp8 requantisation floor is ~39% of a lambda=3.5 weight edit, so small\n"
    "# coefficients are not representable in the weights at all.\n"
    "#\n"
    "# Everything here is inert unless WEIGHTLESS_STEER_PATH is set.\n"
    "# ---------------------------------------------------------------------------\n"
    + MARK
    + "\n"
    "\n"
    "# Recorded for provenance in the startup log. The only site implemented at\n"
    "# this anchor is \"residual_stream_post_layer\": on this tree the model loop's\n"
    "# hidden_states IS the layer's pending FFN write (the mHC fold is deferred to\n"
    "# the next layer's fused post/pre call). Every published vLLM-lane DSV4\n"
    "# number was measured here. Measured 2026-09-04 (hook-site experiment, same\n"
    "# direction/layers): FFN writer >> post-layer residual >> attention writer;\n"
    "# FFN window alpha 4-6, residual garbles at 4.0, attention ~4-5/32 at 4.0.\n"
    "# WEIGHTLESS_STEER_HOOK set to anything else fails closed in _load_steering.\n"
    "_WEIGHTLESS_STEER_HOOK = (os.environ.get(\"WEIGHTLESS_STEER_HOOK\") or \"residual_stream_post_layer\").strip()\n"
    "_GLP_HOOK_ALPHA_ENV = (os.environ.get(\"WEIGHTLESS_STEER_ALPHA\") or \"\").strip()\n"
"# Set by the GGUF loader from glp.alpha_default; WEIGHTLESS_STEER_ALPHA wins when set.\n"
"_GLP_FILE_ALPHA = 1.0\n"
    "# layer id -> (k, hidden) orthonormal rows. Populated for inspection and for\n"
    "# offline tooling; NOT read by the forward path. The pre-fold writer-isolation\n"
    "# hooks that used it are deliberately not ported: they branch on\n"
    "# _WEIGHTLESS_STEER_HOOK inside the traced region, and that choice is not part of\n"
    "# vLLM's compile cache key, so a cached artifact would silently ignore a changed\n"
    "# hook site. Reintroduce them behind a separate image if that experiment is\n"
    "# re-run, and clear the compile cache when doing so.\n"
    "_GLP_HOOK_DIRS: dict[int, torch.Tensor] = {}\n"
    "\n"
    + GGUF_SRC
)

# __init__ tail block + _load_steering method, injected between the
# _mtp_hidden_buffer else-branch and embed_input_ids.
INIT_BLOCK = '''\

        # ---- projective activation steering ---------------------------------
        self._steer_alpha_val = (
            float(_GLP_HOOK_ALPHA_ENV) if _GLP_HOOK_ALPHA_ENV else None
        )
        self._steer_dirs: dict[int, torch.Tensor] = {}
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        _dtype = vllm_config.model_config.dtype
        # Allocated unconditionally, zeros when steering is off. A
        # None-when-disabled branch changes the traced graph, and that
        # difference is not part of vLLM's compile cache key: a compiled
        # artifact from a 29-layer run was reused by a 16-layer run and died
        # with KeyError: 23. Clearing torch_compile_cache was not enough,
        # because an AOT artifact kept it. A dense stack indexed by layer id
        # keeps the graph identical for every layer set, so only values change.
        self.register_buffer(
            "_steer_stack",
            torch.zeros(
                config.num_hidden_layers, 1, config.hidden_size,
                device=_dev, dtype=_dtype,
            ),
            persistent=False,
        )
        # alpha is a TENSOR, not a Python float. torch.compile bakes Python
        # scalars into the traced graph as constants and vLLM caches compiled
        # graphs under VLLM_CACHE_ROOT, so a float alpha is frozen at first
        # compile and every later boot silently reuses it. Measured: alpha of
        # 1.0, 2.0 and 4.0 produced byte-identical refusal rates. A tensor is
        # read at runtime, so alpha takes effect without recompiling.
        self.register_buffer(
            "_steer_alpha",
            torch.zeros((), device=_dev, dtype=_dtype),
            persistent=False,
        )
        # Buffers are non-persistent so they never enter the state dict, which
        # would make load_weights report them as unexpected keys.
        self._load_steering(config, _dev, _dtype)

    def _load_steering(self, config, device, dtype) -> None:
        """Fill _steer_stack from WEIGHTLESS_STEER_PATH. No-op when unset.

        Loaded on every rank and indexed by GLOBAL layer id, so this is correct
        under pipeline parallelism: each rank's forward loop only visits its own
        layers and looks them up by the same global index.
        """
        path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
        if not path:
            return
        if _WEIGHTLESS_STEER_HOOK != "residual_stream_post_layer":
            raise RuntimeError(
                f"WEIGHTLESS_STEER_HOOK={_WEIGHTLESS_STEER_HOOK!r} is not "
                f"implemented at this anchor; the only site here is "
                f"residual_stream_post_layer (realised at the pending FFN "
                f"write on this tree). Refusing to serve with a silently "
                f"wrong hook site."
            )
        try:
            if path.endswith(".gguf"):
                raw = _load_gguf_control_vector(path)
            else:
                raw = torch.load(path, map_location="cpu")
            want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
            selected = (
                {int(t) for t in want.replace(" ", "").split(",") if t} if want else None
            )
            fallback = raw.get("global") if isinstance(raw, dict) else None

            for layer_id in range(config.num_hidden_layers):
                if selected is not None and layer_id not in selected:
                    continue
                vec = None
                if isinstance(raw, dict):
                    vec = raw.get(layer_id, raw.get(str(layer_id), fallback))
                if vec is None:
                    continue
                vec = vec.detach().to(torch.float32)
                if vec.ndim == 1:
                    vec = vec.unsqueeze(0)
                vec = vec.reshape(-1, vec.shape[-1])
                # Width guard (from the capture-validated reference lane): a
                # direction that does not match this arch's stream width must
                # fail here, not as an opaque broadcast error at serve time.
                if vec.shape[-1] != config.hidden_size:
                    raise RuntimeError(
                        f"steering vector layer {layer_id} width "
                        f"{vec.shape[-1]} != {config.hidden_size} (hidden_size)"
                    )
                # Rank-k: orthonormalise the basis, otherwise overlapping
                # components get subtracted more than once. Refusal measured
                # only 0.147 rank-1 energy here, so k>1 is expected to matter.
                q, _ = torch.linalg.qr(vec.T)
                vec = q.T[: vec.shape[0]]
                # Both of these are per layer and must stay inside this loop.
                # A previous revision dedented them out, which kept only the
                # final iteration and silently steered one layer instead of 29.
                self._steer_dirs[layer_id] = vec.to(device=device, dtype=dtype)
                _GLP_HOOK_DIRS[layer_id] = vec.to(
                    device=device, dtype=torch.bfloat16
                )

            if isinstance(raw, dict):
                out_of_range = sorted(
                    int(k) for k in raw
                    if str(k).lstrip("-").isdigit()
                    and not 0 <= int(k) < config.num_hidden_layers
                )
                if out_of_range:
                    raise RuntimeError(
                        f"{path}: direction layers {out_of_range} out of range "
                        f"for this model ({config.num_hidden_layers} layers)"
                    )
            if not self._steer_dirs:
                raise RuntimeError(
                    f"WEIGHTLESS_STEER_PATH={path} matched no layers; "
                    f"refusing to run unsteered"
                )

            k = max(v.shape[0] for v in self._steer_dirs.values())
            stack = torch.zeros(
                config.num_hidden_layers, k, config.hidden_size,
                device=device, dtype=dtype,
            )
            for layer_id, vec in self._steer_dirs.items():
                stack[layer_id, : vec.shape[0]] = vec
            self._steer_stack = stack
            if self._steer_alpha_val is None:  # no env override: the file's dose
                self._steer_alpha_val = _GLP_FILE_ALPHA
            self._steer_alpha = torch.tensor(
                self._steer_alpha_val, device=device, dtype=dtype
            )
            logger.info(
                "weightless GLP steering active: hook=%s alpha=%.3f rank=%s "
                "layers=%d %s",
                _WEIGHTLESS_STEER_HOOK,
                self._steer_alpha_val,
                {k_: int(v.shape[0]) for k_, v in sorted(self._steer_dirs.items())},
                len(self._steer_dirs),
                sorted(self._steer_dirs),
            )
        except Exception as exc:
            # Fail closed: a boot asked for steering must not serve unsteered.
            logger.error("DSpark steering load failed (%s); failing closed", exc)
            raise
'''

# Per-layer apply in the forward loop.
FORWARD_BLOCK = '''\
            # Per-layer projection: h <- h - alpha (h.d) d. The apply is
            # injected only when the build configures steering (see the
            # hotfix docstring): an unconfigured image carries none of this,
            # so off is byte-identical to stock — no extra kernels, no
            # extra launches, same IR.
            # hidden_states at this anchor is the layer's PENDING FFN WRITE,
            # shape (tokens, hidden_size) — the mHC fold of that write into the
            # hyper-connection streams is deferred to the next layer's fused
            # post/pre call (0731 tree: verified 2026-09-04 against the
            # 0.25.2 kernel signatures; this karmic-kraken r6v tree: same
            # return convention, re-verified 2026-10-18 through the 172-patch
            # promoted series).
            steer_dirs = self._steer_stack[idx]
            # r6x 0183/SP: the pending FFN write may be a RankOrderParts (this
            # rank's partial of the sum, folded by the next block's post_pre).
            # Steering one rank's partial would edit only that rank's share of
            # every row's bits -- silently wrong. Fold to the summed write
            # (the same bits post_pre consumes), then project.
            if hasattr(hidden_states, "packed"):
                # RankOrderParts: this rank's partials of the shared sum.
                # sum() yields THIS RANK'S ROWS of the summed write -- exactly
                # the rows the next block consumes on this rank. Steer those
                # rows and pass the plain tensor: the wrapper is not needed
                # downstream, and re-wrapping would corrupt shapes.
                hidden_states = hidden_states.sum()
            steer_coef = torch.einsum("...h,kh->...k", hidden_states, steer_dirs)
            hidden_states = hidden_states - self._steer_alpha * torch.einsum(
                "...k,kh->...h", steer_coef, steer_dirs
            )
'''

ANCHOR_IMPORT = (
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "import copy\n"
    "import typing"
)
REPLACEMENT_IMPORT = (
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "import copy\n"
    "import os\n"
    "import typing"
)

ANCHOR_MODULE = (
    "from vllm.v1.worker.ubatching import dbo_current_ubatch_id\n"
)
REPLACEMENT_MODULE = ANCHOR_MODULE + MODULE_BLOCK

ANCHOR_INIT = (
    "            spec_config is not None and spec_config.use_dspark()\n"
    "        )\n"
    "\n"
    "    def prepare_disk_engram(\n"
)
REPLACEMENT_INIT = (
    "            spec_config is not None and spec_config.use_dspark()\n"
    "        )\n"
    + INIT_BLOCK
    + "\n"
    "    def prepare_disk_engram(\n"
)

ANCHOR_FORWARD = (
    "                sp_rows=sp,\n"
    "            )\n"
    "            if boundary_indices is not None:"
)
REPLACEMENT_FORWARD = (
    "                sp_rows=sp,\n"
    "            )\n"
    + FORWARD_BLOCK
    + "            if boundary_indices is not None:"
)

# Off is stock: when no GLP vector is configured, NOTHING is injected —
# not the loader, not the buffers, not the apply. The patched model.py is
# byte-identical to the unpatched one, so an image built without steering
# carries zero overhead in the traced graph and zero extra launches per step.
# Steering is therefore a build-time configuration, not a runtime toggle:
# re-run the hotfix (or bin/spark build prepare with WEIGHTLESS_STEER_PATH
# exported) to produce the steered image.
def steer_requested() -> bool:
    return bool(os.environ.get("WEIGHTLESS_STEER_PATH", "").strip())


PATCHES = (
    ("import os", ANCHOR_IMPORT, REPLACEMENT_IMPORT),
    ("module steering block", ANCHOR_MODULE, REPLACEMENT_MODULE),
    ("__init__/_load_steering", ANCHOR_INIT, REPLACEMENT_INIT),
    ("forward apply", ANCHOR_FORWARD, REPLACEMENT_FORWARD),
) if steer_requested() else ()


def check_vector() -> int:
    """Validate the vector named by WEIGHTLESS_STEER_PATH. 0 ok, 1 bad."""
    path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
    if not path:
        print("[steering-hotfix] --check: WEIGHTLESS_STEER_PATH unset; nothing to check")
        return 0
    if not os.path.isfile(path):
        print(f"[steering-hotfix] --check: {path} not found", file=sys.stderr)
        return 1
    try:
        if path.endswith(".gguf"):
            import torch

            class _StderrLogger:  # stand-in for vllm's logger in the loader
                @staticmethod
                def info(msg, *args):
                    print("[steering-hotfix] --check: " + (msg % args))

                warning = info
                error = info

            # Execute the exact code that gets injected into model.py, with
            # its two free names bound, and run the full spec validation
            # (mode, hook point, layer cross-check), not a partial re-do.
            ns: dict = {"torch": torch, "logger": _StderrLogger}
            exec(GGUF_SRC, ns)  # noqa: S102 - same code that is injected
            out = ns["_load_gguf_control_vector"](path)
            layers = sorted(out)
            print(
                f"[steering-hotfix] --check: {os.path.basename(path)} OK: "
                f"mode=project layers {layers[0]}..{layers[-1]} ({len(layers)})"
            )
        else:
            import torch

            raw = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(raw, dict) or not raw:
                raise ValueError("expected non-empty {layer_id: tensor} dict")
            layers = sorted(int(k) for k in raw if str(k).isdigit())
            print(
                f"[steering-hotfix] --check: {os.path.basename(path)} OK: "
                f"pt layers {layers[0]}..{layers[-1]} ({len(layers)})"
            )
        want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
        if want:
            [int(t) for t in want.replace(" ", "").split(",") if t]
        float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
    except Exception as exc:
        print(f"[steering-hotfix] --check: {path}: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        status_src = P.read_text() if P.is_file() else ""
        print(
            "steering (projective cvec)          :",
            "APPLIED" if MARK in status_src else "NOT APPLIED",
            "| WEIGHTLESS_STEER_PATH",
            "set" if steer_requested() else "unset",
        )
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return check_vector()

    src = P.read_text()
    if MARK in src:
        print(f"[steering-hotfix] already applied to {P}")
        return check_vector() if steer_requested() else 0

    missing = [name for name, old, _ in PATCHES if old not in src]
    if missing:
        msg = f"[steering-hotfix] anchors not found: {missing}; refusing to patch"
        if steer_requested():
            print(msg + " (WEIGHTLESS_STEER_PATH is set; failing closed)", file=sys.stderr)
            return 1
        print(msg + " (steering off; leaving model.py stock)")
        return 0

    for name, old, new in PATCHES:
        assert src.count(old) == 1, f"anchor {name!r} not unique"
        src = src.replace(old, new, 1)
    P.write_text(src)
    print(f"[steering-hotfix] applied to {P} ({len(PATCHES)} anchors)")
    return check_vector() if steer_requested() else 0


if __name__ == "__main__":
    raise SystemExit(main())
