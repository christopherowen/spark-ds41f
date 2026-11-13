#!/usr/bin/env python3
"""Fail-closed fold-site steering patch (GLP spec_version 3,
hc_fold_pre_engram) for the spark3 candidate vllm tree
(/opt/spark3/candidate/vllm), image vllm-ds41f-kkref:19f2c20ed4d6-1022-4.

Ported from refusal-research/experiments/20261024-dsv41-1017-weaponize-glp/
staging/patch_dsv41.py, which targets the dsv41-feat PR #56201 reference.
This tree (1022) differs where it matters:

- The mHC fold is FUSED into `mhc.pre`/`mhc.post_pre` and only materializes
  on Engram layers (`residual = self.mhc.post(x, residual, post_mix,
  res_mix)`); the reference called a free `mhc_post_tilelang(...)` at every
  layer. Here we materialize the fold at the top of every layer (residual
  not None), steer, then continue with previous_output=None — the exact
  pattern this tree already uses on Engram layers ("do not fuse across it").
  Steering is post-fold, pre-Engram = the spec's hc_fold_pre_engram site.
  (The old residual-stream hotfix steered the loop-level hidden_states =
  the pending FFN write pre-fold; that is the wrong site for this vector.)
- The attention `_o_proj` call site passes keywords
  (`self._o_proj(output, positions, is_prefill=..., sp_rows=...)`), so the
  companion-writer wrap must forward *args/**kwargs.

Steering op at both fold sites (in-layer start-fold and post-loop fold):
    h = folded.mean(dim=1)
    folded <- folded - alpha * (h . d) d   (broadcast to all 4 copies)
Layer i's start-fold consumes layer i-1's FFN write -> direction[i-1]
(stack row 0 stays zero: direction.0 is invalid per the GLP spec); the
post-loop fold applies direction[end_layer-1]. The companion attention
writer (o_proj output) is armed with the same stack and alpha at load —
mandatory under spec_version 3 (fold alone under-doses).

Files patched, all fail-closed:
  1. vllm/models/deepseek_v4_1/nvidia/model.py  (loader, fold sites, ctrl)
  2. vllm/models/deepseek_v4_1/attention.py     (companion-writer globals)
  3. vllm/v1/worker/gpu/model_runner.py         (V2 runner: ctrl poll)
  4. vllm/v1/worker/gpu_model_runner.py         (V1 runner: ctrl poll, best
     effort — warn, not fail, if the anchor is absent; V2 is the live path)

Env vars: WEIGHTLESS_STEER_PATH (.pt {layer: tensor(5120)}),
  WEIGHTLESS_STEER_ALPHA (default 1.0), WEIGHTLESS_STEER_LAYERS,
  DSPARK_STEER_RAW_NORMS, DSPARK_STEER_FOLD_OFF (debug),
  DSPARK_STEER_CTRL (JSON {"alpha", "steer_path", "steer_layers",
  "off"}, mtime-polled per step), DSPARK_STEER_ACK (ack JSON, rank 0
  writes).

Off is stock (the lesson from the r6v hook experiment): until a vector
arms steering, the fold sites run the tree's own fused path —
byte-identical to the unpatched tree, zero extra kernels, no CUDA-graph
delta. The first _ds41_load_steer (boot env or ctrl steer_path) flips
_ds41_steer_on; ctrl {"off": true} restores the fused path.
Exit codes: 0 patched (or already patched), 1 anchor mismatch.
"""

import os as _os
import py_compile
import sys

VLLM_ROOT = _os.environ.get(
    "DSPARK_VLLM_ROOT", "/opt/spark3/candidate/vllm/vllm")
MODEL_FILE = _os.environ.get(
    "DSPARK_V41_MODEL_PY", f"{VLLM_ROOT}/models/deepseek_v4_1/nvidia/model.py")
ATTN_FILE = _os.environ.get(
    "DSPARK_V41_ATTN_PY", f"{VLLM_ROOT}/models/deepseek_v4_1/attention.py")
RUNNER2_FILE = _os.environ.get(
    "DSPARK_V41_RUNNER2_PY", f"{VLLM_ROOT}/v1/worker/gpu/model_runner.py")
RUNNER1_FILE = _os.environ.get(
    "DSPARK_V41_RUNNER_PY", f"{VLLM_ROOT}/v1/worker/gpu_model_runner.py")
MARKER = "dspark dsv41 fold steering"


def _replace_once(src: str, anchor: str, replacement: str, name: str,
                  path: str) -> str:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(
            f"PATCH FAILED: anchor {name!r} found {n} times (expected 1) in "
            f"{path}. The tree has drifted from the pinned reference "
            "(1022-4); refusing to guess. Diff and update this patch.")
    return src.replace(anchor, replacement)


def patch_model(path: str) -> None:
    src = open(path).read()
    if MARKER in src:
        print(f"already patched: {path}")
        return

    # 1. imports -------------------------------------------------------------
    src = _replace_once(
        src,
        "import typing\nfrom collections.abc import Callable, Iterable",
        "import os\nimport typing\nfrom collections.abc import Callable, "
        "Iterable",
        "imports", path)

    # 2. module-level registry + steer loader + ctrl applier -----------------
    src = _replace_once(
        src,
        'logger = init_logger(__name__)',
        'logger = init_logger(__name__)\n'
        '\n'
        '\n'
        '# ---- dspark dsv41 fold steering (GLP v3, hc_fold_pre_engram) ---\n'
        '# Live DeepseekV4Model instances for the runner-side ctrl poll.\n'
        'DSPARK_V41_REGISTRY: list = []\n'
        '# zero-alpha scalar for untagged rebuilt layers (inert fallback)\n'
        '_ds41_zero_alpha = 0.0\n'
        '\n'
        '\n'
        'def _ds41_tag_layers(model) -> None:\n'
        '    """Idempotent: point every DeepseekV4DecoderLayer under this\n'
        '    model at the shared steer state. dspark.py builds its OWN layer\n'
        '    ModuleList for the draft stack, and loaders may rebuild layers\n'
        '    after __init__; this walks model.modules() so every instance\n'
        '    gets the attributes. Fold rows: a DecoderLayer at position i\n'
        '    applies direction[i-1] (its start fold consumes layer i-1\'s\n'
        '    FFN write)."""\n'
        '    _n_layers = model._steer_stack.shape[0]\n'
        '    _lays = [\n'
        '        _m for _m in model.modules()\n'
        '        if type(_m).__name__ == "DeepseekV4DecoderLayer"]\n'
        '    for _i, _l in enumerate(_lays):\n'
        '        _l._ds41_steer_dir = (\n'
        '            model._steer_stack[_i - 1]\n'
        '            if _i >= 1 else model._steer_zero)\n'
        '        _l._ds41_steer_alpha = model._steer_alpha\n'
        '        _l._ds41_fold_off = model._ds41_fold_off\n'
        '        _l._ds41_steer_on = model._ds41_steer_on\n'
        '    return len(_lays)\n'
        '\n'
        '\n'
        'def _ds41_unit(v):\n'
        '    return v / (v.norm() + 1e-9)\n'
        '\n'
        '\n'
        'def _ds41_load_steer(model, path: str, layers_spec: str = "") -> int:\n'
        '    """(Re)fill model._steer_stack from a .pt {layer: tensor} file\n'
        '    and arm the companion attention writer with the same stack.\n'
        '    Returns the number of layers armed. Fail-closed at boot; ctrl\n'
        '    reload errors are caught by _ds41_apply_ctrl."""\n'
        '    try:\n'
        '        _raw = torch.load(path, map_location="cpu",\n'
        '                          weights_only=True)\n'
        '    except Exception:\n'
        '        _raw = torch.load(path, map_location="cpu",\n'
        '                          weights_only=False)\n'
        '    if isinstance(_raw, dict) and isinstance(\n'
        '            _raw.get("per_layer"), dict):\n'
        '        _raw = _raw["per_layer"]\n'
        '    if not isinstance(_raw, dict):\n'
        '        raise RuntimeError(\n'
        '            f"{path}: expected {{layer: tensor}} or "\n'
        '            "{\'per_layer\': {...}}")\n'
        '    _sel = None\n'
        '    if layers_spec.strip():\n'
        '        _sel = {int(t) for t in\n'
        '                layers_spec.replace(" ", "").split(",") if t}\n'
        '    _n_layers = model._steer_stack.shape[0]\n'
        '    _hidden = model._steer_stack.shape[1]\n'
        '    _dirs = {}\n'
        '    for _k, _v in _raw.items():\n'
        '        _lid = int(_k)\n'
        '        if not (0 <= _lid < _n_layers):\n'
        '            continue\n'
        '        if _sel is not None and _lid not in _sel:\n'
        '            continue\n'
        '        _v = _v.detach().to(torch.float32).reshape(-1)\n'
        '        if _v.numel() != _hidden:\n'
        '            raise RuntimeError(\n'
        '                f"{path}: steering vector layer {_lid} width "\n'
        '                f"{_v.numel()} != {_hidden} (hidden_size)")\n'
        '        # DSPARK_STEER_RAW_NORMS: honor stored row norms (per-layer\n'
        '        # dosing). Default keeps the historical unit-norm behaviour.\n'
        '        _dirs[_lid] = (_v if os.environ.get("DSPARK_STEER_RAW_NORMS")\n'
        '                       else _ds41_unit(_v))\n'
        '    if not _dirs:\n'
        '        raise RuntimeError(\n'
        '            f"{path} matched no layers; refusing to run unsteered")\n'
        '    model._steer_stack.zero_()\n'
        '    for _lid, _v in _dirs.items():\n'
        '        model._steer_stack[_lid] = _v.to(\n'
        '            model._steer_stack.dtype)\n'
        '    # companion attention writer (GLP v3: mandatory — fold alone\n'
        '    # under-doses). Mirror the stack into the attention module\n'
        '    # globals, then wrap each layer\'s attn._o_proj. The candidate\n'
        '    # tree\'s call site passes keywords, so the wrap forwards\n'
        '    # *args/**kwargs. Zero row / alpha 0 = numeric no-op.\n'
        '    try:\n'
        '        import vllm.models.deepseek_v4_1.attention as _am\n'
        '        _am._ds41_attn_dir = (\n'
        '            _am._ds41_attn_dir.to(\n'
        '                next(model.parameters()).device))\n'
        '        _am._ds41_attn_dir.zero_()\n'
        '        for _lid, _v in _dirs.items():\n'
        '            _am._ds41_attn_dir[_lid] = _v.to(\n'
        '                _am._ds41_attn_dir.device,\n'
        '                _am._ds41_attn_dir.dtype)\n'
        '        _am._ds41_attn_alpha.fill_(float(model._steer_alpha_val))\n'
        '    except Exception:\n'
        '        _am = None\n'
'    _lays = [\n'
'        _m for _m in model.modules()\n'
'        if type(_m).__name__ == "DeepseekV4DecoderLayer"]\n'
'    for _li, _m in enumerate(_lays, start=1):\n'
'        _m._ds41_lid = _li\n'
'        _at = getattr(_m, "attn", None)\n'
'        if (_at is None or _am is None\n'
'                or getattr(_at, "_ds41_wrapped", False)):\n'
'            continue\n'
'        _orig = _at._o_proj\n'
'        _dsd = (_am._ds41_attn_dir[_li] if 1 <= _li < 40\n'
'                else _am._ds41_attn_dir[0])\n'
'        _dsskip = (not (1 <= _li < 40)\n'
'                or float(_dsd.norm()) <= 0.0)\n'
'        def _wrap(_o, _pos, *args, _self=_m, _f=_orig, _d=_dsd,\n'
'                _skip=_dsskip, **kwargs):\n'
'            _out = _f(_o, _pos, *args, **kwargs)\n'
'            if _skip:\n'
'                return _out\n'
'            _h = _out.float()\n'
'            return (_h - (\n'
'                (_am._ds41_attn_alpha\n'
'                 * (_h @ _d)).unsqueeze(-1)\n'
'                * _d)).to(_out.dtype)\n'
'        _at._o_proj = _wrap\n'
'        _at._ds41_wrapped = True\n'
'    _ds41_tag_layers(model)\n'
'    return len(_dirs)\n'
        '\n'
        '\n'
        'def _ds41_apply_ctrl(owner) -> None:\n'
        '    """Poll the steer-ctrl JSON (mtime-checked) and apply changes:\n'
        '    alpha and/or a new steer stack. Runs OUTSIDE the CUDA graph in\n'
        '    the patched runner. Writes the ack file on every change; a bad\n'
        '    ctrl keeps the previous steering state and sets ok=false."""\n'
        '    import json\n'
        '    ctrl = os.environ.get("DSPARK_STEER_CTRL", "").strip()\n'
        '    if not ctrl:\n'
        '        return\n'
        '    try:\n'
        '        mt = os.path.getmtime(ctrl)\n'
        '    except OSError:\n'
        '        return\n'
        '    if mt == owner._ds41_ctrl_mtime:\n'
        '        return\n'
        '    owner._ds41_ctrl_mtime = mt\n'
        '    ack = {"mtime": mt, "alpha": None, "steer_path": None,\n'
        '           "ok": True, "error": ""}\n'
        '    try:\n'
        '        spec = json.load(open(ctrl))\n'
        '        if spec.get("steer_path"):\n'
        '            owner._ds41_steer_on = True\n'
        '            _ds41_load_steer(owner, spec["steer_path"],\n'
        '                             spec.get("steer_layers") or "")\n'
        '        if spec.get("off"):\n'
        '            owner._ds41_steer_on = False\n'
        '            _ds41_tag_layers(owner)\n'
        '            owner._steer_alpha.fill_(0.0)\n'
        '            try:\n'
        '                import vllm.models.deepseek_v4_1.attention as _am0\n'
        '                _am0._ds41_attn_alpha.fill_(0.0)\n'
        '            except Exception:\n'
        '                pass\n'
        '        if spec.get("alpha") is not None:\n'
        '            try:\n'
        '                import vllm.models.deepseek_v4_1.attention as _am\n'
        '                _am._ds41_attn_alpha.fill_(float(spec["alpha"]))\n'
        '            except Exception:\n'
        '                pass\n'
        '        ack["alpha"] = float(owner._steer_alpha)\n'
        '    except Exception as exc:\n'
        '        ack["ok"] = False\n'
        '        ack["error"] = str(exc)[:400]\n'
        '        try:\n'
        '            ack["alpha"] = float(owner._steer_alpha)\n'
        '        except Exception:\n'
        '            pass\n'
        '    ack_path = os.environ.get("DSPARK_STEER_ACK", "").strip()\n'
        '    if ack_path and get_tensor_model_parallel_rank() == 0:\n'
        '        try:\n'
        '            with open(ack_path, "w") as _f:\n'
        '                json.dump(ack, _f)\n'
        '        except Exception:\n'
        '            pass\n'
        '    logger.info("dspark dsv41 steer-ctrl applied: %s", ack)\n',
        "registry", path)

    # 3. steering state at the end of DeepseekV4Model.__init__ ----------------
    src = _replace_once(
        src,
        "        if get_pp_group().is_last_rank and needs_mtp_hidden_states:\n"
        "            self._mtp_hidden_buffer = torch.empty(\n"
        "                vllm_config.scheduler_config.max_num_batched_tokens,\n"
        "                self.hc_dim,\n"
        "                dtype=vllm_config.model_config.dtype,\n"
        "            )\n"
        "        else:\n"
        "            self._mtp_hidden_buffer = None\n",
        "        if get_pp_group().is_last_rank and needs_mtp_hidden_states:\n"
        "            self._mtp_hidden_buffer = torch.empty(\n"
        "                vllm_config.scheduler_config.max_num_batched_tokens,\n"
        "                self.hc_dim,\n"
        "                dtype=vllm_config.model_config.dtype,\n"
        "            )\n"
        "        else:\n"
        "            self._mtp_hidden_buffer = None\n"
        "\n"
        "        # ---- dspark dsv41 fold steering (GLP v3) ----------------\n"
        "        # h <- h - alpha * (h . d) d on the post-fold hc stream,\n"
        "        # pre-Engram. Inert unless WEIGHTLESS_STEER_PATH is set or a\n"
        "        # steer-ctrl file arms it. Dense zero-padded stack + alpha as\n"
        "        # tensor buffers: compile/graph-safe (pure tensor ops in\n"
        "        # forward; the ctrl poll mutates buffers OUTSIDE the graph).\n"
        "        self._steer_alpha_val = float(\n"
        "            os.environ.get(\"WEIGHTLESS_STEER_ALPHA\", \"1.0\") or 1.0)\n"
        "        _ds_dtype = vllm_config.model_config.dtype\n"
        "        self.register_buffer(\n"
        "            \"_steer_stack\",\n"
        "            torch.zeros(config.num_hidden_layers, config.hidden_size,\n"
        "                        dtype=_ds_dtype),\n"
        "            persistent=False,\n"
        "        )\n"
        "        self.register_buffer(\n"
        "            \"_steer_alpha\",\n"
        "            torch.zeros((), dtype=_ds_dtype),\n"
        "            persistent=False,\n"
        "        )\n"
        "        self.register_buffer(\n"
        "            \"_steer_zero\",\n"
        "            torch.zeros(config.hidden_size, dtype=_ds_dtype),\n"
        "            persistent=False,\n"
        "        )\n"
        "        self._ds41_ctrl_mtime = None\n"
        "        # off is stock: until a vector loads, the fold sites run the\n"
        "        # tree's own fused path — byte-identical to the unpatched\n"
        "        # tree, zero extra kernels, no graph delta (lane discipline).\n"
        "        self._ds41_steer_on = False\n"
        "        self._ds41_fold_off = bool(\n"
        "            os.environ.get(\"DSPARK_STEER_FOLD_OFF\", \"\"))\n"
        "        # Per-layer direction rows via the shared idempotent tagger\n"
        "        # (dspark.py builds its OWN DecoderLayer stack; the loader\n"
        "        # also rebuilds layers, and both re-tag on every load).\n"
        "        _ds41_tag_layers(self)\n"
        "        DSPARK_V41_REGISTRY.append(self)\n"
        "        _steer_path = os.environ.get(\"WEIGHTLESS_STEER_PATH\", \"\")."
        "strip()\n"
        "        if _steer_path:\n"
        "            self._ds41_steer_on = True\n"
        "            _n = _ds41_load_steer(\n"
        "                self, _steer_path,\n"
        "                os.environ.get(\"WEIGHTLESS_STEER_LAYERS\", \"\"))\n"
        "            self._steer_alpha.fill_(self._steer_alpha_val)\n"
        "            logger.info(\n"
        "                \"dspark fold-site steering active: alpha=%.3f \"\n"
        "                \"layers=%d width=%d\", self._steer_alpha_val, _n,\n"
        "                config.hidden_size)\n",
        "init", path)

    # 4. steering apply at the in-layer fold site -----------------------------
    src = _replace_once(
        src,
        "        if residual is None:\n"
        "            residual = x\n"
        "        elif apply_engram:\n"
        "            # Engram mutates the reconstructed residual; do not fuse across it.\n"
        "            residual = self.mhc.post(x, residual, post_mix, res_mix)\n"
        "            previous_output = None\n",
        "        if residual is None:\n"
        "            residual = x\n"
        "        elif not getattr(self, \"_ds41_steer_on\", False):\n"
        "            # (draft-stack layers are never tagged; stock path)\n"
        "            # steering off: the stock fused path, byte-identical to\n"
        "            # the unpatched tree (zero extra kernels, no graph delta)\n"
        "            if apply_engram:\n"
        "                # Engram mutates the reconstructed residual; do not fuse across it.\n"
        "                residual = self.mhc.post(x, residual, post_mix, res_mix)\n"
        "                previous_output = None\n"
        "        else:\n"
        "            # ---- dspark dsv41 fold steering: materialize the fold\n"
        "            # (it is fused into mhc.pre on non-Engram layers), steer\n"
        "            # the folded stream, pre-Engram — the GLP v3\n"
        "            # hc_fold_pre_engram site. h = mean over the hc copies;\n"
        "            # subtracting the rank-1 term from every copy shifts the\n"
        "            # mean by exactly alpha*(h.d)d. Same materialization this\n"
        "            # tree already uses on Engram layers ('do not fuse\n"
        "            # across it').\n"
        "            residual = self.mhc.post(x, residual, post_mix, res_mix)\n"
        "            previous_output = None\n"
        "            _ds41_h = residual.mean(dim=1)\n"
        "            _ds41_d = getattr(\n"
        "                self, \"_ds41_steer_dir\", None)\n"
        "            if _ds41_d is None:\n"
        "                # an untagged rebuilt layer (dspark draft stack that\n"
        "                # was constructed after the last tag pass): inert.\n"
        "                _ds41_d = torch.zeros(\n"
        "                    residual.shape[-1],\n"
        "                    dtype=residual.dtype, device=residual.device)\n"
        "            if not getattr(\n"
        "                self, \"_ds41_fold_off\", False):\n"
        "                residual = residual - (\n"
        "                    (getattr(\n"
        "                        self, \"_ds41_steer_alpha\",\n"
        "                        _ds41_zero_alpha) * (_ds41_h @ _ds41_d))\n"
        "                    .unsqueeze(-1) * _ds41_d).unsqueeze(1)\n",
        "steer-fold", path)

    # 5. steering apply at the post-loop fold (post-layer end_layer-1) --------
    src = _replace_once(
        src,
        "        if layer is not None:\n"
        "            # Reuse if the last layer was captured as an aux hidden state\n"
        "            if self.end_layer in self.aux_hidden_state_layers:\n"
        "                hidden_states = final_aux_recon\n"
        "            else:\n"
        "                hidden_states = layer.mhc.post(\n"
        "                    hidden_states, residual, post_mix, res_mix\n"
        "                )\n",
        "        if layer is not None:\n"
        "            # Reuse if the last layer was captured as an aux hidden state\n"
        "            if self.end_layer in self.aux_hidden_state_layers:\n"
        "                hidden_states = final_aux_recon\n"
        "            else:\n"
        "                hidden_states = layer.mhc.post(\n"
        "                    hidden_states, residual, post_mix, res_mix\n"
        "                )\n"
        "            if self._ds41_steer_on:\n"
        "                # ---- dspark dsv41 fold steering: post-layer-\n"
        "                # (end_layer-1) on the final folded stream (both\n"
        "                # branches produce it). Same op as the in-layer fold\n"
        "                # site; absent when steering is off.\n"
        "                _ds41_h = hidden_states.mean(dim=1)\n"
        "                _ds41_d = self._steer_stack[self.end_layer - 1]\n"
        "                hidden_states = hidden_states - (\n"
        "                    (self._steer_alpha * (_ds41_h @ _ds41_d))\n"
        "                    .unsqueeze(-1) * _ds41_d).unsqueeze(1)\n",
        "steer-final", path)

    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"patched OK: {path}")


def patch_attention(path: str) -> None:
    """Module globals the companion writer reads: zero-padded direction
    stack + scalar alpha (single-writer init in _ds41_load_steer, read-only
    forward)."""
    src = open(path).read()
    if MARKER in src or "_ds41_attn_dir" in src:
        print(f"already patched: {path}")
        return
    src = _replace_once(
        src,
        "import torch\n",
        "import torch\n"
        "# ---- dspark dsv41 fold steering: companion-writer state -------\n"
        "_ds41_attn_dir = torch.zeros(40, 5120, dtype=torch.float32)\n"
        "_ds41_attn_alpha = torch.zeros((), dtype=torch.float32)\n",
        "attn-globals", path)
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"patched OK: {path}")


def _runner_ctrl_block(indent: str) -> str:
    return (
        f"{indent}# ---- dspark dsv41 fold steering: poll the steer-ctrl\n"
        f"{indent}# file. Runs OUTSIDE the CUDA graph.\n"
        f"{indent}try:\n"
        f"{indent}    from vllm.models.deepseek_v4_1.nvidia import (\n"
        f"{indent}        model as _ds_m,\n"
        f"{indent}    )\n"
        f"{indent}    if _ds_m.DSPARK_V41_REGISTRY:\n"
        f"{indent}        _ds_m._ds41_apply_ctrl(_ds_m.DSPARK_V41_REGISTRY[0])\n"
        f"{indent}except Exception as _ds_exc:\n"
        f"{indent}    logger.warning(\n"
        f"{indent}        \"dspark dsv41 steer-ctrl failed: %s\", _ds_exc)\n"
        f"{indent}\n")


def patch_runner_v2(path: str) -> None:
    src = open(path).read()
    if MARKER in src:
        print(f"already patched: {path}")
        return
    src = _replace_once(
        src,
        "        # Run model.\n        if batch_desc.cg_mode == CUDAGraphMode.FULL:",
        _runner_ctrl_block("        ") +
        "        # Run model.\n        if batch_desc.cg_mode == CUDAGraphMode.FULL:",
        "gate-v2", path)
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"patched OK: {path}")


def patch_runner_v1(path: str) -> None:
    """Best effort: the lane runs V2 (VLLM_USE_V2_MODEL_RUNNER=1); a missing
    V1 anchor warns instead of failing the whole patch."""
    src = open(path).read()
    if MARKER in src:
        print(f"already patched: {path}")
        return
    anchor = "            model_output = self._model_forward("
    if src.count(anchor) != 1:
        print(f"WARN: V1 runner anchor absent in {path}; V1 fallback runs "
              "without ctrl polling (boot env steering still applies)")
        return
    src = src.replace(
        anchor,
        _runner_ctrl_block("            ") + anchor)
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"patched OK: {path}")


def main() -> None:
    patch_model(MODEL_FILE)
    patch_attention(ATTN_FILE)
    patch_runner_v2(RUNNER2_FILE)
    patch_runner_v1(RUNNER1_FILE)
    with open("/tmp/dspark_dsv41_fold_patch.ok", "w") as f:
        f.write(f"patched {MODEL_FILE} + {ATTN_FILE} + {RUNNER2_FILE} + "
                f"{RUNNER1_FILE}\n")


if __name__ == "__main__":
    sys.exit(main())
