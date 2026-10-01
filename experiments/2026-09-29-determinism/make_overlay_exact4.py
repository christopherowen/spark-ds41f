"""attn-exact3 -> attn-exact4 (usage: make_exact4.py SRC DST [VLLM_TREE]): an explicit row offset in every record (layout 2), WO's
sequence-parallel local rows recorded at their offset, and a per-step SP record (tag 32)."""
import sys

INVENTORY = """def _describe_plan(plan, depth: int = 0):
    \"\"\"A prepared plan's selected configuration (or its variants', for the MoE).\"\"\"
    selection = getattr(plan, "selection", None)
    config = getattr(selection, "config", None) if selection is not None else None
    if config is not None:
        query = getattr(plan, "query", None) or getattr(plan, "caps", None)
        return f"{str(config)[:400]} | {str(query)[:400]}" if query is not None else str(config)[:400]
    variants = getattr(plan, "variants", None)  # composite plans hold a read-only mapping
    if variants is not None and hasattr(variants, "items") and depth < 2:
        return {str(k): _describe_plan(v, depth + 1) for k, v in sorted(variants.items(), key=lambda kv: str(kv[0]))}
    return type(plan).__name__


def _dump_inventory(rank: int) -> None:
    \"\"\"Once per process: every live object's prepared plans by capacity, with configs.

    Walks the heap for objects whose attributes hold plans (anything named
    *plan* that carries a selection or variants), so each linear, mHC, MoE,
    WO, attention, indexer and LM-head plan set is listed with the capacities
    it was prepared for. The transition map starts here.
    \"\"\"
    path = os.path.join(DIRECTORY, f"inventory-rank{rank}.json")
    if os.path.exists(path):
        return
    import gc
    import json

    out = {}
    # vLLM freezes the heap after startup; frozen objects are invisible to
    # gc.get_objects(), so thaw them for this one walk and freeze them again.
    frozen = gc.get_freeze_count()
    if frozen:
        gc.unfreeze()
    try:
        objects = gc.get_objects()
    finally:
        if frozen:
            gc.freeze()
    for o in objects:
        try:
            state = o.__dict__ if not isinstance(o, type) else None
        except Exception:
            continue
        if not isinstance(state, dict):
            continue
        entry = {}
        for attr, value in list(state.items()):
            if "plan" not in attr.lower():
                continue
            try:
                if isinstance(value, dict):
                    items = list(value.items())
                elif isinstance(value, (list, tuple)):
                    items = list(enumerate(value))
                else:
                    items = [(None, value)]
                described = {}
                for key, item in items:
                    if hasattr(item, "selection") or hasattr(item, "variants"):
                        described[str(key)] = _describe_plan(item)
                if described:
                    entry[attr] = described.get("None", described) if list(described) == ["None"] else described
            except Exception:
                continue
        if not entry:
            continue
        for extra in ("b12x_capacities", "b12x_linear_capacities", "_projection_capacities"):
            value = state.get(extra)
            if value is not None:
                try:
                    entry[extra] = sorted(value) if not isinstance(value, (list, tuple)) else list(value)
                except Exception:
                    pass
        try:
            name = _name(o)
        except Exception:
            name = ""
        key = f"{type(o).__name__} {name}"
        if key in out:
            key = f"{key} #{id(o)}"
        out[key] = entry
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)


"""

CED = """def record_ced(indices: torch.Tensor) -> None:
    \"\"\"Record the step's CED decoder rows (their batch indices); eager steps only.

    From the CED boundary layer on, the model computes only the decoder rows (a
    prefill's last window and every decode row), so those layers' records hold
    one row per decoder row: the analysis maps them through these indices.
    \"\"\"
    if torch.cuda.is_current_stream_capturing():
        return
    n = min(int(indices.numel()), _ROWS)
    log = _log("ced", indices.device, 2 + _ROWS, _SCHEDULE_CAPACITY, _ROWS)
    stage = log.stage[0]
    stage.zero_()
    stage[0].fill_(_step[0] - 1)
    stage[1].fill_(int(indices.numel()))
    stage[2 : 2 + n].copy_(indices.reshape(-1)[:n])
    log.append(0)


# SPARK3_DEBUG_INDEX_CAPTURE="2,8:1530:1545": keep the full selected index lists
# of these layers' rows whose positions fall in [1530, 1545) (eager steps only).
_INDEX_CAPTURE = os.environ.get("SPARK3_DEBUG_INDEX_CAPTURE", "")
_index_captures: list[dict] = []


def capture_indices(owner, positions: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor) -> None:
    if not _INDEX_CAPTURE or torch.cuda.is_current_stream_capturing():
        return
    import re

    layers, lo, hi = _INDEX_CAPTURE.split(":")
    match = re.search(r"layers[.]([0-9]+)", _name(owner))
    if match is None or int(match.group(1)) not in {int(v) for v in layers.split(",")}:
        return
    rows = indices.shape[0]
    pos = positions[:rows]
    mask = (pos >= int(lo)) & (pos < int(hi))
    if not bool(mask.any()):
        return
    _index_captures.append({"step": _step[0] - 1, "layer": int(match.group(1)), "positions": pos[mask].cpu(),
                            "indices": indices[mask].cpu(), "lengths": lengths[:rows][mask].cpu()})


def capture_wanted(owner) -> bool:
    \"\"\"Whether this layer's index selection is captured (scores and inputs too).\"\"\"
    if not _INDEX_CAPTURE or torch.cuda.is_current_stream_capturing():
        return False
    import re

    match = re.search(r"layers[.]([0-9]+)", _name(owner))
    return match is not None and int(match.group(1)) in {int(v) for v in _INDEX_CAPTURE.split(":")[0].split(",")}


def capture_select(owner, positions, selected, scores, q_data, q_scales, weights, lengths) -> None:
    \"\"\"Keep one rank's scored rows in the capture window: selection, top-k scores, inputs.\"\"\"
    import re

    _, lo, hi = _INDEX_CAPTURE.split(":")
    mask = (positions >= int(lo)) & (positions < int(hi))
    if not bool(mask.any()):
        return
    layer = int(re.search(r"layers[.]([0-9]+)", _name(owner)).group(1))
    _index_captures.append({"step": _step[0] - 1, "layer": layer, "kind": "select",
                            "positions": positions[mask].cpu(), "indices": selected[mask].cpu(),
                            "scores": scores[mask].cpu(), "q_data": q_data[mask].cpu(),
                            "q_scales": q_scales[mask].cpu(), "weights": weights[mask].float().cpu(),
                            "lengths": lengths[mask].cpu()})


"""

src, dst = sys.argv[1], sys.argv[2]
tree = sys.argv[3] if len(sys.argv) > 3 else None


def edit(path_in, path_out, subs):
    s = open(path_in).read()
    for old, new in subs:
        assert s.count(old) == 1, (path_in, old, s.count(old))
        s = s.replace(old, new)
    open(path_out, "w").write(s)


edit(f"{src}/checksum_debug.py", f"{dst}/checksum_debug.py", [
    ('_WIDTH = {"runner": lambda m: 2 + 3 * m, "tags": lambda m: 3 + m, "attn": lambda m: 3 + m}\n',
     '# Layout 2: every record is (slot, rows, tag, row offset, values...); a record\n'
     '# covers the step\'s rows [offset, offset + rows). Runner records carry tag -2.\n'
     '_LAYOUT = 2\n'
     '_WIDTH = {"runner": lambda m: 4 + 3 * m, "tags": lambda m: 4 + m, "attn": lambda m: 4 + m}\n'),
    ('def record(owner, x: torch.Tensor, shared: torch.Tensor | None, routed: torch.Tensor) -> None:\n'
     '    """Append one runner record; call only when ENABLED."""\n',
     'def record(owner, x: torch.Tensor, shared: torch.Tensor | None, routed: torch.Tensor,\n'
     '           offset: int = 0) -> None:\n'
     '    """Append one runner record; call only when ENABLED."""\n'),
    ('    stage[0].fill_(slot)\n'
     '    stage[1].fill_(rows)\n'
     '    log.row_sums(x, stage[2 : 2 + rows])\n'
     '    if shared is not None:\n'
     '        log.row_sums(shared, stage[2 + m : 2 + m + rows])\n'
     '    log.row_sums(routed, stage[2 + 2 * m : 2 + 2 * m + rows])\n',
     '    stage[0].fill_(slot)\n'
     '    stage[1].fill_(rows)\n'
     '    stage[2].fill_(-2)\n'
     '    stage[3].fill_(offset)\n'
     '    log.row_sums(x, stage[4 : 4 + rows])\n'
     '    if shared is not None:\n'
     '        log.row_sums(shared, stage[4 + m : 4 + m + rows])\n'
     '    log.row_sums(routed, stage[4 + 2 * m : 4 + 2 * m + rows])\n'),
    ('def record_tag(owner, tag: int, x: torch.Tensor) -> None:\n',
     'def record_tag(owner, tag: int, x: torch.Tensor, offset: int = 0) -> None:\n'),
    ('    stage[0].fill_(slot)\n'
     '    stage[1].fill_(rows)\n'
     '    stage[2].fill_(tag)\n'
     '    log.row_sums(x, stage[3 : 3 + rows])\n'
     '    log.append(slot)\n\n\n'
     'def record_attn(owner, tag: int, x: torch.Tensor) -> None:\n',
     '    stage[0].fill_(slot)\n'
     '    stage[1].fill_(rows)\n'
     '    stage[2].fill_(tag)\n'
     '    stage[3].fill_(offset)\n'
     '    log.row_sums(x, stage[4 : 4 + rows])\n'
     '    log.append(slot)\n\n\n'
     'def record_attn(owner, tag: int, x: torch.Tensor, offset: int = 0) -> None:\n'),
    ('    """Append one attention record (slot, rows, tag, per-row sums); only when ENABLED."""\n'
     '    rows = x.shape[0]\n'
     '    log, _ = _log_for("attn", rows, x.device)\n'
     '    if log is None:\n'
     '        return\n'
     '    slot = log.slot(owner)\n'
     '    stage = log.stage[slot]\n'
     '    stage.zero_()\n'
     '    stage[0].fill_(slot)\n'
     '    stage[1].fill_(rows)\n'
     '    stage[2].fill_(tag)\n'
     '    log.row_sums(x, stage[3 : 3 + rows])\n',
     '    """Append one attention record (slot, rows, tag, offset, per-row fingerprints)."""\n'
     '    rows = x.shape[0]\n'
     '    log, _ = _log_for("attn", rows, x.device)\n'
     '    if log is None:\n'
     '        return\n'
     '    slot = log.slot(owner)\n'
     '    stage = log.stage[slot]\n'
     '    stage.zero_()\n'
     '    stage[0].fill_(slot)\n'
     '    stage[1].fill_(rows)\n'
     '    stage[2].fill_(tag)\n'
     '    stage[3].fill_(offset)\n'
     '    log.row_sums(x, stage[4 : 4 + rows])\n'),
    ('    for name, rows in ((n, r) for r in sizes for n in ("runner", "tags", "attn")):\n'
     '        step_col = 2 if name == "runner" else 3\n',
     '    for name, rows in ((n, r) for r in sizes for n in ("runner", "tags", "attn")):\n'
     '        step_col = 3\n'),
    ('        mark[1].fill_(padded)\n'
     '        if name != "runner":\n'
     '            mark[2].fill_(_MARKER)\n',
     '        mark[1].fill_(padded)\n'
     '        mark[2].fill_(_MARKER)\n'),
    ('"names": list(log.names), "max_rows": log.max_rows,',
     '"names": list(log.names), "max_rows": log.max_rows, "layout": _LAYOUT,'),
    ('def record_attn_values(owner, tag: int, values: torch.Tensor) -> None:\n'
     '    """Append one attention record of per-row values (one float each)."""\n'
     '    record_attn(owner, tag, values.reshape(values.shape[0], 1).to(torch.float32))\n',
     'def record_attn_values(owner, tag: int, values: torch.Tensor, offset: int = 0) -> None:\n'
     '    """Append one attention record of per-row values, stored as they are.\n\n'
     '    The values are exact integers below 2**24 in float32 (index sums modulo the\n'
     '    prime, lengths, step labels), so they compare as exactly as fingerprints.\n'
     '    """\n'
     '    rows = values.shape[0]\n'
     '    log, _ = _log_for("attn", rows, values.device)\n'
     '    if log is None:\n'
     '        return\n'
     '    slot = log.slot(owner)\n'
     '    stage = log.stage[slot]\n'
     '    stage.zero_()\n'
     '    stage[0].fill_(slot)\n'
     '    stage[1].fill_(rows)\n'
     '    stage[2].fill_(tag)\n'
     '    stage[3].fill_(offset)\n'
     '    stage[4 : 4 + rows].copy_(values.reshape(rows))\n'
     '    log.append(slot)\n'),
    ('_schedule_host: list[dict] = []\n',
     INVENTORY + CED + '_schedule_host: list[dict] = []\n'),
    ('    moe, mhc = set(), set()\n'
     '    for o in gc.get_objects():\n',
     '    moe, mhc = set(), set()\n'
     '    frozen = gc.get_freeze_count()  # frozen objects are invisible to get_objects()\n'
     '    if frozen:\n'
     '        gc.unfreeze()\n'
     '    try:\n'
     '        objects = gc.get_objects()\n'
     '    finally:\n'
     '        if frozen:\n'
     '            gc.freeze()\n'
     '    for o in objects:\n'),
    ('                if kind == "dump":\n'
     '                    try:\n'
     '                        _dump_plans(rank)\n'
     '                    except Exception as error:  # never block the schedule dump\n'
     '                        print(f"checksum_debug plan dump failed: {error!r}", flush=True)\n',
     '                if kind == "dump":\n'
     '                    try:\n'
     '                        _dump_plans(rank)\n'
     '                    except Exception as error:  # never block the schedule dump\n'
     '                        print(f"checksum_debug plan dump failed: {error!r}", flush=True)\n'
     '                    try:\n'
     '                        _dump_inventory(rank)\n'
     '                    except Exception as error:\n'
     '                        print(f"checksum_debug inventory dump failed: {error!r}", flush=True)\n'
     '                if kind == "dump" and _index_captures:\n'
     '                    torch.save(list(_index_captures),\n'
     '                               os.path.join(DIRECTORY, f"rank{rank}-index-capture-{len(_index_captures)}.pt"))\n'
     '                if kind == "reset":\n'
     '                    _index_captures.clear()\n'),
])
edit(f"{src}/attention.py", f"{dst}/attention.py", [
    ('        if checksum_debug.ENABLED:\n'
     '            checksum_debug.record_attn(self, 22, out)\n'
     '        return out\n',
     '        if checksum_debug.ENABLED:\n'
     '            # Under prefill SP, WO returns this rank\'s rows after the reduce-scatter.\n'
     '            checksum_debug.record_attn(self, 22, out, sp_rows.start if sp_rows is not None else 0)\n'
     '            if not torch.cuda.is_current_stream_capturing():\n'
     '                checksum_debug.record_attn_values(self, 32, torch.tensor(\n'
     '                    [float(sp_rows is not None), float(sp_rows.start if sp_rows else 0),\n'
     '                     float(sp_rows.local_rows if sp_rows else 0)],\n'
     '                    dtype=torch.float32, device=out.device))\n'
     '        return out\n'),
    ('        if checksum_debug.ENABLED:\n'
     '            checksum_debug.record_attn(self, 16, iw)\n',
     '        if checksum_debug.ENABLED:\n'
     '            # Under an SP indexer split these are this rank\'s rows [first, last).\n'
     '            checksum_debug.record_attn(self, 16, iw, first)\n'),
    ('        if checksum_debug.ENABLED:\n'
     '            checksum_debug.record_attn_values(self, 20, swa_lengths[:rows])\n'
     '            if main is not None:\n',
     '        if checksum_debug.ENABLED:\n'
     '            checksum_debug.record_attn_values(self, 20, swa_lengths[:rows])\n'
     '            if main is not None:\n'
     '                checksum_debug.capture_indices(self, positions, owner.topk_indices_buffer[:rows],\n'
     '                                               top_lengths[:rows])\n'),
    ('                iq_data, iq_scale, iw = query\n'
     '                index_rows = iq_data.shape[0]\n',
     '                iq_data, iq_scale, iw = query\n'
     '                index_rows = iq_data.shape[0]\n'
     '                capture = checksum_debug.ENABLED and checksum_debug.capture_wanted(self)\n'
     '                if capture:  # debug: this rank\'s top-k scores for the capture window\n'
     '                    scores_out = torch.full((index_rows, 512), float("nan"), dtype=torch.float32,\n'
     '                                            device=iq_data.device)\n'),
    ('                        output_indices=selected[offset:end],\n'
     '                        **candidate_args,\n'
     '                    )\n',
     '                        output_indices=selected[offset:end],\n'
     '                        **candidate_args,\n'
     '                        **({"output_scores": scores_out[offset:end]} if capture else {}),\n'
     '                    )\n'),
    ('                    dsa_indexer.score(binding)\n'
     '                    dsa_indexer.select(binding)\n'
     '\n'
     '            if split is None:\n',
     '                    dsa_indexer.score(binding)\n'
     '                    dsa_indexer.select(binding)\n'
     '                if capture:\n'
     '                    checksum_debug.capture_select(\n'
     '                        self, positions[first : first + index_rows], selected[:index_rows], scores_out,\n'
     '                        iq_data, iq_scale, iw, im.cache_lengths[first : first + index_rows])\n'
     '\n'
     '            if split is None:\n'),
])
print("ok")
if tree:
    edit(f"{tree}/vllm/models/deepseek_v4_1/nvidia/model.py", f"{dst}/model41.py", [
        ("    ) -> torch.Tensor | IntermediateTensors:\n"
         "        sp = None\n"
         "        # With CED the encoder layers",
         "    ) -> torch.Tensor | IntermediateTensors:\n"
         "        from vllm.model_executor.layers.fused_moe.runner import checksum_debug\n"
         "\n"
         "        if checksum_debug.ENABLED and ced_indices is not None:\n"
         "            checksum_debug.record_ced(ced_indices)  # debug: decoder rows per step\n"
         "        sp = None\n"
         "        # With CED the encoder layers"),
    ])
    print("model41 ok")
