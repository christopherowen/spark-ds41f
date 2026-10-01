"""attn-exact3 -> attn-exact4: an explicit row offset in every record (layout 2), WO's
sequence-parallel local rows recorded at their offset, and a per-step SP record (tag 32)."""
import sys

INVENTORY = """def _describe_plan(plan, depth: int = 0):
    \"\"\"A prepared plan's selected configuration (or its variants', for the MoE).\"\"\"
    selection = getattr(plan, "selection", None)
    config = getattr(selection, "config", None) if selection is not None else None
    if config is not None:
        return str(config)[:400]
    variants = getattr(plan, "variants", None)
    if isinstance(variants, dict) and depth < 2:
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
    for o in gc.get_objects():
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

src, dst = sys.argv[1], sys.argv[2]


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
     INVENTORY + '_schedule_host: list[dict] = []\n'),
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
     '                        print(f"checksum_debug inventory dump failed: {error!r}", flush=True)\n'),
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
])
print("ok")
