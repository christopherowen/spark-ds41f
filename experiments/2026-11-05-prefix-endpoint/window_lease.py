"""Reject serving requests after ownership loss or a stale coordinator heartbeat."""
import json
from pathlib import Path
from branch_watchdog import ours, read_hold, stale


def require_window(state_path):
    if state_path is None:
        raise RuntimeError('An owned window state is required before serving requests')
    state = json.loads(Path(state_path).read_text())
    hold = read_hold(Path(state['hold_path']))
    if not ours(state, hold):
        raise RuntimeError('Window ownership changed; refusing serving requests')
    if stale(hold):
        raise RuntimeError('Coordinator heartbeat is stale; refusing serving requests')
    return state
