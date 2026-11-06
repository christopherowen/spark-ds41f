"""Reject serving requests after ownership loss, management latch, or a stale coordinator heartbeat."""
import json
from pathlib import Path
from branch_watchdog import latch_management_loss, ours, read_hold, stale


def require_window(state_path):
    if state_path is None:
        raise RuntimeError('An owned window state is required before serving requests')
    state = json.loads(Path(state_path).read_text())
    hold = read_hold(Path(state['hold_path']))
    if not ours(state, hold, review=True):
        raise RuntimeError('Window ownership changed; refusing serving requests')
    if hold.get('management_loss_latched') or hold.get('holder') != state['holder']:
        raise RuntimeError('Management or guard telemetry loss latched; refusing serving requests')
    if stale(hold):
        raise RuntimeError('Coordinator heartbeat is stale; refusing serving requests')
    return state
