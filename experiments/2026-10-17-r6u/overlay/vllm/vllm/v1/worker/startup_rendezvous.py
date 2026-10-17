# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A startup rendezvous on the host that waits on liveness, not on a deadline.

Ranks finish loading and preparing kernels at their own pace: minutes apart when one
rank's caches are cold. Waiting for the others on the host costs nothing, while waiting
inside a phase's first GPU collective would stop the GPU at the collective's timeout.

Each rank publishes a heartbeat in the distributed store from a daemon thread: a
counter, its startup phase and its progress (kernels prepared so far). A rank at a
rendezvous waits as long as every missing rank's heartbeat keeps changing, reports what
the missing ranks are doing, and fails, naming the rank, only when a missing rank's
heartbeat stops changing for ``VLLM_STARTUP_HEARTBEAT_TIMEOUT_S`` (its process died or
hung). Staleness is measured on the waiting rank's own clock, so nodes need not agree
on the time. How long a rank takes to prepare is never bounded.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

HEARTBEAT_INTERVAL_S = 2.0
REPORT_INTERVAL_S = 30.0


def _heartbeat_key(rank: int) -> str:
    return f"vllm/startup/heartbeat/{rank}"


def _arrival_key(phase: str, rank: int) -> str:
    return f"vllm/startup/arrived/{phase}/{rank}"


class StartupHeartbeat:
    """This rank's heartbeat: a counter, its phase and its progress."""

    def __init__(
        self, store, rank: int, progress: Callable[[], str] = lambda: ""
    ) -> None:
        self.store = store
        self.rank = rank
        self.progress = progress
        self.phase = "starting"
        self._count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vllm-startup-heartbeat", daemon=True
        )

    def start(self) -> StartupHeartbeat:
        self._beat()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _beat(self) -> None:
        self._count += 1
        detail = self.progress()
        self.store.set(
            _heartbeat_key(self.rank),
            f"{self._count}|{self.phase}" + (f"|{detail}" if detail else ""),
        )

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_S):
            try:
                self._beat()
            except Exception:  # the store is gone: the process is shutting down
                return


def _describe(value: str) -> str:
    parts = value.split("|")
    return ", ".join(parts[1:]) if len(parts) > 1 else "no heartbeat yet"


def rendezvous(
    store,
    rank: int,
    world_size: int,
    phase: str,
    heartbeat_timeout_s: float,
    report: Callable[[str], None],
    poll_s: float = 0.25,
) -> float:
    """Wait until every rank reaches ``phase``; return the seconds waited.

    Raises RuntimeError naming a missing rank whose heartbeat stopped changing.
    """
    store.set(_arrival_key(phase, rank), "1")
    started = last_report = time.monotonic()
    seen: dict[int, tuple[str, float]] = {}
    while True:
        missing = [
            r for r in range(world_size) if not store.check([_arrival_key(phase, r)])
        ]
        if not missing:
            return time.monotonic() - started
        now = time.monotonic()
        for r in missing:
            key = _heartbeat_key(r)
            value = store.get(key).decode() if store.check([key]) else ""
            previous = seen.get(r)
            if previous is None or previous[0] != value:
                seen[r] = (value, now)
            elif now - previous[1] > heartbeat_timeout_s:
                raise RuntimeError(
                    f"rank {r} stopped responding before {phase}: its heartbeat "
                    f"has not changed for {now - previous[1]:.0f} s "
                    f"(last: {_describe(value)})"
                )
        if now - last_report >= REPORT_INTERVAL_S:
            report(
                f"waited {now - started:.0f} s before {phase} for "
                + "; ".join(f"rank {r} ({_describe(seen[r][0])})" for r in missing)
            )
            last_report = now
        time.sleep(poll_s)
