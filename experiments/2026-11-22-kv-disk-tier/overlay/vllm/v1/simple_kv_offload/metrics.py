# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics of SimpleCPUOffloadConnector's offload tier.

The scheduler-side manager counts events and reports them through
``get_kv_connector_stats``; workers report nothing here. Series (one child per
engine label set):

- ``vllm:kv_offload_disk_degraded`` gauge: 1 once a disk fault on any rank
  degraded the tier (no external hits, no stores until the engine restarts).
- ``vllm:kv_offload_disk_loads_total``: loads finished on every rank.
- ``vllm:kv_offload_disk_load_failures_total``: loads that failed on a rank and
  fell back to a cache miss.
- ``vllm:kv_offload_disk_stored_blocks_total``: blocks stored on every rank and
  published in the tier's index.
- ``vllm:kv_offload_disk_store_failed_blocks_total``: blocks whose store failed
  on a rank and were dropped from the index.
- ``vllm:kv_offload_disk_io_errors_total``: disk faults reported by ranks
  (each rank reports its first fault).
- ``vllm:kv_offload_disk_load_seconds_max`` gauge: the slowest load since the
  engine started, in seconds from the step that sent it to the step that saw
  it finished on every rank (queue time included, failed loads included).

External prefix-cache hits keep their generic series,
``vllm:external_prefix_cache_hits_total`` and
``vllm:external_prefix_cache_queries_total`` (tokens, counted at lookup).
"""

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

COUNTERS = {
    "loads": (
        "vllm:kv_offload_disk_loads",
        "Offload-tier loads that finished on every rank.",
    ),
    "load_failures": (
        "vllm:kv_offload_disk_load_failures",
        "Offload-tier loads that failed on a rank and fell back to a cache miss.",
    ),
    "stored_blocks": (
        "vllm:kv_offload_disk_stored_blocks",
        "Blocks stored on every rank and published in the offload tier's index.",
    ),
    "store_failed_blocks": (
        "vllm:kv_offload_disk_store_failed_blocks",
        "Blocks whose store failed on a rank and were dropped from the index.",
    ),
    "io_errors": (
        "vllm:kv_offload_disk_io_errors",
        "Disk faults reported by ranks (each rank reports its first fault).",
    ),
}
DEGRADED = (
    "vllm:kv_offload_disk_degraded",
    "1 when a disk fault degraded the offload tier until the engine restarts.",
)
LOAD_SECONDS_MAX = (
    "vllm:kv_offload_disk_load_seconds_max",
    "Slowest offload-tier load since the engine started, in seconds from the "
    "step that sent it to the step that saw it finished on every rank.",
)
# Keys whose reports merge by max, not by sum.
_MAX_KEYS = ("degraded", "load_seconds_max")


@dataclass
class SimpleCPUOffloadStats(KVConnectorStats):
    """Counters since the last report, plus ``degraded`` (0 or 1) and
    ``load_seconds_max`` when it rose."""

    def reset(self) -> None:
        self.data = {}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        merged = dict(self.data)
        for key, value in other.data.items():
            if key in _MAX_KEYS:
                merged[key] = max(merged.get(key, 0), value)
            else:
                merged[key] = merged.get(key, 0) + value
        return SimpleCPUOffloadStats(data=merged)

    def reduce(self) -> dict[str, int | float]:
        return {f"disk_{key}": value for key, value in sorted(self.data.items())}

    def is_empty(self) -> bool:
        return not self.data


class SimpleCPUOffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.counters = {
            key: create_metric_per_engine(
                self._counter_cls(name=name, documentation=doc, labelnames=labelnames),
                per_engine_labelvalues,
            )
            for key, (name, doc) in COUNTERS.items()
        }
        self.degraded, self.load_seconds_max = (
            create_metric_per_engine(
                self._gauge_cls(
                    name=name,
                    documentation=doc,
                    multiprocess_mode="mostrecent",
                    labelnames=labelnames,
                ),
                per_engine_labelvalues,
            )
            for name, doc in (DEGRADED, LOAD_SECONDS_MAX)
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for key, value in transfer_stats_data.items():
            if key == "degraded":
                self.degraded[engine_idx].set(value)
            elif key == "load_seconds_max":
                self.load_seconds_max[engine_idx].set(value)
            elif key in self.counters and value:
                self.counters[key][engine_idx].inc(value)
