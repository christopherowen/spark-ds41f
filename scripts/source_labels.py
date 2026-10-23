"""Source labels: ``bin/spark build`` records each source's upstream, ref, revision, patch head
and tree as ``local.spark-ds41f.<source>.<field>``. Images built before the release after
spark-ds41f-1022 carry the same labels as ``local.spark3.<source>.<field>``; a lookup reads
either, so a profile or an image of an earlier release keeps resolving."""

from __future__ import annotations

PREFIX = "local.spark-ds41f."
LEGACY_PREFIX = "local.spark3."


def key(labels: dict, field: str) -> str:
    """The label holding ``field`` (such as ``vllm.tree``) in ``labels``: the current name,
    unless only the name of images before the rename is present."""
    if PREFIX + field not in labels and LEGACY_PREFIX + field in labels:
        return LEGACY_PREFIX + field
    return PREFIX + field


def get(labels: dict, field: str) -> str | None:
    """``field``'s value in an image's labels or a profile's ``expected_labels``."""
    return labels.get(key(labels, field))
