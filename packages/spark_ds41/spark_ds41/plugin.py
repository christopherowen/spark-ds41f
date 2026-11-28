# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin entry point for spark_ds41."""

from __future__ import annotations


def register() -> None:
    """Register Spark DS4.1 direct-IO model loaders with vLLM."""
    from vllm.model_executor.model_loader import register_model_loader

    from spark_ds41.direct_io.direct_loader import DirectModelLoader

    register_model_loader("direct")(DirectModelLoader)
