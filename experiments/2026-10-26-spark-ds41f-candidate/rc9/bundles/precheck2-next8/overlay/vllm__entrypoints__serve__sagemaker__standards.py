# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``model_hosting_container_standards.sagemaker``, imported without changing the
process's root logging.

When it is first imported, ``model_hosting_container_standards`` (its
``logging_config``) sets the root logger to ``SAGEMAKER_CONTAINER_LOG_LEVEL``, else
``LOG_LEVEL``, else ERROR, and raises every root handler below that level to it.
The API server imports it for the SageMaker routes, so with neither variable set,
third-party warnings, which propagate to the root logger, were dropped from the
server's log (vLLM's own loggers have their own handler). Unless one of the
variables is set, the root logger's level and its handlers' levels are put back
after the import; a variable that is set keeps the library's documented behavior.
"""

import logging
import os
from types import ModuleType

LEVEL_VARIABLES = ("SAGEMAKER_CONTAINER_LOG_LEVEL", "LOG_LEVEL")


def _import_sagemaker_standards() -> ModuleType:
    root = logging.getLogger()
    level = root.level
    handler_levels = [(handler, handler.level) for handler in root.handlers]
    import model_hosting_container_standards.sagemaker as standards

    if not any(name in os.environ for name in LEVEL_VARIABLES):
        root.setLevel(level)
        for handler, handler_level in handler_levels:
            handler.setLevel(handler_level)
    return standards


sagemaker_standards = _import_sagemaker_standards()
