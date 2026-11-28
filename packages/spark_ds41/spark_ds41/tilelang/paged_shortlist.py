# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base shortlist plan interface for page-summary indexers."""


class PageShortlistPlan:
    """Buffer-free serving plan base class; caller owns all scratch."""

    def scratch_specs(self, rows):
        return tuple(self._specs(rows).values())
