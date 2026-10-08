# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conservative current-cell neighbor search without stale geometry capacities."""

from __future__ import annotations

from enum import Enum

from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks.neighbor_list import NeighborListHook


class VariableCellNeighborListHook(NeighborListHook):
    """Refresh geometry-dependent search ranges on every model evaluation.

    Uses the parent configuration/format/capacity handling, but never reuses
    positional-skin flags or cell-dependent staging/scratch allocations across
    evaluations. This includes force priming and cell-only motion with atoms
    at the origin. Register at BEFORE_COMPUTE with frequency=1.

    This correctness-first path deliberately permits host capacity calculation
    and allocation. It is not a fullgraph hook or a performance-certified
    variable-cell allocator. Existing fixed-cell hooks are unchanged. For small
    validation cases, explicitly request method="naive".
    """

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Build complete periodic neighbors using the cell visible to the model.

        Parameters
        ----------
        ctx : HookContext
            Current batch after publishing both positions and cell.
        stage : Enum
            Dispatch stage, normally BEFORE_COMPUTE.
        """
        self._alloc_N = self._alloc_B = None
        self._ref_positions = self._rebuild_flags = None
        self._first_build = True
        self._rebuild(ctx.batch)
