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

"""Field initialization from the measured conventional workflow."""

from __future__ import annotations

import torch

def schema(batch):
    """Model and observer fields preserved through inflight membership changes."""
    for key, value, level in (
        ("forces", torch.zeros_like(batch.positions), "node"),
        ("initial_raw_forces", torch.zeros_like(batch.positions), "node"),
        (
            "initial_raw_stress",
            batch.positions.new_zeros(batch.num_graphs, 3, 3),
            "system",
        ),
        ("initial_energy", batch.positions.new_zeros(batch.num_graphs, 1), "system"),
        ("energy", batch.positions.new_zeros(batch.num_graphs, 1), "system"),
        ("stress", batch.positions.new_zeros(batch.num_graphs, 3, 3), "system"),
        (
            "convergence_streak",
            torch.zeros(batch.num_graphs, 1, dtype=torch.long, device=batch.device),
            "system",
        ),
        (
            "initial_observed",
            torch.zeros(batch.num_graphs, 1, dtype=torch.bool, device=batch.device),
            "system",
        ),
        (
            "physical_fmax",
            batch.positions.new_full((batch.num_graphs, 1), torch.inf),
            "system",
        ),
        (
            "physical_stress_norm",
            batch.positions.new_full((batch.num_graphs, 1), torch.inf),
            "system",
        ),
    ):
        if key not in batch:
            parts = (
                list(value.split(batch.num_nodes_per_graph.tolist()))
                if level == "node"
                else list(value.split(1))
            )
            batch.add_key(key, parts, level=level)
