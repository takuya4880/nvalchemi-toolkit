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
"""Opt-in per-candidate failure isolation for standalone inflight relaxation."""

from __future__ import annotations

from collections import OrderedDict
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from nvalchemi.dynamics.base import DynamicsStage, InflightTerminationReason

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics


class SafetyFailureReason(IntEnum):
    """Separate safety failures from physical convergence and step exhaustion."""

    NONE = 0
    NONFINITE_GEOMETRY = 1
    INVALID_CELL = 2
    NEAR_OVERLAP = 3
    NONFINITE_MODEL_OUTPUT = 4


class InflightSafety:
    """Quarantine bad rows before downstream hooks and reuse ordinary refill.

    Parameters
    ----------
    min_distance : float
        Failure threshold in Angstrom, not a chemical bond-length criterion.
    min_cell_length : float
        Minimum cell singular value in Angstrom.
    min_cell_ratio : float
        Minimum ratio of smallest to largest cell singular value.

    Notes
    -----
    This policy is opt-in and supports standalone inflight runs with refill
    frequency one. Matrix/COO neighbor distances are checked after rebuilding.
    Arbitrary model exceptions, CUDA OOM and fused/pipeline execution are not
    silently swallowed. Failed rows keep their last checked finite geometry,
    are excluded from subsequent model evaluations, and leave at next refill.
    Successful rows retain the original numerical updates and stopping rule.
    """

    def __init__(
        self,
        min_distance: float = 1e-3,
        min_cell_length: float = 1e-4,
        min_cell_ratio: float = 1e-8,
    ) -> None:
        if min(min_distance, min_cell_length, min_cell_ratio) <= 0:
            raise ValueError("Safety thresholds must be positive")
        self.min_distance = min_distance
        self.min_cell_length = min_cell_length
        self.min_cell_ratio = min_cell_ratio
        self.failures: list[dict] = []

    def install(self, optimizer: BaseDynamics) -> None:
        """Attach checks without replacing optimizer formulas or convergence."""
        if optimizer.sampler is None or optimizer.refill_frequency != 1:
            raise ValueError("Safety requires standalone inflight refill_frequency=1")
        if hasattr(optimizer, "inflight_safety"):
            raise ValueError("Safety is already installed")
        optimizer.inflight_safety = self
        original_init = optimizer.init_fn
        original_hooks = optimizer._call_hooks
        original_compute = optimizer.compute

        def initialize(batch):
            self._initialize(optimizer, batch)
            if original_init is not None:
                original_init(batch)

        def hooks(stage, batch):
            self._restore(optimizer, batch)
            if stage in (
                DynamicsStage.BEFORE_STEP,
                DynamicsStage.AFTER_PRE_UPDATE,
                DynamicsStage.BEFORE_COMPUTE,
                DynamicsStage.AFTER_POST_UPDATE,
            ):
                self._geometry(optimizer, batch)
            if stage == DynamicsStage.AFTER_COMPUTE:
                self._outputs(optimizer, batch)
            result = original_hooks(stage, batch)
            if stage == DynamicsStage.BEFORE_COMPUTE:
                self._overlap(optimizer, batch)
            return result

        def compute(batch):
            failed = batch.safety_failure.view(-1) != 0
            if not bool(failed.any()):
                return original_compute(batch)
            keep = torch.where(~failed)[0]
            outputs = OrderedDict()
            if keep.numel():
                valid = batch.index_select(keep)
                outputs = original_compute(valid)
                nodes = ~failed[batch.batch_idx.long()]
                for key, attr in optimizer._OUTPUT_KEY_TO_BATCH_ATTR.items():
                    if key not in outputs:
                        continue
                    target = getattr(batch, attr)
                    mask = nodes if attr == "forces" else ~failed
                    target.zero_()
                    target[mask] = getattr(valid, attr)
                    outputs[key] = target.detach()
            else:
                for key, attr in optimizer._OUTPUT_KEY_TO_BATCH_ATTR.items():
                    value = getattr(batch, attr, None)
                    if isinstance(value, torch.Tensor):
                        value.zero_()
                        outputs[key] = value.detach()
            optimizer._last_outputs = outputs
            return outputs

        optimizer.init_fn = initialize
        optimizer._call_hooks = hooks
        optimizer.compute = compute

    def _initialize(self, optimizer: BaseDynamics, batch: Batch) -> None:
        """Create identical safety schema on initial and replacement batches."""
        if all(
            key in batch
            for key in ("safety_failure", "safety_positions", "safety_cell")
        ):
            return
        optimizer._ensure_bookkeeping_fields(batch)
        optimizer._ensure_standalone_inflight_fields(batch)
        for key, value, level in (
            (
                "safety_failure",
                torch.zeros(batch.num_graphs, 1, dtype=torch.long, device=batch.device),
                "system",
            ),
            ("safety_positions", batch.positions.detach().clone(), "node"),
            ("safety_cell", batch.cell.detach().clone(), "system"),
        ):
            if key not in batch:
                parts = value.split(
                    batch.num_nodes_per_graph.tolist() if level == "node" else 1
                )
                batch.add_key(key, list(parts), level=level)

    @staticmethod
    def _reduce_bad(batch: Batch, values: torch.Tensor) -> torch.Tensor:
        """Reduce non-finite atomic values to a graph mask."""
        bad = ~torch.isfinite(values).reshape(batch.num_nodes, -1).all(-1)
        counts = torch.zeros(batch.num_graphs, device=batch.device, dtype=torch.long)
        counts.index_add_(0, batch.batch_idx.long(), bad.long())
        return counts > 0

    def _geometry(self, optimizer: BaseDynamics, batch: Batch) -> None:
        """Check cell solves before neighbors, projection or model evaluation."""
        self._initialize(optimizer, batch)
        bad = self._reduce_bad(batch, batch.positions) | ~torch.isfinite(
            batch.cell
        ).all((-1, -2))
        self._fail(optimizer, batch, bad, SafetyFailureReason.NONFINITE_GEOMETRY)
        cells = batch.cell.double()
        singular = torch.linalg.svdvals(cells)
        bad = (singular[:, -1] < self.min_cell_length) | (
            singular[:, -1] < self.min_cell_ratio * singular[:, 0]
        )
        bad |= (
            torch.linalg.det(cells) * torch.linalg.det(batch.safety_cell.double()) <= 0
        )
        self._fail(optimizer, batch, bad, SafetyFailureReason.INVALID_CELL)

    def _overlap(self, optimizer: BaseDynamics, batch: Batch) -> None:
        """Use freshly built neighbors, excluding same-atom periodic images."""
        if "neighbor_list" in batch:
            i, j = batch.neighbor_list.long().unbind(-1)
            shifts = getattr(batch, "neighbor_list_shifts", None)
        elif "neighbor_matrix" in batch:
            width = batch.neighbor_matrix.shape[1]
            mask = torch.arange(width, device=batch.device)[
                None
            ] < batch.num_neighbors.view(-1, 1)
            i = torch.arange(batch.num_nodes, device=batch.device)[:, None].expand(
                -1, width
            )[mask]
            j = batch.neighbor_matrix[mask].long()
            shift_matrix = getattr(batch, "neighbor_matrix_shifts", None)
            shifts = shift_matrix[mask] if shift_matrix is not None else None
        else:
            i = j = torch.empty(0, dtype=torch.long, device=batch.device)
            shifts = None
        graph = batch.batch_idx[i].long()
        delta = batch.positions[j].double() - batch.positions[i].double()
        if shifts is not None:
            delta += torch.bmm(
                shifts.double()[:, None], batch.cell[graph].double()
            ).squeeze(1)
        distance = delta.square().sum(-1)
        bad_edges = (i != j) & (distance < self.min_distance**2)
        counts = torch.zeros(batch.num_graphs, dtype=torch.long, device=batch.device)
        counts.index_add_(0, graph, bad_edges.long())
        self._fail(optimizer, batch, counts > 0, SafetyFailureReason.NEAR_OVERLAP)
        active = batch.safety_failure.view(-1) == 0
        with torch.no_grad():
            batch.safety_positions[active[batch.batch_idx.long()]] = batch.positions[
                active[batch.batch_idx.long()]
            ]
            batch.safety_cell[active] = batch.cell[active]

    def _outputs(self, optimizer: BaseDynamics, batch: Batch) -> None:
        """Stop non-finite forces/stress before projection or optimizer kicks."""
        bad = self._reduce_bad(batch, batch.forces)
        for name in ("energy", "stress"):
            value = getattr(batch, name, None)
            if value is not None:
                bad |= ~torch.isfinite(value).reshape(batch.num_graphs, -1).all(-1)
        self._fail(optimizer, batch, bad, SafetyFailureReason.NONFINITE_MODEL_OUTPUT)

    def _fail(
        self,
        optimizer: BaseDynamics,
        batch: Batch,
        mask: torch.Tensor,
        reason: SafetyFailureReason,
    ) -> None:
        """Mark only new failures; quarantine their data until ordinary refill."""
        mask = (
            mask
            & (batch.safety_failure.view(-1) == 0)
            & (batch.status.view(-1) < optimizer.exit_status)
        )
        if not bool(mask.any()):
            return
        ids = getattr(batch, "candidate_id", batch.system_id).view(-1)
        for index in torch.where(mask)[0].tolist():
            self.failures.append(
                dict(
                    candidate_id=int(ids[index]),
                    reason=reason.name,
                    updates=int(batch.inflight_step_count[index]),
                    global_step=optimizer.step_count,
                )
            )
        batch.safety_failure.view(-1)[mask] = int(reason)
        self._restore(optimizer, batch)

    def _restore(self, optimizer: BaseDynamics, batch: Batch) -> None:
        """Keep previously quarantined rows harmless until they graduate."""
        if "safety_failure" not in batch:
            return
        mask = batch.safety_failure.view(-1) != 0
        if not bool(mask.any()):
            return
        nodes = mask[batch.batch_idx.long()]
        with torch.no_grad():
            batch.inflight_termination.view(-1)[mask] = InflightTerminationReason.FAILED
            batch.status.view(-1)[mask] = optimizer.exit_status
            batch.positions[nodes] = batch.safety_positions[nodes]
            batch.cell[mask] = batch.safety_cell[mask]
            # Malformed admission may have no valid cached geometry.
            invalid_cell = ~torch.isfinite(batch.cell).all((-1, -2))
            invalid_cell |= (
                torch.linalg.det(torch.nan_to_num(batch.cell.double())).abs() < 1e-12
            )
            invalid_cell &= mask
            batch.cell[invalid_cell] = (
                torch.eye(3, device=batch.device, dtype=batch.cell.dtype) * 5
            )
            batch.positions[nodes] = torch.nan_to_num(batch.positions[nodes])
            for name in ("forces", "velocities"):
                value = getattr(batch, name, None)
                if value is not None:
                    value[nodes] = 0
            for name in ("stress", "energy", "cell_velocities"):
                value = getattr(batch, name, None)
                if value is not None:
                    value[mask] = 0
