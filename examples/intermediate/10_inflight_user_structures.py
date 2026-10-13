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
"""
Inflight relaxation of user structures
======================================

A small composition recipe using the existing sampler, optimizer and hooks.
Call ``relax_structures`` from your driver with your model and convergence
hook. Inputs can be ASE Atoms from CIF, XYZ or an in-memory generator.
Symmetry constraints and per-structure failure isolation are independent
options. This example does not load a model or run a calculation at import.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    DynamicsStage,
    FIRE2VariableCell,
    HostMemory,
    SizeAwareSampler,
)
from nvalchemi.dynamics.hooks import FixSymmetryHook, prepare_symmetry
from nvalchemi.dynamics.inflight_safety import InflightSafety

if TYPE_CHECKING:
    from ase import Atoms

    from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook
    from nvalchemi.models.base import BaseModelMixin


class StructureDataset:
    """Clone-on-read input collection for the existing SizeAwareSampler."""

    def __init__(self, samples: Sequence[AtomicData]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        """Return the number of input structures."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[AtomicData, dict]:
        """Return an independent structure with empty loader metadata."""
        return self.samples[index].clone(), {}

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return atom count and the pre-neighbor edge count."""
        return self.samples[index].num_nodes, 0


def relax_structures(
    atoms: Sequence[Atoms],
    *,
    model: BaseModelMixin,
    convergence_hook: ConvergenceHook,
    device: str,
    max_batch_size: int,
    max_atoms: int,
    max_steps_per_system: int,
    optimizer_kwargs: dict[str, Any],
    symmetry: bool = False,
    failure_isolation: bool = True,
    symprec: float = 0.01,
    optimizer_type: type[BaseDynamics] = FIRE2VariableCell,
) -> tuple[Batch, list[dict]]:
    """Compose existing APIs without imposing a model or stopping protocol.

    Parameters
    ----------
    atoms : sequence of ase.Atoms
        Input structures in the desired ID order.
    model : BaseModelMixin
        Caller-provided model wrapper, placed on the selected device.
    convergence_hook : ConvergenceHook
        Caller-provided stopping rule, including cell criteria when needed.
    device : str
        Explicit execution device, for example ``cuda:0`` or ``cpu``.
    max_batch_size, max_atoms : int
        Explicit sampler capacities; no hardware-specific preset is used.
    max_steps_per_system : int
        Update budget for each structure.
    symmetry, failure_isolation : bool
        Independent controls for constraints and failed-row isolation.
    symprec : float
        ASE/spglib tolerance when preparing optional symmetry records.
    optimizer_type : type[BaseDynamics]
        Existing standalone optimizer, e.g. FIRE2 or FIRE2VariableCell.
    optimizer_kwargs : dict
        Optimizer parameters, e.g. timestep and displacement limits.

    Returns
    -------
    Batch
        Completed structures on CPU, matched by ``candidate_id`` to input order.
    list of dict
        Failure records. Failed endpoints are diagnostic, not physical results.
    """
    if not atoms or min(max_batch_size, max_atoms, max_steps_per_system) <= 0:
        raise ValueError("Provide nonempty inputs and positive capacities/budgets")
    samples, records = [], []
    for index, structure in enumerate(atoms):
        data = AtomicData.from_atoms(structure, device="cpu", dtype=torch.float32)
        if symmetry:
            data, record = prepare_symmetry(data, symmetry_id=index, symprec=symprec)
            records.append(record)
        data.add_system_property("candidate_id", torch.tensor([[index]]))
        samples.append(data)
    if any(data.num_nodes > max_atoms for data in samples):
        raise ValueError("An input exceeds max_atoms; increase the sampler capacity")
    sink = HostMemory(capacity=len(samples))
    hooks = [FixSymmetryHook(records=records)] if symmetry else []

    def initialize(batch: Batch) -> None:
        """Allocate runtime fields and prime only newly admitted structures."""
        for name, value, level in (
            ("forces", torch.zeros_like(batch.positions), "node"),
            ("velocities", torch.zeros_like(batch.positions), "node"),
            ("energy", batch.positions.new_zeros(batch.num_graphs, 1), "system"),
            ("stress", torch.zeros_like(batch.cell), "system"),
        ):
            if name not in batch:
                parts = batch.num_nodes_per_graph.tolist() if level == "node" else 1
                batch.add_key(name, list(value.split(parts)), level=level)
        if symmetry:
            constraint = hooks[0]
            constraint._bind_batch(batch)
            constraint._layout_signature = constraint._current_layout_signature(batch)
        optimizer._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
        optimizer.compute(batch)
        optimizer._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)

    optimizer = optimizer_type(
        model=model,
        device_type=device,
        convergence_hook=convergence_hook,
        hooks=hooks,
        init_fn=initialize,
        sampler=SizeAwareSampler(
            StructureDataset(samples),
            max_batch_size=max_batch_size,
            max_atoms=max_atoms,
            shuffle=False,
        ),
        sinks=[sink],
        max_steps_per_system=max_steps_per_system,
        refill_frequency=1,
        **optimizer_kwargs,
    )
    for hook in model.make_neighbor_hooks():
        optimizer.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
    safety = InflightSafety()
    if failure_isolation:
        safety.install(optimizer)
    optimizer.run(n_steps=len(samples) * (max_steps_per_system + 1))
    if optimizer.active_batch is not None or len(sink) != len(samples):
        raise RuntimeError("The run did not collect every input structure")
    return sink.read(), safety.failures
