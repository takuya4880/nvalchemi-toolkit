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

"""Conventional-only production relaxation using the original PR2 scheduler."""

from __future__ import annotations

import gzip
import io
import time
from pathlib import Path

import torch

import fields as engine
from neighbors import VariableCellNeighborListHook
from nvalchemi.data import AtomicData
from nvalchemi.dynamics import DynamicsStage, FIRE2, FIRE2VariableCell, SizeAwareSampler
from nvalchemi.dynamics.hooks import FixSymmetryHook, SymmetryRecord
from nvalchemi.dynamics.sinks import HostMemory
from production_stop import Stop, frechet_forces

def schema(batch):
    """Initialize model/observer fields once, preserving survivors on refill."""
    engine.schema(batch)
    for key in ("production_fmax", "production_atom_fmax", "production_cell_fmax", "frechet_log_error"):
        if key not in batch:
            batch.add_key(key, list(batch.positions.new_full((batch.num_graphs, 1), torch.inf).split(1)), level="system")
    if "production_reference_cell" not in batch:
        batch.add_key("production_reference_cell", list(batch.cell.clone().split(1)), level="system")


class Residuals:
    """Compute common projected full-space ASE stopping observables."""

    stage, frequency = DynamicsStage.AFTER_COMPUTE, 1

    def __init__(self, records, variable, constrained):
        self.projector = FixSymmetryHook(records) if constrained else None
        self.variable = variable

    def __call__(self, ctx, stage):
        batch = ctx.batch
        forces, stress = batch.forces, batch.stress
        if self.projector is not None:
            projector = self.projector
            signature = projector._current_layout_signature(batch)
            if projector._layout_signature != signature:
                projector._bind_batch(batch)
                projector._layout_signature = signature
            forces = projector._project_rank1_batch(forces, batch.cell)
            stress = projector._project_rank2_batch(stress, batch.cell)
        physical = forces.norm(dim=-1).new_zeros(batch.num_graphs).scatter_reduce(0, batch.batch_idx.long(), forces.norm(dim=-1), reduce="amax", include_self=True)
        batch.physical_fmax.copy_(physical[:, None])
        batch.physical_stress_norm.copy_(stress.norm(dim=(-1, -2))[:, None])
        if self.variable:
            atom, cell, error = frechet_forces(forces, stress, batch.cell, batch.production_reference_cell, batch.batch_idx, batch.num_nodes_per_graph)
            atom_max = cell.new_zeros(batch.num_graphs).scatter_reduce(0, batch.batch_idx.long(), atom.norm(dim=-1), reduce="amax", include_self=True)
            cell_max = cell.norm(dim=-1).amax(-1)
        else:
            atom_max = physical
            cell_max = torch.zeros_like(physical)
            error = torch.zeros_like(physical)
        batch.production_atom_fmax.copy_(atom_max[:, None])
        batch.production_cell_fmax.copy_(cell_max[:, None])
        batch.production_fmax.copy_(torch.maximum(atom_max, cell_max)[:, None])
        batch.frechet_log_error.copy_(error[:, None])
        unseen = ~batch.initial_observed.view(-1)
        nodes = unseen[batch.batch_idx.long()]
        batch.initial_raw_forces[nodes] = batch.forces[nodes]
        batch.initial_raw_stress[unseen] = batch.stress[unseen]
        batch.initial_energy[unseen] = batch.energy[unseen]
        batch.initial_observed.fill_(True)


class Dataset:
    """A full unchanged queue; one independent graph per admission."""

    def __init__(self, samples, ids):
        self.samples, self.ids = samples, ids

    def __len__(self):
        return len(self.ids)

    def get_metadata(self, index):
        return self.samples[self.ids[index]].num_nodes, 0

    def __getitem__(self, index):
        return self.samples[self.ids[index]].clone(), {}


def load_inputs(path: Path, device: str = "cuda"):
    """Load Cartesian starts and immutable symmetry records without chart compilers."""
    payload = torch.load(io.BytesIO(gzip.decompress(path.read_bytes())), weights_only=True)
    samples, records = {}, []
    for item in payload["candidates"]:
        key = int(item["candidate_id"])
        data = AtomicData(positions=item["positions"].to(device).clone(), atomic_numbers=item["atomic_numbers"].to(device).clone(), cell=item["cell"].to(device)[None].clone(), pbc=torch.ones(1, 3, dtype=torch.bool, device=device))
        for name in ("candidate_id", "symmetry_id"):
            data.add_system_property(name, torch.tensor([[key]], device=device))
        samples[key] = data
        records.append(SymmetryRecord(key, item["atomic_numbers"].clone(), item["rotations"].clone(), item["translations"].clone(), item["maps"].clone()))
    return samples, records


def fresh_samples(initial, ids, endpoints=None):
    """Reset geometry/state from Cartesian starts or the previous stage endpoints."""
    samples = {}
    for key in ids:
        data = initial[key].clone()
        if endpoints is not None:
            data.positions.copy_(endpoints[key].positions)
            data.cell.copy_(endpoints[key].cell)
        samples[key] = data
    return samples


def make(model, records, ids, samples, variable, constrained, capacity, sink, max_steps, device="cuda"):
    """Configure unchanged PR2 FIRE2 with the measured admission and stopping hooks."""
    neighbors = VariableCellNeighborListHook(model.model_config.neighbor_config, skin=0, method="naive", stage=DynamicsStage.BEFORE_COMPUTE)
    hooks = [neighbors, Residuals(records, variable, constrained)]
    options = dict(convergence_hook=Stop(), device_type=device, sinks=[sink], max_steps_per_system=max_steps, hooks=hooks,
                   sampler=SizeAwareSampler(Dataset(samples, ids), max_batch_size=capacity, max_atoms=capacity * max(samples[i].num_nodes for i in ids), shuffle=False, bin_width=1_000_000))
    if constrained:
        hooks.append(FixSymmetryHook(records, adjust_cell=variable))
    optimizer = (FIRE2VariableCell if variable else FIRE2)(model, dt=0.02, tmax=0.08, tmin=0.001, delaystep=10, maxstep=0.05, **options)

    def admit(batch):
        schema(batch)
        if constrained:
            constraint = hooks[-1]
            constraint._bind_batch(batch)
            constraint._layout_signature = constraint._current_layout_signature(batch)
        optimizer._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
        optimizer.compute(batch)
        optimizer._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)

    optimizer.init_fn = admit
    return optimizer


def campaign(model, monitor, records, ids, samples, variable=True, constrained=True, capacity=64, max_steps=500):
    """Measure the whole queue, including its original refill and tail."""
    sink = HostMemory(capacity=len(ids))
    monitor.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer = make(model, records, ids, samples, variable, constrained, capacity, sink, max_steps)
    monitor.bind(optimizer)
    optimizer.run(n_steps=(max_steps + 1) * len(ids))
    if optimizer.active_batch is not None:
        raise RuntimeError("Queue did not drain")
    result = sink.read()
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    profiling, trace = monitor.finish(capacity)
    actual_ids = result.candidate_id.view(-1).cpu().tolist()
    if sorted(actual_ids) != sorted(ids):
        raise RuntimeError("Coverage mismatch")
    for key in ("positions", "cell", "energy", "production_fmax"):
        if not bool(torch.isfinite(getattr(result, key)).all()):
            raise RuntimeError(f"Non-finite {key}")
    if float(result.frechet_log_error.max()) >= 1e-8:
        raise RuntimeError("Frechet log quadrature left validated accuracy range")
    converged = result.inflight_termination.view(-1) == 1
    if not bool((result.production_fmax.view(-1)[converged] < 0.02).all()):
        raise RuntimeError("False convergence")
    metrics = dict(status="PASS", wall_s=wall, completed=len(ids), converged=int(converged.sum()), budget_exhausted=int((~converged).sum()), completed_per_s=len(ids) / wall, qualified_per_s=int(converged.sum()) / wall,
                   structure_steps=int(result.inflight_step_count.sum()), method="A", constrained=constrained, variable_cell=variable, capacity=capacity, max_steps=max_steps,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, frechet_log_error_max=float(result.frechet_log_error.max()), **profiling)
    return result, metrics, trace
