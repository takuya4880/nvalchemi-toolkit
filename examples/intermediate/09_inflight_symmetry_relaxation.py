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
Inflight Symmetry-Constrained Relaxation
========================================

This example compares two campaigns of TensorNet-powered
:class:`~nvalchemi.dynamics.FIRE2VariableCell` relaxations.  Both campaigns
contain diamond Si, BCC Fe, and FCC Al crystals with different atom counts and
initial strains, use the same force-and-stress convergence criteria, and
preserve each structure's initial space group.

The fixed-batch campaign leaves a completed slot idle until every structure in
that batch has finished.  The inflight campaign uses
:class:`~nvalchemi.dynamics.SizeAwareSampler` to replace completed structures
at step boundaries.  Its primary metrics are completed and converged
structures per second, rather than steady-state structure-steps per second.

ASE/spglib preprocessing is intentionally outside the timed region.  Each
input receives a stable ``symmetry_id`` and a precomputed
:class:`~nvalchemi.dynamics.hooks.SymmetryRecord`; the runtime symmetry hook
only rebinds the records for the current graph layout.  Result rows retain a
separate ``structure_id`` so reordered and replaced structures can be matched
to their original inputs.

The default workload can be adjusted with these environment variables:

* ``NVALCHEMI_INFLIGHT_CYCLES`` (default ``6``)
* ``NVALCHEMI_INFLIGHT_MAX_ATOMS`` (default ``96``)
* ``NVALCHEMI_INFLIGHT_MAX_BATCH_SIZE`` (default ``4``)
* ``NVALCHEMI_INFLIGHT_MAX_STEPS`` (default ``100``)
* ``NVALCHEMI_INFLIGHT_REFILL_FREQUENCY`` (default ``1``)
* ``NVALCHEMI_INFLIGHT_WARMUP_STEPS`` (default ``2``)
* ``NVALCHEMI_INFLIGHT_OUTPUT`` (default ``inflight_symmetry_results.json``)

TensorNet support requires ``pip install 'matgl>=2.1.2'``.  The first run
downloads the pretrained model selected by ``NVALCHEMI_TENSORNET_MODEL``
(default ``TensorNet-PES-MatPES-PBE-2025.2``).  Sphinx Gallery sets
``NVALCHEMI_SPHINX_BUILD=1`` to skip model loading and simulation.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from ase import Atoms
from ase.build import bulk
from ase.spacegroup.symmetrize import check_symmetry

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    ConvergenceHook,
    DynamicsStage,
    FIRE2VariableCell,
    HostMemory,
    InflightTerminationReason,
    SizeAwareSampler,
)
from nvalchemi.dynamics.hooks import FixSymmetryHook, SymmetryRecord, prepare_symmetry

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin


DTYPE = torch.float32
SYMPREC = 1.0e-3
FORCE_THRESHOLD = 5.0e-3
STRESS_THRESHOLD = 5.0e-4


@dataclass(frozen=True)
class CrystalSpec:
    """Definition of one crystal family."""

    label: str
    formula: str
    crystalstructure: str
    lattice_constant: float


@dataclass(frozen=True)
class PreparedStructure:
    """A preprocessed campaign input and its expected space group."""

    data: AtomicData
    record: SymmetryRecord
    label: str
    initial_space_group: tuple[int, str]


@dataclass(frozen=True)
class CampaignResult:
    """Wall time and structures collected by one campaign."""

    name: str
    elapsed: float
    batch: Batch


CRYSTALS = (
    CrystalSpec("diamond Si", "Si", "diamond", 5.43),
    CrystalSpec("BCC Fe", "Fe", "bcc", 2.87),
    CrystalSpec("FCC Al", "Al", "fcc", 4.05),
)
# Isotropic repeats keep the complete point group representable in the chosen
# supercell basis. An anisotropic repeat such as (2, 2, 1) can expose a smaller
# operation set even when spglib recognizes a higher-symmetry primitive cell.
REPEATS = ((1, 1, 1), (2, 2, 2))
STRAINS = (1.02, 1.035, 1.05, 1.065)


def load_tensornet(device: torch.device) -> BaseModelMixin:
    """Load the configured pretrained TensorNet potential."""

    import matgl
    from matgl.ext.alchmtk import TensorNetWrapper

    potential = matgl.load_model(
        os.getenv("NVALCHEMI_TENSORNET_MODEL", "TensorNet-PES-MatPES-PBE-2025.2")
    )
    model = TensorNetWrapper.from_potential(potential).to(device)
    model.eval()
    return model


def make_atoms(spec: CrystalSpec, repeat: tuple[int, int, int], strain: float) -> Atoms:
    """Build one fully periodic, isotropically strained crystal."""

    atoms = bulk(
        spec.formula,
        crystalstructure=spec.crystalstructure,
        a=spec.lattice_constant,
        cubic=True,
    ).repeat(repeat)
    atoms.set_cell(atoms.cell * strain, scale_atoms=True)
    return atoms


def to_atoms(data: AtomicData) -> Atoms:
    """Convert one atomic graph to an ASE structure."""

    return Atoms(
        numbers=data.atomic_numbers.detach().cpu().numpy(),
        positions=data.positions.detach().cpu().numpy(),
        cell=data.cell.reshape(3, 3).detach().cpu().numpy(),
        pbc=data.pbc.reshape(3).detach().cpu().numpy(),
    )


def space_group(data: AtomicData) -> tuple[int, str]:
    """Return the international space-group number and symbol."""

    dataset = check_symmetry(to_atoms(data), symprec=SYMPREC, verbose=False)
    return int(dataset.number), str(dataset.international)


def prepare_campaign(cycles: int) -> list[PreparedStructure]:
    """Create varied structures and perform all CPU symmetry preprocessing."""

    prepared: list[PreparedStructure] = []
    structure_id = 0
    for cycle in range(cycles):
        for family_index, spec in enumerate(CRYSTALS):
            variant = cycle * len(CRYSTALS) + family_index
            atoms = make_atoms(
                spec,
                REPEATS[variant % len(REPEATS)],
                STRAINS[variant % len(STRAINS)],
            )
            data = AtomicData.from_atoms(atoms, device="cpu", dtype=DTYPE)
            refined, record = prepare_symmetry(
                data,
                symmetry_id=structure_id,
                symprec=SYMPREC,
            )
            refined.add_system_property(
                "structure_id", torch.tensor([[structure_id]], dtype=torch.long)
            )
            prepared.append(
                PreparedStructure(
                    data=refined,
                    record=record,
                    label=f"{spec.label} repeat={REPEATS[variant % len(REPEATS)]}",
                    initial_space_group=space_group(refined),
                )
            )
            structure_id += 1
    return prepared


class PreparedDataset:
    """Minimal clone-on-read dataset for :class:`SizeAwareSampler`."""

    def __init__(self, structures: Sequence[PreparedStructure]) -> None:
        self.structures = list(structures)

    def __len__(self) -> int:
        """Return the number of structures."""

        return len(self.structures)

    def __getitem__(self, index: int) -> tuple[AtomicData, dict[str, Any]]:
        """Return an independent structure and empty loader metadata."""

        return self.structures[index].data.clone(), {}

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return atom and pre-neighbor-list edge counts."""

        return self.structures[index].data.num_nodes, 0


def initialize_runtime_fields(batch: Batch) -> None:
    """Initialize fields required by FIRE2VariableCell and TensorNet."""

    batch["forces"] = torch.zeros(batch.num_nodes, 3, device=batch.device, dtype=DTYPE)
    batch["energy"] = torch.zeros(batch.num_graphs, 1, device=batch.device, dtype=DTYPE)
    batch["stress"] = torch.zeros(
        batch.num_graphs, 3, 3, device=batch.device, dtype=DTYPE
    )
    batch["velocities"] = torch.zeros(
        batch.num_nodes, 3, device=batch.device, dtype=DTYPE
    )


def convergence_hook() -> ConvergenceHook:
    """Build the shared force-and-stress stopping criteria."""

    return ConvergenceHook(
        criteria=[
            {
                "key": "forces",
                "threshold": FORCE_THRESHOLD,
                "reduce_op": "norm",
                "reduce_dims": -1,
            },
            {
                "key": "stress",
                "threshold": STRESS_THRESHOLD,
                "reduce_op": "norm",
                "reduce_dims": [-2, -1],
            },
        ]
    )


def make_optimizer(
    model: BaseModelMixin,
    records: Sequence[SymmetryRecord],
    **kwargs: Any,
) -> FIRE2VariableCell:
    """Construct a TensorNet FIRE2 optimizer with dynamic symmetry binding."""

    optimizer = FIRE2VariableCell(
        model=model,
        dt=0.02,
        delaystep=10,
        tmax=0.08,
        maxstep=0.05,
        hooks=[FixSymmetryHook(records=records)],
        convergence_hook=convergence_hook(),
        **kwargs,
    )
    for hook in model.make_neighbor_hooks():
        optimizer.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
    return optimizer


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA when measuring wall-clock time."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fixed_groups(
    structures: Sequence[PreparedStructure], max_atoms: int, max_batch_size: int
) -> list[list[PreparedStructure]]:
    """Pack sequential fixed batches under the inflight capacity limits."""

    groups: list[list[PreparedStructure]] = []
    current: list[PreparedStructure] = []
    current_atoms = 0
    for structure in structures:
        num_atoms = structure.data.num_nodes
        if num_atoms > max_atoms:
            raise ValueError(
                f"Structure {structure.label} has {num_atoms} atoms, exceeding "
                f"max_atoms={max_atoms}."
            )
        if current and (
            current_atoms + num_atoms > max_atoms or len(current) >= max_batch_size
        ):
            groups.append(current)
            current = []
            current_atoms = 0
        current.append(structure)
        current_atoms += num_atoms
    if current:
        groups.append(current)
    return groups


def run_fixed_campaign(
    structures: Sequence[PreparedStructure],
    model: BaseModelMixin,
    device: torch.device,
    max_atoms: int,
    max_batch_size: int,
    max_steps: int,
) -> CampaignResult:
    """Run static batches, retaining finished structures in occupied slots."""

    sink = HostMemory(capacity=len(structures))
    jobs: list[tuple[Batch, FIRE2VariableCell]] = []
    records = [item.record for item in structures]
    for group in fixed_groups(structures, max_atoms, max_batch_size):
        batch = Batch.from_data_list(
            [item.data.clone() for item in group], device=device
        )
        initialize_runtime_fields(batch)
        batch["status"] = torch.zeros(
            batch.num_graphs, 1, dtype=torch.long, device=device
        )
        batch["inflight_step_count"] = torch.zeros_like(batch.status)
        batch["inflight_termination"] = torch.zeros_like(batch.status)
        jobs.append((batch, make_optimizer(model, records)))

    synchronize(device)
    start = time.perf_counter()
    for batch, optimizer in jobs:
        for _ in range(max_steps):
            active = batch.status.view(-1) < optimizer.exit_status
            batch, converged = optimizer.step(batch)
            batch.inflight_step_count.view(-1)[active] += 1
            converged_mask = torch.zeros_like(active)
            if converged is not None:
                converged = converged[active[converged]]
                converged_mask[converged] = True
                batch.status.view(-1)[converged] = optimizer.exit_status
                batch.inflight_termination.view(-1)[converged] = (
                    InflightTerminationReason.CONVERGED
                )
            reached_limit = (
                active
                & ~converged_mask
                & (batch.inflight_step_count.view(-1) >= max_steps)
            )
            batch.status.view(-1)[reached_limit] = optimizer.exit_status
            batch.inflight_termination.view(-1)[reached_limit] = (
                InflightTerminationReason.MAX_STEPS
            )
            if bool((batch.status >= optimizer.exit_status).all().item()):
                break
        sink.write(batch)
    synchronize(device)
    return CampaignResult("fixed", time.perf_counter() - start, sink.read())


def run_inflight_campaign(
    structures: Sequence[PreparedStructure],
    model: BaseModelMixin,
    device: torch.device,
    max_atoms: int,
    max_batch_size: int,
    max_steps: int,
    refill_frequency: int,
) -> CampaignResult:
    """Run one sampler-driven campaign with step-boundary replacement."""

    dataset = PreparedDataset(structures)
    sampler = SizeAwareSampler(
        dataset,
        max_atoms=max_atoms,
        max_edges=None,
        max_batch_size=max_batch_size,
        shuffle=False,
    )
    sink = HostMemory(capacity=len(structures))
    optimizer = make_optimizer(
        model,
        [item.record for item in structures],
        sampler=sampler,
        sinks=[sink],
        init_fn=initialize_runtime_fields,
        max_steps_per_system=max_steps,
        refill_frequency=refill_frequency,
        device_type=device.type,
    )

    # In the worst case, one full batch drains every max_steps iterations.
    global_step_limit = max_steps * len(structures)
    synchronize(device)
    start = time.perf_counter()
    remainder = optimizer.run(n_steps=global_step_limit)
    synchronize(device)
    elapsed = time.perf_counter() - start
    if remainder is not None:
        raise RuntimeError(
            "Inflight campaign reached its global safety limit before all "
            "structures were collected."
        )
    return CampaignResult("inflight", elapsed, sink.read())


def validate_results(
    result: CampaignResult, structures: Sequence[PreparedStructure]
) -> tuple[int, int]:
    """Check IDs and space groups, returning completed and converged counts."""

    rows = result.batch.to_data_list()
    by_id = {int(row.structure_id.item()): row for row in rows}
    expected_ids = set(range(len(structures)))
    if set(by_id) != expected_ids or len(rows) != len(structures):
        raise AssertionError(
            f"{result.name}: expected each structure exactly once; got IDs "
            f"{sorted(by_id)}."
        )

    for structure_id, expected in enumerate(structures):
        observed = space_group(by_id[structure_id])
        if observed != expected.initial_space_group:
            raise AssertionError(
                f"{result.name} structure {structure_id} ({expected.label}): "
                f"space group changed from {expected.initial_space_group} to "
                f"{observed}."
            )
    reasons = result.batch.inflight_termination.view(-1)
    converged = int((reasons == InflightTerminationReason.CONVERGED).sum().item())
    return len(rows), converged


def compare_final_structures(
    fixed: CampaignResult, inflight: CampaignResult
) -> tuple[float, float, float]:
    """Compare fixed and inflight final states by stable structure ID."""

    fixed_by_id = {
        int(row.structure_id.item()): row for row in fixed.batch.to_data_list()
    }
    inflight_by_id = {
        int(row.structure_id.item()): row for row in inflight.batch.to_data_list()
    }
    max_position_delta = 0.0
    max_cell_delta = 0.0
    max_energy_delta_per_atom = 0.0
    for structure_id, fixed_row in fixed_by_id.items():
        inflight_row = inflight_by_id[structure_id]
        max_position_delta = max(
            max_position_delta,
            float((fixed_row.positions - inflight_row.positions).abs().max()),
        )
        max_cell_delta = max(
            max_cell_delta,
            float((fixed_row.cell - inflight_row.cell).abs().max()),
        )
        energy_delta = float((fixed_row.energy - inflight_row.energy).abs().max())
        max_energy_delta_per_atom = max(
            max_energy_delta_per_atom, energy_delta / fixed_row.num_nodes
        )
    if max_position_delta > 5.0e-3 or max_cell_delta > 5.0e-3:
        raise AssertionError(
            "Fixed and inflight final geometries differ beyond 5e-3 Å: "
            f"positions={max_position_delta:.3e}, cell={max_cell_delta:.3e}."
        )
    if max_energy_delta_per_atom > 1.0e-3:
        raise AssertionError(
            "Fixed and inflight energies differ beyond 1e-3 eV/atom: "
            f"{max_energy_delta_per_atom:.3e}."
        )
    return max_position_delta, max_cell_delta, max_energy_delta_per_atom


def write_results(
    path: Path,
    results: Sequence[CampaignResult],
    structures: Sequence[PreparedStructure],
    configuration: dict[str, int | float | str],
    device: torch.device,
) -> None:
    """Write reproducible campaign configuration and per-structure rows."""

    payload: dict[str, Any] = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch_version": torch.__version__,
        "configuration": configuration,
        "campaigns": {},
    }
    for result in results:
        rows = []
        for data in result.batch.to_data_list():
            structure_id = int(data.structure_id.item())
            reason = InflightTerminationReason(int(data.inflight_termination.item()))
            rows.append(
                {
                    "structure_id": structure_id,
                    "label": structures[structure_id].label,
                    "num_atoms": data.num_nodes,
                    "steps": int(data.inflight_step_count.item()),
                    "termination": reason.name.lower(),
                    "initial_space_group": structures[structure_id].initial_space_group,
                    "final_space_group": space_group(data),
                    "energy_eV": float(data.energy.item()),
                }
            )
        payload["campaigns"][result.name] = {
            "elapsed_seconds": result.elapsed,
            "structures_per_second": len(rows) / result.elapsed,
            "rows": sorted(rows, key=lambda row: row["structure_id"]),
        }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run and compare fixed and inflight symmetry-relaxation campaigns."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cycles = int(os.getenv("NVALCHEMI_INFLIGHT_CYCLES", "6"))
    max_atoms = int(os.getenv("NVALCHEMI_INFLIGHT_MAX_ATOMS", "96"))
    max_batch_size = int(os.getenv("NVALCHEMI_INFLIGHT_MAX_BATCH_SIZE", "4"))
    max_steps = int(os.getenv("NVALCHEMI_INFLIGHT_MAX_STEPS", "100"))
    refill_frequency = int(os.getenv("NVALCHEMI_INFLIGHT_REFILL_FREQUENCY", "1"))
    warmup_steps = int(os.getenv("NVALCHEMI_INFLIGHT_WARMUP_STEPS", "2"))
    output_path = Path(
        os.getenv("NVALCHEMI_INFLIGHT_OUTPUT", "inflight_symmetry_results.json")
    )
    if (
        min(
            cycles,
            max_atoms,
            max_batch_size,
            max_steps,
            refill_frequency,
            warmup_steps,
        )
        < 1
    ):
        raise ValueError("All inflight example configuration values must be positive.")

    structures = prepare_campaign(cycles)
    print(
        f"Prepared {len(structures)} structures with "
        f"{min(item.data.num_nodes for item in structures)}-"
        f"{max(item.data.num_nodes for item in structures)} atoms on {device}."
    )
    model = load_tensornet(device)

    # Exercise both control-flow paths and representative graph shapes before
    # timing so one mode does not pay TensorNet/Warp lazy initialization on
    # behalf of the other.
    print(f"Warming both campaign modes for {warmup_steps} step(s) per structure...")
    run_fixed_campaign(
        structures, model, device, max_atoms, max_batch_size, warmup_steps
    )
    run_inflight_campaign(
        structures,
        model,
        device,
        max_atoms,
        max_batch_size,
        warmup_steps,
        refill_frequency,
    )

    fixed = run_fixed_campaign(
        structures, model, device, max_atoms, max_batch_size, max_steps
    )
    inflight = run_inflight_campaign(
        structures,
        model,
        device,
        max_atoms,
        max_batch_size,
        max_steps,
        refill_frequency,
    )

    print("\nCampaign throughput (model loading and symmetry preparation excluded)")
    print("mode       elapsed (s)  completed  converged  completed/s  converged/s")
    for result in (fixed, inflight):
        completed, converged = validate_results(result, structures)
        step_counts = sorted(result.batch.inflight_step_count.view(-1).tolist())
        median_steps = step_counts[len(step_counts) // 2]
        print(
            f"{result.name:<10} {result.elapsed:11.4f}  {completed:9d}  "
            f"{converged:9d}  {completed / result.elapsed:11.3f}  "
            f"{converged / result.elapsed:11.3f}"
        )
        print(
            "  per-structure steps min/median/max: "
            f"{step_counts[0]}/{median_steps}/{step_counts[-1]}"
        )
    position_delta, cell_delta, energy_delta = compare_final_structures(fixed, inflight)
    print(f"Inflight elapsed-time speedup: {fixed.elapsed / inflight.elapsed:.2f}x")
    print(
        "Fixed/inflight maximum differences: "
        f"positions={position_delta:.3e} Å, cell={cell_delta:.3e} Å, "
        f"energy={energy_delta:.3e} eV/atom"
    )
    configuration = {
        "cycles": cycles,
        "max_atoms": max_atoms,
        "max_batch_size": max_batch_size,
        "max_steps_per_system": max_steps,
        "refill_frequency": refill_frequency,
        "warmup_steps": warmup_steps,
        "force_threshold_eV_per_A": FORCE_THRESHOLD,
        "stress_threshold_eV_per_A3": STRESS_THRESHOLD,
        "symprec_A": SYMPREC,
        "model": os.getenv(
            "NVALCHEMI_TENSORNET_MODEL", "TensorNet-PES-MatPES-PBE-2025.2"
        ),
    }
    write_results(output_path, (fixed, inflight), structures, configuration, device)
    print(f"Wrote per-structure results to {output_path}.")


if __name__ == "__main__" and os.environ.get("NVALCHEMI_SPHINX_BUILD") != "1":
    main()
