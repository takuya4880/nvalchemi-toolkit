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
Batched Symmetry-Constrained FIRE2 Relaxation with TensorNet
============================================================

This example relaxes periodic crystals while preserving each graph's initial
space-group symmetry. Diamond Si, BCC Fe, and FCC Al supercells are
isotropically strained, converted to :class:`~nvalchemi.data.AtomicData`, and
relaxed together in one
:class:`~nvalchemi.data.Batch` with
:class:`~nvalchemi.dynamics.FIRE2VariableCell`.

The :class:`~nvalchemi.dynamics.hooks.FixSymmetryHook` discovers the symmetry
operations once with ASE/spglib and projects positions, forces, cells, and
stress on the batch device during every optimization step. A pretrained MatGL
TensorNet potential provides energy, forces, and stress. After relaxation, the
example asserts that the space-group number and symbol of every graph are
unchanged.

The final section runs a small steady-state throughput check for batches of
identical diamond-Si crystals. It uses fixed step counts, untimed warm-up steps,
CUDA synchronization, and the median of repeated measurements.  The printed
speedup uses ``batch=1 median time × batch size`` as its serial-time estimate;
it is a lightweight demonstration rather than a formal benchmark, and no
minimum speedup is asserted.

The default workload can be adjusted with these environment variables:

* ``NVALCHEMI_SYMMETRY_RELAX_STEPS`` (default ``40``)
* ``NVALCHEMI_SYMMETRY_BENCH_STEPS`` (default ``100``)
* ``NVALCHEMI_SYMMETRY_BENCH_WARMUP`` (default ``1``)
* ``NVALCHEMI_SYMMETRY_BENCH_REPEATS`` (default ``2``)
* ``NVALCHEMI_SYMMETRY_BATCH_SIZES`` (default ``1,2,4,8,16,32``)

TensorNet support requires ``pip install 'matgl>=2.1.2'``. The first run
downloads the pretrained model selected by ``NVALCHEMI_TENSORNET_MODEL``
(default ``TensorNet-PES-MatPES-PBE-2025.2``) and therefore needs network
access. Sphinx Gallery sets ``NVALCHEMI_SPHINX_BUILD=1`` to skip the download
and simulation while still importing this module.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from ase import Atoms
from ase.build import bulk
from ase.spacegroup.symmetrize import check_symmetry

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ConvergenceHook, DynamicsStage, FIRE2VariableCell
from nvalchemi.dynamics.hooks import FixSymmetryHook

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

# %%
# Runtime configuration
# ---------------------
DTYPE = torch.float32
SYMPREC = 1.0e-3

# %%
# TensorNet model with stress
# ---------------------------


def load_tensornet(device: torch.device) -> BaseModelMixin:
    """Load and wrap the configured pretrained TensorNet potential."""

    import matgl
    from matgl.ext.alchmtk import TensorNetWrapper

    potential = matgl.load_model(
        os.getenv("NVALCHEMI_TENSORNET_MODEL", "TensorNet-PES-MatPES-PBE-2025.2")
    )
    model = TensorNetWrapper.from_potential(potential).to(device)
    model.eval()
    return model


@dataclass(frozen=True)
class CrystalSpec:
    """Definition of one cubic crystal.

    Parameters
    ----------
    label : str
        Human-readable structure label.
    formula : str
        Chemical formula passed to ASE.
    crystalstructure : str
        ASE ``bulk`` crystal-structure name.
    lattice_constant : float
        Approximate unstrained lattice constant in Å.
    expected_space_group : int
        Expected international space-group number.
    """

    label: str
    formula: str
    crystalstructure: str
    lattice_constant: float
    expected_space_group: int


CRYSTALS = (
    CrystalSpec("diamond Si", "Si", "diamond", 5.43, 227),
    CrystalSpec("BCC Fe", "Fe", "bcc", 2.87, 229),
    CrystalSpec("FCC Al", "Al", "fcc", 4.05, 225),
)
ISOTROPIC_STRAIN = 1.04
SUPERCELL_REPEAT = (2, 2, 2)


def make_crystal(spec: CrystalSpec) -> Atoms:
    """Build an isotropically expanded cubic supercell.

    Parameters
    ----------
    spec : CrystalSpec
        Crystal definition.

    Returns
    -------
    ase.Atoms
        Fully periodic, strained supercell.
    """

    atoms = bulk(
        spec.formula,
        crystalstructure=spec.crystalstructure,
        a=spec.lattice_constant,
        cubic=True,
    ).repeat(SUPERCELL_REPEAT)
    atoms.set_cell(atoms.cell * ISOTROPIC_STRAIN, scale_atoms=True)
    return atoms


def atoms_to_data(atoms: Atoms, device: torch.device) -> AtomicData:
    """Convert ASE atoms and allocate the fields required by FIRE2.

    Parameters
    ----------
    atoms : ase.Atoms
        Periodic input structure.
    device : torch.device
        Destination device.

    Returns
    -------
    AtomicData
        Structure on the selected device with zeroed dynamics fields.
    """

    data = AtomicData.from_atoms(atoms, device=device, dtype=DTYPE)
    data.forces = torch.zeros(data.num_nodes, 3, device=device, dtype=DTYPE)
    data.energy = torch.zeros(1, 1, device=device, dtype=DTYPE)
    data.add_node_property(
        "velocities",
        torch.zeros(data.num_nodes, 3, device=device, dtype=DTYPE),
    )
    return data


def make_batch(structures: list[Atoms], device: torch.device) -> Batch:
    """Convert a list of ASE structures into a stress-enabled batch.

    Parameters
    ----------
    structures : list[ase.Atoms]
        Structures to batch.
    device : torch.device
        Destination device.

    Returns
    -------
    Batch
        Batched structures with a graph-level stress placeholder.
    """

    batch = Batch.from_data_list([atoms_to_data(atoms, device) for atoms in structures])
    batch["stress"] = torch.zeros(batch.num_graphs, 3, 3, device=device, dtype=DTYPE)
    return batch


def data_to_atoms(data: AtomicData) -> Atoms:
    """Convert one graph back to ASE for symmetry analysis.

    Parameters
    ----------
    data : AtomicData
        One graph from a batch.

    Returns
    -------
    ase.Atoms
        CPU ASE representation.
    """

    return Atoms(
        numbers=data.atomic_numbers.detach().cpu().numpy(),
        positions=data.positions.detach().cpu().numpy(),
        cell=data.cell.squeeze(0).detach().cpu().numpy(),
        pbc=data.pbc.squeeze(0).detach().cpu().numpy(),
    )


def symmetry_id(atoms: Atoms) -> tuple[int, str]:
    """Return the international space-group number and symbol.

    Parameters
    ----------
    atoms : ase.Atoms
        Periodic structure to classify with ASE/spglib.

    Returns
    -------
    tuple[int, str]
        Space-group number and international symbol.
    """

    dataset = check_symmetry(atoms, symprec=SYMPREC, verbose=False)
    return int(dataset.number), str(dataset.international)


def batch_symmetry_ids(batch: Batch) -> list[tuple[int, str]]:
    """Return symmetry identifiers for every graph in a batch.

    Parameters
    ----------
    batch : Batch
        Structures to analyze.

    Returns
    -------
    list[tuple[int, str]]
        One ``(number, symbol)`` pair per graph.
    """

    return [symmetry_id(data_to_atoms(data)) for data in batch.to_data_list()]


def graph_fmax(batch: Batch) -> torch.Tensor:
    """Compute maximum atomic force norm for each graph.

    Parameters
    ----------
    batch : Batch
        Batch with per-atom forces.

    Returns
    -------
    torch.Tensor
        Per-graph fmax tensor with shape ``[num_graphs]``.
    """

    force_norm = torch.linalg.vector_norm(batch.forces, dim=-1)
    result = torch.zeros(batch.num_graphs, device=batch.device, dtype=force_norm.dtype)
    result.scatter_reduce_(
        0, batch.batch_idx.long(), force_norm, reduce="amax", include_self=True
    )
    return result


def make_optimizer(
    batch: Batch,
    model: BaseModelMixin,
    *,
    convergence_hook: ConvergenceHook | None = None,
) -> FIRE2VariableCell:
    """Create a symmetry-constrained variable-cell FIRE2 optimizer.

    Parameters
    ----------
    batch : Batch
        The exact graph layout that will be optimized.  ``FixSymmetryHook``
        refines it in-place while discovering symmetry operations.
    model : BaseModelMixin
        TensorNet model wrapper.
    convergence_hook : ConvergenceHook | None, optional
        Optional relaxation stopping criteria.  Omit for fixed-step timing.

    Returns
    -------
    FIRE2VariableCell
        Configured optimizer with neighbor-list and symmetry hooks.
    """

    symmetry_hook = FixSymmetryHook(batch, symprec=SYMPREC)
    optimizer = FIRE2VariableCell(
        model=model,
        dt=0.02,
        delaystep=10,
        tmax=0.08,
        maxstep=0.05,
        hooks=[symmetry_hook],
        convergence_hook=convergence_hook,
    )
    for hook in model.make_neighbor_hooks():
        optimizer.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
    return optimizer


# %%
# Relax three different symmetries in one batch
# ------------------------------------------------
# The lattice constants are uniformly expanded by 4%, which keeps every
# symmetry operation intact while creating nonzero cell stress. Atomic fmax
# alone is insufficient for ideal crystals, so convergence requires both fmax
# and the Frobenius norm of stress.


def run_relaxation(
    model: BaseModelMixin, device: torch.device, relax_steps: int
) -> None:
    """Relax the mixed-symmetry batch and validate its space groups."""

    initial_structures = [make_crystal(spec) for spec in CRYSTALS]
    relax_batch = make_batch(initial_structures, device)
    relaxation = make_optimizer(
        relax_batch,
        model,
        convergence_hook=ConvergenceHook(
            criteria=[
                {
                    "key": "forces",
                    "threshold": 5.0e-3,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                },
                {
                    "key": "stress",
                    "threshold": 5.0e-4,
                    "reduce_op": "norm",
                    "reduce_dims": [-2, -1],
                },
            ]
        ),
    )
    initial_symmetry = batch_symmetry_ids(relax_batch)
    initial_volumes = torch.linalg.det(relax_batch.cell).abs().detach().cpu()
    for spec, found in zip(CRYSTALS, initial_symmetry, strict=True):
        if found[0] != spec.expected_space_group:
            raise AssertionError(
                f"{spec.label}: expected space group "
                f"{spec.expected_space_group}, got {found}"
            )

    relax_batch = relaxation.run(relax_batch, n_steps=relax_steps)
    final_symmetry = batch_symmetry_ids(relax_batch)
    final_volumes = torch.linalg.det(relax_batch.cell).abs().detach().cpu()
    final_fmax = graph_fmax(relax_batch).detach().cpu()
    final_stress = torch.linalg.matrix_norm(relax_batch.stress).detach().cpu()

    print("\nSymmetry-constrained FIRE2 variable-cell relaxation")
    print("structure       space group       volume (Å^3)       fmax (eV/Å)  |stress|")
    failures: list[str] = []
    for index, spec in enumerate(CRYSTALS):
        before = initial_symmetry[index]
        after = final_symmetry[index]
        if after != before:
            failures.append(f"{spec.label}: symmetry changed from {before} to {after}")
        volume_change = float(final_volumes[index] - initial_volumes[index])
        if abs(volume_change) <= 1.0e-5 * float(initial_volumes[index]):
            failures.append(f"{spec.label}: cell volume did not change")
        print(
            f"{spec.label:>13}  {before[0]:3d} {before[1]:<10} -> "
            f"{after[0]:3d} {after[1]:<10}  "
            f"{initial_volumes[index]:9.3f} -> {final_volumes[index]:9.3f}  "
            f"{final_fmax[index]:12.4e}  {final_stress[index]:8.3e}"
        )
    if failures:
        raise AssertionError("; ".join(failures))
    print(f"FIRE2 steps completed: {relaxation.step_count}/{relax_steps}")


# %%
# Lightweight batched-throughput scaling
# --------------------------------------
# Setup, symmetry discovery, and optimizer construction are outside the timed
# region.  Each repetition first executes untimed steps on the same batch and
# optimizer, which warms neighbor-list buffers and CUDA kernels.  The timed run
# then measures a fixed number of additional FIRE2 steps. All final diamond-Si
# graphs are checked against their initial space group.


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA for accurate wall-clock timing when needed."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_batch(
    batch_size: int,
    model: BaseModelMixin,
    device: torch.device,
    *,
    timed_steps: int,
    warmup_steps: int,
    repeats: int,
) -> float:
    """Return median fixed-step runtime for a batch of diamond-Si crystals.

    Parameters
    ----------
    batch_size : int
        Number of identical diamond-Si graphs in the batch.
    model : BaseModelMixin
        TensorNet model wrapper.
    device : torch.device
        Inference device.
    timed_steps : int
        Number of timed FIRE2 steps.
    warmup_steps : int
        Number of untimed warm-up steps.
    repeats : int
        Number of timing samples.

    Returns
    -------
    float
        Median elapsed time in seconds.
    """

    elapsed_samples: list[float] = []
    for _ in range(repeats):
        diamond = make_crystal(CRYSTALS[0])
        batch = make_batch([diamond.copy() for _ in range(batch_size)], device)
        optimizer = make_optimizer(batch, model)
        reference_symmetry = batch_symmetry_ids(batch)

        if warmup_steps:
            batch = optimizer.run(batch, n_steps=warmup_steps)
        synchronize(device)
        start = time.perf_counter()
        batch = optimizer.run(batch, n_steps=timed_steps)
        synchronize(device)
        elapsed_samples.append(time.perf_counter() - start)

        observed_symmetry = batch_symmetry_ids(batch)
        if observed_symmetry != reference_symmetry:
            raise AssertionError(
                f"batch={batch_size}: diamond-Si symmetry changed from "
                f"{reference_symmetry} to {observed_symmetry}"
            )

    return statistics.median(elapsed_samples)


def main() -> None:
    """Run the TensorNet relaxation and batched scaling demonstration."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relax_steps = int(os.getenv("NVALCHEMI_SYMMETRY_RELAX_STEPS", "40"))
    benchmark_steps = int(os.getenv("NVALCHEMI_SYMMETRY_BENCH_STEPS", "100"))
    benchmark_warmup = int(os.getenv("NVALCHEMI_SYMMETRY_BENCH_WARMUP", "1"))
    benchmark_repeats = int(os.getenv("NVALCHEMI_SYMMETRY_BENCH_REPEATS", "2"))
    benchmark_batch_sizes = tuple(
        int(value)
        for value in os.getenv("NVALCHEMI_SYMMETRY_BATCH_SIZES", "1,2,4,8,16,32").split(
            ","
        )
    )
    if relax_steps < 2:
        raise ValueError("Relaxation requires at least two FIRE2 steps.")
    if benchmark_steps < 1:
        raise ValueError("Benchmark step count must be positive.")
    if benchmark_warmup < 0 or benchmark_repeats < 1:
        raise ValueError("Warm-up must be non-negative and repeats must be positive.")
    if not benchmark_batch_sizes or min(benchmark_batch_sizes) < 1:
        raise ValueError("Benchmark batch sizes must be positive integers.")
    if 1 not in benchmark_batch_sizes:
        raise ValueError(
            "Benchmark batch sizes must include 1 for the serial estimate."
        )

    print(f"Device: {device}")
    model = load_tensornet(device)
    run_relaxation(model, device, relax_steps)

    timings = {
        batch_size: benchmark_batch(
            batch_size,
            model,
            device,
            timed_steps=benchmark_steps,
            warmup_steps=benchmark_warmup,
            repeats=benchmark_repeats,
        )
        for batch_size in benchmark_batch_sizes
    }
    baseline_per_structure = timings[1]

    print(
        "\nFixed-step batched throughput "
        f"({benchmark_steps} timed steps, {benchmark_repeats} repeats, median)"
    )
    print(
        "batch  elapsed (s)  structure-steps/s  vs batch=1*  "
        "estimated serial (s)  speedup"
    )
    for batch_size in benchmark_batch_sizes:
        elapsed = timings[batch_size]
        throughput = batch_size * benchmark_steps / elapsed
        estimated_serial = baseline_per_structure * batch_size
        baseline_throughput = benchmark_steps / baseline_per_structure
        throughput_ratio = throughput / baseline_throughput
        speedup = estimated_serial / elapsed
        print(
            f"{batch_size:5d}  {elapsed:11.5f}  {throughput:12.3f}  "
            f"{throughput_ratio:11.2f}x  {estimated_serial:20.5f}  "
            f"{speedup:7.2f}x"
        )
    print(
        "* Relative to the per-structure throughput of batch=1; "
        "serial time is estimated from that median."
    )


if __name__ == "__main__" and os.environ.get("NVALCHEMI_SPHINX_BUILD") != "1":
    main()
