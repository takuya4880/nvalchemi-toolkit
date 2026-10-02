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
"""Space-group symmetry constraint for batched geometry relaxation."""

from __future__ import annotations

import math
import warnings
from enum import Enum

import torch

from nvalchemi import OptionalDependency
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import DynamicsContext

__all__ = ["FixSymmetryHook"]


@OptionalDependency.ASE.require
class FixSymmetryHook:
    """Preserve each graph's initial space-group symmetry during relaxation.

    This is the batched Torch equivalent of ASE's
    :class:`ase.constraints.FixSymmetry`. ASE and spglib are used once during
    construction to refine each initial structure and discover its symmetry
    operations. All per-step projections then run with Torch on the batch's
    device.

    Parameters
    ----------
    batch : Batch
        Initial structures. Every graph must provide a non-singular cell and
        periodic-boundary flags. Positions and cells are refined in-place.
    symprec : float, optional
        Symmetry tolerance passed to ASE. Default ``0.01``.
    adjust_positions : bool, optional
        Project position steps and atomic velocities. Default ``True``.
    adjust_cell : bool, optional
        Project cell steps and cell velocities. Default ``True``.
    verbose : bool, optional
        Forward verbose symmetry information from ASE. Default ``False``.
    frequency : int, optional
        Constraint frequency. Symmetry constraints must run on every step, so
        only ``1`` is accepted.

    Notes
    -----
    The hook must be constructed from the same graph layout and atomic-number
    ordering used by the relaxation. It validates that identity on its first
    successful invocation and raises :class:`ValueError` if the initial batch
    is replaced, reordered, or resized. The graph layout and atomic-number
    ordering must remain unchanged afterward. Consequently, this hook is not
    compatible with in-flight replacement or graph graduation that changes the
    active batch layout.
    ``FusedStage`` is not supported because its sub-stage lifecycle omits
    ``AFTER_PRE_UPDATE``, where position and cell steps must be projected.
    The hook also uses Python graph loops and dynamic safety checks and is not
    compatible with ``torch.compile(fullgraph=True)``. It is intended for
    correctness-oriented relaxation batches rather than compiled hot paths.
    """

    stage = DynamicsStage.BEFORE_PRE_UPDATE
    supports_fused_stage = False
    _projector_workspace_bytes = 64 * 1024**2

    @staticmethod
    def _ase_symmetry_data(
        batch: Batch,
        symprec: float,
        verbose: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Refine graphs with ASE and return symmetry tensors."""
        from ase import Atoms
        from ase.spacegroup.symmetrize import prep_symmetry, refine_symmetry

        rotations: list[torch.Tensor] = []
        translations: list[torch.Tensor] = []
        symm_maps: list[torch.Tensor] = []
        refined_positions: list[torch.Tensor] = []
        refined_cells: list[torch.Tensor] = []
        batch_ptr = batch.batch_ptr[: batch.num_graphs + 1].detach().cpu().tolist()

        with torch.no_grad():
            for graph_index, (start, end) in enumerate(
                zip(batch_ptr[:-1], batch_ptr[1:], strict=True)
            ):
                atoms = Atoms(
                    numbers=batch.atomic_numbers[start:end].detach().cpu().numpy(),
                    positions=batch.positions[start:end].detach().cpu().numpy(),
                    cell=batch.cell[graph_index].detach().cpu().numpy(),
                    pbc=batch.pbc[graph_index].detach().cpu().numpy(),
                )
                refine_symmetry(atoms, symprec=symprec, verbose=verbose)
                graph_rotations, graph_translations, graph_symm_map = prep_symmetry(
                    atoms, symprec=symprec, verbose=verbose
                )

                refined_positions.append(
                    torch.as_tensor(
                        atoms.positions,
                        dtype=batch.positions.dtype,
                        device=batch.positions.device,
                    )
                )
                refined_cells.append(
                    torch.as_tensor(
                        atoms.cell.array,
                        dtype=batch.cell.dtype,
                        device=batch.cell.device,
                    )
                )
                rotations.append(
                    torch.as_tensor(
                        graph_rotations,
                        dtype=batch.positions.dtype,
                        device=batch.positions.device,
                    )
                )
                translations.append(
                    torch.as_tensor(
                        graph_translations,
                        dtype=batch.positions.dtype,
                        device=batch.positions.device,
                    )
                )
                symm_maps.append(
                    torch.as_tensor(
                        graph_symm_map,
                        dtype=torch.long,
                        device=batch.positions.device,
                    )
                )

            # Refine atomically: do not partially mutate the input batch when
            # symmetry discovery fails for any later graph.
            batch.positions.copy_(torch.cat(refined_positions, dim=0))
            batch.cell.copy_(torch.stack(refined_cells, dim=0))

        return rotations, translations, symm_maps

    def __init__(
        self,
        batch: Batch,
        symprec: float = 0.01,
        adjust_positions: bool = True,
        adjust_cell: bool = True,
        verbose: bool = False,
        frequency: int = 1,
    ) -> None:
        if frequency != 1:
            raise ValueError(
                "FixSymmetryHook is a constraint and requires frequency=1."
            )
        if batch.num_graphs == 0:
            raise ValueError("FixSymmetryHook requires at least one graph.")
        if getattr(batch, "cell", None) is None:
            raise ValueError("FixSymmetryHook requires a cell for every graph.")
        if getattr(batch, "pbc", None) is None:
            raise ValueError("FixSymmetryHook requires pbc for every graph.")
        if batch.cell.shape != (batch.num_graphs, 3, 3):
            raise ValueError(
                "FixSymmetryHook requires cell shape "
                f"[{batch.num_graphs}, 3, 3], got {list(batch.cell.shape)}."
            )
        if batch.pbc.shape != (batch.num_graphs, 3):
            raise ValueError(
                "FixSymmetryHook requires pbc shape "
                f"[{batch.num_graphs}, 3], got {list(batch.pbc.shape)}."
            )
        if not bool(batch.pbc.all().item()):
            raise ValueError(
                "FixSymmetryHook requires fully periodic boundary conditions "
                "for every graph."
            )
        if not bool((torch.linalg.det(batch.cell).abs() > 0).all().item()):
            raise ValueError("FixSymmetryHook requires non-singular cells.")

        self.frequency = frequency
        self.symprec = symprec
        self.adjust_positions = adjust_positions
        self.adjust_cell = adjust_cell
        self.verbose = verbose
        self._active_stages = frozenset(
            {
                DynamicsStage.BEFORE_PRE_UPDATE,
                DynamicsStage.AFTER_PRE_UPDATE,
                DynamicsStage.AFTER_COMPUTE,
                DynamicsStage.AFTER_POST_UPDATE,
            }
        )
        self.rotations, self.translations, self.symm_maps = self._ase_symmetry_data(
            batch, symprec, verbose
        )
        self._batch_ptr = batch.batch_ptr[: batch.num_graphs + 1].detach().clone()
        self._batch_ptr_values = tuple(int(value) for value in self._batch_ptr.tolist())
        self._atomic_numbers = batch.atomic_numbers.detach().clone()
        self._batch_validated = False
        self._saved_positions: torch.Tensor | None = None
        self._saved_cells: torch.Tensor | None = None
        self._awaiting_after_pre_update = False
        self._rank2_rotation_cache: dict[
            int,
            tuple[
                tuple[int, torch.device, torch.dtype],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._rank2_batch_cache: (
            tuple[
                tuple[tuple[int, torch.device, torch.dtype], ...],
                tuple[torch.Tensor, ...],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether the hook participates in *stage*."""
        return stage in self._active_stages

    def _validate_batch(self, batch: Batch) -> None:
        """Validate graph boundaries and atomic identity against initialization."""
        current_ptr = batch.batch_ptr[: batch.num_graphs + 1].detach()
        expected_ptr = self._batch_ptr.to(
            device=current_ptr.device, dtype=current_ptr.dtype
        )
        self._batch_ptr = expected_ptr
        if current_ptr.shape != self._batch_ptr.shape or not torch.equal(
            current_ptr, expected_ptr
        ):
            raise ValueError(
                "FixSymmetryHook batch graph layout differs from its initial batch_ptr."
            )
        current_numbers = batch.atomic_numbers.detach()
        expected_numbers = self._atomic_numbers.to(
            device=current_numbers.device, dtype=current_numbers.dtype
        )
        self._atomic_numbers = expected_numbers
        if current_numbers.shape != self._atomic_numbers.shape or not torch.equal(
            current_numbers, expected_numbers
        ):
            raise ValueError(
                "FixSymmetryHook batch atomic_numbers differ from the initial ordering."
            )

    def _graph_symmetry(
        self, graph_index: int, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rotations and atom maps matching a reference tensor."""
        rotations = self.rotations[graph_index]
        if rotations.device != reference.device or rotations.dtype != reference.dtype:
            rotations = rotations.to(device=reference.device, dtype=reference.dtype)
            self.rotations[graph_index] = rotations
        symm_map = self.symm_maps[graph_index]
        if symm_map.device != reference.device:
            symm_map = symm_map.to(device=reference.device)
            self.symm_maps[graph_index] = symm_map
        translations = self.translations[graph_index]
        if (
            translations.device != reference.device
            or translations.dtype != reference.dtype
        ):
            self.translations[graph_index] = translations.to(
                device=reference.device, dtype=reference.dtype
            )
        return rotations, symm_map

    def _project_rank1_graph(
        self,
        vectors: torch.Tensor,
        cell: torch.Tensor,
        graph_index: int,
    ) -> torch.Tensor:
        """Apply ASE's Cartesian rank-1 symmetry projector to one graph."""
        rotations, symm_map = self._graph_symmetry(graph_index, vectors)
        inv_cell = torch.linalg.inv_ex(
            cell.to(device=vectors.device, dtype=vectors.dtype),
            check_errors=False,
        ).inverse
        scaled_vectors_t = inv_cell.T @ vectors.T
        projected_t = torch.zeros_like(scaled_vectors_t)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(
                "FixSymmetryHook rotations must have shape [S, 3, 3], got "
                f"{list(rotations.shape)}."
            )
        if symm_map.shape != (rotations.shape[0], vectors.shape[0]):
            raise ValueError(
                "FixSymmetryHook symmetry maps must have shape "
                f"[{rotations.shape[0]}, {vectors.shape[0]}], got "
                f"{list(symm_map.shape)}."
            )

        # Account for the batched matmul result, its flattened permutation,
        # and a conservatively materialized scatter index. ``scatter_add_``
        # also preserves every contribution for non-bijective maps instead of
        # relying on advanced-index assignment semantics.
        bytes_per_operation = max(
            1,
            vectors.shape[0]
            * (6 * vectors.element_size() + 3 * symm_map.element_size())
            + 9 * rotations.element_size(),
        )
        chunk_size = max(1, self._projector_workspace_bytes // bytes_per_operation)
        for start in range(0, rotations.shape[0], chunk_size):
            end = min(start + chunk_size, rotations.shape[0])
            transformed = torch.matmul(rotations[start:end], scaled_vectors_t)
            values = transformed.permute(1, 0, 2).reshape(3, -1)
            destinations = symm_map[start:end].reshape(1, -1).expand(3, -1)
            projected_t.scatter_add_(1, destinations, values)
        projected_t /= rotations.shape[0]
        return (cell.to(device=vectors.device, dtype=vectors.dtype).T @ projected_t).T

    def _rank2_rotations(
        self,
        graph_index: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unique rotations and exact multiplicities for one graph."""
        rotations, _ = self._graph_symmetry(graph_index, reference)
        cache_key = (
            rotations._version,
            rotations.device,
            rotations.dtype,
        )
        cached = self._rank2_rotation_cache.get(graph_index)
        if cached is None or cached[0] != cache_key or cached[1] is not rotations:
            unique_rotations, multiplicities = torch.unique(
                rotations,
                dim=0,
                return_counts=True,
            )
            multiplicities = multiplicities.to(dtype=reference.dtype)
            cached = (cache_key, rotations, unique_rotations, multiplicities)
            self._rank2_rotation_cache[graph_index] = cached
        return cached[2], cached[3]

    def _project_rank2_graph(
        self,
        tensor: torch.Tensor,
        cell: torch.Tensor,
        graph_index: int,
    ) -> torch.Tensor:
        """Apply ASE's Cartesian rank-2 symmetry projector to one graph."""
        rotations, multiplicities = self._rank2_rotations(graph_index, tensor)
        cell = cell.to(device=tensor.device, dtype=tensor.dtype)
        inv_cell = torch.linalg.inv_ex(cell, check_errors=False).inverse
        scaled_tensor = cell @ tensor @ cell.T
        transformed = torch.matmul(
            rotations.transpose(-1, -2),
            torch.matmul(scaled_tensor, rotations),
        )
        projected = (transformed * multiplicities.reshape(-1, 1, 1)).sum(
            dim=0
        ) / self.rotations[graph_index].shape[0]
        return inv_cell @ projected @ inv_cell.T

    def _rank2_batch_rotations(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded rotations, normalized weights, and active lanes."""
        for graph_index in range(len(self.rotations)):
            self._graph_symmetry(graph_index, reference)
        sources = tuple(self.rotations)
        cache_key = tuple(
            (source._version, source.device, source.dtype) for source in sources
        )
        cached = self._rank2_batch_cache
        if (
            cached is None
            or cached[0] != cache_key
            or any(
                cached_source is not source
                for cached_source, source in zip(cached[1], sources, strict=True)
            )
        ):
            graph_rotations = []
            graph_multiplicities = []
            for graph_index in range(len(sources)):
                unique_rotations, multiplicities = self._rank2_rotations(
                    graph_index, reference
                )
                graph_rotations.append(unique_rotations)
                graph_multiplicities.append(multiplicities)

            max_rotations = max(rotation.shape[0] for rotation in graph_rotations)
            padded_rotations = reference.new_zeros(len(sources), max_rotations, 3, 3)
            weights = reference.new_zeros(len(sources), max_rotations)
            active = torch.zeros(
                len(sources),
                max_rotations,
                dtype=torch.bool,
                device=reference.device,
            )
            for graph_index, (rotations, multiplicities) in enumerate(
                zip(graph_rotations, graph_multiplicities, strict=True)
            ):
                count = rotations.shape[0]
                padded_rotations[graph_index, :count].copy_(rotations)
                weights[graph_index, :count].copy_(
                    multiplicities / sources[graph_index].shape[0]
                )
                active[graph_index, :count] = True
            cached = (
                cache_key,
                sources,
                padded_rotations,
                weights,
                active,
            )
            self._rank2_batch_cache = cached
        return cached[2], cached[3], cached[4]

    def _project_rank2_batch(
        self,
        tensors: torch.Tensor,
        cells: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Cartesian rank-2 projector to all graphs together."""
        expected_shape = (len(self.rotations), 3, 3)
        if tensors.shape != expected_shape:
            raise ValueError(
                "FixSymmetryHook rank-2 tensors must have shape "
                f"{list(expected_shape)}, got {list(tensors.shape)}."
            )
        cells = cells.to(device=tensors.device, dtype=tensors.dtype)
        rotations, weights, active = self._rank2_batch_rotations(tensors)
        inv_cells = torch.linalg.inv_ex(cells, check_errors=False).inverse
        scaled_tensors = cells @ tensors @ cells.transpose(-1, -2)
        transformed = torch.matmul(
            rotations.transpose(-1, -2),
            torch.matmul(scaled_tensors[:, None], rotations),
        )
        # Padding can produce non-finite values when the input is non-finite.
        # Mask those lanes before weighting so 0 * inf cannot contaminate sums.
        transformed.masked_fill_(~active[..., None, None], 0)
        projected = (transformed * weights[..., None, None]).sum(dim=1)
        return inv_cells @ projected @ inv_cells.transpose(-1, -2)

    def _project_rank1(self, vectors: torch.Tensor, cells: torch.Tensor) -> None:
        """Project a concatenated per-atom rank-1 tensor in-place."""
        with torch.no_grad():
            for graph_index, (start, end) in enumerate(
                zip(
                    self._batch_ptr_values[:-1],
                    self._batch_ptr_values[1:],
                    strict=True,
                )
            ):
                vectors[start:end].copy_(
                    self._project_rank1_graph(
                        vectors[start:end], cells[graph_index], graph_index
                    )
                )

    def _project_rank2(self, tensors: torch.Tensor, cells: torch.Tensor) -> None:
        """Project a batched per-graph rank-2 tensor in-place."""
        with torch.no_grad():
            tensors.copy_(self._project_rank2_batch(tensors, cells))

    def _project_position_and_cell_steps(self, batch: Batch) -> None:
        """Project the coordinate and deformation steps made by pre-update."""
        if self._saved_cells is None:
            raise RuntimeError(
                "FixSymmetryHook did not snapshot cells before pre_update."
            )
        saved_positions = self._saved_positions
        if self.adjust_positions and saved_positions is None:
            raise RuntimeError(
                "FixSymmetryHook did not snapshot positions before pre_update."
            )

        # A variable-cell update contains an affine position remap. Only its
        # internal-coordinate remainder should be projected as a rank-1 step.
        proposed_cells = batch.cell.clone()
        proposed_positions = batch.positions.clone()

        if self.adjust_cell:
            with torch.no_grad():
                identity = torch.eye(
                    3,
                    dtype=proposed_cells.dtype,
                    device=proposed_cells.device,
                )
                delta_deformations = (
                    torch.linalg.solve_ex(
                        self._saved_cells,
                        proposed_cells,
                        check_errors=False,
                    ).result.transpose(-1, -2)
                    - identity
                )
                max_delta = float(delta_deformations.abs().amax().item())
                if not math.isfinite(max_delta):
                    raise RuntimeError(
                        "FixSymmetryHook adjust_cell produced a non-finite "
                        "deformation gradient step."
                    )
                if max_delta > 0.25:
                    raise RuntimeError(
                        "FixSymmetryHook adjust_cell produced a deformation "
                        f"gradient step of {max_delta:.6g}, exceeding 0.25."
                    )
                if max_delta > 0.15:
                    warnings.warn(
                        "FixSymmetryHook adjust_cell may be ill behaved: "
                        "deformation gradient step exceeds 0.15.",
                        UserWarning,
                        stacklevel=2,
                    )
                projected = self._project_rank2_batch(
                    delta_deformations, self._saved_cells
                )
                batch.cell.copy_(
                    self._saved_cells @ (projected + identity).transpose(-1, -2)
                )

        if self.adjust_positions:
            with torch.no_grad():
                for graph_index, (start, end) in enumerate(
                    zip(
                        self._batch_ptr_values[:-1],
                        self._batch_ptr_values[1:],
                        strict=True,
                    )
                ):
                    old_cell = self._saved_cells[graph_index]
                    proposed_cell = proposed_cells[graph_index]
                    projected_cell = batch.cell[graph_index]
                    old_positions = saved_positions[start:end]
                    proposed_scaled = torch.linalg.solve_ex(
                        proposed_cell.T,
                        proposed_positions[start:end].T,
                        check_errors=False,
                    ).result.T
                    proposed_in_old_cell = proposed_scaled @ old_cell
                    internal_step = self._project_rank1_graph(
                        proposed_in_old_cell - old_positions,
                        old_cell,
                        graph_index,
                    )
                    corrected_scaled = torch.linalg.solve_ex(
                        old_cell.T,
                        (old_positions + internal_step).T,
                        check_errors=False,
                    ).result.T
                    batch.positions[start:end].copy_(corrected_scaled @ projected_cell)

    def _project_velocities(self, ctx: DynamicsContext) -> None:
        """Project atomic and optimizer cell velocities when present."""
        batch = ctx.batch
        if self.adjust_positions:
            velocities = getattr(batch, "velocities", None)
            if isinstance(velocities, torch.Tensor):
                self._project_rank1(velocities, batch.cell)

        if not self.adjust_cell or ctx.workflow is None:
            return
        state = getattr(ctx.workflow, "_state", None)
        if state is None:
            return
        for name in ("cell_velocity", "cell_velocities"):
            value = getattr(state, name, None)
            if isinstance(value, torch.Tensor):
                if value.shape != (batch.num_graphs, 3, 3):
                    raise ValueError(
                        f"FixSymmetryHook {name} must have shape "
                        f"[{batch.num_graphs}, 3, 3], got {list(value.shape)}."
                    )
                with torch.no_grad():
                    cells = batch.cell.to(device=value.device, dtype=value.dtype)
                    # Optimizers store Hdot, whereas the symmetry projector
                    # acts on the deformation-rate tensor (inv(H) @ Hdot).T.
                    deformation_rates = torch.linalg.solve_ex(
                        cells,
                        value,
                        check_errors=False,
                    ).result.transpose(-1, -2)
                    projected = self._project_rank2_batch(deformation_rates, cells)
                    value.copy_(cells @ projected.transpose(-1, -2))

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:
        """Apply symmetry constraints at the appropriate dynamics stage."""
        if ctx.batch is None:
            raise ValueError("FixSymmetryHook requires a dynamics batch.")
        batch = ctx.batch
        needs_validation = not self._batch_validated
        if needs_validation:
            self._validate_batch(batch)

        if stage == DynamicsStage.BEFORE_PRE_UPDATE:
            self._awaiting_after_pre_update = True
            self._saved_positions = batch.positions.clone()
            self._saved_cells = batch.cell.clone()
        elif stage == DynamicsStage.AFTER_PRE_UPDATE:
            self._project_position_and_cell_steps(batch)
            self._project_velocities(ctx)
            self._awaiting_after_pre_update = False
        elif stage == DynamicsStage.AFTER_COMPUTE:
            if self._awaiting_after_pre_update:
                raise RuntimeError(
                    "FixSymmetryHook does not support FusedStage because its "
                    "lifecycle omits AFTER_PRE_UPDATE. Run the optimizer as a "
                    "standalone dynamics stage."
                )
            forces = getattr(batch, "forces", None)
            if isinstance(forces, torch.Tensor):
                self._project_rank1(forces, batch.cell)
            stress = getattr(batch, "stress", None)
            if isinstance(stress, torch.Tensor):
                self._project_rank2(stress, batch.cell)
        elif stage == DynamicsStage.AFTER_POST_UPDATE:
            self._project_velocities(ctx)

        if needs_validation:
            self._batch_validated = True
