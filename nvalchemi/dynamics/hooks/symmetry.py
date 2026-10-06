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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

import torch

from nvalchemi import OptionalDependency
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import DynamicsContext

__all__ = ["FixSymmetryHook", "SymmetryRecord", "prepare_symmetry"]


@dataclass(frozen=True, slots=True)
class SymmetryRecord:
    """Precomputed space-group operations for one atomic structure.

    The tensors are stored on CPU so records can be prepared before an
    in-flight dynamics run without occupying GPU memory.  The hook transfers
    the active records to the batch device when the active layout changes.

    Parameters
    ----------
    symmetry_id : int
        Stable identifier stored on the corresponding :class:`AtomicData`.
    atomic_numbers : torch.Tensor
        Atomic numbers in the ordering used by ``symm_map``.
    rotations : torch.Tensor
        Fractional-coordinate rotations with shape ``[S, 3, 3]``.
    translations : torch.Tensor
        Fractional-coordinate translations with shape ``[S, 3]``.
    symm_map : torch.Tensor
        Atom permutations with shape ``[S, N]``.
    """

    symmetry_id: int
    atomic_numbers: torch.Tensor
    rotations: torch.Tensor
    translations: torch.Tensor
    symm_map: torch.Tensor


@OptionalDependency.ASE.require
def prepare_symmetry(
    data: AtomicData,
    symmetry_id: int,
    symprec: float = 0.01,
    verbose: bool = False,
) -> tuple[AtomicData, SymmetryRecord]:
    """Refine one periodic structure and precompute its symmetry operations.

    This function performs all ASE/spglib work before dynamics starts.  The
    returned structure is a clone of *data* carrying a graph-level
    ``symmetry_id``; the input structure is not modified.

    Parameters
    ----------
    data : AtomicData
        One fully periodic structure with a non-singular cell.
    symmetry_id : int
        Stable non-negative identifier for the prepared symmetry record.
    symprec : float, optional
        Symmetry tolerance passed to ASE. Default ``0.01``.
    verbose : bool, optional
        Forward verbose symmetry information from ASE. Default ``False``.

    Returns
    -------
    tuple[AtomicData, SymmetryRecord]
        Refined structure and its reusable CPU symmetry metadata.
    """
    from ase import Atoms
    from ase.spacegroup.symmetrize import prep_symmetry, refine_symmetry

    if isinstance(symmetry_id, bool) or not isinstance(symmetry_id, int):
        raise TypeError("symmetry_id must be an integer.")
    if symmetry_id < 0:
        raise ValueError("symmetry_id must be non-negative.")
    if data.cell is None:
        raise ValueError("prepare_symmetry requires a cell.")
    if data.pbc is None:
        raise ValueError("prepare_symmetry requires periodic-boundary flags.")
    if data.cell.shape not in ((3, 3), (1, 3, 3)):
        raise ValueError(
            "prepare_symmetry requires cell shape [3, 3] or [1, 3, 3], "
            f"got {list(data.cell.shape)}."
        )
    if data.pbc.shape not in ((3,), (1, 3)):
        raise ValueError(
            "prepare_symmetry requires pbc shape [3] or [1, 3], "
            f"got {list(data.pbc.shape)}."
        )
    cell = data.cell.reshape(3, 3)
    pbc = data.pbc.reshape(3)
    if not bool(pbc.all().item()):
        raise ValueError("prepare_symmetry requires fully periodic boundaries.")
    if not bool((torch.linalg.det(cell).abs() > 0).item()):
        raise ValueError("prepare_symmetry requires a non-singular cell.")

    atoms = Atoms(
        numbers=data.atomic_numbers.detach().cpu().numpy(),
        positions=data.positions.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
        pbc=pbc.detach().cpu().numpy(),
    )
    refine_symmetry(atoms, symprec=symprec, verbose=verbose)
    rotations, translations, symm_map = prep_symmetry(
        atoms, symprec=symprec, verbose=verbose
    )

    refined = data.clone()
    with torch.no_grad():
        refined.positions.copy_(
            torch.as_tensor(
                atoms.positions,
                dtype=refined.positions.dtype,
                device=refined.positions.device,
            )
        )
        refined.cell.copy_(
            torch.as_tensor(
                atoms.cell.array,
                dtype=refined.cell.dtype,
                device=refined.cell.device,
            ).reshape_as(refined.cell)
        )
    refined.add_system_property(
        "symmetry_id",
        torch.tensor([[symmetry_id]], dtype=torch.long, device=refined.device),
    )
    record = SymmetryRecord(
        symmetry_id=symmetry_id,
        atomic_numbers=refined.atomic_numbers.detach().cpu().clone(),
        rotations=torch.as_tensor(rotations).detach().cpu().clone(),
        translations=torch.as_tensor(translations).detach().cpu().clone(),
        symm_map=torch.as_tensor(symm_map, dtype=torch.long).detach().cpu().clone(),
    )
    return refined, record


@dataclass(slots=True)
class _Rank1BatchCache:
    """Packed metadata for the batched rank-1 projector."""

    rotation_keys: tuple[tuple[int, torch.device, torch.dtype], ...]
    map_keys: tuple[tuple[int, torch.device, torch.dtype], ...]
    rotation_sources: tuple[torch.Tensor, ...]
    map_sources: tuple[torch.Tensor, ...]
    batch_ptr_values: tuple[int, ...]
    reference_device: torch.device
    reference_dtype: torch.dtype
    packed_rotations: torch.Tensor
    source_indices: torch.Tensor
    destination_indices: torch.Tensor
    operation_indices: torch.Tensor
    atom_graph: torch.Tensor
    normalizer: torch.Tensor


@OptionalDependency.ASE.require
class FixSymmetryHook:
    """Preserve prepared space-group symmetries during batched relaxation.

    This is the batched Torch equivalent of ASE's
    :class:`ase.constraints.FixSymmetry`. Call :func:`prepare_symmetry` once per
    input structure before dynamics starts.  During dynamics this hook selects
    the records identified by the active batch's graph-level ``symmetry_id``
    and performs all projections with Torch on the batch's device.

    Parameters
    ----------
    records : Mapping[int, SymmetryRecord] or Iterable[SymmetryRecord]
        Prepared symmetry metadata keyed by stable symmetry ID.  Iterable
        records are keyed by their :attr:`SymmetryRecord.symmetry_id` values.
    adjust_positions : bool, optional
        Project position steps and atomic velocities. Default ``True``.
    adjust_cell : bool, optional
        Project cell steps and cell velocities. Default ``True``.
    frequency : int, optional
        Constraint frequency. Symmetry constraints must run on every step, so
        only ``1`` is accepted.

    Notes
    -----
    The active layout may change at complete step boundaries.  At the next
    ``BEFORE_PRE_UPDATE`` invocation, the hook rebinds records in graph order,
    validates graph sizes and atomic-number ordering, and invalidates all
    layout-dependent projector caches. ASE/spglib are never called while the
    hook is running.
    ``FusedStage`` is not supported because its sub-stage lifecycle omits
    ``AFTER_PRE_UPDATE``, where position and cell steps must be projected.
    The hook also uses Python graph loops and dynamic safety checks and is not
    compatible with ``torch.compile(fullgraph=True)``. It is intended for
    correctness-oriented relaxation batches rather than compiled hot paths.
    """

    stage = DynamicsStage.BEFORE_PRE_UPDATE
    supports_fused_stage = False
    _projector_workspace_bytes = 64 * 1024**2

    def __init__(
        self,
        records: Mapping[int, SymmetryRecord] | Iterable[SymmetryRecord],
        adjust_positions: bool = True,
        adjust_cell: bool = True,
        frequency: int = 1,
    ) -> None:
        if frequency != 1:
            raise ValueError(
                "FixSymmetryHook is a constraint and requires frequency=1."
            )
        if isinstance(records, Mapping):
            prepared_records = dict(records)
            for key, record in prepared_records.items():
                if key != record.symmetry_id:
                    raise ValueError(
                        "FixSymmetryHook record mapping keys must match "
                        "SymmetryRecord.symmetry_id."
                    )
        else:
            prepared_records = {}
            for record in records:
                if record.symmetry_id in prepared_records:
                    raise ValueError(
                        f"Duplicate symmetry record ID {record.symmetry_id}."
                    )
                prepared_records[record.symmetry_id] = record
        if not prepared_records:
            raise ValueError("FixSymmetryHook requires at least one symmetry record.")

        self.frequency = frequency
        self.adjust_positions = adjust_positions
        self.adjust_cell = adjust_cell
        self.records = prepared_records
        self._active_stages = frozenset(
            {
                DynamicsStage.BEFORE_PRE_UPDATE,
                DynamicsStage.AFTER_PRE_UPDATE,
                DynamicsStage.AFTER_COMPUTE,
                DynamicsStage.AFTER_POST_UPDATE,
            }
        )
        self.rotations: list[torch.Tensor] = []
        self.translations: list[torch.Tensor] = []
        self.symm_maps: list[torch.Tensor] = []
        self._batch_ptr_values: tuple[int, ...] = ()
        self._layout_signature: tuple[object, ...] | None = None
        self._saved_positions: torch.Tensor | None = None
        self._saved_cells: torch.Tensor | None = None
        self._awaiting_after_pre_update = False
        self._rank1_batch_cache: _Rank1BatchCache | None = None
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

    @staticmethod
    def _current_layout_signature(batch: Batch) -> tuple[object, ...]:
        """Return a synchronization-free signature for layout mutations."""
        symmetry_ids = getattr(batch, "symmetry_id", None)
        if not isinstance(symmetry_ids, torch.Tensor):
            raise ValueError(
                "FixSymmetryHook requires graph-level symmetry_id metadata. "
                "Prepare inputs with prepare_symmetry()."
            )
        batch_ptr = batch.batch_ptr
        return (
            id(batch),
            batch.num_graphs,
            batch.num_nodes,
            symmetry_ids.data_ptr(),
            symmetry_ids._version,
            batch.atomic_numbers.data_ptr(),
            batch.atomic_numbers._version,
            batch_ptr.data_ptr(),
            batch_ptr._version,
        )

    def _bind_batch(self, batch: Batch) -> None:
        """Bind active records in graph order and reset projector caches."""
        if batch.num_graphs == 0:
            raise ValueError("FixSymmetryHook requires at least one active graph.")
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

        symmetry_ids = batch.symmetry_id.reshape(-1).detach().cpu().tolist()
        if len(symmetry_ids) != batch.num_graphs:
            raise ValueError(
                "FixSymmetryHook requires one symmetry_id per graph, got "
                f"{len(symmetry_ids)} for {batch.num_graphs} graphs."
            )
        batch_ptr_values = tuple(
            int(value)
            for value in batch.batch_ptr[: batch.num_graphs + 1].detach().cpu().tolist()
        )
        atomic_numbers = batch.atomic_numbers.detach().cpu()
        active_records = []
        for graph_index, (symmetry_id, start, end) in enumerate(
            zip(
                symmetry_ids,
                batch_ptr_values[:-1],
                batch_ptr_values[1:],
                strict=True,
            )
        ):
            record = self.records.get(int(symmetry_id))
            if record is None:
                raise ValueError(
                    f"FixSymmetryHook has no record for symmetry_id {symmetry_id} "
                    f"at graph {graph_index}."
                )
            expected = record.atomic_numbers.reshape(-1).to(dtype=torch.long)
            actual = atomic_numbers[start:end].to(dtype=torch.long)
            if expected.shape != actual.shape or not torch.equal(expected, actual):
                raise ValueError(
                    "FixSymmetryHook atomic_numbers do not match record "
                    f"{symmetry_id} at graph {graph_index}."
                )
            if record.symm_map.shape != (record.rotations.shape[0], end - start):
                raise ValueError(
                    "FixSymmetryHook symmetry map shape does not match the active "
                    f"graph for record {symmetry_id}."
                )
            active_records.append(record)

        self.rotations = [
            record.rotations.to(
                device=batch.positions.device, dtype=batch.positions.dtype
            )
            for record in active_records
        ]
        self.translations = [
            record.translations.to(
                device=batch.positions.device, dtype=batch.positions.dtype
            )
            for record in active_records
        ]
        self.symm_maps = [
            record.symm_map.to(device=batch.positions.device, dtype=torch.long)
            for record in active_records
        ]
        self._batch_ptr_values = batch_ptr_values
        self._rank1_batch_cache = None
        self._rank2_rotation_cache.clear()
        self._rank2_batch_cache = None

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

    def _rank1_batch_metadata(self, reference: torch.Tensor) -> _Rank1BatchCache:
        """Return exact-size packed metadata for all rank-1 operations."""
        for graph_index in range(len(self.rotations)):
            self._graph_symmetry(graph_index, reference)
        rotation_sources = tuple(self.rotations)
        map_sources = tuple(self.symm_maps)
        rotation_keys = tuple(
            (source._version, source.device, source.dtype)
            for source in rotation_sources
        )
        map_keys = tuple(
            (source._version, source.device, source.dtype) for source in map_sources
        )
        cached = self._rank1_batch_cache
        if (
            cached is not None
            and cached.rotation_keys == rotation_keys
            and cached.map_keys == map_keys
            and all(
                cached_source is source
                for cached_source, source in zip(
                    cached.rotation_sources, rotation_sources, strict=True
                )
            )
            and all(
                cached_source is source
                for cached_source, source in zip(
                    cached.map_sources, map_sources, strict=True
                )
            )
            and cached.batch_ptr_values == self._batch_ptr_values
            and cached.reference_device == reference.device
            and cached.reference_dtype == reference.dtype
        ):
            return cached

        packed_rotations = []
        source_indices = []
        destination_indices = []
        operation_indices = []
        atom_graph = torch.empty(
            self._batch_ptr_values[-1],
            dtype=torch.long,
            device=reference.device,
        )
        normalizer = reference.new_empty(self._batch_ptr_values[-1])
        operation_offset = 0
        for graph_index, (start, end) in enumerate(
            zip(
                self._batch_ptr_values[:-1],
                self._batch_ptr_values[1:],
                strict=True,
            )
        ):
            rotations = rotation_sources[graph_index]
            symm_map = map_sources[graph_index]
            atom_count = end - start
            if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
                raise ValueError(
                    "FixSymmetryHook rotations must have shape [S, 3, 3], got "
                    f"{list(rotations.shape)}."
                )
            if symm_map.shape != (rotations.shape[0], atom_count):
                raise ValueError(
                    "FixSymmetryHook symmetry maps must have shape "
                    f"[{rotations.shape[0]}, {atom_count}], got "
                    f"{list(symm_map.shape)}."
                )

            operation_count = rotations.shape[0]
            atoms = torch.arange(start, end, device=reference.device)
            operations = torch.arange(
                operation_offset,
                operation_offset + operation_count,
                device=reference.device,
            )
            packed_rotations.append(rotations)
            source_indices.append(atoms.repeat(operation_count))
            destination_indices.append(symm_map.reshape(-1) + start)
            operation_indices.append(operations.repeat_interleave(atom_count))
            atom_graph[start:end] = graph_index
            normalizer[start:end] = operation_count
            operation_offset += operation_count

        cached = _Rank1BatchCache(
            rotation_keys=rotation_keys,
            map_keys=map_keys,
            rotation_sources=rotation_sources,
            map_sources=map_sources,
            batch_ptr_values=self._batch_ptr_values,
            reference_device=reference.device,
            reference_dtype=reference.dtype,
            packed_rotations=torch.cat(packed_rotations),
            source_indices=torch.cat(source_indices),
            destination_indices=torch.cat(destination_indices),
            operation_indices=torch.cat(operation_indices),
            atom_graph=atom_graph,
            normalizer=normalizer,
        )
        self._rank1_batch_cache = cached
        return cached

    def _project_rank1_batch(
        self,
        vectors: torch.Tensor,
        cells: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Cartesian rank-1 projector to all graphs together."""
        expected_shape = (self._batch_ptr_values[-1], 3)
        if vectors.shape != expected_shape:
            raise ValueError(
                "FixSymmetryHook rank-1 vectors must have shape "
                f"{list(expected_shape)}, got {list(vectors.shape)}."
            )
        cells = cells.to(device=vectors.device, dtype=vectors.dtype)
        metadata = self._rank1_batch_metadata(vectors)
        inv_cells = torch.linalg.inv_ex(cells, check_errors=False).inverse
        scaled_vectors = torch.bmm(
            vectors.unsqueeze(1), inv_cells[metadata.atom_graph]
        ).squeeze(1)
        projected = torch.zeros_like(vectors)
        bytes_per_pair = max(
            1,
            15 * vectors.element_size() + 3 * metadata.source_indices.element_size(),
        )
        chunk_size = max(1, self._projector_workspace_bytes // bytes_per_pair)
        pair_count = metadata.source_indices.shape[0]
        for start in range(0, pair_count, chunk_size):
            end = min(start + chunk_size, pair_count)
            transformed = torch.bmm(
                metadata.packed_rotations[metadata.operation_indices[start:end]],
                scaled_vectors[metadata.source_indices[start:end]].unsqueeze(-1),
            ).squeeze(-1)
            destinations = metadata.destination_indices[start:end, None].expand(-1, 3)
            projected.scatter_add_(0, destinations, transformed)
        projected /= metadata.normalizer[:, None]
        return torch.bmm(projected.unsqueeze(1), cells[metadata.atom_graph]).squeeze(1)

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
            vectors.copy_(self._project_rank1_batch(vectors, cells))

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

        if stage == DynamicsStage.BEFORE_PRE_UPDATE:
            signature = self._current_layout_signature(batch)
            if signature != self._layout_signature:
                self._bind_batch(batch)
                # Binding only reads batch tensors, so the pre-bind signature
                # remains valid for detecting replacement before the next step.
                self._layout_signature = signature
            self._awaiting_after_pre_update = True
            self._saved_positions = batch.positions.clone()
            self._saved_cells = batch.cell.clone()
            return

        if self._layout_signature is None:
            raise RuntimeError(
                "FixSymmetryHook must run at BEFORE_PRE_UPDATE before other stages."
            )
        if self._current_layout_signature(batch) != self._layout_signature:
            raise RuntimeError(
                "FixSymmetryHook batch layout changed during an optimization step. "
                "Replace or reorder graphs only at a complete step boundary."
            )

        if stage == DynamicsStage.AFTER_PRE_UPDATE:
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
