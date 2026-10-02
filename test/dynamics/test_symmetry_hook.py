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
"""Tests for the batched space-group symmetry constraint hook."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
pytest.importorskip("spglib")

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    FIRE,
    FIRE2,
    ConvergenceHook,
    FIRE2VariableCell,
    FIREVariableCell,
    FusedStage,
)
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import FixSymmetryHook
from nvalchemi.hooks import DynamicsContext, Hook
from nvalchemi.models.base import BaseModelMixin, ModelConfig


def _make_batch(device: str = "cpu", dtype: torch.dtype = torch.float64) -> Batch:
    """Create two periodic graphs with different sizes and space groups."""
    cubic = AtomicData(
        atomic_numbers=torch.tensor([14]),
        positions=torch.zeros(1, 3, dtype=dtype),
        cell=torch.eye(3, dtype=dtype).unsqueeze(0) * 4.0,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    orthorhombic = AtomicData(
        atomic_numbers=torch.tensor([14, 8]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.46, 0.93, 1.64]], dtype=dtype),
        cell=torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=dtype)).unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    batch = Batch.from_data_list([cubic, orthorhombic]).to(device)
    batch.forces = torch.zeros(3, 3, dtype=dtype, device=device)
    batch.stress = torch.zeros(2, 3, 3, dtype=dtype, device=device)
    batch.velocities = torch.zeros(3, 3, dtype=dtype, device=device)
    return batch


def _make_space_group_batch(
    structure: str,
    device: str,
    dtype: torch.dtype,
) -> Batch:
    """Create one representative structure for projector-oracle tests."""
    from ase import Atoms
    from ase.build import bulk

    if structure == "p1":
        atoms = Atoms(
            numbers=[14, 8],
            scaled_positions=[[0.13, 0.21, 0.37], [0.61, 0.42, 0.89]],
            cell=[[3.1, 0.0, 0.0], [0.4, 3.7, 0.0], [0.2, 0.3, 4.2]],
            pbc=True,
        )
    elif structure == "wurtzite":
        atoms = bulk("ZnS", "wurtzite", a=3.82, c=6.26)
    elif structure == "diamond":
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    else:
        raise ValueError(f"Unknown test structure: {structure}")

    data = AtomicData.from_atoms(atoms, device=device, dtype=dtype)
    return Batch.from_data_list([data])


def _context(batch: Batch, state: object | None = None) -> DynamicsContext:
    """Create a hook context with an optional optimizer state."""
    workflow = SimpleNamespace(_state=state) if state is not None else None
    return DynamicsContext(batch=batch, workflow=workflow)


class _AsymmetricModel(torch.nn.Module, BaseModelMixin):
    """Return deliberately non-symmetric forces and stress for hook tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            supports_pbc=True,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding outputs."""
        return {}

    def compute_embeddings(self, data: Batch, **kwargs: object) -> Batch:
        """Return *data* unchanged."""
        del kwargs
        return data

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Return fixed asymmetric model outputs."""
        dtype = batch.positions.dtype
        device = batch.device
        raw_force = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
        raw_stress = torch.tensor(
            [[2.0, 0.3, 0.4], [0.3, -1.0, 0.5], [0.4, 0.5, 0.7]],
            dtype=dtype,
            device=device,
        )
        return {
            "energy": torch.zeros(batch.num_graphs, 1, dtype=dtype, device=device),
            "forces": raw_force.expand_as(batch.positions).clone(),
            "stress": raw_stress.expand(batch.num_graphs, -1, -1).clone(),
        }


def _make_cubic_batch(device: str = "cpu") -> Batch:
    """Create a one-atom primitive cubic batch for optimizer integration."""
    dtype = torch.float64
    data = AtomicData(
        atomic_numbers=torch.tensor([14]),
        atomic_masses=torch.tensor([28.085], dtype=dtype),
        positions=torch.zeros(1, 3, dtype=dtype),
        velocities=torch.zeros(1, 3, dtype=dtype),
        forces=torch.zeros(1, 3, dtype=dtype),
        energy=torch.zeros(1, 1, dtype=dtype),
        stress=torch.zeros(1, 3, 3, dtype=dtype),
        cell=torch.eye(3, dtype=dtype).unsqueeze(0) * 4.0,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    return Batch.from_data_list([data]).to(device)


def _expected_rank1(
    vectors: torch.Tensor,
    cell: torch.Tensor,
    rotations: torch.Tensor,
    symm_map: torch.Tensor,
) -> torch.Tensor:
    """Independent transcription of ASE's rank-1 projector."""
    scaled_t = torch.linalg.inv(cell).T @ vectors.T
    result_t = torch.zeros_like(scaled_t)
    for rotation, atom_map in zip(rotations, symm_map, strict=True):
        result_t[:, atom_map] += rotation @ scaled_t
    return (cell.T @ (result_t / rotations.shape[0])).T


def _expected_rank2(
    tensor: torch.Tensor,
    cell: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Independent transcription of ASE's rank-2 projector."""
    inverse = torch.linalg.inv(cell)
    scaled = cell @ tensor @ cell.T
    result = torch.zeros_like(scaled)
    for rotation in rotations:
        result += rotation.T @ scaled @ rotation
    return inverse @ (result / rotations.shape[0]) @ inverse.T


def _set_controlled_symmetry(hook: FixSymmetryHook, batch: Batch) -> None:
    """Install deterministic graph-specific operations for projection tests."""
    dtype = batch.positions.dtype
    device = batch.device
    identity = torch.eye(3, dtype=dtype, device=device)
    c2z = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=dtype, device=device))
    hook.rotations = [
        torch.stack([identity, c2z]),
        torch.stack([identity, c2z]),
    ]
    hook.translations = [
        torch.zeros(2, 3, dtype=dtype, device=device),
        torch.zeros(2, 3, dtype=dtype, device=device),
    ]
    hook.symm_maps = [
        torch.tensor([[0], [0]], dtype=torch.long, device=device),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device),
    ]


class TestFixSymmetryHook:
    """Behavior tests for :class:`FixSymmetryHook`."""

    def test_constructor_refines_and_records_each_graph(self, device: str) -> None:
        """Each graph gets independent ASE symmetry metadata on its device."""
        batch = _make_batch(device)
        hook = FixSymmetryHook(batch)

        assert len(hook.rotations) == 2
        assert len(hook.translations) == 2
        assert len(hook.symm_maps) == 2
        assert hook.rotations[0].shape[0] != hook.rotations[1].shape[0]
        assert hook.symm_maps[0].shape[1] == 1
        assert hook.symm_maps[1].shape[1] == 2
        assert all(rotation.device.type == device for rotation in hook.rotations)

    def test_constructor_refinement_is_atomic(self) -> None:
        """A later graph failure leaves all input positions and cells unchanged."""
        batch = _make_batch()
        positions = batch.positions.clone()
        cells = batch.cell.clone()
        calls = 0

        def fake_refine(atoms, symprec, verbose):
            del symprec, verbose
            atoms.positions += 1.0

        def fake_prep(atoms, symprec, verbose):
            nonlocal calls
            del symprec, verbose
            calls += 1
            if calls == 2:
                raise RuntimeError("second graph failed")
            count = len(atoms)
            return (
                np.eye(3, dtype=int)[None],
                np.zeros((1, 3)),
                np.arange(count)[None],
            )

        with (
            patch("ase.spacegroup.symmetrize.refine_symmetry", fake_refine),
            patch("ase.spacegroup.symmetrize.prep_symmetry", fake_prep),
            pytest.raises(RuntimeError, match="second graph failed"),
        ):
            FixSymmetryHook(batch)

        assert torch.equal(batch.positions, positions)
        assert torch.equal(batch.cell, cells)

    def test_rank1_position_force_and_velocity_projection(self, device: str) -> None:
        """Position steps, forces, and velocities use the ASE rank-1 formula."""
        batch = _make_batch(device)
        hook = FixSymmetryHook(batch, adjust_cell=False)
        _set_controlled_symmetry(hook, batch)
        ctx = _context(batch)
        hook(ctx, DynamicsStage.BEFORE_PRE_UPDATE)

        step = torch.tensor(
            [[1.0, 2.0, 3.0], [0.4, -0.2, 0.8], [-0.7, 0.3, 0.1]],
            dtype=batch.positions.dtype,
            device=batch.device,
        )
        initial = batch.positions.clone()
        expected_step = torch.cat(
            [
                _expected_rank1(
                    step[:1], batch.cell[0], hook.rotations[0], hook.symm_maps[0]
                ),
                _expected_rank1(
                    step[1:], batch.cell[1], hook.rotations[1], hook.symm_maps[1]
                ),
            ]
        )
        batch.positions.add_(step)
        batch.velocities.copy_(step * 2.0)
        hook(ctx, DynamicsStage.AFTER_PRE_UPDATE)
        assert torch.allclose(batch.positions, initial + expected_step)
        assert torch.allclose(batch.velocities, expected_step * 2.0)

        batch.forces.copy_(step * 3.0)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        assert torch.allclose(batch.forces, expected_step * 3.0)

    def test_rank2_stress_and_cell_projection(self, device: str) -> None:
        """Stress and deformation-gradient steps use the ASE rank-2 formula."""
        batch = _make_batch(device)
        hook = FixSymmetryHook(batch, adjust_positions=False)
        _set_controlled_symmetry(hook, batch)
        ctx = _context(batch)
        hook(ctx, DynamicsStage.BEFORE_PRE_UPDATE)
        old_cells = batch.cell.clone()
        deltas = torch.tensor(
            [
                [[0.02, 0.03, 0.04], [0.01, -0.02, 0.05], [0.06, 0.07, 0.01]],
                [[-0.01, 0.02, 0.08], [0.03, 0.04, 0.09], [0.05, 0.06, 0.02]],
            ],
            dtype=batch.positions.dtype,
            device=batch.device,
        )
        expected_deltas = torch.stack(
            [
                _expected_rank2(deltas[i], old_cells[i], hook.rotations[i])
                for i in range(2)
            ]
        )
        identity = torch.eye(3, dtype=batch.positions.dtype, device=batch.device)
        batch.cell.copy_(old_cells @ (deltas + identity).transpose(-1, -2))
        hook(ctx, DynamicsStage.AFTER_PRE_UPDATE)
        assert torch.allclose(
            batch.cell,
            old_cells @ (expected_deltas + identity).transpose(-1, -2),
        )

        raw_stress = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                [[2.0, 1.0, 4.0], [1.0, 3.0, 5.0], [4.0, 5.0, 7.0]],
            ],
            dtype=batch.positions.dtype,
            device=batch.device,
        )
        expected_stress = torch.stack(
            [
                _expected_rank2(raw_stress[i], batch.cell[i], hook.rotations[i])
                for i in range(2)
            ]
        )
        batch.stress.copy_(raw_stress)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        assert torch.allclose(batch.stress, expected_stress)

    def test_projectors_match_ase_oracle(self) -> None:
        """Torch projectors match ASE for a skew cell and non-trivial map."""
        from ase.spacegroup.symmetrize import symmetrize_rank1, symmetrize_rank2

        batch = _make_batch()
        hook = FixSymmetryHook(batch)
        _set_controlled_symmetry(hook, batch)
        graph_index = 1
        vectors = torch.tensor(
            [[0.4, -0.2, 0.8], [-0.1, 0.7, 0.3]], dtype=batch.positions.dtype
        )
        tensor = torch.tensor(
            [[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, -1.0]],
            dtype=batch.positions.dtype,
        )
        cell = torch.tensor(
            [[3.0, 0.0, 0.0], [0.5, 2.5, 0.0], [0.2, 0.3, 4.0]],
            dtype=batch.positions.dtype,
        )
        rotations = hook.rotations[graph_index]
        translations = hook.translations[graph_index]
        symm_map = hook.symm_maps[graph_index]

        expected_vectors = symmetrize_rank1(
            cell.numpy(),
            torch.linalg.inv(cell).numpy(),
            vectors.numpy(),
            rotations.numpy(),
            translations.numpy(),
            symm_map.numpy(),
        )
        expected_tensor = symmetrize_rank2(
            cell.numpy(),
            torch.linalg.inv(cell).numpy(),
            tensor.numpy(),
            rotations.numpy(),
        )
        assert torch.allclose(
            hook._project_rank1_graph(vectors, cell, graph_index),
            torch.from_numpy(expected_vectors),
        )
        assert torch.allclose(
            hook._project_rank2_graph(tensor, cell, graph_index),
            torch.from_numpy(expected_tensor),
        )

    @pytest.mark.parametrize("structure", ["p1", "wurtzite", "diamond"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_vectorized_projectors_match_ase_across_space_groups(
        self,
        structure: str,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Projectors match ASE for low, nonsymmorphic, and cubic symmetry."""
        from ase.spacegroup.symmetrize import symmetrize_rank1, symmetrize_rank2

        batch = _make_space_group_batch(structure, device, dtype)
        hook = FixSymmetryHook(batch)
        atom_count = batch.positions.shape[0]
        vectors = (
            torch.arange(
                atom_count * 3,
                dtype=dtype,
                device=device,
            ).reshape(atom_count, 3)
            / 7.0
            - 0.4
        )
        tensor = torch.tensor(
            [[1.1, 0.2, -0.3], [0.4, -0.7, 0.5], [0.6, -0.8, 1.3]],
            dtype=dtype,
            device=device,
        )
        cell = batch.cell[0]
        rotations = hook.rotations[0]
        translations = hook.translations[0]
        symm_map = hook.symm_maps[0]

        expected_vectors = symmetrize_rank1(
            cell.detach().cpu().numpy(),
            torch.linalg.inv(cell).detach().cpu().numpy(),
            vectors.detach().cpu().numpy(),
            rotations.detach().cpu().numpy(),
            translations.detach().cpu().numpy(),
            symm_map.detach().cpu().numpy(),
        )
        expected_tensor = symmetrize_rank2(
            cell.detach().cpu().numpy(),
            torch.linalg.inv(cell).detach().cpu().numpy(),
            tensor.detach().cpu().numpy(),
            rotations.detach().cpu().numpy(),
        )
        tolerance = 2e-5 if dtype == torch.float32 else 1e-12

        assert torch.allclose(
            hook._project_rank1_graph(vectors, cell, 0).cpu(),
            torch.from_numpy(expected_vectors).to(dtype=dtype),
            atol=tolerance,
            rtol=tolerance,
        )
        assert torch.allclose(
            hook._project_rank2_graph(tensor, cell, 0).cpu(),
            torch.from_numpy(expected_tensor).to(dtype=dtype),
            atol=tolerance,
            rtol=tolerance,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rank1_preserves_maps_for_duplicate_rotations(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Equal rotations with distinct atom maps remain distinct operations."""
        batch = _make_batch(device, dtype)
        hook = FixSymmetryHook(batch)
        graph_index = 1
        identity = torch.eye(3, dtype=dtype, device=device)
        rotations = torch.stack([identity, identity, identity])
        symm_map = torch.tensor(
            [[0, 1], [1, 0], [0, 1]],
            dtype=torch.long,
            device=device,
        )
        hook.rotations[graph_index] = rotations
        hook.translations[graph_index] = torch.zeros(3, 3, dtype=dtype, device=device)
        hook.symm_maps[graph_index] = symm_map
        vectors = torch.tensor(
            [[1.2, -0.7, 0.3], [-0.4, 0.8, 2.1]],
            dtype=dtype,
            device=device,
        )
        cell = batch.cell[graph_index]
        expected = _expected_rank1(vectors, cell, rotations, symm_map)

        assert torch.allclose(
            hook._project_rank1_graph(vectors, cell, graph_index),
            expected,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rank2_weights_uneven_duplicate_rotations(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Rank-2 compression preserves multiplicities and invalidates caches."""
        batch = _make_batch(device, dtype)
        hook = FixSymmetryHook(batch)
        graph_index = 1
        identity = torch.eye(3, dtype=dtype, device=device)
        c2z = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=dtype, device=device))
        tensor = torch.tensor(
            [[1.0, 0.2, 0.7], [0.4, -0.5, 0.6], [-0.3, 0.8, 1.4]],
            dtype=dtype,
            device=device,
        )
        cell = batch.cell[graph_index]

        rotations = torch.stack([identity, identity, identity, c2z])
        hook.rotations[graph_index] = rotations
        expected = _expected_rank2(tensor, cell, rotations)
        assert torch.allclose(
            hook._project_rank2_graph(tensor, cell, graph_index),
            expected,
        )

        # Mutate the source tensor in-place after priming the derived cache.
        # The tensor identity stays fixed, so invalidation must observe its
        # version rather than only replacement of the list entry.
        updated_rotations = torch.stack([identity, c2z, c2z, c2z])
        hook.rotations[graph_index].copy_(updated_rotations)
        updated_expected = _expected_rank2(tensor, cell, updated_rotations)
        assert torch.allclose(
            hook._project_rank2_graph(tensor, cell, graph_index),
            updated_expected,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batched_rank2_matches_mixed_space_group_oracle(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """A mixed batch matches graph oracles with one batched inverse."""
        batch = _make_batch(device, dtype)
        hook = FixSymmetryHook(batch)
        assert hook.rotations[0].shape[0] != hook.rotations[1].shape[0]
        tensors = torch.tensor(
            [
                [[1.0, 0.2, 0.7], [0.4, -0.5, 0.6], [-0.3, 0.8, 1.4]],
                [[-0.2, 0.5, 0.1], [0.7, 1.3, -0.4], [0.9, 0.3, -0.8]],
            ],
            dtype=dtype,
            device=device,
        )
        expected = torch.stack(
            [
                _expected_rank2(tensors[i], batch.cell[i], hook.rotations[i])
                for i in range(batch.num_graphs)
            ]
        )
        original_inv_ex = torch.linalg.inv_ex

        with patch.object(
            torch.linalg,
            "inv_ex",
            wraps=original_inv_ex,
        ) as inv_ex:
            projected = hook._project_rank2_batch(tensors, batch.cell)

        tolerance = 2e-5 if dtype == torch.float32 else 1e-12
        assert torch.allclose(projected, expected, atol=tolerance, rtol=tolerance)
        assert inv_ex.call_count == 1
        assert inv_ex.call_args.args[0].shape == (batch.num_graphs, 3, 3)
        assert inv_ex.call_args.kwargs["check_errors"] is False

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batched_rank2_cache_tracks_rotation_identity_and_version(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Padded rotation cache preserves weights and invalidates safely."""
        batch = _make_batch(device, dtype)
        hook = FixSymmetryHook(batch)
        identity = torch.eye(3, dtype=dtype, device=device)
        c2z = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=dtype, device=device))
        hook.rotations = [
            torch.stack([identity, identity, identity]),
            torch.stack([identity, c2z, c2z, c2z]),
        ]
        tensors = torch.tensor(
            [
                [[1.0, 0.2, 0.7], [0.4, -0.5, 0.6], [-0.3, 0.8, 1.4]],
                [[-0.2, 0.5, 0.1], [0.7, 1.3, -0.4], [0.9, 0.3, -0.8]],
            ],
            dtype=dtype,
            device=device,
        )

        expected = torch.stack(
            [
                _expected_rank2(tensors[i], batch.cell[i], hook.rotations[i])
                for i in range(batch.num_graphs)
            ]
        )
        projected = hook._project_rank2_batch(tensors, batch.cell)
        first_cache = hook._rank2_batch_cache
        assert first_cache is not None
        assert first_cache[4].sum(dim=1).tolist() == [1, 2]
        assert torch.allclose(
            first_cache[3].sum(dim=1),
            torch.ones(2, dtype=dtype, device=device),
        )
        assert torch.allclose(projected, expected)

        hook.rotations[1].copy_(torch.stack([identity, identity, identity, c2z]))
        updated = hook._project_rank2_batch(tensors, batch.cell)
        in_place_cache = hook._rank2_batch_cache
        updated_expected = torch.stack(
            [
                _expected_rank2(tensors[i], batch.cell[i], hook.rotations[i])
                for i in range(batch.num_graphs)
            ]
        )
        assert in_place_cache is not first_cache
        assert torch.allclose(updated, updated_expected)

        hook.rotations[0] = torch.stack([identity, c2z])
        replaced = hook._project_rank2_batch(tensors, batch.cell)
        replaced_expected = torch.stack(
            [
                _expected_rank2(tensors[i], batch.cell[i], hook.rotations[i])
                for i in range(batch.num_graphs)
            ]
        )
        assert hook._rank2_batch_cache is not in_place_cache
        assert torch.allclose(replaced, replaced_expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rank1_chunk_boundaries_do_not_change_projection(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Single-operation chunks match an unchunked rank-1 projection."""
        batch = _make_batch(device, dtype)
        hook = FixSymmetryHook(batch)
        graph_index = 1
        identity = torch.eye(3, dtype=dtype, device=device)
        c2x = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=dtype, device=device))
        c2y = torch.diag(torch.tensor([-1.0, 1.0, -1.0], dtype=dtype, device=device))
        rotations = torch.stack([identity, c2x, c2y, identity, c2x])
        symm_map = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
            dtype=torch.long,
            device=device,
        )
        hook.rotations[graph_index] = rotations
        hook.translations[graph_index] = torch.zeros(5, 3, dtype=dtype, device=device)
        hook.symm_maps[graph_index] = symm_map
        vectors = torch.tensor(
            [[0.4, -0.2, 0.8], [-0.1, 0.7, 0.3]],
            dtype=dtype,
            device=device,
        )
        cell = batch.cell[graph_index]

        hook._projector_workspace_bytes = 1
        chunked = hook._project_rank1_graph(vectors, cell, graph_index)
        hook._projector_workspace_bytes = 1 << 30
        unchunked = hook._project_rank1_graph(vectors, cell, graph_index)
        expected = _expected_rank1(vectors, cell, rotations, symm_map)
        tolerance = 2e-5 if dtype == torch.float32 else 1e-12

        assert torch.allclose(chunked, unchunked, atol=tolerance, rtol=tolerance)
        assert torch.allclose(chunked, expected, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize("field", ["cell_velocity", "cell_velocities"])
    def test_cell_velocity_projects_deformation_rate(
        self, field: str, device: str
    ) -> None:
        """FIRE and FIRE2 Hdot state is projected through deformation rate."""
        batch = _make_batch(device)
        hook = FixSymmetryHook(batch, adjust_positions=False)
        _set_controlled_symmetry(hook, batch)
        raw = torch.tensor(
            [
                [[0.2, 0.3, 0.4], [0.1, -0.2, 0.5], [0.6, 0.7, 0.1]],
                [[-0.1, 0.2, 0.8], [0.3, 0.4, 0.9], [0.5, 0.6, 0.2]],
            ],
            dtype=batch.positions.dtype,
            device=batch.device,
        )
        state = SimpleNamespace(**{field: raw.clone()})
        expected = []
        for i in range(2):
            rate = torch.linalg.solve(batch.cell[i], raw[i]).T
            projected_rate = _expected_rank2(rate, batch.cell[i], hook.rotations[i])
            expected.append(batch.cell[i] @ projected_rate.T)

        hook(_context(batch, state), DynamicsStage.AFTER_POST_UPDATE)
        assert torch.allclose(getattr(state, field), torch.stack(expected))

    def test_cell_step_warning_and_error_thresholds(self) -> None:
        """Large deformation steps follow ASE's warning and error thresholds."""
        warning_batch = _make_batch()
        warning_hook = FixSymmetryHook(warning_batch, adjust_positions=False)
        warning_hook(_context(warning_batch), DynamicsStage.BEFORE_PRE_UPDATE)
        warning_batch.cell[0].mul_(1.2)
        with pytest.warns(UserWarning, match="exceeds 0.15"):
            warning_hook(_context(warning_batch), DynamicsStage.AFTER_PRE_UPDATE)

        error_batch = _make_batch()
        error_hook = FixSymmetryHook(error_batch, adjust_positions=False)
        error_hook(_context(error_batch), DynamicsStage.BEFORE_PRE_UPDATE)
        error_batch.cell[0].mul_(1.3)
        with pytest.raises(RuntimeError, match="exceeding 0.25"):
            error_hook(_context(error_batch), DynamicsStage.AFTER_PRE_UPDATE)

    def test_cell_step_threshold_uses_whole_batch_before_mutation(self) -> None:
        """A later graph triggers one batch-wide warning or atomic rejection."""
        warning_batch = _make_batch()
        warning_hook = FixSymmetryHook(warning_batch, adjust_positions=False)
        warning_hook(_context(warning_batch), DynamicsStage.BEFORE_PRE_UPDATE)
        warning_batch.cell[1].mul_(1.2)
        with pytest.warns(UserWarning, match="exceeds 0.15") as records:
            warning_hook(_context(warning_batch), DynamicsStage.AFTER_PRE_UPDATE)
        assert len(records) == 1

        error_batch = _make_batch()
        error_hook = FixSymmetryHook(error_batch, adjust_positions=False)
        error_hook(_context(error_batch), DynamicsStage.BEFORE_PRE_UPDATE)
        proposed_cells = error_batch.cell.clone()
        proposed_cells[0].mul_(1.1)
        proposed_cells[1].mul_(1.3)
        error_batch.cell.copy_(proposed_cells)
        with pytest.raises(RuntimeError, match="exceeding 0.25"):
            error_hook(_context(error_batch), DynamicsStage.AFTER_PRE_UPDATE)
        assert torch.equal(error_batch.cell, proposed_cells)

    def test_cell_projection_failure_preserves_proposed_batch(self) -> None:
        """A failed batched projection cannot partially update graph cells."""
        batch = _make_batch()
        hook = FixSymmetryHook(batch, adjust_positions=False)
        ctx = _context(batch)
        hook(ctx, DynamicsStage.BEFORE_PRE_UPDATE)
        batch.cell[0].mul_(1.01)
        batch.cell[1].mul_(0.99)
        proposed_cells = batch.cell.clone()

        with (
            patch.object(
                hook,
                "_project_rank2_batch",
                side_effect=RuntimeError("projection failed"),
            ),
            pytest.raises(RuntimeError, match="projection failed"),
        ):
            hook(ctx, DynamicsStage.AFTER_PRE_UPDATE)

        assert torch.equal(batch.cell, proposed_cells)

    def test_cell_step_rejects_non_finite_deformation(self) -> None:
        """Unchecked linear algebra still rejects non-finite cell steps."""
        batch = _make_batch()
        hook = FixSymmetryHook(batch, adjust_positions=False)
        hook(_context(batch), DynamicsStage.BEFORE_PRE_UPDATE)
        batch.cell[1, 0, 0] = torch.nan

        with pytest.raises(RuntimeError, match="non-finite deformation"):
            hook(_context(batch), DynamicsStage.AFTER_PRE_UPDATE)

    def test_affine_cell_step_preserves_fractional_positions(self, device: str) -> None:
        """Cell-coupled FIRE steps keep off-origin atoms on their Wyckoff sites."""
        from ase import Atoms
        from ase.spacegroup.symmetrize import check_symmetry

        dtype = torch.float64
        data = AtomicData(
            atomic_numbers=torch.tensor([26, 26]),
            atomic_masses=torch.full((2,), 55.845, dtype=dtype),
            positions=torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=dtype),
            velocities=torch.zeros(2, 3, dtype=dtype),
            forces=torch.zeros(2, 3, dtype=dtype),
            energy=torch.zeros(1, 1, dtype=dtype),
            stress=torch.zeros(1, 3, 3, dtype=dtype),
            cell=torch.eye(3, dtype=dtype).unsqueeze(0) * 4.0,
            pbc=torch.ones(1, 3, dtype=torch.bool),
        )
        batch = Batch.from_data_list([data]).to(device)
        hook = FixSymmetryHook(batch)
        ctx = _context(batch)
        initial_scaled = torch.linalg.solve(
            batch.cell[0].T, batch.positions.T
        ).T.clone()

        hook(ctx, DynamicsStage.BEFORE_PRE_UPDATE)
        batch.cell.mul_(1.02)
        batch.positions.mul_(1.02)
        hook(ctx, DynamicsStage.AFTER_PRE_UPDATE)

        final_scaled = torch.linalg.solve(batch.cell[0].T, batch.positions.T).T
        assert torch.allclose(final_scaled, initial_scaled, atol=1e-12)
        symmetry = check_symmetry(
            Atoms(
                numbers=batch.atomic_numbers.cpu().numpy(),
                positions=batch.positions.cpu().numpy(),
                cell=batch.cell[0].cpu().numpy(),
                pbc=True,
            ),
            symprec=hook.symprec,
            verbose=False,
        )
        assert symmetry.number == 229

    def test_batch_identity_validation(self) -> None:
        """First-use graph boundaries and atomic ordering fail explicitly."""
        batch = _make_batch()
        changed_numbers = _make_batch()
        numbers_hook = FixSymmetryHook(batch)
        changed_numbers.atomic_numbers[0] = 8
        with pytest.raises(ValueError, match="atomic_numbers"):
            numbers_hook(_context(changed_numbers), DynamicsStage.BEFORE_PRE_UPDATE)
        assert not numbers_hook._batch_validated
        numbers_hook(_context(batch), DynamicsStage.BEFORE_PRE_UPDATE)
        assert numbers_hook._batch_validated

        changed_layout = Batch.from_data_list(
            [batch.get_data(0), batch.get_data(0), batch.get_data(0)]
        )
        layout_hook = FixSymmetryHook(batch)
        with pytest.raises(ValueError, match="batch graph layout"):
            layout_hook(_context(changed_layout), DynamicsStage.BEFORE_PRE_UPDATE)
        assert not layout_hook._batch_validated

    def test_batch_identity_is_validated_once(self) -> None:
        """Batch identity checks run only on the first successful invocation."""
        batch = _make_batch()
        hook = FixSymmetryHook(batch, adjust_positions=False, adjust_cell=False)
        ctx = _context(batch)
        stages = (
            DynamicsStage.BEFORE_PRE_UPDATE,
            DynamicsStage.AFTER_PRE_UPDATE,
            DynamicsStage.AFTER_COMPUTE,
            DynamicsStage.AFTER_POST_UPDATE,
        )

        with patch.object(
            hook, "_validate_batch", wraps=hook._validate_batch
        ) as validate:
            for _ in range(2):
                for stage in stages:
                    hook(ctx, stage)

        assert validate.call_count == 1
        assert hook._batch_validated

    def test_projection_lifecycle_uses_check_free_linalg(self) -> None:
        """Per-step projections avoid error-checking linear algebra calls."""
        batch = _make_batch()
        hook = FixSymmetryHook(batch)
        _set_controlled_symmetry(hook, batch)
        state = SimpleNamespace(cell_velocity=torch.full_like(batch.cell, 0.01))
        ctx = _context(batch, state)
        original_inv_ex = torch.linalg.inv_ex
        original_solve_ex = torch.linalg.solve_ex

        with (
            patch.object(
                torch.linalg,
                "inv",
                side_effect=AssertionError("legacy inv must not run per step"),
            ),
            patch.object(
                torch.linalg,
                "solve",
                side_effect=AssertionError("legacy solve must not run per step"),
            ),
            patch.object(
                torch.linalg,
                "inv_ex",
                wraps=original_inv_ex,
            ) as inv_ex,
            patch.object(
                torch.linalg,
                "solve_ex",
                wraps=original_solve_ex,
            ) as solve_ex,
        ):
            hook(ctx, DynamicsStage.BEFORE_PRE_UPDATE)
            batch.cell.mul_(1.01)
            batch.positions.add_(0.01)
            batch.velocities.fill_(0.02)
            hook(ctx, DynamicsStage.AFTER_PRE_UPDATE)
            batch.forces.fill_(0.03)
            batch.stress.fill_(0.04)
            hook(ctx, DynamicsStage.AFTER_COMPUTE)
            hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert inv_ex.call_count > 0
        assert solve_ex.call_count > 0
        assert all(
            call.kwargs["check_errors"] is False for call in inv_ex.call_args_list
        )
        assert all(
            call.kwargs["check_errors"] is False for call in solve_ex.call_args_list
        )

    def test_batch_validation_is_not_latched_when_stage_fails(self) -> None:
        """A failed first stage leaves identity validation active for retry."""
        batch = _make_batch()
        hook = FixSymmetryHook(batch)

        with pytest.raises(RuntimeError, match="did not snapshot cells"):
            hook(_context(batch), DynamicsStage.AFTER_PRE_UPDATE)
        assert not hook._batch_validated

        batch.atomic_numbers[0] = 8
        with pytest.raises(ValueError, match="atomic_numbers"):
            hook(_context(batch), DynamicsStage.BEFORE_PRE_UPDATE)
        assert not hook._batch_validated

    def test_protocol_stages_and_frequency_validation(self) -> None:
        """The public hook implements the multi-stage hook protocol."""
        batch = _make_batch()
        hook = FixSymmetryHook(batch)
        assert isinstance(hook, Hook)
        assert hook.stage == DynamicsStage.BEFORE_PRE_UPDATE
        assert hook._active_stages == frozenset(
            {
                DynamicsStage.BEFORE_PRE_UPDATE,
                DynamicsStage.AFTER_PRE_UPDATE,
                DynamicsStage.AFTER_COMPUTE,
                DynamicsStage.AFTER_POST_UPDATE,
            }
        )
        with pytest.raises(ValueError, match="frequency=1"):
            FixSymmetryHook(_make_batch(), frequency=2)

    def test_fused_stage_is_rejected_explicitly(self) -> None:
        """FusedStage construction rejects the missing AFTER_PRE_UPDATE stage."""
        batch = _make_cubic_batch()
        model = _AsymmetricModel()
        constrained = FIRE(
            model=model, dt=0.01, hooks=[FixSymmetryHook(batch, adjust_cell=False)]
        )
        other = FIRE(model=model, dt=0.01)
        with pytest.raises(ValueError, match="FixSymmetryHook"):
            _ = constrained + other

    @pytest.mark.parametrize(
        "registration",
        ["constructor", "register_hook", "register_fused_hook"],
    )
    def test_fused_stage_rejects_outer_hook_paths(self, registration: str) -> None:
        """Every outer FusedStage registration path enforces capabilities."""
        batch = _make_cubic_batch()
        model = _AsymmetricModel()
        stages = [(0, FIRE(model=model, dt=0.01))]
        hook = FixSymmetryHook(batch, adjust_cell=False)

        with pytest.raises(ValueError, match="FixSymmetryHook"):
            if registration == "constructor":
                FusedStage(sub_stages=stages, hooks=[hook])
            else:
                fused = FusedStage(sub_stages=stages)
                getattr(fused, registration)(hook)

    def test_missing_after_pre_update_has_runtime_guard(self) -> None:
        """Nonstandard lifecycles cannot silently skip coordinate projection."""
        batch = _make_cubic_batch()
        hook = FixSymmetryHook(batch)
        ctx = _context(batch)
        hook(ctx, DynamicsStage.BEFORE_PRE_UPDATE)
        with pytest.raises(RuntimeError, match="does not support FusedStage"):
            hook(ctx, DynamicsStage.AFTER_COMPUTE)


@pytest.mark.parametrize(
    ("optimizer_type", "cell_state_field"),
    [
        (FIRE, None),
        (FIREVariableCell, "cell_velocity"),
        (FIRE2, None),
        (FIRE2VariableCell, "cell_velocities"),
    ],
)
def test_real_optimizer_run_preserves_symmetry(
    optimizer_type: type, cell_state_field: str | None, device: str
) -> None:
    """All FIRE variants preserve symmetry through multiple real steps."""
    from ase import Atoms
    from ase.spacegroup.symmetrize import check_symmetry

    batch = _make_cubic_batch(device)
    hook = FixSymmetryHook(batch)
    initial_number = check_symmetry(
        Atoms(
            numbers=batch.atomic_numbers.cpu().numpy(),
            positions=batch.positions.cpu().numpy(),
            cell=batch.cell[0].cpu().numpy(),
            pbc=True,
        ),
        symprec=hook.symprec,
        verbose=False,
    ).number
    optimizer = optimizer_type(
        model=_AsymmetricModel(), dt=0.01, n_steps=3, hooks=[hook]
    )
    result = optimizer.run(batch)

    final_number = check_symmetry(
        Atoms(
            numbers=result.atomic_numbers.cpu().numpy(),
            positions=result.positions.cpu().numpy(),
            cell=result.cell[0].cpu().numpy(),
            pbc=True,
        ),
        symprec=hook.symprec,
        verbose=False,
    ).number
    assert optimizer.step_count == 3
    assert final_number == initial_number
    assert torch.allclose(result.forces, torch.zeros_like(result.forces), atol=1e-12)

    if cell_state_field is not None:
        cell_velocity = getattr(optimizer._state, cell_state_field)
        assert cell_velocity.shape == (1, 3, 3)
        cell = result.cell[0]
        rate = torch.linalg.solve(cell, cell_velocity[0]).T
        expected = hook._project_rank2_graph(rate, cell, 0)
        assert torch.allclose(rate, expected, atol=1e-10)


def test_projected_force_controls_fmax_convergence(device: str) -> None:
    """Convergence observes AFTER_COMPUTE-projected forces, not raw forces."""
    batch = _make_cubic_batch(device)
    model = _AsymmetricModel()
    raw_force_norm = torch.linalg.vector_norm(model(batch)["forces"], dim=-1).max()
    optimizer = FIRE(
        model=model,
        dt=0.01,
        hooks=[FixSymmetryHook(batch, adjust_cell=False)],
        convergence_hook=ConvergenceHook.from_fmax(0.1),
    )

    _, converged = optimizer.step(batch)
    assert raw_force_norm > 0.1
    assert torch.equal(converged, torch.tensor([0], device=batch.device))
    assert torch.allclose(batch.forces, torch.zeros_like(batch.forces), atol=1e-12)
