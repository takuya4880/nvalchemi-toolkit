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

"""Common ASE FrechetCellFilter stopping observable, independent of optimizer."""

from __future__ import annotations

import numpy as np
import torch


def frechet_forces(forces, stress, cells, reference_cells, node_graph, counts):
    """Return ASE atomic/cell generalized forces and matrix-log reconstruction error.

    Zero external pressure, full cell mask, exp_cell_factor = atom count.
    A 32-point Gauss integral computes log(F) without eigenvector degeneracy
    at identity or shear. The block exponential evaluates its exact Frechet
    adjoint. All small-matrix work stays on the input device in float64.
    """
    cell = cells.double()
    deformation = torch.linalg.solve(reference_cells.double(), cell).mT
    eye = torch.eye(3, device=cell.device, dtype=torch.float64)
    delta = deformation - eye
    nodes, weights = np.polynomial.legendre.leggauss(32)
    nodes = torch.as_tensor((nodes + 1) / 2, device=cell.device)
    weights = torch.as_tensor(weights / 2, device=cell.device)
    integral = torch.linalg.solve(
        eye + nodes[None, :, None, None] * delta[:, None],
        delta[:, None].expand(-1, 32, -1, -1),
    )
    log_f = (integral * weights[None, :, None, None]).sum(1)
    virial = -torch.linalg.det(cell).abs()[:, None, None] * stress.double()
    unit_gradient = torch.linalg.solve(deformation, virial.mT).mT
    block = cell.new_zeros(len(cell), 6, 6)
    block[:, :3, :3] = log_f.mT
    block[:, 3:, 3:] = log_f.mT
    block[:, :3, 3:] = unit_gradient
    exponential = torch.matrix_exp(block)
    cell_forces = exponential[:, :3, 3:] / counts.double()[:, None, None]
    atom_forces = torch.bmm(
        forces.double()[:, None], deformation[node_graph.long()]
    ).squeeze(1)
    error = (exponential[:, :3, :3].mT - deformation).abs().amax((-1, -2))
    return atom_forces, cell_forces, error


class Stop:
    """Single production check; no development stress threshold or streak."""

    frequency = 1

    def evaluate(self, batch):
        """Terminate only finite residuals strictly below the ASE fmax limit."""
        good = (batch.production_fmax.view(-1) < 0.02) & (
            batch.frechet_log_error.view(-1) < 1e-8
        )
        index = torch.where(good & (batch.status.view(-1) < 1))[0]
        return index if index.numel() else None
