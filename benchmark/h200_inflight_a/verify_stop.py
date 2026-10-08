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

"""Independent comparison against ASE's actual filter, including shear and threshold crossings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from production_stop import frechet_forces


def verify(device="cpu"):
    rng = np.random.default_rng(20261008)
    checks = []
    for n in (1, 4, 17, 64):
        for kind in ("identity", "shear", "symmetric", "rotated", "expanded", "compressed"):
            initial = np.array([[4.0, 0.3, 0.1], [0.0, 5.0, 0.4], [0.2, 0.0, 6.0]])
            f = np.eye(3)
            if kind == "shear":
                f[0, 1] = 0.4
            elif kind == "symmetric":
                draw = rng.normal(0, 0.12, (3, 3))
                f += (draw + draw.T) / 2
            elif kind == "rotated":
                theta = 0.3
                f = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            elif kind == "expanded":
                f = np.diag([3.0, 1.2, 0.7])
            elif kind == "compressed":
                f = np.diag([0.25, 0.8, 1.1])
            atoms = Atoms("H" * n, positions=rng.random((n, 3)), cell=initial, pbc=True)
            filt = FrechetCellFilter(atoms)
            atoms.set_cell(initial @ f.T, scale_atoms=True)
            force = rng.normal(0, 0.03, (n, 3))
            stress = rng.normal(0, 0.001, (3, 3))
            stress = (stress + stress.T) / 2
            atoms.calc = SinglePointCalculator(atoms, energy=0, forces=force, stress=stress)
            expected = filt.get_forces()
            actual_atom, actual_cell, error = frechet_forces(
                torch.tensor(force, device=device), torch.tensor(stress[None], device=device),
                torch.tensor(atoms.cell.array[None], device=device), torch.tensor(initial[None], device=device),
                torch.zeros(n, dtype=torch.long, device=device), torch.tensor([n], device=device),
            )
            actual = torch.cat((actual_atom, actual_cell[0])).cpu().numpy()
            maximum_error = float(np.abs(actual - expected).max())
            assert maximum_error < 1e-9, (n, kind, maximum_error)
            assert float(error.max()) < 1e-8
            # Independent crossings on both sides of the production threshold.
            norm = np.linalg.norm(expected, axis=1).max()
            for target in (0.0199, 0.0201):
                scale = target / norm
                assert (np.linalg.norm(actual * scale, axis=1).max() < 0.02) == (target < 0.02)
            checks.append(dict(atoms=n, deformation=kind, force_error=maximum_error, log_error=float(error.max())))
    return dict(status="PASS", device=device, cases=len(checks), threshold_crossings=2 * len(checks), checks=checks)


if __name__ == "__main__":
    import sys
    result = verify(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    (Path(__file__).parent / f"ase_parity_{result['device']}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["device"], result["cases"], max(x["force_error"] for x in result["checks"]))
