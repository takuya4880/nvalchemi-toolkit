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

"""TensorNet wrapper used by the historical conventional measurements."""

from __future__ import annotations

from typing import Any

MODEL = "TensorNet-PES-MatPES-PBE-2025.2"

def load_model(device: str) -> tuple[Any, Any]:
    """Load the cached real TensorNet with full tensile-positive eV/A^3 stress.

    Parameters
    ----------
    device : str
        Worker device, shared across methods.

    Returns
    -------
    tuple
        Toolkit wrapper and native MatGL Potential for independent ASE checks.
    """
    import matgl
    from matgl.ext.alchmtk import TensorNetWrapper

    class BalancedTensorNet(TensorNetWrapper):
        """Balance input tensor identity even on forward failure or energy-only AD."""

        def forward(self, data: Any, **kwargs: Any) -> Any:
            """Evaluate unchanged TensorNet and restore original geometry leaves."""
            positions, cells = data.positions, data.cell
            try:
                return super().forward(data, **kwargs)
            finally:
                data["positions"], data["cell"] = positions, cells

    potential = matgl.load_model(MODEL).to(device).eval()
    potential.calc_stresses = True
    model = BalancedTensorNet.from_potential(potential).to(device).eval()
    model.set_config("active_outputs", {"energy", "forces", "stress"})
    return model, potential
