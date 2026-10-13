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
"""Healthy trajectory parity and failed-row isolation through real refill."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import SizeAwareSampler
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.inflight_safety import InflightSafety, SafetyFailureReason
from nvalchemi.dynamics.optimizers.fire2 import FIRE2VariableCell
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.base import BaseModelMixin, ModelConfig


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Exercise GPU execution only when an allocated device is available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requires an allocated GPU")
    return request.param


class LocalModel(torch.nn.Module, BaseModelMixin):
    """Independent harmonic systems provide a deterministic regression oracle."""

    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}), supports_pbc=True
        )

    @property
    def embedding_shapes(self):
        """The oracle does not have learned embeddings."""
        return {}

    def compute_embeddings(self, data, **kwargs):
        """Embedding computation is unnecessary for this oracle."""
        raise NotImplementedError

    def forward(self, data, **kwargs):
        """Return graph-independent harmonic energy, forces and pressure."""
        energy = data.positions.new_zeros((data.num_graphs, 1)).index_add(
            0,
            data.batch_idx.long(),
            0.025 * data.positions.square().sum(-1, keepdim=True),
        )
        stress = (
            torch.eye(3, dtype=data.positions.dtype, device=data.device).expand(
                data.num_graphs, 3, 3
            )
            * 0.001
        )
        return OrderedDict(energy=energy, forces=-0.05 * data.positions, stress=stress)


def catalog(dtype, device):
    """Construct five periodic inputs with heterogeneous atom counts."""
    result = {}
    for key, count in ((101, 1), (707, 1), (22, 2), (900, 2), (55, 1)):
        positions = torch.full((count, 3), 1.2, dtype=dtype, device=device)
        if count == 2:
            positions[1] += 1.1
        data = AtomicData(
            positions=positions,
            atomic_numbers=torch.full((count,), 14, dtype=torch.long, device=device),
            cell=(torch.eye(3, dtype=dtype, device=device) * 5)[None],
            pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
        )
        data.add_system_property("candidate_id", torch.tensor([[key]], device=device))
        result[key] = data
    return result


class Dataset:
    """Supply independent copies so reference and protected runs share inputs."""

    def __init__(self, samples):
        self.samples = samples
        self.ids = list(samples)

    def __len__(self):
        """Return the number of inputs."""
        return len(self.ids)

    def get_metadata(self, index):
        """Return atom counts and zero required edges for the harmonic oracle."""
        return self.samples[self.ids[index]].num_nodes, 0

    def __getitem__(self, index):
        """Start each admission from an independent copy of its input."""
        return self.samples[self.ids[index]].clone(), {}


class InjectModel(LocalModel):
    """Return a bad force for one immutable ID, independent of neighbors."""

    def forward(self, data, **kwargs) -> OrderedDict:
        """Inject only candidate22 and preserve every other output."""
        result = super().forward(data, **kwargs)
        selected = (data.candidate_id.view(-1) == 22)[data.batch_idx.long()]
        field = getattr(self, "bad_field", "forces")
        result[field][
            selected if field == "forces" else data.candidate_id.view(-1) == 22
        ] = float("nan")
        return result


def campaign(
    safety=True,
    model=None,
    hooks=None,
    convergence=None,
    dtype=torch.float64,
    device="cpu",
):
    """Run a five-member heterogeneous queue with capacity two."""
    c = catalog(dtype, device)
    sink = HostMemory(capacity=5)
    options = dict(
        device_type=device,
        hooks=hooks,
        sampler=SizeAwareSampler(
            Dataset(c), max_batch_size=2, max_atoms=4, shuffle=False, bin_width=1000
        ),
        sinks=[sink],
        max_steps_per_system=5,
        convergence_hook=convergence,
    )
    optimizer = FIRE2VariableCell(model or LocalModel(), dt=0.02, **options)

    def initialize(batch):
        for field, value, level in (
            ("forces", torch.zeros_like(batch.positions), "node"),
            ("velocities", torch.zeros_like(batch.positions), "node"),
            ("energy", batch.positions.new_zeros(batch.num_graphs, 1), "system"),
            ("stress", torch.zeros_like(batch.cell), "system"),
        ):
            if field in batch:
                continue
            batch.add_key(
                field,
                list(
                    value.split(
                        batch.num_nodes_per_graph.tolist() if level == "node" else 1
                    )
                ),
                level=level,
            )
        optimizer.compute(batch)
        optimizer._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)

    optimizer.init_fn = initialize
    policy = InflightSafety()
    if safety:
        policy.install(optimizer)
    optimizer.run(n_steps=40)
    assert optimizer.active_batch is None
    result = sink.read()
    assert sorted(result.candidate_id.view(-1).tolist()) == sorted(c)
    return result, policy


def endpoints(batch):
    """Index results by immutable candidate IDs."""
    return {
        int(batch.get_data(i).candidate_id.item()): batch.get_data(i)
        for i in range(batch.num_graphs)
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("converged", [False, True])
def test_healthy_trajectory_unchanged(converged, dtype, device):
    """With no failures, every numerical result matches disabled safety."""

    class Stop:
        def evaluate(self, batch):
            good = (batch.inflight_step_count.view(-1) >= 2) & (
                batch.status.view(-1) == 0
            )
            return torch.where(good)[0]

    stop = Stop() if converged else None
    baseline, _ = campaign(False, convergence=stop, dtype=dtype, device=device)
    guarded, policy = campaign(convergence=stop, dtype=dtype, device=device)
    assert not policy.failures
    a, b = endpoints(baseline), endpoints(guarded)
    for key in a:
        for field in (
            "positions",
            "cell",
            "energy",
            "forces",
            "stress",
            "inflight_step_count",
            "inflight_termination",
        ):
            torch.testing.assert_close(
                getattr(a[key], field), getattr(b[key], field), rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "kind", ["model", "model_stress", "model_energy", "geometry", "cell"]
)
def test_one_failure_does_not_abort_or_reset_survivors(kind, device):
    """A failed member leaves once, survivors retain updates and newcomers run."""

    class Inject:
        stage, frequency = DynamicsStage.AFTER_PRE_UPDATE, 1

        def __call__(self, ctx, stage):
            batch = ctx.batch
            rows = batch.candidate_id.view(-1) == 22
            if not bool(rows.any()):
                return
            if kind == "geometry":
                batch.positions[rows[batch.batch_idx.long()]] = float("nan")
            elif kind == "cell":
                batch.cell[rows] = 0

    baseline, _ = campaign(False, device=device)
    model = InjectModel() if kind.startswith("model") else None
    if model is not None:
        model.bad_field = {
            "model": "forces",
            "model_stress": "stress",
            "model_energy": "energy",
        }[kind]
    guarded, policy = campaign(
        model=model, hooks=None if model is not None else [Inject()], device=device
    )
    a, b = endpoints(baseline), endpoints(guarded)
    assert len(policy.failures) == 1
    assert policy.failures[0]["candidate_id"] == 22
    assert int(b[22].inflight_termination) == 3
    assert int(b[22].safety_failure) != 0
    assert all(int(b[key].inflight_termination) == 2 for key in b if key != 22)
    for key in a:
        if key == 22:
            continue
        for field in ("positions", "cell", "energy", "forces", "inflight_step_count"):
            torch.testing.assert_close(
                getattr(a[key], field), getattr(b[key], field), rtol=1e-12, atol=1e-12
            )


def test_overlap_is_quarantined_before_model_call(device):
    """Fresh matrix distances detect near-overlap before model evaluation."""
    c = catalog(torch.float64, device)
    batch = Batch.from_data_list([c[22]])
    optimizer = FIRE2VariableCell(
        LocalModel(),
        dt=0.02,
        device_type=device,
        sampler=SizeAwareSampler(Dataset(c), max_batch_size=1),
    )
    policy = InflightSafety()
    policy.install(optimizer)
    policy._initialize(optimizer, batch)
    batch.positions[1] = batch.positions[0] + 1e-5
    batch.add_key(
        "neighbor_matrix", [torch.tensor([[1], [0]], device=device)], level="node"
    )
    batch.add_key(
        "num_neighbors", [torch.ones(2, dtype=torch.long, device=device)], level="node"
    )
    policy._overlap(optimizer, batch)
    assert int(batch.safety_failure) == SafetyFailureReason.NEAR_OVERLAP
    assert int(batch.inflight_termination) == 3
    assert int(batch.status) == optimizer.exit_status


@pytest.mark.parametrize("symmetry", [False, True])
@pytest.mark.parametrize("variable_cell", [False, True])
def test_user_cif_recipe(tmp_path, device, symmetry, variable_cell):
    """Read user CIFs and complete real refill with optional symmetry constraints."""
    import runpy
    from pathlib import Path

    from ase.build import bulk
    from ase.io import read, write
    from ase.spacegroup.symmetrize import check_symmetry

    from nvalchemi.dynamics import FIRE2, ConvergenceHook

    namespace = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/intermediate/10_inflight_user_structures.py"
        )
    )
    inputs = []
    for index, repeat in enumerate([(1, 1, 1), (2, 1, 1), (1, 1, 1)]):
        path = tmp_path / f"input_{index}.cif"
        write(path, bulk("Si", "diamond", a=5.43).repeat(repeat))
        inputs.append(read(path))
    model = LocalModel()
    if symmetry:
        # Isotropic pressure and zero atomic forces are symmetry compatible.
        class SymmetricModel(LocalModel):
            def forward(self, data, **kwargs):
                """Return symmetric outputs to exercise the projection hook."""
                result = super().forward(data, **kwargs)
                result["forces"].zero_()
                return result

        model = SymmetricModel()
    results, failures = namespace["relax_structures"](
        inputs,
        model=model,
        convergence_hook=ConvergenceHook.from_fmax(threshold=1e-10),
        device=device,
        max_batch_size=2,
        max_atoms=8,
        max_steps_per_system=3,
        symmetry=symmetry,
        failure_isolation=True,
        optimizer_type=FIRE2VariableCell if variable_cell else FIRE2,
        optimizer_kwargs={"dt": 0.02},
    )
    assert not failures
    assert sorted(results.candidate_id.view(-1).tolist()) == [0, 1, 2]
    assert torch.isfinite(results.positions).all()
    if symmetry:
        from ase import Atoms

        for index in range(results.num_graphs):
            data = results.get_data(index)
            atoms = Atoms(
                numbers=data.atomic_numbers.cpu().numpy(),
                positions=data.positions.cpu().numpy(),
                cell=data.cell.reshape(3, 3).cpu().numpy(),
                pbc=True,
            )
            assert check_symmetry(atoms, symprec=0.01).number == 227


def test_symmetry_and_failure_isolation_together(device):
    """A failed constrained member leaves while healthy records rebind on refill."""
    import runpy
    from pathlib import Path

    from ase.build import bulk

    from nvalchemi.dynamics import ConvergenceHook

    class BrokenMember(LocalModel):
        def forward(self, data, **kwargs):
            """Keep good forces symmetric and inject one invalid member."""
            output = super().forward(data, **kwargs)
            output["forces"].zero_()
            rows = data.candidate_id.view(-1) == 1
            output["stress"][rows] = float("nan")
            return output

    namespace = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/intermediate/10_inflight_user_structures.py"
        )
    )
    results, failures = namespace["relax_structures"](
        [bulk("Si", "diamond", a=5.43) for _ in range(3)],
        model=BrokenMember(),
        convergence_hook=ConvergenceHook.from_fmax(1e-10),
        device=device,
        max_batch_size=2,
        max_atoms=4,
        max_steps_per_system=3,
        optimizer_kwargs={"dt": 0.02},
        symmetry=True,
        failure_isolation=True,
    )
    assert len(failures) == 1
    assert failures[0]["candidate_id"] == 1
    rows = endpoints(results)
    assert int(rows[1].inflight_termination) == 3
    assert int(rows[0].inflight_termination) == 1
    assert int(rows[2].inflight_termination) == 1
    assert torch.isfinite(results.positions).all()
