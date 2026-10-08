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

"""Standalone conventional H200 pilot; no reduced optimizer imports."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter

import runtime as pilot
from model import load_model
from telemetry import Monitor
from verify_stop import verify

ROOT = Path(__file__).resolve().parent
FIELDS = ("positions", "cell", "atomic_numbers", "energy", "forces", "stress", "candidate_id", "physical_fmax", "physical_stress_norm", "production_fmax", "production_atom_fmax", "production_cell_fmax", "production_reference_cell", "frechet_log_error", "inflight_step_count", "inflight_termination")

def checkpoint(result):
    return {int(result.get_data(k).candidate_id.item()): {key: getattr(result.get_data(k), key).detach().cpu().clone() for key in FIELDS} for k in range(result.num_graphs)}


def ase_endpoint_audit(result, records, constrained, variable):
    """Compare actual endpoint observations against an independent ASE filter."""
    import torch
    projector = pilot.FixSymmetryHook(records) if constrained else None
    if projector:
        projector._bind_batch(result)
        forces = projector._project_rank1_batch(result.forces, result.cell)
        stress = projector._project_rank2_batch(result.stress, result.cell)
    else:
        forces, stress = result.forces, result.stress
    checks = []
    indices = np.linspace(0, result.num_graphs - 1, min(16, result.num_graphs)).round().astype(int)
    for index in indices:
        a, b = result.batch_ptr[index:index + 2].tolist()
        force = forces[a:b].detach().cpu().double().numpy()
        sigma = stress[index].detach().cpu().double().numpy()
        cell = result.cell[index].detach().cpu().double().numpy()
        if variable:
            initial = result.production_reference_cell[index].cpu().double().numpy()
            atoms = Atoms(numbers=result.atomic_numbers[a:b].cpu().numpy(), positions=result.positions[a:b].cpu().numpy(), cell=initial, pbc=True)
            filt = FrechetCellFilter(atoms)
            atoms.set_cell(cell, scale_atoms=False)
            atoms.calc = SinglePointCalculator(atoms, energy=0, forces=force, stress=sigma)
            expected = float(np.linalg.norm(filt.get_forces(), axis=1).max())
        else:
            expected = float(np.linalg.norm(force, axis=1).max())
        actual = float(result.production_fmax[index].item())
        error = abs(actual - expected)
        if error > 2e-6 * max(1, expected):
            raise RuntimeError(f"ASE endpoint parity failed: {actual}, {expected}")
        checks.append(dict(candidate_id=int(result.candidate_id[index].item()), ase_fmax=expected, common_fmax=actual, error=error))
    return dict(status="PASS", checks=checks)


def rattle(samples):
    """Use the archived CrySPR numpy strain/rattle algorithm and paired ID seed."""
    seeds = {}
    for identifier, data in samples.items():
        seed = 2026100800 + identifier
        rng = np.random.default_rng(seed)
        draw = rng.normal(0, 0.01, (3, 3))
        strain = 0.5 * (draw + draw.T)
        atoms = Atoms(numbers=data.atomic_numbers.cpu().numpy(), positions=data.positions.cpu().numpy(), cell=data.cell[0].cpu().numpy(), pbc=True)
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
        atoms.rattle(stdev=0.05, seed=int(rng.integers(1, 2**31 - 1)))
        data.positions.copy_(torch.as_tensor(atoms.positions, device=data.device, dtype=data.positions.dtype))
        data.cell.copy_(torch.as_tensor(atoms.cell.array[None], device=data.device, dtype=data.cell.dtype))
        seeds[identifier] = seed
    return seeds


def main():
    """Run one intact conventional queue or the four-stage transition pilot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("smoke", "acceleration", "fourstage"), required=True)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=ROOT / "starts_cartesian.pt.gz")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261008)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-capable GPU.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    label = f"{args.kind}_A_c{args.capacity}"
    output = args.output_dir / f"{label}.json"
    packed = gzip.decompress(args.inputs.read_bytes())
    metadata = json.loads((ROOT / "dataset_metadata.json").read_text())
    summary = dict(status="RUNNING", kind=args.kind, method="A", capacity=args.capacity,
        gpu=torch.cuda.get_device_name(), pbs_job_id=os.environ.get("PBS_JOBID"),
        PR2_base="e479d16855521d35c1fe511bfb2bfe049b312e1b",
        source_commit=metadata["source_commit"], starts_sha256=hashlib.sha256(packed).hexdigest(),
        protocol=dict(fmax=0.02, cell_stop="ASE FrechetCellFilter atomic/cell generalized force, stage reference cell, exp_cell_factor=N", pressure=0, strict_less_than=True, max_updates_per_stage=500, consecutive_checks=1, dt=0.02, tmax=0.08, min_dt=0.001, delaystep=10, maxstep=0.05, fire_adaptation=dict(grow=1.05, shrink=0.75, initial_alpha=0.09, alpha_shrink=0.985)),
        standalone_changes=metadata["input_export_note"],
        warmup="two fresh conventional capacity2 queues, two updates each; excluded from queue timing",
        implementation_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("run.py", "runtime.py", "production_stop.py", "model.py", "neighbors.py", "telemetry.py")}, runs={})
    output.write_text(json.dumps(summary, indent=2)+"\n")
    saved = {}
    try:
        initial, records = pilot.load_inputs(args.inputs, "cuda")
        ids = list(initial)
        model, _ = load_model("cuda")
        monitor = Monitor(model)
        for _ in range(2):
            small = ids[:2]
            pilot.campaign(model, monitor, records, small, pilot.fresh_samples(initial, small), capacity=2, max_steps=2)
        if args.kind == "smoke":
            summary["ase_cuda_parity"] = verify("cuda")
            selected = metadata["pilot_ids"][:8]
            endpoints = None
            for name, variable, constrained in (("fixed", False, True), ("cell", True, True), ("free", True, False)):
                result, metrics, _ = pilot.campaign(model, monitor, records, selected, pilot.fresh_samples(initial, selected, endpoints), variable=variable, constrained=constrained, capacity=4, max_steps=3)
                metrics["ase_endpoint_audit"] = ase_endpoint_audit(result, records, constrained, variable)
                summary["runs"][name] = metrics
                endpoints = {int(result.get_data(i).candidate_id.item()): result.get_data(i) for i in range(result.num_graphs)}
        elif args.kind == "acceleration":
            probe = ids[:args.capacity]
            result, metrics, _ = pilot.campaign(model, monitor, records, probe, pilot.fresh_samples(initial, probe), capacity=args.capacity, max_steps=3)
            summary["capacity_probe"] = metrics
            del result
            print("CAPACITY PROBE PASS", json.dumps(metrics), flush=True)
            result, metrics, trace = pilot.campaign(model, monitor, records, ids, pilot.fresh_samples(initial, ids), capacity=args.capacity)
            metrics["ase_endpoint_audit"] = ase_endpoint_audit(result, records, True, True)
            summary["runs"]["symmetry_joint"] = metrics
            saved["symmetry_joint"] = dict(endpoints=checkpoint(result), trace=trace)
        else:
            ids = metadata["pilot_ids"]
            endpoints = None
            for stage in range(1, 5):
                begin = time.perf_counter()
                samples = pilot.fresh_samples(initial, ids, endpoints)
                if stage == 4:
                    summary["rattle_seeds"] = rattle(samples)
                transition = time.perf_counter()-begin
                result, metrics, trace = pilot.campaign(model, monitor, records, ids, samples, variable=stage != 1, constrained=stage <= 2, capacity=args.capacity)
                metrics["transition_host_s"] = transition
                metrics["ase_endpoint_audit"] = ase_endpoint_audit(result, records, stage <= 2, stage != 1)
                summary["runs"][f"stage{stage}"] = metrics
                endpoints = {int(result.get_data(i).candidate_id.item()): result.get_data(i) for i in range(result.num_graphs)}
                saved[f"stage{stage}"] = dict(endpoints=checkpoint(result), trace=trace)
                torch.save(saved, args.output_dir/f"{label}.pt")
                output.write_text(json.dumps(summary, indent=2)+"\n")
                print("STAGE DONE", stage, json.dumps(metrics), flush=True)
            kept = {}
            for key in ids:
                s3, s4 = saved["stage3"]["endpoints"][key], saved["stage4"]["endpoints"][key]
                delta = float((s4["energy"]-s3["energy"]).item())/len(s3["positions"])
                kept[key] = dict(kept_stage=4 if delta < -0.001 else 3, delta_energy_ev_atom=delta)
            saved["selection"] = kept
            summary["accepted_stage4"] = sum(x["kept_stage"] == 4 for x in kept.values())
            summary["fourstage_wall_s"] = sum(x["wall_s"]+x["transition_host_s"] for x in summary["runs"].values())
            summary["starts"] = len(ids)
        if saved:
            torch.save(saved, args.output_dir/f"{label}.pt")
        summary["status"] = "PASS"
        output.write_text(json.dumps(summary, indent=2)+"\n")
        print("PILOT PASS", label, flush=True)
    except Exception as error:
        summary.update(status="FAIL", error=repr(error))
        output.write_text(json.dumps(summary, indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
