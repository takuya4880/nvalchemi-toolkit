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

"""Compact asynchronous CUDA timing and active/evaluated workload accounting."""

from __future__ import annotations

import time
from collections import defaultdict

import torch


class Monitor:
    """Collect timing events without adding a synchronization every step."""

    def __init__(self, model):
        self.model = model
        self.original_forward = model.forward
        self.reset()
        model.forward = self.forward

    def reset(self):
        """Start a fresh measurement window."""
        self.phase = "update"
        self.events = defaultdict(list)
        self.counts, self.calls, self.phases = [], [], []
        self.begin = time.perf_counter()

    def pair(self):
        """Create and start a CUDA event pair."""
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        return begin, end

    def forward(self, data, **kwargs):
        """Count actual neighbor edges before TensorNet adapts the batch."""
        active = (
            data.status.view(-1) < 1
            if "status" in data
            else torch.ones(data.num_graphs, dtype=torch.bool, device=data.device)
        )
        nodes = active[data.batch_idx.long()]
        if "neighbor_list" in data:
            edge_nodes = data.neighbor_list[:, 0].long()
            active_edges = nodes[edge_nodes].sum()
            edge_count = data.neighbor_list.shape[0]
        else:
            active_edges = data.num_neighbors[nodes].sum()
            edge_count = data.num_neighbors.sum()
        evaluated_edges = torch.as_tensor(edge_count, device=data.device)
        self.counts.append(
            torch.stack(
                (
                    active.sum(),
                    nodes.sum(),
                    active_edges,
                    active.new_tensor(data.num_graphs, dtype=torch.long),
                    active.new_tensor(data.num_nodes, dtype=torch.long),
                    evaluated_edges,
                )
            ).detach()
        )
        self.calls.append(time.perf_counter() - self.begin)
        self.phases.append(self.phase)
        begin, end = self.pair()
        result = self.original_forward(data, **kwargs)
        end.record()
        self.events["model_" + self.phase].append((begin, end))
        return result

    def wrap(self, owner, name, event=None, phase=None):
        """Time an existing method, preserving its arguments and return value."""
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            previous = self.phase
            if phase is not None:
                self.phase = phase
            pair = self.pair() if event else None
            try:
                return original(*args, **kwargs)
            finally:
                if pair:
                    pair[1].record()
                    self.events[event].append(pair)
                self.phase = previous

        setattr(owner, name, measured)

    def bind(self, optimizer):
        """Label initial and refill priming separately from normal updates."""
        for name in ("pre_update", "post_update"):
            self.wrap(optimizer, name, name)
        self.wrap(optimizer, "refill_check", "refill_boundary")
        if hasattr(optimizer, "_prime_forces"):
            self.wrap(optimizer, "_prime_forces", phase="priming")
        if optimizer.init_fn is not None:
            self.wrap(optimizer, "init_fn", phase="priming")
        for hook in optimizer.hooks:
            if hasattr(hook, "_rebuild"):
                self.wrap(hook, "_rebuild", "neighbors")
        original = optimizer._call_hooks

        def hooks(stage, batch):
            from nvalchemi.dynamics import DynamicsStage

            if stage != DynamicsStage.AFTER_COMPUTE:
                return original(stage, batch)
            pair = self.pair()
            result = original(stage, batch)
            pair[1].record()
            self.events["residual_and_symmetry_hooks"].append(pair)
            return result

        optimizer._call_hooks = hooks

    def finish(self, capacity):
        """Aggregate once, retaining one compact workload trace per run."""
        torch.cuda.synchronize()
        times = {
            name: sum(a.elapsed_time(b) for a, b in pairs) / 1000
            for name, pairs in self.events.items()
        }
        trace = torch.stack(self.counts).cpu() if self.counts else torch.empty(0, 6)
        # Call timestamp intervals describe host-observed occupancy, including tails.
        stamps = torch.tensor(self.calls, dtype=torch.float64)
        weights = torch.diff(
            torch.cat((stamps, torch.tensor([time.perf_counter() - self.begin])))
        )
        weights = weights.clamp_min(0)
        occupancy = trace[:, 0].double() / capacity
        means = (trace.double() * weights[:, None]).sum(0) / weights.sum()
        metrics = dict(
            cuda_s=times,
            model_calls=len(trace),
            priming_calls=self.phases.count("priming"),
            model_cuda_s=times.get("model_update", 0) + times.get("model_priming", 0),
            mean_counts_host_time_weighted=means.tolist(),
            count_columns=[
                "active_graphs",
                "active_atoms",
                "active_edges",
                "evaluated_graphs",
                "evaluated_atoms",
                "evaluated_edges",
            ],
            occupancy_host_time_weighted=float(
                (occupancy * weights).sum() / weights.sum()
            ),
            full_capacity_host_time_fraction=float(
                (weights * (trace[:, 0] == capacity)).sum() / weights.sum()
            ),
            priming_is_subset_of_model_cuda=True,
            refill_can_overlap_priming=True,
        )
        return metrics, dict(counts=trace, host_call_s=stamps, phases=self.phases)
