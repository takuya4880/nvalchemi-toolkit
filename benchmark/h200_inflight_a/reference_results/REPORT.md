# H200 conventional inflight results and overlap investigation

The conventional implementation is based on PR2 `e479d16`. The benchmark
supplies production stopping, admission, neighbor rebuilding and model settings.
All runs below use the same 1280 candidate IDs and a 500-update budget.
Queue PASS means execution completed; it does not mean every start converged.

| Experiment | Capacity | Queue status | Wall (s) | Converged | Budget exhausted |
| --- | ---: | --- | ---: | ---: | ---: |
| Historical campaign | 64 | PASS | 503.74 | 985 | 295 |
| Historical campaign | 128 | PASS | 463.67 | 985 | 295 |
| Historical campaign | 256 | FAIL | Incomplete | Unavailable | Unavailable |
| Packaged conventional benchmark | 256 | PASS | 462.90 | 987 | 293 |
| Original-configuration replay | 256 | PASS | 475.93 | 984 | 296 |

The last replay restores the original fixture reconstruction and warmup sequence
and uses exception-only recording. No numerical optimizer changes, retry or
candidate skipping were introduced. The packaged benchmark instead loads its
exported Cartesian fixture and uses a standalone warmup. These are separate
measurements; neither is claimed to be a bitwise recreation of the historical
trajectory. Capacity256 is not a demonstrated universal failure limit.

## Candidate229: completion still leaves an invalid near-overlap endpoint

The historical capacity256 crash was localized in an earlier diagnostic to
g22/trial9, candidate229, at its446th update. Positions and cell were initially
finite at the captured boundary, but model energy, force and stress were
non-finite and the minimum periodic pair distance was0 Angstrom. The subsequent
cell update became non-finite and the symmetry hook raised its guard error.
The original nearest distance was0.475648 Angstrom.

| Candidate229 experiment | Updates | Converged | Minimum distance (Angstrom) | Production fmax |
| --- | ---: | --- | ---: | ---: |
| Historical capacity64 endpoint | 500 | No | 0.000455 | 0.414 |
| Historical capacity128 endpoint | 500 | No | 0.000144 | 1.246 |
| Historical capacity256 failure snapshot | 446 | No | 0 | Non-finite |
| Packaged capacity256 endpoint | 500 | No | 0.000144 | 1.344 |
| Original-configuration capacity256 replay endpoint | 500 | No | 0.0000069 | 19.811 |

Both new runs finish the queue, but candidate229 remains unconverged and its
atoms are still almost coincident. Numerical differences may determine whether
an overlap rounds to exactly zero and triggers non-finite model outputs, or
remains tiny and lets execution survive. This is a plausible interpretation,
not proof of the first failing operation. A finite endpoint here is not a
physically reasonable optimized structure, and execution PASS must not hide it.
Candidate229 has not been tested as an isolated single-structure run in this
investigation.

## Input and stopping

Candidate229 contains62 atoms (Sm2H36I6O18), space group59, with27 positional
and3 cell degrees of freedom. Stop is the maximum ASE FrechetCellFilter
atomic/cell generalized-force row norm <0.02, zero pressure, full cell mask,
exp_cell_factor=N and the initial cell of the stage as reference. There is no
separate stress2e-5 criterion.

## Other historical pilot

The160-start four-stage conventional pilot completed in93.60s including
transitions. Per-stage convergence was160/160,158/160,160/160,136/160; final kept
convergence was136/160. Stage4 was accepted for76 starts. This is a transition
and cost check, not target-recovery accuracy.

## Review questions and artifacts

Investigate why this candidate enters the overlap region, the first non-finite
model/update quantity, and candidate-level failure isolation. Neither GPU memory
exhaustion nor a specific batch-state migration defect is established as the
cause. Allocator warnings also occurred in completed capacity256 replays.

`c256_replays.json` records the new measurements and source-result hashes.
`a256_failure_state.pt.gz` preserves the historical before/after snapshot;
`a256_case229_analysis.json` contains the historical endpoint comparisons.
Historical and new measurements remain separate rather than overwriting them.
