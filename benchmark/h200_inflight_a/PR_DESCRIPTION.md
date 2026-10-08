# Draft: H200 conventional inflight benchmark and near-overlap investigation

A historical 1280-start conventional capacity256 campaign crashed after a
62-atom candidate (g22/trial9, candidate229) reached coincident atoms and
non-finite model outputs. Two subsequent capacity256 runs completed, but the
same candidate still exhausted500 updates without convergence and ended with
minimum distances of0.000144 and0.0000069 Angstrom. Execution completion does
not resolve the underlying near-overlap trajectory.

This branch supplies the conventional benchmark,1280 packaged inputs,
production stopping configuration, CUDA telemetry, PBS template and historical
failure snapshot. It is based on `feat/inflight-symmetry-relaxation` at
`e479d16`; optimizer, scheduler, symmetry hook and numerical kernels are unchanged.
The public fork's CI workflow is retained for publishing compatibility.

Stop: max ASE FrechetCellFilter atomic/cell generalized-force row norm <0.02,
500 updates per candidate; no separate stress2e-5 condition.

The packaged capacity256 run completed in462.90s (987/1280 converged).
An original-configuration replay completed in475.93s (984/1280 converged).
They are distinct from historical measurements: the packaged driver uses an
exported Cartesian fixture and standalone warmup; the original-configuration
replay restores the original fixture reconstruction and warmup sequence.
Small numerical differences may explain exact-zero overlap and a crash versus
near-zero overlap and survival; this is a hypothesis, not an established cause.

Review focus: overlap trajectories, first non-finite values, production stopping
and how to isolate bad candidates while preserving unrelated work.
GPU smoke checks passed; CPU regressions87 passed,33 CUDA cases skipped on the
login node. See VALIDATION.json and reference_results/REPORT.md.
This is a Draft for reproduction/review, not a completed failure-handling fix.
