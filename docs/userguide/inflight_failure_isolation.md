# Inflight relaxation and failure isolation

Standalone inflight execution, symmetry constraints and failure isolation are
separate components. The existing `SizeAwareSampler` and optimizer own GPU
batching, refill and state migration. `FixSymmetryHook` optionally constrains
atomic/cell updates. `InflightSafety` optionally quarantines a failed structure
at the execution layer and lets ordinary refill replace it. The caller supplies
the model, optimizer, capacities, device and convergence hook.

## User structures

The composition recipe is
`examples/intermediate/10_inflight_user_structures.py`. It accepts any sequence
of ASE `Atoms`; reading CIF is simply `ase.io.read(path)`. It uses the existing
`AtomicData.from_atoms`, `prepare_symmetry`, `SizeAwareSampler`, optimizer and
`HostMemory` APIs. It introduces no new package-level relaxation adapter.

```python
from pathlib import Path
import runpy
from ase.io import read
from nvalchemi.dynamics import FIRE2, FIRE2VariableCell

recipe = runpy.run_path("examples/intermediate/10_inflight_user_structures.py")
atoms = [read(path) for path in sorted(Path("inputs").glob("*.cif"))]

# Set model, stopping_rule, device and capacities in your application.
results, failures = recipe["relax_structures"](
    atoms, model=model, convergence_hook=stopping_rule, device=device,
    max_batch_size=batch_capacity, max_atoms=atom_capacity,
    max_steps_per_system=update_budget,
    optimizer_type=FIRE2VariableCell, optimizer_kwargs={"dt": timestep},
    symmetry=False, failure_isolation=True,
)
```

Use `symmetry=True` for Cartesian symmetry-constrained relaxation; this prepares
one record per input and rebinds it after refill. Use `FIRE2` for fixed-cell
relaxation. Both choices use the same inflight machinery. Stopping rules remain
caller-owned: variable-cell runs need appropriate atomic and cell criteria.
There is no fixed force or stress threshold in the recipe or safety policy.
Results retain `candidate_id`, the zero-based input index, even if completion
order differs. Newcomers are initialized independently; survivors keep their
existing optimizer state. Models with moving-cell neighbor caches require a
neighbor hook that rebuilds when the cell changes; choose this in the model
wrapper's existing `make_neighbor_hooks` configuration.

## Attach isolation to an existing run

```python
from nvalchemi.dynamics.inflight_safety import InflightSafety

safety = InflightSafety(min_distance=1e-3)
safety.install(optimizer)
optimizer.run(n_steps=step_budget)
print(safety.failures)
```

Install once, before running, on a standalone sampler-driven optimizer with
`refill_frequency=1`. Existing model, optimizer and convergence configuration
remain in effect. Initializers that directly evaluate the model must invoke
`BEFORE_COMPUTE` hooks before that evaluation, as in the composition recipe.

Checks cover finite positions/cells, cell conditioning and orientation,
near-coincident distinct atoms using fresh COO or matrix neighbors, and finite
model energy, forces and stress before downstream hooks. Default geometric
thresholds are 0.001 Angstrom separation, 0.0001 Angstrom minimum cell singular
value and 1e-8 minimum singular-value ratio. They are numerical guards, not
chemical validity criteria. Without a neighbor list, overlap checks are absent.

A failed structure receives `InflightTerminationReason.FAILED` (3) and a separate
`safety_failure` reason. It is excluded from later model evaluation until refill
removes it. Its cached finite geometry is retained; malformed admission may
require a placeholder. Failed endpoint energies/forces are neutralized and
must not be interpreted as optimized physical results. Inspect termination
codes and failure records before further calculations.

The policy does not catch arbitrary model exceptions, recover CUDA OOM, or
support fused/distributed pipelines. It adds checking overhead. No-failure
CPU trajectories are tested against unguarded execution; GPU tests use the
same small deterministic oracle to validate isolation and state migration.
