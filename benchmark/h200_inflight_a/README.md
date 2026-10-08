# H200 conventional symmetry-constrained inflight benchmark

This is a conventional-only reproduction branch based on Takuya PR2 at
`e479d16855521d35c1fe511bfb2bfe049b312e1b`. The package's conventional optimizer,
symmetry hook, inflight scheduler and numerical kernels are unchanged. Added
files are benchmark code, inputs and review material. The public fork's existing
CI workflow is retained to avoid the workflow-scope publishing restriction.

The measured conventional workflow uses FIRE2/FIRE2VariableCell,
FixSymmetryHook, stable candidate/symmetry IDs, cell-aware neighbor rebuilding,
newcomer field initialization and force priming. These experiment settings are
explicitly supplied by the driver; this is not an unconfigured run of PR2's example.
There is no reduced optimizer, chart compiler or reduced-state migration here.

## Scope and inputs

- Acceleration: one symmetry-constrained joint position/cell relaxation stage,
  one intact1280-start queue, capacities64/128/256, one round per configuration.
- Optional four-stage transition/cost pilot:160 starts from16 genes, capacity64;
  fixed/symmetric, cell/symmetric, cell/free, paired strain-rattle/cell/free.
- Starts:128 genes with ten original PyXtal candidates each; all candidates and
  their original atom ordering, symmetry operations and immutable IDs are retained.
- Source commit: `bdd803bfab6178ab7b999b63b1479980121a0a02`, tree identical to the
  subsequently fetched projection-sampler candidate tree. No structures are
  regenerated, standardized, refined, origin-shifted or wrapped.

`starts_cartesian.pt.gz` contains ordinary Cartesian coordinates, atomic numbers,
cells and symmetry records. `manifest.csv` and `dataset_metadata.json` pin IDs,
source paths, CIF hashes and queue order. `initial_cifs.tar.gz` keeps original
CIFs together in one file. Model weights are not included.

## Production stopping

| Setting | Value |
| --- | --- |
| Fixed-cell symmetric stage | Maximum norm of symmetry-projected Cartesian forces <0.02 |
| Variable-cell stages | Maximum row norm across ASE FrechetCellFilter atomic/cell generalized forces <0.02 |
| Filter settings | Zero pressure, full cell mask, exp_cell_factor=N, original cell of each stage as reference |
| Stress | Included through cell generalized forces; no separate stress2e-5 condition |
| Checks and budget | One check per update;500 updates per candidate per stage |
| Numerical guard | Matrix-log reconstruction error <1e-8 and finite outputs |
| FIRE settings | dt0.02, tmax0.08, tmin0.001, delay10, maxstep0.05 |
| Inherited FIRE2 adaptation | Growth1.05, shrink0.75, alpha0.09, alpha shrink0.985 |

`runtime.Residuals` evaluates the common full-coordinate stopping observable.
`production_stop.Stop` determines termination. This criterion is distinct from
the archived prod640 development gate (.002 projected force and stress2e-5 for
three checks). It does not reproduce ASE BFGS trajectories or zero-update stopping.

## Installation and CPU check

Use an editable checkout in Python3.12 with a compatible Torch/CUDA environment.
The measured environment used torch2.14.1+cu126, toolkit-ops0.4.1, MatGL2.1.2,
ASE3.29.0, spglib2.7.0 and Warp1.18.0. MatGL and its TensorNet dependencies must
be configured separately; use TensorNet-PES-MatPES-PBE-2025.2.

```bash
git clone --branch benchmark/h200-inflight-a --single-branch https://github.com/YuyaoHuang330/nvalchemi-toolkit.git
cd nvalchemi-toolkit
python -m pip install -e '.[ase]'
cd benchmark/h200_inflight_a
export WARP_CACHE_PATH=/YOUR/WRITABLE/PATH/warp-cache
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python verify_stop.py cpu
```

The CPU check compares24 independent ASE filter cases and48 threshold crossings.
`model.py` uses the historical TensorNet wrapper; it restores input geometry
tensor references after model evaluations and uses unchanged potential weights.
`neighbors.py` is the historical current-cell rebuilding hook, local to this
benchmark rather than a package modification.

## CUDA GPU execution

Historical results were measured on H200. The runner accepts any CUDA GPU;
A100 execution has not yet been validated. Capacity may need to be reduced
to fit GPU memory. CUDA-compatible Torch, toolkit-ops, Warp and MatGL builds
are required. The ASPIRE2B PBS template is site-specific.

On an allocated CUDA GPU, use a new result directory for each invocation:

```bash
python run.py --kind smoke --output-dir /YOUR/WRITABLE/PATH/smoke
python run.py --kind acceleration --capacity 64 --output-dir /YOUR/WRITABLE/PATH/a64
python run.py --kind acceleration --capacity 128 --output-dir /YOUR/WRITABLE/PATH/a128
python run.py --kind acceleration --capacity 256 --output-dir /YOUR/WRITABLE/PATH/a256
```

Each configuration can run on an independent GPU; its internal queue is not
partitioned. On ASPIRE2B, `run_one.pbs` requests one GPU with project52001011.
Set TASK_SOURCE, TASK_PY and OUTPUT_DIR before submitting, e.g.:

```bash
qsub -v TASK_SOURCE=/YOUR/SOURCE/nvalchemi-toolkit,TASK_PY=/YOUR/ENV/bin/python,OUTPUT_DIR=/YOUR/RUNS/a64,CAPACITY=64 run_one.pbs
```

For the optional160-start cost/transition pilot:

```bash
python run.py --kind fourstage --capacity 64 --output-dir /YOUR/WRITABLE/PATH/fourstage
```

The strain/rattle seed is2026100800+candidate_id, strain standard deviation.01,
ASE rattle standard deviation.05 Angstrom. Stage4 replaces stage3 only if energy
improves by >1meV/atom. This pilot does not score target-recovery accuracy.

## Fidelity and timing boundaries

This is a standalone extraction of the measured conventional setup, not an
unchanged copy of the original cross-method driver. Differences are explicit:

1. The fixture is exported by CPU float32 Cartesian expansion. Original initial
   CUDA expansion was not captured bitwise; CPU/CUDA arithmetic can differ in
   the last digits. Original physical geometry and ordering are preserved, but
   identical long trajectories or exact failure steps are not certified.
2. The original warmup ran two methods. This driver instead runs two fresh
   conventional two-update queues. Both are excluded from reported queue timers.
3. Four-stage transitions copy Cartesian endpoints directly; the original
   chart-rebasing audit is omitted. Four-stage transition timing can differ.
4. Paths and output handling are standalone; caught exceptions write FAIL rather
   than leaving a startup RUNNING summary. One candidate exception still aborts
   its worker; no per-candidate recovery or mid-queue acceleration checkpoint exists.

The preserved historical results were measured before extraction. They are
references and are retained separately from the subsequent GPU validation.
New capacity256 measurements are recorded in reference_results/c256_replays.json.
The historical crash did not recur, but severe near-overlap persists; an exact
failure step is not a guaranteed replay point.

Wall time includes setup, admission priming, refill, synchronization, tail and
result collection; model load and warmup are excluded. Telemetry records active
graphs/atoms/edges and evaluated workloads, model CUDA time, priming and refill.
Event intervals can overlap and must not be added as independent costs.
One summary JSON and one aggregate endpoint/trace PT are written per successful
case; logs and failure summaries remain compact. Budget exhaustion is separate
from convergence; campaign PASS does not mean every structure converged.

See [historical results](reference_results/REPORT.md), `PROVENANCE.json` and
[Draft PR material](PR_DESCRIPTION.md) for the measured outcomes and limitations.

## Latest GPU check

The packaged capacity256 campaign completed on H200 in462.90s, with987/1280
converged. An original-configuration replay also completed, in475.93s with984
converged. Candidate229 remained unconverged and almost coincident in both runs.
See `reference_results/REPORT.md` and `c256_replays.json` for the distinction
between queue completion and unresolved physical near-overlap. A100 is untested.
