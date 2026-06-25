# Running HyQMOM.jl on PACE (GT) — quickstart & gotchas

Practical recipe for running the solver on the GT PACE cluster, including the
non-obvious MPI / module / precompile steps. If you only read one thing, read
**§1 TL;DR** and use **`scripts/pace_mpi.sh`**.

---

## 1. TL;DR

```bash
# from the repo root
scripts/pace_mpi.sh 16 examples/run_3d_jets_timeseries.jl --no-viz --Nx 128 --Ny 128
```

`scripts/pace_mpi.sh <NRANKS> <script.jl> [args...]` loads the right modules,
sets the required env vars, precompiles once, and launches under `mpiexec`.
Everything below explains what it does and why.

---

## 2. Modules (REQUIRED)

```bash
module load julia/1.11.3 openmpi/4.1.5
```

- **Julia 1.11.3** — the project is compiled against v1.11.
- **openmpi/4.1.5** (or any `openmpi/4.1.x`) — **must match the ABI pinned in
  `LocalPreferences.toml`** (`abi = "OpenMPI"`). The cluster's *default* MPI is
  MVAPICH2 (MPICH ABI); loading it instead gives:
  ```
  could not load symbol "ompi_mpi_comm_null": ... undefined symbol
  ```
  If you see that, you loaded the wrong MPI. Load an `openmpi` module.

Shell state does **not** persist between separate shell invocations — load the
modules in the *same* shell/script that runs Julia.

---

## 3. Headless env vars (REQUIRED on clusters)

```bash
export HYQMOM_SKIP_PLOTTING=true
export CI=true
```

Without these, `using HyQMOM` tries to load GLMakie, which fails on a node with
no display (`GLFW`/`GLX` init error) and aborts the whole load.

> Note: on this branch the visualization packages (GLMakie/ColorSchemes/FileIO/
> LaTeXStrings) and MAT are **not** in `[deps]` (only viz-optional). If you ever
> see `KeyError: ... "ColorSchemes" not found`, your `Project.toml` and
> `Manifest.toml` disagree — run `julia --project=. -e 'using Pkg; Pkg.resolve()'`.

---

## 4. Multi-rank MPI on a single node (the UCX gotcha)

PACE login/compute nodes are **shared** and the UCX (InfiniBand) transport often
fails to wire up for intra-node jobs:

```
UCX ERROR  ep ...: no remote ep address for lane[4]->remote_lane[4]
MPIError(16): MPI_ERR_OTHER
```

Fix — restrict UCX to shared-memory transports (all ranks are on one node):

```bash
export UCX_TLS=sm,self
mpiexec -n 16 julia --project=. your_script.jl
```

`UCX_TLS=sm,self` (or `tcp,self`) avoids the IB lanes entirely. Single-rank
(`-n 1`) works without this, but multi-rank needs it.

---

## 5. Precompile ONCE before launching many ranks

The first `using HyQMOM` after any source edit triggers precompilation. If N
ranks start simultaneously they all contend on the precompile-cache lock and the
job appears to hang (low CPU, no output for minutes). Precompile once, serially,
first:

```bash
julia --project=. -e 'using HyQMOM'          # builds the cache (single process)
mpiexec -n 16 julia --project=. your_script.jl   # ranks load the cached image
```

`scripts/pace_mpi.sh` does this automatically.

---

## 6. Shared-node etiquette

- `nproc` may report 96 cores, but **other users' jobs run on the same node**
  (check `ps -eo pid,etime,pcpu,comm | grep julia`). Don't assume you own all
  cores. For Np=128, **16 ranks** is a good balance; 64 ranks oversubscribes and
  contends.
- Background long runs and poll the log; don't block.
- For a guaranteed-exclusive node, request one with Slurm (`salloc`/`sbatch`,
  see `slurm/`).

---

## 7. Reading MATLAB `.mat` results in Julia

`MAT.jl` is **not** a dependency of the solver (it pulls in HDF5). Use the
separate analysis environment so you don't perturb the solver's `Project.toml`:

```bash
cd ../mat_analysis            # sibling env with MAT + JLD2 (see below)
julia --project=. -e 'using MAT; M = matread("path/to/file.mat")["M"]'
```

To (re)create that env:

```bash
mkdir -p ../mat_analysis && cd ../mat_analysis
julia --project=. -e 'using Pkg; Pkg.add(["MAT","JLD2"])'
```

(HDF5 precompiles fine in a clean env; it only failed inside the solver's test
env due to a stale cross-branch Manifest.)

---

## 8. Reproducing the MATLAB jet-crossing cases (Ma=0 / Ma=2)

The solver has a faithful port of the MATLAB IC, selected with
`ic_type = :crossing_matlab` (3D jets `Uc = Ma/√3`, diagonally-offset cubes
`[Np/2−Csize : Np/2]³` and `[Np/2+1 : Np/2+1+Csize]³`, `Csize = floor(0.1*Np)`,
background `rhor`). Matching MATLAB `main_crossing_3DHyQMOM35.m`:

```julia
params = (
    Nx=128, Ny=128, Nz=128, Nmom=35,
    tmax=0.008, Kn=1000.0, Ma=0.0, flag2D=0, CFL=1/3,
    nnmax=100000, dtmax=1000.0,           # dtmax = Kn
    rhol=1.0, rhor=0.001, T=1.0, r110=0.0, r101=0.0, r011=0.0,
    symmetry_check_interval=1000, homogeneous_z=false, debug_output=false,
    snapshot_interval=0,
    ic_type=:crossing_matlab,
)
M_final, t, steps, grid = simulation_runner(params)
```

The `dt` rule matches MATLAB (spread-bound `rp = max(0,v⁺)−min(0,v⁻)` **and**
max-|eigenvalue| bound, per direction, capped by `dtmax`). Reference `.mat`
files: `Code_Riemann_3D_35mom_july2026_GT/src/riemann_full3D_..._Ma{0,2}.mat`.

A ready runner is `test/repro/run_crossing.jl` (set `REPRO_MA`), e.g.:

```bash
REPRO_MA=0.0 scripts/pace_mpi.sh 16 test/repro/run_crossing.jl
REPRO_MA=2.0 scripts/pace_mpi.sh 16 test/repro/run_crossing.jl
```

---

## 9. Running the test suite

```bash
module load julia/1.11.3 openmpi/4.1.5
HYQMOM_SKIP_PLOTTING=true CI=true mpiexec -n 1 julia --project=. -e 'using Pkg; Pkg.test()'
```

Run under `mpiexec -n 1` because some tests call `MPI.Init`. (The MATLAB-golden
and MAT-dependent tests skip when MATLAB / the golden dir / MAT aren't present.)
```
