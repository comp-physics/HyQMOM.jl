# HyQMOM.jl

[![CI](https://github.com/comp-physics/HyQMOM.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/comp-physics/HyQMOM.jl/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/hyqmomjl/badge/?version=latest)](https://hyqmomjl.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/comp-physics/HyQMOM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/comp-physics/HyQMOM.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17682195.svg)](https://doi.org/10.5281/zenodo.17682195)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia Version](https://img.shields.io/badge/julia-v1.9+-blue.svg)](https://julialang.org/downloads/)

3D Hyperbolic Quadrature Method of Moments (HyQMOM) solver for the Boltzmann–BGK equation with MPI and optional interactive 3D visualization.

**Docs:** https://hyqmomjl.readthedocs.io/en/latest/

**High-order spatial scheme (status, usage, limitations):** see [`HIGHORDER.md`](HIGHORDER.md).
**Running on GT PACE (modules, MPI, gotchas):** see [`RUNNING.md`](RUNNING.md).

---

## 1. TL;DR – run a 3D jets demo

```bash
git clone https://github.com/comp-physics/HyQMOM.jl.git
cd HyQMOM.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Serial, interactive 3D viewer (desktop)
julia --project=. examples/run_3d_jets_timeseries.jl

# MPI (4 ranks)
mpiexec -n 4 julia --project=. examples/run_3d_jets_timeseries.jl --Nx 80 --Ny 80
```

To visualize `.jld2` snapshots (from examples or HPC runs):

```bash
julia visualize_jld2.jl
```

---

## 2. What this repo gives you

- 3D moment-based kinetic solver for the Boltzmann–BGK equation
- MPI domain decomposition in x–y (z replicated on all ranks)
- Streaming snapshots to `.jld2` for post‑processing
- Optional GLMakie-based interactive 3D viewer (density / velocity / moments)

Use `examples/run_3d_jets_timeseries.jl` for standard crossing jets,  
and `examples/run_3d_custom_jets.jl` for custom jet configurations.

---

## 3. CLI parameters (most important)

All examples share the same parser (`examples/parse_params.jl`):

```bash
--Nx N               # Grid points in x (default: 40)
--Ny N               # Grid points in y (default: 40)
--Nz N               # Grid points in z (default: 20)
--tmax T             # Final time (default: 0.05)
--Ma M               # Mach number (default: 0.0)
--Kn K               # Knudsen number (default: 1.0)
--CFL C              # CFL number (default: 0.7)
--snapshot-interval N  # Save snapshots every N steps (0 = off)
--no-viz             # Disable visualization (clusters / CI)
--config NAME        # Initial-condition config (crossing, triple-jet, ...)
--help               # Show all options for the given example
```

Typical runs:

```bash
# Quick low‑resolution sanity check
julia --project=. examples/run_3d_jets_timeseries.jl --Nx 20 --Ny 20 --tmax 0.01

# Higher resolution + MPI
mpiexec -n 8 julia --project=. examples/run_3d_jets_timeseries.jl \
  --Nx 120 --Ny 120 --Nz 60 --snapshot-interval 5

# Custom jets
julia --project=. examples/run_3d_custom_jets.jl --config triple-jet --snapshot-interval 5
```

More detail: `examples/README.md`.

---

## 4. Visualization vs headless mode

Desktop / interactive:

```bash
julia --project=. examples/run_3d_jets_timeseries.jl
```

Headless (HPC / CI) – run solver only, save snapshots:

```bash
julia --project=. examples/run_3d_jets_timeseries.jl --no-viz --snapshot-interval 5
mpiexec -n 4 julia --project=. examples/run_3d_custom_jets.jl --config crossing --no-viz
```

Then visualize `.jld2` files on a machine with a display:

```bash
julia visualize_jld2.jl snapshots_*.jld2
```

To completely skip plotting deps (CI / lightweight installs):

```bash
export HYQMOM_SKIP_PLOTTING=true   # or: export CI=true
```

More: `docs/src/tutorials/interactive_visualization.md`.

---

## 5. MPI & HPC

All examples support MPI:

```bash
# Serial
julia --project=. examples/run_3d_jets_timeseries.jl --Nx 40 --Ny 40

# 4‑rank MPI
mpiexec -n 4 julia --project=. examples/run_3d_jets_timeseries.jl --Nx 100 --Ny 100
```

Domain decomposition:

- x–y plane split across ranks
- z direction replicated on all ranks

For Slurm / clusters and sweeps:

- `slurm/README.md`
- `docs/src/hpc_quickstart.md`
- `docs/src/mpi.md`

---

## 6. Minimal API example

```julia
using HyQMOM
using MPI

MPI.Init()

params = (
    Nx = 40,
    Ny = 40,
    Nz = 40,
    tmax = 0.1,
    Ma = 0.0,
    Kn = 1.0,
    CFL = 0.7,
)

# With snapshots: returns (snapshot_filename, grid)
snapshot_filename, grid = simulation_runner(params)

# If you set snapshot_interval = 0 you instead get:
# M_final, final_time, time_steps, grid = simulation_runner(params)

MPI.Finalize()
```

Key exported entry points (see `src/HyQMOM.jl` for full list):

- `simulation_runner(params)`
- `run_simulation_with_snapshots(params; snapshot_interval)`
- `interactive_3d_timeseries_streaming(filename, grid, params)`

---

## 7. Tests and common issues

```bash
# All tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Include MPI tests
cd test && bash run_mpi_tests.sh
```

If something goes wrong:

- **NaNs / instability** – lower `--CFL`, lower `--Ma`, or increase `--Nx`, `--Ny`, `--Nz`
- **Out of memory** – reduce resolution or increase / disable `--snapshot-interval`
- **MPI problems** – check `mpiexec --version`, reduce ranks, verify MPI.jl config
- **GLMakie on clusters** – use `--no-viz` and/or `HYQMOM_SKIP_PLOTTING=true`

---

## 8. Docs & layout

- Online docs: https://hyqmomjl.readthedocs.io/en/latest/
- Local docs: `docs/src/quickstart.md`, `docs/src/user_guide.md`,
  `docs/src/hpc_quickstart.md`, `docs/src/mpi.md`

Code structure:

- Core solver: `src/`
- Examples / CLIs: `examples/`
- Tests: `test/`
- Visualization helpers: `src/visualization/`

## License

MIT – see `license.md`.


