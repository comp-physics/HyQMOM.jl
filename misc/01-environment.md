# Environment

This is written so it works on **any** machine with an NVIDIA GPU — not just the PACE
node it was developed on. Read "Requirements" and "Portable setup" first; the
PACE-specific paths at the bottom are one concrete example, not a requirement. Nothing
here assumes a particular Julia or MPI *version* — discover what your machine has.

## Requirements (version-agnostic)

| need | minimum / note |
|---|---|
| **Julia** | 1.10+ recommended (CUDA.jl + the GPU env). The CPU package also runs on 1.9–1.11. Use whatever Julia your system provides — `module load julia`, your package manager, or [julialang.org/downloads](https://julialang.org/downloads). Don't assume a specific build path. |
| **NVIDIA GPU + driver** | Any CUDA-capable GPU. You need the **driver**; you do NOT need a system CUDA toolkit — CUDA.jl downloads a matching toolkit artifact by default. fp64 is used throughout (weak-FP64 consumer GPUs work but are slow). |
| **MPI** | Only for multi-GPU. MPI.jl ships a bundled MPI (MPICH_jll) that works out of the box — no system MPI required. CUDA-aware MPI is **not** needed (halos are host-staged). |

Discover what you have:
```bash
julia --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
which mpiexec 2>/dev/null   # only relevant if you want to bind a system MPI
```

## Portable setup (works anywhere)

Create the GPU project (CUDA + MPI). The simplest, most portable choice is to let
CUDA.jl manage the toolkit and let MPI.jl use its **bundled** MPI — no system paths, no
`LocalPreferences.toml`:

```bash
julia --project=gpu/gpuenv2 -e 'using Pkg; Pkg.add(["CUDA","MPI","MPIPreferences","StaticArrays"]); using CUDA; CUDA.precompile_runtime()'
```

That's enough to run everything below. Then:

```bash
# single-GPU
julia --project=gpu/gpuenv2 gpu/validate_residual3d_gpu.jl
# multi-GPU (bundled MPI provides mpiexec)
julia --project=gpu/gpuenv2 -e 'using MPI; run(`$(mpiexec()) -n 2 julia --project=gpu/gpuenv2 gpu/validate_timestep3d_mpi.jl`)'
```

Each rank binds its own GPU via `CUDA.device!(rank % CUDA.ndevices())`, so multiple
ranks map onto the available GPUs automatically.

### When to bind a *system* MPI instead of the bundled one

Only if you must match an existing MPI (e.g. the cluster's `srun`/PMIx, or the CPU
solver's MPI for ABI consistency). Then:
```bash
julia --project=gpu/gpuenv2 -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
```
This writes `gpu/gpuenv2/LocalPreferences.toml`. After that, put that MPI's `lib` on
`LD_LIBRARY_PATH` when running. If you don't need this, skip it — the bundled MPI is
simpler and more portable.

## Scratch / quota

Put the Julia depot and temp files wherever you have space and write permission:
```bash
export JULIA_DEPOT_PATH=<scratch>/julia_depot:$HOME/.julia
export TMPDIR=<scratch>/tmp
```
Reference `.f64` data and scratch scripts live in a `DATA` dir of your choosing
(see [`03-running-and-validation.md`](03-running-and-validation.md) for what's needed
and how to regenerate it). On a machine with a normal HOME quota you can use the
defaults and skip these.

---

## Concrete example: the PACE (Georgia Tech) node it was developed on

These are **this cluster's** specifics — substitute your own. Notably, HOME here is
**over quota**, so everything is redirected to scratch, the GPU env is bound to the
**system** OpenMPI (to match the CPU solver), and the system CUDA toolkit is reused.

```bash
# toolchain (versions/paths are THIS node's — yours will differ)
JULIA=/usr/local/pace-apps/manual/packages/julia/1.11.3/bin/julia      # or: module load julia/1.11.3
OMPI=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/openmpi-4.1.8-iit4xaslnjxkchcc6n62b5kluzibl2v2
CUDA=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/cuda-12.6.1-cu7yzjlutjf36tdszcz65iva7h3skek5

# HOME is over quota -> everything on scratch
export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:$HOME/.julia
export TMPDIR=/storage/scratch1/6/sbryngelson3/tmp
export LD_LIBRARY_PATH=$OMPI/lib:$LD_LIBRARY_PATH          # because gpuenv2 binds the system OpenMPI
# DATA dir for reference .f64 files:
#   /storage/scratch1/6/sbryngelson3/gpudata
# GPU node: 2x Quadro RTX 6000 (24 GB, sm_75, weak FP64 ~1:32)
```

`gpu/gpuenv2/LocalPreferences.toml` on this node pins the system OpenMPI
(`binary="system"`, `abi="OpenMPI"`), matching the main project. This is the
`use_system_binary()` step above — it is a PACE choice, not a requirement.

### Running on PACE (SLURM + PMIx, system MPI, host-staged halos)

```bash
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader            # disable flaky UCX; host-staged needs none
# multi-GPU
srun --mpi=pmix -n 2 --gpus=2 $JULIA --project=gpu/gpuenv2 gpu/validate_timestep3d_mpi.jl
# single-GPU
srun --mpi=pmix -n 1 --gpus=1 $JULIA --project=gpu/gpuenv2 gpu/validate_residual3d_gpu.jl
```

Singleton fallback (no SLURM step), and the MAT/HDF5 ABI caveat, are in
[`04-gotchas.md`](04-gotchas.md). Harmless `PMIX ERROR` launcher lines can be filtered.
