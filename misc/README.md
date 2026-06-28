# `misc/` — GPU + MPI developer reference

Everything a developer needs to build, run, validate, extend, and debug the GPU /
multi-GPU path of HyQMOM.jl. Written against the prototype on branch
`gpu-single-source-port`, validated on a 2× Quadro RTX 6000 PACE node (2026-06-28).

## Read in this order

1. [`01-environment.md`](01-environment.md) — PACE setup: Julia, CUDA, MPI, the `gpuenv2`
   project, the scratch depot, every env var, and the exact `srun` invocations
   (single-GPU, multi-GPU, singleton). **Start here.** HOME is over quota — read the
   quota rule first.
2. [`02-architecture.md`](02-architecture.md) — how the GPU code is organized: the
   single-source `src/*_dev` kernels shared with the CPU, the module dependency
   closure, the rectangular residual, the z-slab decomposition, and what is
   *intentionally* CPU/GPU-specific (the eigensolvers).
3. [`03-running-and-validation.md`](03-running-and-validation.md) — every validator and
   benchmark, the command to run it, the reference data it needs, how to regenerate
   that data, and the expected result.
4. [`04-gotchas.md`](04-gotchas.md) — the non-obvious failure modes (FP parity,
   MPI/CUDA/HDF5 ABI clashes, CI flakes, fp32, tuple-splat, scope). Read before
   debugging anything weird.
5. [`05-results.md`](05-results.md) — consolidated performance + accuracy tables.

The narrative milestone log (how each piece was built and why) lives in
[`../gpu/README.md`](../gpu/README.md); `misc/` is the operational reference.

## 60-second quickstart

Versions/paths are machine-specific — use *your* Julia (1.10+) and GPU. See
[`01-environment.md`](01-environment.md) for the full portable setup; don't copy a
specific Julia/MPI/CUDA path or version, discover your own.

**Portable (any machine, bundled MPI — no system paths):**
```bash
cd <repo>            # HyQMOM.jl checkout; `julia` = your Julia 1.10+
# one-time: julia --project=gpu/gpuenv2 -e 'using Pkg; Pkg.add(["CUDA","MPI","MPIPreferences","StaticArrays"])'
julia --project=gpu/gpuenv2 -e 'using MPI; run(`$(mpiexec()) -n 2 julia --project=gpu/gpuenv2 gpu/validate_timestep3d_mpi.jl`)'
# expect: dt-sequence + final field max abs diff = 0.000e+00  BIT-IDENTICAL PASS
```

**On the PACE node it was built on** (system OpenMPI + SLURM; substitute your paths):
```bash
# set the PACE env block from 01-environment.md (JULIA, OMPI, scratch depot, …), then:
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader        # host-staged halos, no UCX
srun --mpi=pmix -n 2 --gpus=2 $JULIA --project=gpu/gpuenv2 gpu/validate_timestep3d_mpi.jl
```

## What the GPU path can do today

- Per-cell physics (flux closure, realizability projection, wave speeds) on GPU,
  single-sourced with the CPU (`src/*_dev.jl`).
- Full **3D** order-2 SSP-RK3 timestep on one GPU (`march3d_gpu!`).
- **Multi-GPU** 3D timestep via z-slab decomposition + host-staged halo exchange
  (`march3d_slab_gpu!`), bit-identical to single-GPU.
- **2D** (`nz=1`) timestep on one GPU, validated against MATLAB to machine precision.

Not done (by design / open): multi-GPU 2D (slab is z-only), CUDA-aware MPI (system
OpenMPI is `--without-cuda`), overlapping halo exchange with compute.
