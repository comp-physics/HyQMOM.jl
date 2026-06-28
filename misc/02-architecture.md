# GPU architecture

## Single-source: device kernels live in `src/`, CPU delegates

The per-cell physics is written **once** as allocation-free scalar/NTuple device
functions in `src/`, used by BOTH the CPU solver (which delegates to them, preserving
its public signatures) and the GPU kernels (which inline them). No duplicated math.

| physics | shared device kernel (`src/`) | CPU entry that delegates |
|---|---|---|
| flux closure | `src/numerics/flux_closure_dev.jl` (`FluxClosureDev.flux_closure35_dev`) | `Flux_closure35_3D` |
| reconstruction | `src/numerics/recon_dev.jl` (`ReconDev.to_recon_vars_dev`/`from_recon_vars_dev`) | `to_recon_vars` / `from_recon_vars` |
| realizability | `src/realizability/realize_dev.jl` (`RealizeDev.realizable_3D_M4_dev`, `…_corr_dev`) | `realizable_3D_M4` |

These are golden-clean: the CPU output is byte-identical (or matches the autogen to
machine precision) — see the golden battery `debug/golden_kernels.jl`. The
load-bearing detail is that the shared `@fastmath` central-moment helpers are
`@noinline` (see [`04-gotchas.md`](04-gotchas.md)).

## GPU driver dependency closure (the whole 3D + multi-GPU capability)

```
gpu/timestep3d_gpu.jl   (module Timestep3DGPU)  — march3d_gpu! (single) + march3d_slab_gpu! (multi)
  ├── gpu/residual3d_gpu.jl   (Residual3DGPU)   — residual3d_box_gpu! (rectangular) + residual3d_gpu! (cubic wrapper)
  │     ├── gpu/wavespeed_dev.jl (WavespeedDev)  — realize_and_speed_*_dev, jac15_eig_dev
  │     │     └── gpu/schur4.jl (Schur4)         — custom 4×4 real-Schur eig (no LAPACK)
  │     ├── src/numerics/flux_closure_dev.jl
  │     ├── src/numerics/recon_dev.jl
  │     └── src/realizability/realize_dev.jl
  └── gpu/realize_gpu.jl      (RealizeGPU)        — realizable_batched! / realizable_batched (CUDA kernel)
        ├── src/numerics/recon_dev.jl
        └── src/realizability/realize_dev.jl
```

That is the complete set of driver files. `gpu/schur4.jl`, `wavespeed_dev.jl`,
`bench_eig.jl`, `test_cuda.jl` round it out. Everything else under `gpu/` is a
`validate_*`/`dump_*`/`bench_*` harness.

### Module-wiring rule (avoid double-include of `ReconDev`)

`realize_dev.jl`/`residual3d_gpu.jl` reference the already-loaded `ReconDev` via
`using ..ReconDev` (not a second `include`). Each consumer `include`s `recon_dev.jl`
as a sibling **first**. Re-including a module file defines a stale second copy — this
bit the recon/realize ports; keep the include order and the `..ReconDev` reference.

## Residual: one rectangular kernel set

`residual3d_box_gpu!(R, M, nx, ny, nz, dx, Ma; …, flat=nothing)` is the order-2
unsplit `Lx+Ly+Lz` HLL residual on a rectangular `(35,nx,ny,nz)` field with **outflow
on all 6 faces** (index-clamp). It is a *strict generalization*: with `nx==ny==nz` it
reproduces the old cubic kernels bit-for-bit, so the cubic `residual3d_gpu!` is a thin
wrapper that reshapes its `Fbuf` as the box face-scratch and delegates. There is one
set of 6 kernels (`_fhat_{x,y,z}_g!`, `_diff_{x,y,z}_g!`), not two.

- 2D = `nz=1` (the velocity space is always 3D/35-moment; `flag2D` is a legacy no-op).
  `Lz=0` on z-uniform data.
- The optional `flat` kwarg lets the timestep loop preallocate the face-scratch and
  reuse it every step (alloc-free).

## Multi-GPU: z-slab decomposition, halo-only host transfer

Each rank owns a z-slab interior `(35,n,n,nz_loc)` **resident** on its GPU. Per RK
stage:
1. copy the interior into an **extended** slab `(35,n,n,nz_loc+2·halo)`, `halo=2`;
2. refresh the `halo` z-ghost planes by host-staging them (GPU→host `copyto!` →
   `MPI.Sendrecv!` → host→GPU), with **outflow replicas** at the global z-boundary
   (matching the cubic clamp);
3. run `residual3d_box_gpu!` on the extended slab and keep only the interior planes.

Because interior cells never reach the extended z-edges (`halo=2` > stencil reach 2),
they read their real ±2 neighbors → the interior residual is **bit-identical** to the
single-GPU full-domain residual. The global CFL `dt` is `Allreduce(max)` of per-rank
max wave speed (`max` is exact → `dt`, and the whole march, are bit-identical to
single-GPU). Only `35·n·n·halo` floats cross the host per stage.

## The one timestep module: shared RK3 body + speed kernel

`Timestep3DGPU` holds both marches. They share **one** `_rk3_step!` (the SSP-RK3 stage
sequence + per-cell projection) and **one** rectangular `_speed_box_kernel!` (CFL). They
differ only in their residual operator `L!` (cube vs extended-slab-with-exchange) and
`dt` (local vs `Allreduce`). `march3d_gpu!` accepts any rectangular extent (incl. 2D
`nz=1`); `march3d_slab_gpu!` asserts `nz_loc >= halo` (z-slab is 3D-only — 2D is
single-GPU by design).

## What is intentionally NOT single-sourced: the eigensolvers

The wave-speed eig is two *legitimately different* implementations, not duplication —
LAPACK cannot run inside a CUDA kernel:

- **4×4 non-symmetric block**: CPU `jac4_realpart_minmax` (LAPACK `dgeev`) vs GPU
  `schur4` (custom Francis double-shift QR). They differ by **rel ~4.6e-8** on
  ill-conditioned high-Ma companion blocks — above the golden 1e-10, so the CPU keeps
  LAPACK and the GPU residual matches the CPU only to ~1e-6 (the wave-speed-limited
  floor). fp64 is mandatory; fp32 is catastrophic on these blocks.
- **6×6 symmetric (realizability) + closure**: CPU LAPACK vs GPU analytic/Jacobi
  in-kernel. Same math, different numerics by platform necessity.

Everything else (the algebra) is single-sourced.
