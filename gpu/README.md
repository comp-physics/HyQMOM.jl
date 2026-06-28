# GPU acceleration — prototype & findings

The 3D profile (`docs/diffusion-reduction-results.md`, profiling notes) shows the high-order step is
~60% **small-matrix eigenvalue computation** — two consumers:
- the **non-symmetric 4×4** wave-speed block (`jac4_realpart_minmax`, LAPACK `dgeev`), and
- the **symmetric** realizability (`delta2star3D`, 6×6) + closure (symmetric tridiagonal) eigensolves.

By Amdahl, a GPU port that leaves the eigensolves on the CPU caps total speedup at ~2.6×, so the
eigensolve is the decisive port target. This directory holds the de-risking prototype.

## Result (Quadro RTX 6000, FP64)

Batched **symmetric 6×6** eigenvalues (the realizability matrices), GPU cuSOLVER `syevjBatched` vs
single-core CPU LAPACK, 2,097,152 matrices (= 128³):

| | throughput | speedup | accuracy |
|---|---|---|---|
| CPU LAPACK (1 core) | 0.21 Mmat/s | — | — |
| GPU end-to-end (incl H2D) | 1.36 Mmat/s | **6.4×** | 1.9e-14 vs CPU |
| GPU solve-only (data resident) | 2.28 Mmat/s | **11×** | machine-identical |

**This is a conservative floor:** the RTX 6000 (Turing/Quadro) has weak FP64 (~1:32); a datacenter GPU
(V100/A100/H100) would give substantially more. The "solve-only" number is the realistic one for an
all-GPU solver where the moment field lives on-device (no per-step transfer).

**Validated:** the GPU eigensolve path is accurate (machine precision) and fast for the **symmetric**
eig (realizability + closure). cuSOLVER `syevjBatched` is the right tool; `version="local"` toolkit works.

## Non-symmetric 4×4 wave-speed eig — SOLVED with a custom batched kernel

No GPU library batches non-symmetric eig (cuSOLVER and MAGMA both confirmed lacking; `cusolverDnXgeev`
is one-matrix-per-call). A Gershgorin bound is far too loose (3,483×–23M×) and an analytic quartic is
numerically fragile near the defective wave-speed extremes. So we built a **custom batched real-Schur QR
kernel** (`schur4.jl` = CPU prototype, `schur4_gpu.jl` = CUDA kernel): scale → Householder Hessenberg →
Francis implicit double-shift QR + deflation → 1×1/2×2 block real parts, **eigenvalues-only, fp64, one
matrix per thread**, with a `status` flag → CPU/LAPACK fallback for the rare flagged matrices.

Validated vs LAPACK on **262,144 real evolved Ma=10/100 blocks**: max relative error **6.3e-8, 0% flagged**
(matches the CPU prototype). 200k random non-symmetric: 9.5e-14, 0.045% → fallback by design.

| 4×4 non-sym (B=2.1M, fp64) | throughput | speedup |
|---|---|---|
| CPU LAPACK (1 core) | 0.185 Mmat/s | — |
| GPU solve-only (resident) | 78.5 Mmat/s | **425×** |
| GPU end-to-end (incl H2D) | 15.4 Mmat/s | 83× |

(425× is vs single-thread; production CPU uses buffered `dgeev` + MPI many-core, so a fair GPU-vs-socket
number is smaller — but solve-only is the right metric for an all-GPU solver where data stays on device.)

**fp64 is required:** in fp32 the ill-conditioned high-Ma companion blocks hit percent-level error.

### Net: the entire eigensolve bottleneck (~60% of the step) is now GPU-viable
- symmetric (realizability 6×6 + closure) → cuSOLVER `syevjBatched` (11×, above)
- non-symmetric (wave-speed 4×4) → this custom kernel (425× solve-only)

## Flux closure on GPU — DONE

`Flux_closure35_3D` (pure per-cell arithmetic) ported to an alloc-free device function
`flux_closure35_dev(35 scalars) -> NTuple{105}` (`src/numerics/flux_closure_dev.jl`) + CUDA kernel
(`flux_closure_gpu.jl`). Validated vs CPU on 21,296 real states: **max rel error 4.0e-14**.

| flux closure (B=2.1M, fp64) | throughput | speedup |
|---|---|---|
| CPU 1-thread (alloc-free dev) | 5.4 Mcell/s | — |
| GPU solve-only (resident) | 65.6 Mcell/s | **12.2×** |
| GPU end-to-end (incl H2D/D2H) | 1.5 Mcell/s | 0.3× (PCIe-bound) |

(12× is vs an already-optimized alloc-free CPU baseline — a conservative, honest number. End-to-end is
transfer-bound by design; the closure runs on resident data in a real GPU solver.)

## Wave-speed path + end-to-end first-order residual — DONE

- **Wave-speed path** (`realize_and_speed` = jacobian15 3×3/4×4 blocks → eig3 + Schur kernel + symmetric
  closure `v5` + hyperbolicity correction + `max(v5,v6)`): `wavespeed_dev.jl`/`wavespeed_gpu.jl`. Validated
  vs CPU on 8192 real states × 3 axes: **max rel err 6.4e-13**, hyperbolicity-correction branch matches.
  **85× solve-only** / 49× end-to-end. *(@fastmath must stay OFF here — GPU rsqrt flips the complex-root
  discriminant at the hyperbolicity boundary.)*
- **End-to-end first-order 1D residual** (`residual1d_gpu.jl`): composes flux + wave-speed + HLL + stencil
  on device. Validated vs CPU `residual_1d(order=1)` on N=256 Ma=100: **max rel err 2.3e-9** (worst cell
  agrees to 9 digits). The full first-order HLL residual of the 35-moment scheme runs end-to-end on GPU.

## Full solver on GPU — reconstruction, projection, 3D residual, timestep

- **High-order reconstruction** (`recon_dev.jl`) → order-2 1D residual: **7e-12** vs CPU.
- **Realizability projection** `realizable_3D_M4` (`src/realizability/realize_dev.jl`/`realize_gpu.jl`, in-kernel 6×6
  symmetric Jacobi min-eig): **3.3e-15** vs CPU, sign decision matches on every cell, 64× solve-only.
- **3D order-2 residual** (`residual3d_gpu.jl`): **1.4e-10** vs CPU on gradient-rich real states.
- **3D timestep loop** (`timestep3d_gpu.jl`): SSP-RK3 + per-stage projection + 3D-CFL dt, fully resident.

| 3D order-2 residual (real states) | throughput | speedup |
|---|---|---|
| CPU `residual_ho_3d!` (1 thread) | 0.0054 Mcell/s | — |
| GPU (n=128) | 1.15 Mcell/s | **~210× vs 1 thread** |

(≈4–9× vs a full MPI CPU socket; and this is a weak-FP64 Quadro RTX 6000 — a datacenter GPU would be more.)

**Multi-step validation caveat (physics, not a bug):** at the crossing-jet shock the highest-order moments
are FP-conditioning-limited (`dt·R ≫ M`) — CPU itself diverges O(1) under a 1e-10 perturbation at the same
cell/moment as GPU-vs-CPU. So the GPU march is validated by **per-step bit-match** (residual 1e-10,
projection 1e-14, dt exact) + **multi-step conserved/low-order moments** (density ~3e-4, momentum ~1e-3) +
stability/ρ-range match — not by a high-order-moment multi-step bit-gate (meaningless here, for CPU too).

## Status: the full 3D high-order solver runs on one GPU

The entire pipeline — eigensolves, flux closure, wave-speed path, reconstruction, realizability projection,
3D residual, and the SSP-RK3 timestep — runs on GPU, each piece validated vs CPU (1e-10–1e-15 per step).
**No algorithmic blockers remain.**

**Remaining for production:** kernel-fusion / per-stage-split perf work (the full step is ~0.34 Mcell/s,
bottlenecked by the per-cell 6×6-Jacobi min-eig in the projection), and CUDA-aware MPI halo exchange for
multi-GPU (the 1024³ target).

## Environment (PACE)

CUDA.jl artifact downloads exceed the home-dir quota, and large artifacts can hit network "Data Error".
Use a scratch Julia depot (home as read-only fallback):

```bash
export JULIA_DEPOT_PATH=/storage/scratch1/6/$USER/julia_depot:$HOME/.julia
export TMPDIR=/storage/scratch1/6/$USER/tmp          # must exist
# system CUDA toolkit (avoids re-downloading the big runtime artifact):
CUH=$(module show cuda/12.6.1 2>&1 | sed -n 's/.*CUDA_HOME","\(.*\)").*/\1/p')
export CUDA_PATH=$CUH
```

Run on a GPU node (`gpu-rtx6000`/`gpu-v100`):
```bash
julia gpu/test_cuda.jl     # toolchain check: CUDA.functional() + trivial kernel
NBATCH=2097152 julia gpu/bench_eig.jl   # batched-eig GPU vs CPU benchmark
```
The scripts `Pkg.activate(@__DIR__)` — first run `Pkg.add("CUDA")` in this dir (writes to the scratch
depot). `gpu/test_cuda.jl` also writes `LocalPreferences.toml` with `[CUDA_Runtime_jll] version="local"`
if you want the system toolkit instead of artifacts.
