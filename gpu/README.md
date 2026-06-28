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
`flux_closure35_dev(35 scalars) -> NTuple{105}` (`flux_closure_dev.jl`) + CUDA kernel
(`flux_closure_gpu.jl`). Validated vs CPU on 21,296 real states: **max rel error 4.0e-14**.

| flux closure (B=2.1M, fp64) | throughput | speedup |
|---|---|---|
| CPU 1-thread (alloc-free dev) | 5.4 Mcell/s | — |
| GPU solve-only (resident) | 65.6 Mcell/s | **12.2×** |
| GPU end-to-end (incl H2D/D2H) | 1.5 Mcell/s | 0.3× (PCIe-bound) |

(12× is vs an already-optimized alloc-free CPU baseline — a conservative, honest number. End-to-end is
transfer-bound by design; the closure runs on resident data in a real GPU solver.)

## Status & remaining for a full GPU residual

**On GPU now (per-cell physics):** eigensolves (symmetric cuSOLVER + non-symmetric Schur kernel) and the
flux closure. No remaining *algorithmic* blockers — the rest is arithmetic + array ops.

**Remaining:** reconstruction (`to_recon_vars`/MUSCL, per-face arithmetic — ports like the flux); the GPU
wave-speed path (`realize_and_speed` = non-sym Schur `v6` + symmetric closure `v5` + hyperbolicity
correction); HLL combine; residual stencil assembly; SSP-RK3 on device; `projection35` realizability; and
CUDA-aware MPI halo exchange for multi-GPU.

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
