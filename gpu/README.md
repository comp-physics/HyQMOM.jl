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

**Still open:** the **non-symmetric 4×4** wave-speed eig has no batched cuSOLVER routine. Options:
(a) a custom in-kernel batched solver (fixed-iteration QR), or (b) a cheaper wave-speed *bound*
(e.g. Gershgorin) that over-estimates speeds → slightly more HLL diffusion but no eigensolve. Analytic
quartic was rejected (numerically fragile near defective eigenvalues — exactly the wave-speed extremes).

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
