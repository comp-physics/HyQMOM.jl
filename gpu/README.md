# GPU acceleration — prototype & findings

## Multi-GPU + MPI (validated, 2× Quadro RTX 6000)

GPU+MPI works on this node. **CUDA-aware MPI is NOT required** — the system OpenMPI 4.1.8 is built
`--without-cuda`, so halos are **host-staged** (GPU→host, `MPI.Sendrecv!`/`Gatherv!`, host→GPU). Each
MPI rank binds a distinct device with `CUDA.device!(rank % CUDA.ndevices())`.

Two validators (run with `srun --mpi=pmix -n 2 --gpus=2`, env per the recipe below):
- `validate_gpu_mpi_smoke.jl` — rank↔GPU binding, host-staged ring halo exchange, cross-GPU `Allreduce`.
  Result: rank0→GPU0, rank1→GPU1, halo + reduction correct → **PASS**.
- `validate_gpu_mpi_realize.jl` — the **real** realizability projection kernel, domain-decomposed across
  both GPUs (column slabs), MPI scatter→compute→`Gatherv!`. 21,296 cells (10,648/GPU) vs the CPU
  reference: **max rel 3.3e-15** (machine precision) → **PASS**.

Run recipe (env that makes CUDA.jl + MPI.jl coexist; MPI bound to system OpenMPI, host-staged BTLs):
```bash
export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:$HOME/.julia
export TMPDIR=/storage/scratch1/6/sbryngelson3/tmp
OMPI=/usr/local/pace-apps/spack/.../openmpi-4.1.8-iit4xaslnjxkchcc6n62b5kluzibl2v2
export LD_LIBRARY_PATH=$OMPI/lib:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader     # host-staged: no UCX/CUDA-aware needed
srun --mpi=pmix -n 2 --gpus=2 julia --project=gpu/gpuenv2 gpu/validate_gpu_mpi_realize.jl
```
`gpu/gpuenv2` now carries `MPI` + `MPIPreferences` (bound `binary="system"`, `abi="OpenMPI"` via the
project `LocalPreferences.toml`) alongside `CUDA`. Per-cell kernels (flux, realizability) decompose with
zero halo; the stencil residual (`residual3d_gpu`) would add a z-slab host-staged halo exchange — the
smoke test already proves that exchange path. That domain-decomposed residual is the remaining piece.

### Multi-GPU scaling (resident field, halo-only host transfer) — `bench_gpu_mpi_resident.jl`

The right design: the moment field stays **resident** on each GPU for the whole run; only the thin halo
z-planes are staged host→MPI→host each step (pinned host buffers, preallocated GPU ghost buffers, direct
`copyto!`). Benchmark (n=256 cube = 16.8M cells, 20 steps, realizability kernel as the per-cell workload;
2× Quadro RTX 6000):

| | 1 GPU | 2 GPUs | speedup |
|---|---|---|---|
| compute (resident) | 21.74 s | 11.26 s | **1.93×** (near-ideal) |
| overall (compute + halo) | 21.76 s | 11.88 s | **1.83×** |
| throughput | 15.4 Mcells/s | 28.2 Mcells/s | — |
| halo exchange | — | 0.89 s (7% of wall) | — |

The **compute scales near-linearly (1.93×)** — multi-GPU works. Halo is only 7% of wall (73 MB/face vs a
4.7 GB resident field), and would shrink further by overlapping exchange with compute via CUDA streams.
Contrast the naive *full-field* host round-trip (the convenience `realizable_batched(M_host,Ma)` path):
only 1.48× and ~4.5 Mcells/s — transfer-bound. Keeping data resident and moving only halos is ~6× higher
absolute throughput. (Absolute Mcells/s here is worst-case: random inputs force every cell through the
projection-correction branch; realistic fields where many cells skip correction run ~3–4× faster. The
scaling *ratio* is data-independent.)

### z-slab decomposed 3D STENCIL residual — built + validated bit-for-bit

The full unsplit `Lx+Ly+Lz` HLL residual now runs domain-decomposed across GPUs (not just per-cell work).
`residual3d_box_gpu!` (added to `residual3d_gpu.jl`) generalizes the cubic `residual3d_gpu!` to rectangular
`(nx,ny,nz)` extents with outflow on all 6 faces — a *strict* generalization (nx==ny==nz reproduces the
cubic kernel **bit-for-bit**, verified). A z-slab is then that box run on each rank's EXTENDED slab
`(35, n, n, nz_loc + 2·halo)` whose `halo=2` z-ghost planes are filled by the host-staged halo exchange
(neighbor interior planes, or outflow replicas at the global z-boundary, matching the cubic index-clamp).
Interior cells never reach the extended z-edges, so they read their real ±2 neighbors → identical math.

Validators (`validate_residual3d_box.jl`, `validate_slab_residual_mpi.jl`):
- box vs cubic (single GPU, n=16,24): **max abs diff 0.0** — bit-identical generalization.
- z-slab vs single-GPU full domain residual, **max abs diff 0.0 (BIT-IDENTICAL)** for **1, 2, and 4 ranks**
  (4 ranks on 2 GPUs exercises interior slabs with two-sided halos). Field resident per GPU; only the
  `35·n·n·halo` ghost planes touch the host for the MPI `Sendrecv!`.

This is the actual stencil timestep building block running multi-GPU with resident data + halo-only
transfer — the remaining piece from the scaling note above is now done and correctness-locked.

### Full multi-GPU SSP-RK3 timestep loop — `timestep3d_gpu.jl`, bit-identical

`march3d_slab_gpu!` (module `Timestep3DGPU`, alongside the single-GPU `march3d_gpu!`) wires the z-slab
residual into the complete order-2 SSP-RK3 time loop. Both marches share ONE RK3-step helper
(`_rk3_step!`) and ONE rectangular CFL speed kernel — they differ only in their residual operator `L!`
(cube vs extended-slab-with-halo-exchange) and `dt` (local vs `Allreduce(max)`). Each rank keeps its
interior slab `(35,n,n,nz_loc)` resident; per RK stage it refreshes the halo (host-staged `Sendrecv!`),
runs `residual3d_box_gpu!` on the extended slab, then RK-combines + projects the contiguous interior.
`dt = Allreduce(max)` of per-rank max speed — `max` is exact, so `dt` equals single-GPU bit-for-bit.

Validated against single-GPU `march3d_gpu!` (`validate_timestep3d_mpi.jl`, n=24, 5 steps, on-device CFL on
both): for **1, 2, and 4 ranks** (4 on 2 GPUs = interior slabs), both the **dt sequence** and the **final
field** match with **max abs diff 0.0 — BIT-IDENTICAL**. The full timestep now runs multi-GPU with only
halo planes crossing the host; the per-step compute is the resident-field workload benchmarked above.

### 2D (Nz=1) support — single-GPU, validated against MATLAB

A "2D" run is just a `nz=1` spatial grid (the 35-moment velocity space is always 3D; `flag2D` is a legacy
no-op). `march3d_gpu!` and `residual3d_box_gpu!` take **rectangular** `(nx,ny,nz)` extents, so 2D is
`(35,nx,ny,1)` — `Lz=0` on z-uniform data, no spurious z-transport. (Multi-GPU is **not** supported in 2D:
the slab march decomposes z, which can't split `nz=1`; `march3d_slab_gpu!` asserts `nz_loc >= halo`. 2D runs
on one GPU — by design.)

Validation:
- **Direct vs MATLAB** (`validate_2d_flux_vs_matlab.jl`): the GPU realizability kernel + flux device
  function (the per-cell physics the 2D residual uses) vs the MATLAB golden `test_flux_eigenvalues_golden.mat`
  (`flag2D=1`, Ma=0.5) → **max rel 4.4e-16** (machine precision).
- **Composed residual vs MATLAB-ported CPU** (`validate_2d_residual_vs_cpu.jl`): GPU 2D residual vs CPU
  `residual_ho_3d!` (the MATLAB port) → **rel 3.9e-7, GATE PASS** (the ~e-7 floor is `schur4`(GPU) vs
  LAPACK(CPU) in the wave-speed eig — the one place the GPU intentionally differs).
- **Self-consistency** (`validate_2d_timestep.jl`): GPU 2D `nz=1` SSP-RK3 timestep is **bit-identical (0.0)**
  to the interior plane of a z-homogeneous 3D run.

(Reading `.mat` here needs an MPI-free env — `HDF5_jll`→`OpenMPI_jll` clashes with the system-MPI binding;
`dump_matlab_flux_golden.jl` / `dump_cpu_2d_residual.jl` regenerate the f64 references.)

## Single-source port status (branch `gpu-single-source-port`)

The original prototype kept a separate `gpu/*_dev.jl` copy of each per-cell kernel beside the CPU
implementation — duplicated algebra. That duplication has been **eliminated** for everything that was
genuinely the same math written twice: the shared, alloc-free, GPU-compilable device kernels now live
in `src/`, and the CPU functions **delegate** to them (public signatures unchanged, golden-clean):

| component | shared kernel (in `src/`) | CPU entry that delegates | golden battery |
|---|---|---|---|
| flux closure | `src/numerics/flux_closure_dev.jl` | `Flux_closure35_3D` | 0 fails (rel 2.7e-13) |
| reconstruction | `src/numerics/recon_dev.jl` | `to_recon_vars` / `from_recon_vars` | 0 fails (rel 2.7e-13) |
| realizability | `src/realizability/realize_dev.jl` | `realizable_3D_M4` | 0 fails (rel 2.7e-13) |

**FP-parity gotcha (load-bearing):** the shared `@fastmath` central-moment helpers
(`_recon_centrals`, `_c4tom4_35`) are `@noinline`, NOT `@inline`. `@fastmath` lets LLVM reassociate the
cancellation-heavy central-moment formulas depending on surrounding context; inlined, they drift ~1 ULP
from the standalone autogen `M4toC4_3D`/`C4toM4_3D`, which the `1/sC200^k` standardization amplifies to
~2e-7 on deep-vacuum (ρ~1e-5) cells. `@noinline` pins them to the autogen's compilation. The realizability
CPU path delegates only the **correction** (`realizable_3D_M4_corr_dev`) and finishes with the autogen
`standardized_to_M4`, so the reconstructed moments stay bit-for-bit with the reference.

**Wave-speed is intentionally NOT single-sourced** (`gpu/wavespeed_dev.jl` + `gpu/schur4.jl` stay here).
Its core is the eigenvalue computation, and the CPU/GPU versions are two *legitimately different*
implementations — not duplication — because LAPACK cannot run inside a CUDA kernel:
- 4×4 non-symmetric block: CPU `jac4_realpart_minmax` (LAPACK `dgeev`) vs GPU `schur4` (custom Francis QR).
  These differ by rel ~4.6e-8 on ill-conditioned high-Ma companion blocks (verified, `validate_schur4.jl`)
  — far above the golden 1e-10, so the CPU cannot adopt `schur4` without breaking byte-parity.
- closure / 3×3: CPU uses LAPACK (Golub-Welsch Chebyshev) and a LAPACK block eig; the device path uses
  analytic in-kernel solvers. Same math, different numerics by platform necessity.

So the port is **complete**: the genuinely-duplicated algebra is shared; the only per-platform code left
is the eigenvalue backend, which *must* differ between CPU (LAPACK, golden-accurate) and GPU (custom
kernels, LAPACK-free).

---


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

> **Note (prototype scripts pruned for minimal footprint).** The sections below document the *incremental
> 1D / single-component* milestones that led to the full 3D solver. Their standalone batched benchmark/
> validation scripts (`schur4_gpu.jl`, `flux_closure_gpu.jl`, `wavespeed_gpu.jl`, `residual1d_gpu.jl`,
> `residual2_gpu.jl`, `timestep_gpu.jl` + their `validate_*`/`bench_*`) have been **removed** — they are not
> part of the 3D multi-GPU capability (whose closure is just `timestep3d_gpu` → `residual3d_gpu` /
> `realize_gpu` → `wavespeed_dev` → `schur4`, plus the `src/*_dev` single-source kernels). The speedup
> numbers below are **historical milestones**; the scripts that produced them are recoverable from git
> history (branch `gpu-single-source-port`, before the prune commit). The custom Schur eig itself lives on
> in `schur4.jl` (validated by `validate_schur4.jl`) and is used by the 3D path via `wavespeed_dev.jl`.

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
