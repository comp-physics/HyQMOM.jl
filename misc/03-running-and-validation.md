# Running & validation

Commands below are written in the **PACE form** (`$JULIA` = your Julia 1.10+, launched
with `srun --mpi=pmix` against a system MPI). On any other machine use the portable form
from [`01-environment.md`](01-environment.md) instead: `julia --project=gpu/gpuenv2 <script>`
for single-GPU, and `mpiexec() -n N julia …` (bundled MPI) for multi-GPU. Nothing here
depends on a specific Julia/MPI/CUDA *version* — substitute your own. Set the env block
from `01-environment.md` first; for MPI runs on PACE also export
`OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader`. Reference data lives in your `DATA` dir
(PACE: `/storage/scratch1/6/sbryngelson3/gpudata`). Filter harmless `PMIX ERROR` lines.

## Validators (gpu/) and expected results

| validator | ranks/GPUs | what it checks | expected |
|---|---|---|---|
| `test_cuda.jl` | 1 | CUDA functional, device visible | prints device |
| `bench_eig.jl` | 1 | cuSOLVER batched **symmetric 6×6** eig vs CPU LAPACK | ~11× solve-only, accuracy 1.9e-14 |
| `validate_schur4.jl` | 1 | custom 4×4 `schur4` vs LAPACK on real jacobian blocks | rel ≤1e-6 (note ~4.6e-8 on companion blocks) |
| `validate_realize_gpu.jl` | 1 | batched realizability kernel vs CPU `realizable_3D_M4` | machine precision |
| `validate_residual3d_box.jl` | 1 | box residual == cubic residual (n=16,24) | max abs **0.0** |
| `validate_residual3d_gpu.jl` | 1 | GPU 3D residual vs CPU `residual_ho_3d!` (the MATLAB port) | rel ~5e-11, GATE PASS |
| `validate_timestep3d_gpu.jl` | 1 | single-GPU SSP-RK3 timestep vs CPU | dt EXACT; density rel ~1.9e-5; high-order moments conditioning-limited (see note) |
| `validate_gpu_mpi_smoke.jl` | 2 | rank↔GPU binding, host-staged ring halo, `Allreduce` | PASS |
| `validate_gpu_mpi_realize.jl` | 2 | realizability split across 2 GPUs vs CPU ref | rel 3.3e-15 |
| `validate_slab_residual_mpi.jl` | 1/2/4 | z-slab residual vs single-GPU full domain | max abs **0.0** (all rank counts) |
| `validate_timestep3d_mpi.jl` | 1/2/4 | multi-GPU timestep vs single-GPU (dt seq + final field) | **0.0** BIT-IDENTICAL |
| `bench_gpu_mpi_resident.jl` | 1 vs 2 | resident-field + halo-only scaling (n=256) | compute 1.93×, overall 1.83× |
| `validate_2d_timestep.jl` | 1 | 2D `nz=1` timestep vs z-homogeneous 3D slice | max abs **0.0** |
| `validate_2d_flux_vs_matlab.jl` | 1 | GPU realize+flux vs **MATLAB** golden (flag2D=1, Ma=0.5) | max rel **4.4e-16** |
| `validate_2d_residual_vs_cpu.jl` | 1 | GPU 2D residual vs MATLAB-ported CPU `residual_ho_3d!` | rel 3.9e-7, GATE PASS |

> **Timestep-vs-CPU note:** the multi-step high-order *cross moments* are FP-conditioning
> limited at shocks (`dt·R ≫ M`) and are NOT bit-reproducible CPU↔GPU — a CPU
> self-perturbation of 1e-15 also diverges O(1). The meaningful gates are the EXACT dt
> sequence and the density/low-order moments (rel ~1e-5). Don't chase the high-order
> moment "GATE FAIL" — it is documented, expected physics-of-FP, not a bug.

## Reference data files (`DATA`) and how to regenerate

Most validators read pre-dumped fp64 references so they don't need the CPU/MATLAB at
run time. Layout convention: cell `(i,j,k)` → 35 contiguous, then i,j,k = `(35,…)`
column-major.

| files | used by | generator |
|---|---|---|
| `proj_M.f64`, `proj_ref.f64`, `proj_Ma.f64`, `proj.meta` | realize, slab, 2D | CPU `realizable_3D_M4` dump (build_states) |
| `r3d_M.f64`, `r3d_R.f64`, `r3d.meta` | `validate_residual3d_gpu` | CPU `residual_ho_3d!` (3D) dump |
| `r2d_M.f64`, `r2d_R.f64`, `r2d.meta` | `validate_2d_residual_vs_cpu` | **`gpu/dump_cpu_2d_residual.jl`** (main project, CPU) |
| `flxg_in.f64`, `flxg_out.f64` | `validate_2d_flux_vs_matlab` | **`gpu/dump_matlab_flux_golden.jl`** (MPI-free env w/ MAT) |
| `real_blocks.f64`, `real_lapack.f64` | `validate_schur4` | CPU jacobian-block + LAPACK dump |
| `step3d_M0.f64`, `step3d_Mf.f64`, `step3d_dts.f64` | `validate_timestep3d_gpu` | CPU `march`/`residual_ho_3d!` dump |

The two dump scripts that are committed and may need re-running:

```bash
# 2D CPU residual reference (MATLAB-ported CPU; run in the MAIN project, not gpuenv2)
#   needs the system-OpenMPI LD_LIBRARY_PATH; uses the singleton env vars
env <singleton vars> $JULIA --project=. gpu/dump_cpu_2d_residual.jl

# 2D MATLAB flux golden -> f64  (run in an MPI-FREE env that can load MAT/HDF5;
#   do NOT put system OpenMPI on LD_LIBRARY_PATH — see 04-gotchas)
$JULIA --project=/storage/scratch1/6/sbryngelson3/matenv gpu/dump_matlab_flux_golden.jl
```

The MATLAB goldens themselves are in `legacy/3D_MATLAB/tests/goldenfiles/test_*_golden.mat`
(NOT `test/goldenfiles/`, which holds the full-sim `.mat`).

## Reproduce the headline 2D-vs-MATLAB result

```bash
# 1) dump the MATLAB flux golden (MPI-free env)
$JULIA --project=/storage/scratch1/6/sbryngelson3/matenv gpu/dump_matlab_flux_golden.jl
# 2) compare GPU device kernels to it
srun --mpi=pmix -n 1 --gpus=1 $JULIA --project=gpu/gpuenv2 gpu/validate_2d_flux_vs_matlab.jl
# expect: realizable + Fx/Fy/Fz -> max rel 4.4e-16  MATCHES MATLAB (<1e-10)
```
