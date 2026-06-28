# Results — performance & accuracy

Hardware: 2× Quadro RTX 6000 (sm_75, weak FP64 ~1:32). All numbers are a
**conservative floor** — a datacenter GPU (V100/A100/H100) with real FP64 would be
substantially faster.

## Accuracy (vs CPU / MATLAB / single-GPU)

| comparison | metric |
|---|---|
| batched symmetric 6×6 eig (cuSOLVER) vs CPU LAPACK | 1.9e-14 (machine-identical) |
| `schur4` 4×4 vs LAPACK (general blocks) | LAPACK-accurate; ~4.6e-8 on companion blocks (fp64) |
| realizability kernel vs CPU `realizable_3D_M4` | machine precision |
| 2-GPU realizability vs CPU ref | rel 3.3e-15 |
| GPU 3D residual vs CPU `residual_ho_3d!` | rel ~5e-11 (GATE PASS) |
| box residual == cubic residual | **0.0** (bit-identical) |
| z-slab residual vs single-GPU (1/2/4 ranks) | **0.0** (bit-identical) |
| multi-GPU timestep vs single-GPU (1/2/4 ranks) | **0.0** dt seq + final field |
| **GPU 2D realize+flux vs MATLAB golden** | **rel 4.4e-16** (machine precision) |
| GPU 2D residual vs MATLAB-ported CPU | rel 3.9e-7 (GATE PASS) |
| GPU 2D (nz=1) timestep vs z-homogeneous 3D slice | **0.0** |
| single-GPU timestep vs CPU | dt EXACT; density rel ~1.9e-5; high-order moments conditioning-limited |

## Throughput / speedup

| kernel / path | speedup | notes |
|---|---|---|
| symmetric 6×6 eig | 11× solve-only / 6.4× end-to-end | vs 1-core CPU LAPACK, 128³ matrices |
| non-symmetric 4×4 `schur4` | ~425× vs 1-thread | the original "impossible" piece, custom kernel |
| flux closure | ~12× solve-only | (1D component bench, historical — script pruned) |
| wave-speed | ~85× solve-only | (historical) |
| 3D residual | ~210× vs 1-thread CPU | ~4–9× vs a full MPI CPU socket |
| **multi-GPU resident (n=256, real per-cell kernel)** | **compute 1.93× / overall 1.83×** on 2 GPUs | halo only 7% of wall; near-ideal compute scaling |

### Multi-GPU scaling detail (resident field, halo-only transfer)

n=256 (16.8M cells), 20 steps, realizability kernel as the per-cell workload:

| | 1 GPU | 2 GPUs |
|---|---|---|
| compute (resident) | 21.74 s | 11.26 s (**1.93×**) |
| overall (compute + halo) | 21.76 s | 11.88 s (**1.83×**) |
| throughput | 15.4 Mcells/s | 28.2 Mcells/s |
| halo exchange | — | 0.89 s (7% of wall) |

Contrast the naive *full-field* host round-trip (e.g. the convenience
`realizable_batched(M_host,Ma)`): only 1.48× and ~4.5 Mcells/s — transfer-bound.
Keep data resident, move only halos (~6× higher absolute throughput). Remaining
optimization: overlap halo exchange with interior compute via CUDA streams (would
shrink the 7%).

## Capability matrix

| | single-GPU | multi-GPU (z-slab) |
|---|---|---|
| per-cell physics (flux/realize/wavespeed) | ✅ | ✅ |
| 3D order-2 SSP-RK3 timestep | ✅ `march3d_gpu!` | ✅ `march3d_slab_gpu!`, bit-identical |
| 2D (`nz=1`) timestep | ✅ `march3d_gpu!` | ✗ by design (slab is z-only) |
| CUDA-aware MPI | n/a | ✗ system OpenMPI `--without-cuda` (host-staged) |
