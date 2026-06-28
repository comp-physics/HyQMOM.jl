#!/usr/bin/env julia
# validate_residual3d_gpu.jl
#
# CAPSTONE validation + benchmark for the on-device 3D order-2 (MUSCL) HLL
# residual (`gpu/residual3d_gpu.jl`, module `Residual3DGPU`) vs the CPU
# reference `residual_ho_3d!` (`src/numerics/highorder_3d.jl`, order=2),
# the production-relevant like-for-like comparison.
#
#   * HEADLINE gate: max ABS and REL error over all n^3 x 35 of R_gpu vs the
#     on-disk CPU 3D residual reference. GATE: rel <= 1e-6.
#   * BENCHMARK: GPU 3D residual throughput (Mcell/s, solve-only, resident) at
#     n=64 and n=128, with speedup vs the measured CPU 3D baseline 0.0054 Mcell/s
#     (single-thread; production CPU is MPI many-core, so a fair GPU-vs-socket is
#     lower).
#
# ENV: home is OVER QUOTA — read inputs from /storage/scratch1/6/sbryngelson3/gpudata,
# write nothing under home. Run with gpuenv2, depot on scratch.

import Pkg
Pkg.activate(joinpath(@__DIR__, "gpuenv2"))

using CUDA, Printf
include(joinpath(@__DIR__, "residual3d_gpu.jl"))
using .Residual3DGPU

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const DATA = "/storage/scratch1/6/sbryngelson3/gpudata"
const HO_VACUUM_FLOOR = 0.001
const CPU_BASELINE_MCELLS = 0.0054   # measured single-thread residual_ho_3d! n=24 order2

# ---------------------------------------------------------------------------
# Load reference. Flatten on disk: cell (i,j,k) -> 35 contiguous, then i,j,k.
# i.e. exactly (35, n, n, n) column-major.  r3d.meta: n / dx / Ma / halo.
# ---------------------------------------------------------------------------
meta = split(strip(read(joinpath(DATA, "r3d.meta"), String)), '\n')
n    = parse(Int,     strip(meta[1]))
dx   = parse(Float64, strip(meta[2]))
Ma   = parse(Float64, strip(meta[3]))
halo = parse(Int,     strip(meta[4]))

M    = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_M.f64")))), 35, n, n, n)
Rref = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_R.f64")))), 35, n, n, n)
@printf("loaded n=%d  dx=%.6g  Ma=%.4g  halo=%d  HO_VACUUM_FLOOR=%.3g   (M in [%.3g, %.3g])\n",
        n, dx, Ma, halo, HO_VACUUM_FLOOR, extrema(M)...)

Mh = Array{Float64,4}(M)

# ---------------------------------------------------------------------------
# GPU 3D residual
# ---------------------------------------------------------------------------
Rgpu = residual3d_gpu(Mh, n, dx, Ma; vacuum_floor=HO_VACUUM_FLOOR, project_faces=true)
@assert all(isfinite, Rgpu) "GPU 3D residual produced non-finite values"

# ---------------------------------------------------------------------------
# Compare (all n^3 x 35)
# ---------------------------------------------------------------------------
maxabs = 0.0; maxrel = 0.0; ac = (0,0,0); am = 0; ndiv = 0
refscale = maximum(abs, Rref)
for k in 1:n, j in 1:n, i in 1:n, m in 1:35
    a = abs(Rgpu[m,i,j,k] - Rref[m,i,j,k])
    e = a / max(1.0, abs(Rref[m,i,j,k]))
    if e > maxrel
        global maxrel = e; global ac = (i,j,k); global am = m
    end
    global maxabs = max(maxabs, a)
    e > 1e-6 && (global ndiv += 1)
end

@printf("\n=== HEADLINE (GPU 3D order-2 MUSCL HLL residual vs CPU residual_ho_3d! order=2) ===\n")
@printf("n=%d  cells=%d  moments=35  (total comparisons=%d)\n", n, n^3, n^3*35)
@printf("max REL error |dR|/max(1,|Rref|) = %.3e  (gate <= 1e-6)  [cell %s, moment %d]\n",
        maxrel, ac, am)
@printf("max ABS error = %.3e   (# comparisons > 1e-6: %d / %d)\n", maxabs, ndiv, n^3*35)
@printf("Rref scale (max|Rref|) = %.3e\n", refscale)
@printf("worst cell %s, moment %d:  Rgpu=%.10e  Rref=%.10e\n",
        ac, am, Rgpu[am, ac...], Rref[am, ac...])
@printf("GATE: %s\n", maxrel <= 1e-6 ? "PASS" : "FAIL")

# ---------------------------------------------------------------------------
# BENCHMARK: solve-only, data resident. Tile the validated 24^3 field up to
# n=64 and n=128 by nearest-neighbor index so per-cell work is representative.
# ---------------------------------------------------------------------------
@printf("\n=== BENCHMARK (GPU 3D residual, solve-only, resident) ===\n")
function tile_field(M0::Array{Float64,4}, n0::Int, n::Int)
    Mt = Array{Float64,4}(undef, 35, n, n, n)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        i0 = (i - 1) % n0 + 1; j0 = (j - 1) % n0 + 1; k0 = (k - 1) % n0 + 1
        for m in 1:35
            Mt[m,i,j,k] = M0[m,i0,j0,k0]
        end
    end
    return Mt
end

for nb in (64, 128)
    Mt = tile_field(Mh, n, nb)
    Md   = CuArray(Mt)
    R    = CUDA.zeros(Float64, 35, nb, nb, nb)
    Fbuf = CUDA.zeros(Float64, 35, nb + 1, nb, nb)
    residual3d_gpu!(R, Fbuf, Md, nb, 1.0/nb, Ma; vacuum_floor=HO_VACUUM_FLOOR, project_faces=true)  # warmup
    CUDA.synchronize()
    K = 10
    t = CUDA.@elapsed begin
        for _ in 1:K
            residual3d_gpu!(R, Fbuf, Md, nb, 1.0/nb, Ma; vacuum_floor=HO_VACUUM_FLOOR, project_faces=true)
        end
        CUDA.synchronize()
    end
    mcells = nb^3 * K / t / 1e6
    @printf("GPU n=%-4d : %.2f ms/residual,  %.2f Mcell/s,  speedup vs CPU(%.4f Mcell/s) = %.0fx\n",
            nb, 1e3*t/K, mcells, CPU_BASELINE_MCELLS, mcells / CPU_BASELINE_MCELLS)
    flush(stdout)
    Md = nothing; R = nothing; Fbuf = nothing
    GC.gc(); CUDA.reclaim()
end

@printf("\nNOTE: CPU baseline %.4f Mcell/s is SINGLE-THREAD residual_ho_3d!; production CPU is\n", CPU_BASELINE_MCELLS)
@printf("      MPI many-core, so a fair GPU-vs-full-socket speedup is correspondingly lower.\n")
@printf("\nDONE\n")
