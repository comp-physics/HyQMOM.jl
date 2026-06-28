#!/usr/bin/env julia
# validate_wavespeed_gpu.jl
#
# Validation + benchmark for the batched CUDA wave-speed path
# (`gpu/wavespeed_gpu.jl`, module `WavespeedGPU`), which inlines the validated device
# function `WavespeedDev.realize_and_speed_dev` (the per-cell `realize_and_speed`).
#
#  * HEADLINE gate: GPU (vmin,vmax) for axis 1,2,3 vs the CPU `realize_and_speed`
#    reference battery (8192 real evolved Ma=10/100 states), max REL error
#    |Δv|/max(1,|vref|) over all nb x 3 x 2. GATE ≤ 1e-6 (expect ~1e-8: the 4x4
#    Schur path already matches LAPACK to ~1e-8).
#  * Benchmark: GPU batched throughput (Mcell/s) solve-only (resident) AND
#    end-to-end (incl H2D/D2H) vs a single-thread CPU baseline (the same scalar
#    device function looped on CPU), batch ~2e6 (=128^3), all 3 axes.
#
# ENV: home is OVER QUOTA — read inputs from /storage/scratch1/6/sbryngelson3/gpudata,
# write nothing under home. Run with gpuenv2, depot on scratch.

import Pkg
Pkg.activate(joinpath(@__DIR__, "gpuenv2"))

using CUDA, Printf
include(joinpath(@__DIR__, "wavespeed_gpu.jl"))
using .WavespeedGPU
using .WavespeedGPU.WavespeedDev: realize_and_speed_dev

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const DATA = "/storage/scratch1/6/sbryngelson3/gpudata"

# ---------------------------------------------------------------------------
# 1. Load real battery (HEADLINE)
# ---------------------------------------------------------------------------
nb   = parse(Int, strip(read(joinpath(DATA, "ws.meta"), String)))
M    = reshape(reinterpret(Float64, read(joinpath(DATA, "ws_M.f64"))),  35, nb)  # (35,nb) col=cell
Wref = reshape(reinterpret(Float64, read(joinpath(DATA, "ws_ref.f64"))),  6, nb)  # rows: ax1 vmin,vmax, ax2..., ax3...
@printf("loaded %d real states  (M ∈ [%.3g, %.3g])\n", nb, extrema(M)...)

Mh = Matrix{Float64}(collect(M))

maxrel = 0.0; maxabs = 0.0; argax = 0; argk = 0; argwhich = 0; ndiv = 0
for ax in 1:3
    vmn, vmx = WavespeedGPU.wave_speeds_batched(Mh, ax)
    rmn = @view Wref[2*ax-1, :]
    rmx = @view Wref[2*ax,   :]
    for k in 1:nb
        for (g, r, wch) in ((vmn[k], rmn[k], 0), (vmx[k], rmx[k], 1))
            a = abs(g - r); e = a / max(1.0, abs(r))
            if e > maxrel
                global maxrel = e; global argax = ax; global argk = k; global argwhich = wch
            end
            global maxabs = max(maxabs, a)
            if e > 1e-6; global ndiv += 1; end
        end
    end
end
@printf("\n=== HEADLINE (GPU wave speeds vs CPU realize_and_speed reference) ===\n")
@printf("nb=%d  axes=3  (total comparisons=%d)\n", nb, nb*6)
@printf("max REL error |Δv|/max(1,|vref|) = %.3e  (gate ≤ 1e-6)  [axis %d, cell %d, %s]\n",
        maxrel, argax, argk, argwhich == 0 ? "vmin" : "vmax")
@printf("max ABS error = %.3e   (# comparisons > 1e-6: %d / %d)\n", maxabs, ndiv, nb*6)
@printf("GATE: %s\n", maxrel <= 1e-6 ? "PASS" : "FAIL")

# ---------------------------------------------------------------------------
# 2. Benchmark — batch ~2e6 (128^3), all 3 axes
# ---------------------------------------------------------------------------
Bbench = 2_097_152
@printf("\n=== BENCHMARK (B=%d, summed over 3 axes) ===\n", Bbench)
Mb = Matrix{Float64}(undef, 35, Bbench)
@inbounds for k in 1:Bbench
    src = ((k - 1) % nb) + 1
    for m in 1:35; Mb[m, k] = M[m, src]; end
end

# --- CPU 1-thread baseline: same scalar device fn looped on CPU (one axis) ---
function cpu_baseline(Mb, rng, axis)
    s = 0.0
    @inbounds for k in rng
        a, b = realize_and_speed_dev(
            Mb[1,k],  Mb[2,k],  Mb[3,k],  Mb[4,k],  Mb[5,k],  Mb[6,k],  Mb[7,k],
            Mb[8,k],  Mb[9,k],  Mb[10,k], Mb[11,k], Mb[12,k], Mb[13,k], Mb[14,k],
            Mb[15,k], Mb[16,k], Mb[17,k], Mb[18,k], Mb[19,k], Mb[20,k], Mb[21,k],
            Mb[22,k], Mb[23,k], Mb[24,k], Mb[25,k], Mb[26,k], Mb[27,k], Mb[28,k],
            Mb[29,k], Mb[30,k], Mb[31,k], Mb[32,k], Mb[33,k], Mb[34,k], Mb[35,k],
            axis, 0.0)
        s += a + b
    end
    s
end
cpu_baseline(Mb, 1:1000, 1)   # warmup / compile
ncpu = 50_000                 # subset for CPU timing; scale to Mcell/s
t_cpu = 0.0
for ax in 1:3
    global t_cpu += @elapsed cpu_baseline(Mb, 1:ncpu, ax)
end
cpu_rate = (3 * ncpu) / t_cpu / 1e6
@printf("CPU 1-thread realize_and_speed_dev: %d cell-axes in %.3f s  -> %.4f Mcell/s\n",
        3*ncpu, t_cpu, cpu_rate)

# --- GPU solve-only (resident) and end-to-end (incl H2D + D2H), summed 3 axes ---
Md  = CuArray(Mb)
vmn = CUDA.zeros(Float64, Bbench)
vmx = CUDA.zeros(Float64, Bbench)
for ax in 1:3
    WavespeedGPU.wave_speeds_batched!(vmn, vmx, Md, ax)   # warmup each axis
end
CUDA.synchronize()

t_solve = CUDA.@elapsed begin
    for ax in 1:3
        WavespeedGPU.wave_speeds_batched!(vmn, vmx, Md, ax)
    end
end
solve_rate = (3 * Bbench) / t_solve / 1e6

t_e2e = CUDA.@elapsed begin
    Md2  = CuArray(Mb)
    vmn2 = CUDA.zeros(Float64, Bbench)
    vmx2 = CUDA.zeros(Float64, Bbench)
    for ax in 1:3
        WavespeedGPU.wave_speeds_batched!(vmn2, vmx2, Md2, ax)
    end
    a = Array(vmn2); b = Array(vmx2)
end
e2e_rate = (3 * Bbench) / t_e2e / 1e6

@printf("GPU solve-only (resident):     %d cell-axes in %.4f s  -> %.2f Mcell/s   (%.1f× vs CPU)\n",
        3*Bbench, t_solve, solve_rate, solve_rate / cpu_rate)
@printf("GPU end-to-end (incl H2D/D2H): %d cell-axes in %.4f s  -> %.2f Mcell/s   (%.1f× vs CPU)\n",
        3*Bbench, t_e2e, e2e_rate, e2e_rate / cpu_rate)

@printf("\nDONE\n")
