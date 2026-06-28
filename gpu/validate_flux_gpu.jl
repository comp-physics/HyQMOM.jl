#!/usr/bin/env julia
# validate_flux_gpu.jl
#
# Validation + benchmark for the batched CUDA analytic flux closure
# (`gpu/flux_closure_gpu.jl`, module `FluxClosureGPU`), which inlines the validated
# device function `FluxClosureDev.flux_closure35_dev`.
#
#  * HEADLINE gate: GPU Fx|Fy|Fz vs the CPU `Flux_closure35_3D` reference battery
#    (21296 real evolved Ma=10/100 states), max RELATIVE error
#    |ΔF|/max(1,|Fref|) over all 105 outputs × nb states. GATE ≤ 1e-10.
#  * Benchmark: GPU batched flux throughput (Mcell/s) solve-only (data resident)
#    AND end-to-end (incl H2D) vs a single-thread CPU baseline (the same scalar
#    device function `flux_closure35_dev` looped on the CPU), batch ~2e6 (=128^3).
#
# ENV: home is OVER QUOTA — read inputs from /storage/scratch1/6/sbryngelson3/gpudata,
# write nothing under home. Run with --project=gpu/gpuenv2, depot on scratch.

import Pkg
Pkg.activate(joinpath(@__DIR__, "gpuenv2"))

using CUDA, Printf
include(joinpath(@__DIR__, "flux_closure_gpu.jl"))
using .FluxClosureGPU
using .FluxClosureGPU.FluxClosureDev: flux_closure35_dev

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const DATA = "/storage/scratch1/6/sbryngelson3/gpudata"

# ---------------------------------------------------------------------------
# 1. Load real battery (HEADLINE)
# ---------------------------------------------------------------------------
nb   = parse(Int, strip(read(joinpath(DATA, "flux.meta"), String)))
M    = reshape(reinterpret(Float64, read(joinpath(DATA, "flux_M.f64"))),  35,  nb)  # (35,nb)  col=cell
Fref = reshape(reinterpret(Float64, read(joinpath(DATA, "flux_ref.f64"))), 105, nb) # (105,nb) Fx|Fy|Fz
@printf("loaded %d real states  (M ∈ [%.3g, %.3g])\n", nb, extrema(M)...)

Mh = Matrix{Float64}(collect(M))           # (35, nb) host, already kernel layout
Fg = FluxClosureGPU.flux_closure35_batched(Mh)   # (105, nb) host result

maxrel = 0.0; argn = 0; argk = 0
@inbounds for k in 1:nb, n in 1:105
    e = abs(Fg[n, k] - Fref[n, k]) / max(1.0, abs(Fref[n, k]))
    if e > maxrel
        global maxrel = e; global argn = n; global argk = k
    end
end
@printf("\n=== HEADLINE (GPU flux vs CPU Flux_closure35_3D reference) ===\n")
@printf("nb=%d  outputs=%d  (total comparisons=%d)\n", nb, 105, nb*105)
@printf("max REL error |ΔF|/max(1,|Fref|) = %.3e  (gate ≤ 1e-10)  [worst output #%d, cell %d]\n",
        maxrel, argn, argk)
@printf("GATE: %s\n", maxrel <= 1e-10 ? "PASS" : "FAIL")

# ---------------------------------------------------------------------------
# 2. Benchmark — batch ~2e6 (128^3)
# ---------------------------------------------------------------------------
Bbench = 2_097_152
@printf("\n=== BENCHMARK (B=%d) ===\n", Bbench)
# tile the real states (representative conditioning) into a (35, Bbench) battery
Mb = Matrix{Float64}(undef, 35, Bbench)
@inbounds for k in 1:Bbench
    src = ((k - 1) % nb) + 1
    for m in 1:35; Mb[m, k] = M[m, src]; end
end

# --- CPU 1-thread baseline: the SAME scalar device function looped on CPU ---
function cpu_baseline(Mb, rng)
    s = 0.0
    @inbounds for k in rng
        Fc = flux_closure35_dev(
            Mb[1,k],  Mb[2,k],  Mb[3,k],  Mb[4,k],  Mb[5,k],  Mb[6,k],  Mb[7,k],
            Mb[8,k],  Mb[9,k],  Mb[10,k], Mb[11,k], Mb[12,k], Mb[13,k], Mb[14,k],
            Mb[15,k], Mb[16,k], Mb[17,k], Mb[18,k], Mb[19,k], Mb[20,k], Mb[21,k],
            Mb[22,k], Mb[23,k], Mb[24,k], Mb[25,k], Mb[26,k], Mb[27,k], Mb[28,k],
            Mb[29,k], Mb[30,k], Mb[31,k], Mb[32,k], Mb[33,k], Mb[34,k], Mb[35,k])
        s += Fc[1] + Fc[53] + Fc[105]   # touch outputs so nothing is elided
    end
    s
end
cpu_baseline(Mb, 1:1000)   # warmup / compile
ncpu = 100_000             # subset for CPU timing; scale to Mcell/s
t_cpu = @elapsed cpu_baseline(Mb, 1:ncpu)
cpu_rate = ncpu / t_cpu / 1e6
@printf("CPU 1-thread flux_closure35_dev: %d cells in %.3f s  -> %.4f Mcell/s\n", ncpu, t_cpu, cpu_rate)

# --- GPU solve-only (resident) and end-to-end (incl H2D + D2H) ---
Md = CuArray(Mb)
Fd = CUDA.zeros(Float64, 105, Bbench)
FluxClosureGPU.flux_closure35_batched!(Fd, Md); CUDA.synchronize()   # warmup

t_solve = CUDA.@elapsed begin
    FluxClosureGPU.flux_closure35_batched!(Fd, Md)
end
solve_rate = Bbench / t_solve / 1e6

t_e2e = CUDA.@elapsed begin
    Md2 = CuArray(Mb)
    Fd2 = CUDA.zeros(Float64, 105, Bbench)
    FluxClosureGPU.flux_closure35_batched!(Fd2, Md2)
    Fh = Array(Fd2)
end
e2e_rate = Bbench / t_e2e / 1e6

@printf("GPU solve-only (resident):  %d cells in %.4f s  -> %.2f Mcell/s   (%.1f× vs CPU)\n",
        Bbench, t_solve, solve_rate, solve_rate / cpu_rate)
@printf("GPU end-to-end (incl H2D/D2H): %d cells in %.4f s  -> %.2f Mcell/s   (%.1f× vs CPU)\n",
        Bbench, t_e2e, e2e_rate, e2e_rate / cpu_rate)

@printf("\nDONE\n")
