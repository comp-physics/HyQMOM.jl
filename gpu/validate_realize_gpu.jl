#!/usr/bin/env julia
# validate_realize_gpu.jl
#
# Validation + benchmark for the batched CUDA realizability projection
# (`gpu/realize_gpu.jl`, module `RealizeGPU`), which inlines the device function
# `RealizeDev.realizable_3D_M4_dev` (the per-cell `realizable_3D_M4`).
#
#  * HEADLINE gate: GPU corrected 35-moment vectors vs the CPU `realizable_3D_M4`
#    reference battery (21296 real evolved Ma=10/100 states), max REL error
#    |Δ|/max(1,|ref|) over all nb x 35. GATE ≤ 1e-6. The ~4% of cells the CPU
#    actually corrects (projection branch) must match too — the projection is
#    deterministic, so the only freedom is the (sign-only) min-eig branch decision.
#  * %-corrected: how many cells the GPU changes vs input (‖Mout-Min‖>1e-10),
#    should ≈ the CPU's 4.05% (863/21296).
#  * Benchmark: GPU batched throughput (Mcell/s) solve-only (resident) AND
#    end-to-end (incl H2D/D2H) vs a single-thread CPU baseline (the same scalar
#    device function looped on CPU), batch ~2e6 (=128^3).
#
# ENV: home is OVER QUOTA — read inputs from /storage/scratch1/6/sbryngelson3/gpudata,
# write nothing under home. Run with gpuenv2, depot on scratch.

import Pkg
Pkg.activate(joinpath(@__DIR__, "gpuenv2"))

using CUDA, Printf, LinearAlgebra
include(joinpath(@__DIR__, "realize_gpu.jl"))
using .RealizeGPU
using .RealizeGPU.RealizeDev: realizable_3D_M4_dev

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const DATA = "/storage/scratch1/6/sbryngelson3/gpudata"

# ---------------------------------------------------------------------------
# 1. Load real battery (HEADLINE)
# ---------------------------------------------------------------------------
nb  = parse(Int, strip(read(joinpath(DATA, "proj.meta"), String)))
M   = reshape(reinterpret(Float64, read(joinpath(DATA, "proj_M.f64"))),   35, nb)  # (35,nb) col=cell
Ref = reshape(reinterpret(Float64, read(joinpath(DATA, "proj_ref.f64"))), 35, nb)  # CPU realizable_3D_M4
Ma  = collect(reinterpret(Float64, read(joinpath(DATA, "proj_Ma.f64"))))           # per-cell Ma
@assert length(Ma) == nb "proj_Ma length mismatch"
@printf("loaded %d real states  (M ∈ [%.3g, %.3g], Ma ∈ {%s})\n",
        nb, extrema(M)..., join(sort(unique(Ma)), ","))

Mh = Matrix{Float64}(collect(M))

# GPU projection with per-cell Ma
Gout = RealizeGPU.realizable_batched(Mh, Ma)

# ---------------------------------------------------------------------------
# 2. HEADLINE: max rel err vs CPU ref
# ---------------------------------------------------------------------------
maxrel = 0.0; maxabs = 0.0; argk = 0; argm = 0; ndiv = 0
for k in 1:nb
    for m in 1:35
        g = Gout[m,k]; r = Ref[m,k]
        a = abs(g - r); e = a / max(1.0, abs(r))
        if e > maxrel
            global maxrel = e; global argk = k; global argm = m
        end
        global maxabs = max(maxabs, a)
        if e > 1e-6; global ndiv += 1; end
    end
end

@printf("\n=== HEADLINE (GPU realizable_3D_M4 vs CPU realizable_3D_M4 reference) ===\n")
@printf("nb=%d  (total comparisons=%d)\n", nb, nb*35)
@printf("max REL error |Δ|/max(1,|ref|) = %.3e  (gate ≤ 1e-6)  [cell %d, moment %d, Ma=%g]\n",
        maxrel, argk, argm, Ma[argk])
@printf("max ABS error                  = %.3e\n", maxabs)
@printf("entries exceeding 1e-6         = %d / %d\n", ndiv, nb*35)
@printf("GATE: %s\n", maxrel <= 1e-6 ? "PASS ✅" : "FAIL ❌")

# ---------------------------------------------------------------------------
# 3. %-corrected: GPU vs input, and CPU(ref) vs input
# ---------------------------------------------------------------------------
ncorr_gpu = 0; ncorr_cpu = 0; nmismatch_branch = 0
for k in 1:nb
    dg = 0.0; dc = 0.0
    for m in 1:35
        dg += (Gout[m,k] - M[m,k])^2
        dc += (Ref[m,k]  - M[m,k])^2
    end
    cg = sqrt(dg) > 1e-10
    cc = sqrt(dc) > 1e-10
    global ncorr_gpu += cg
    global ncorr_cpu += cc
    cg != cc && (global nmismatch_branch += 1)
end
@printf("\n=== %%-corrected (‖Mout-Min‖ > 1e-10) ===\n")
@printf("GPU corrected = %d / %d  (%.2f%%)\n", ncorr_gpu, nb, 100*ncorr_gpu/nb)
@printf("CPU corrected = %d / %d  (%.2f%%)\n", ncorr_cpu, nb, 100*ncorr_cpu/nb)
@printf("cells where GPU/CPU disagree on whether-corrected = %d\n", nmismatch_branch)

# ---------------------------------------------------------------------------
# 4. Benchmark: batch ~2e6, scalar Ma=10
# ---------------------------------------------------------------------------
println("\n=== BENCHMARK (batch ~2e6, Ma=10) ===")
Bbench = 128^3
# tile the real battery up to Bbench
reps = cld(Bbench, nb)
Mbig = repeat(Mh, 1, reps)[:, 1:Bbench]
@printf("batch B = %d\n", Bbench)

Mabench = 10.0

# CPU 1-thread baseline (same device scalar function looped)
function cpu_loop!(out, Min, Ma, B)
    @inbounds for k in 1:B
        r = realizable_3D_M4_dev(
            Min[1,k],  Min[2,k],  Min[3,k],  Min[4,k],  Min[5,k],  Min[6,k],  Min[7,k],
            Min[8,k],  Min[9,k],  Min[10,k], Min[11,k], Min[12,k], Min[13,k], Min[14,k],
            Min[15,k], Min[16,k], Min[17,k], Min[18,k], Min[19,k], Min[20,k], Min[21,k],
            Min[22,k], Min[23,k], Min[24,k], Min[25,k], Min[26,k], Min[27,k], Min[28,k],
            Min[29,k], Min[30,k], Min[31,k], Min[32,k], Min[33,k], Min[34,k], Min[35,k],
            Ma)
        for m in 1:35
            out[m,k] = r[m]
        end
    end
    return nothing
end

# warmup + time CPU on a smaller slice (it's slow), then extrapolate via Mcell/s
Bcpu = min(Bbench, 200_000)
outc = similar(Mbig[:, 1:Bcpu])
cpu_loop!(outc, Mbig, Mabench, Bcpu)      # warmup/compile
t_cpu = @elapsed cpu_loop!(outc, Mbig, Mabench, Bcpu)
cpu_rate = Bcpu / t_cpu
@printf("CPU 1-thread : %.3f s for %d cells  -> %.3f Mcell/s\n", t_cpu, Bcpu, cpu_rate/1e6)

# GPU resident (solve-only)
Md = CuArray(Mbig)
Mo = similar(Md)
RealizeGPU.realizable_batched!(Mo, Md, Mabench); CUDA.synchronize()   # warmup
t_gpu = CUDA.@elapsed begin
    RealizeGPU.realizable_batched!(Mo, Md, Mabench)
    CUDA.synchronize()
end
gpu_rate = Bbench / t_gpu
@printf("GPU solve-only: %.4f s for %d cells  -> %.3f Mcell/s\n", t_gpu, Bbench, gpu_rate/1e6)

# GPU end-to-end (incl H2D/D2H)
t_e2e = @elapsed begin
    _ = RealizeGPU.realizable_batched(Mbig, Mabench)
end
e2e_rate = Bbench / t_e2e
@printf("GPU end-to-end: %.4f s for %d cells  -> %.3f Mcell/s\n", t_e2e, Bbench, e2e_rate/1e6)

@printf("\nspeedup (solve-only)  = %.1fx vs CPU 1-thread\n", gpu_rate/cpu_rate)
@printf("speedup (end-to-end)  = %.1fx vs CPU 1-thread\n", e2e_rate/cpu_rate)
