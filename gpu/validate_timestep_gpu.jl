#!/usr/bin/env julia
# validate_timestep_gpu.jl
#
# CAPSTONE validation: the full on-device 1D order-2 SSP-RK3 time-march
# (`gpu/timestep_gpu.jl`, module `TimestepGPU`, `march_gpu!`) vs the CPU
# reference produced by `dump_step.jl` (N=256, Ma=100, NSTEP=20).
#
#   PRIMARY   : march on GPU FEEDING the CPU `step_dts` sequence (identical dt =>
#               isolates the residual+RK3+projection composition). Compare the GPU
#               final state vs `step_Mf`. HEADLINE = max rel err over N x 35 after
#               20 steps. GATE: rel <= 1e-6.
#   SECONDARY : march with GPU-computed dt; report final-state diff vs CPU
#               (small dt-reduction FP divergence may appear; not gated).
#
# HO_VACUUM_FLOOR = 0.001. Data layout on disk: cell-major -> reshape (35,N)
# (col = cell). ENV: read inputs from scratch, write nothing under home.

import Pkg
Pkg.activate(joinpath(@__DIR__, "gpuenv2"))

using CUDA, Printf
include(joinpath(@__DIR__, "timestep_gpu.jl"))
using .TimestepGPU: march_gpu!

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const DATA = "/storage/scratch1/6/sbryngelson3/gpudata"
const HO_VACUUM_FLOOR = 0.001

# --- meta ---
meta  = split(strip(read(joinpath(DATA, "step.meta"), String)), '\n')
N     = parse(Int,     strip(meta[1]))
dx    = parse(Float64, strip(meta[2]))
Ma    = parse(Float64, strip(meta[3]))
NSTEP = parse(Int,     strip(meta[4]))

# --- reference data (col = cell) ---
M0    = reshape(reinterpret(Float64, read(joinpath(DATA, "step_M0.f64"))), 35, N)
Mfref = reshape(reinterpret(Float64, read(joinpath(DATA, "step_Mf.f64"))), 35, N)
dts   = collect(reinterpret(Float64, read(joinpath(DATA, "step_dts.f64"))))
@printf("loaded N=%d dx=%.8g Ma=%.4g NSTEP=%d  HO_VACUUM_FLOOR=%.3g\n", N, dx, Ma, NSTEP, HO_VACUUM_FLOOR)
@printf("M0 rho[min,max]=[%.4g, %.4g]   CPU final rho[min,max]=[%.4g, %.4g]\n",
        extrema(M0[1, :])..., extrema(Mfref[1, :])...)

M0h = Matrix{Float64}(collect(M0))

cmp(A, B) = begin
    maxabs = 0.0; maxrel = 0.0; ac = 0; am = 0
    for c in 1:N, m in 1:35
        a = abs(A[m, c] - B[m, c])
        e = a / max(1.0, abs(B[m, c]))
        if e > maxrel; maxrel = e; ac = c; am = m; end
        maxabs = max(maxabs, a)
    end
    (maxabs, maxrel, ac, am)
end

# ===========================================================================
# PRIMARY: GPU march fed the CPU dt sequence
# ===========================================================================
Md = CuArray(M0h)
march_gpu!(Md, dx, Ma, NSTEP; dts=dts, vacuum_floor=HO_VACUUM_FLOOR)   # warm/compile
Md = CuArray(M0h)
t0 = time(); used = march_gpu!(Md, dx, Ma, NSTEP; dts=dts, vacuum_floor=HO_VACUUM_FLOOR); CUDA.synchronize()
twall = time() - t0
Mf_cpudt = Array(Md)
@assert all(isfinite, Mf_cpudt) "PRIMARY: GPU march produced non-finite values"

maxabs1, maxrel1, ac1, am1 = cmp(Mf_cpudt, Mfref)
@printf("\n=== PRIMARY (GPU march fed CPU step_dts, %d steps) ===\n", NSTEP)
@printf("max REL err |dM|/max(1,|ref|) = %.3e  (gate <= 1e-6) [cell %d, moment %d]\n", maxrel1, ac1, am1)
@printf("max ABS err = %.3e\n", maxabs1)
@printf("dt sequence matches CPU: %s (max |ddt|=%.2e)\n",
        all(used .== dts) ? "EXACT" : "approx", maximum(abs.(used .- dts)))
@printf("GPU final rho[min,max] = [%.6g, %.6g]\n", extrema(Mf_cpudt[1, :])...)
@printf("wall time %.4f s  (%.3f ms/step)\n", twall, 1e3 * twall / NSTEP)
@printf("GATE: %s\n", maxrel1 <= 1e-6 ? "PASS" : "FAIL")

# ===========================================================================
# SECONDARY: GPU march with GPU-computed dt (fully autonomous)
# ===========================================================================
Md2 = CuArray(M0h)
used2 = march_gpu!(Md2, dx, Ma, NSTEP; dts=nothing, vacuum_floor=HO_VACUUM_FLOOR)
Mf_gpudt = Array(Md2)
@assert all(isfinite, Mf_gpudt) "SECONDARY: GPU march produced non-finite values"

maxabs2, maxrel2, ac2, am2 = cmp(Mf_gpudt, Mfref)
@printf("\n=== SECONDARY (GPU march with GPU-computed dt, %d steps) ===\n", NSTEP)
@printf("max REL err vs CPU = %.3e (NOT gated) [cell %d, moment %d]\n", maxrel2, ac2, am2)
@printf("max ABS err vs CPU = %.3e\n", maxabs2)
@printf("GPU-dt vs CPU-dt: max |ddt|=%.3e (sum dt: gpu=%.6e cpu=%.6e)\n",
        maximum(abs.(used2 .- dts)), sum(used2), sum(dts))
@printf("GPU final rho[min,max] = [%.6g, %.6g]\n", extrema(Mf_gpudt[1, :])...)

@printf("\nDONE\n")
