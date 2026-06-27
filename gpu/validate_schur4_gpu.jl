#!/usr/bin/env julia
# validate_schur4_gpu.jl
#
# Validation + benchmark for the batched CUDA 4×4 real-Schur eigensolver
# (`gpu/schur4_gpu.jl`, module `Schur4GPU`).
#
#  * HEADLINE gate: GPU (rmin,rmax) vs the LAPACK reference battery
#    (262144 real 4×4 blocks from evolved Ma=10/100 states),
#    max RELATIVE error |Δ|/max(1,|eig|) over status==0 blocks ≤ 1e-6,
#    plus the flagged (status==1) %.
#  * Extra stress: random non-symmetric + a companion-form matrix.
#  * Benchmark: GPU batched throughput (Mmat/s) solve-only AND end-to-end (incl
#    H2D) vs a single-thread CPU LAPACK `eigvals` baseline, batch ~2e6.
#
# ENV: home is OVER QUOTA — read inputs from /storage/scratch1/6/sbryngelson3/gpudata,
# write nothing under home. Run with:
#   --project=gpu/gpuenv2 (CUDA + StaticArrays), depot on scratch.

import Pkg
Pkg.activate(joinpath(@__DIR__, "gpuenv2"))

using CUDA, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "schur4_gpu.jl"))
using .Schur4GPU

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const DATA = "/storage/scratch1/6/sbryngelson3/gpudata"

# ---------------------------------------------------------------------------
# 1. Real-block battery (HEADLINE)
# ---------------------------------------------------------------------------
nb = parse(Int, strip(read(joinpath(DATA, "real_blocks.meta"), String)))
blocks = reshape(reinterpret(Float64, read(joinpath(DATA, "real_blocks.f64"))), 16, nb)  # (16,nb) col-major/block
lap    = reshape(reinterpret(Float64, read(joinpath(DATA, "real_lapack.f64"))), 2, nb)    # (2,nb) = (lo,hi)
@printf("loaded %d real blocks  (lo∈[%.3g,%.3g], hi∈[%.3g,%.3g])\n",
        nb, extrema(lap[1, :])..., extrema(lap[2, :])...)

A = Matrix{Float64}(collect(blocks))     # (16, nb), already kernel layout
rmin, rmax, st = Schur4GPU.schur4_batched(A)

ok = st .== Int32(0)
nflag = count(!, ok)
maxrel = 0.0; argk = 0
for k in 1:nb
    ok[k] || continue
    e = max(abs(rmin[k] - lap[1, k]), abs(rmax[k] - lap[2, k])) / max(1.0, abs(lap[2, k]), abs(lap[1, k]))
    if e > maxrel
        global maxrel = e; global argk = k
    end
end
@printf("\n=== HEADLINE (real blocks vs LAPACK) ===\n")
@printf("status==0: %d/%d   flagged(status==1): %d  (%.4f%%)\n", count(ok), nb, nflag, 100*nflag/nb)
@printf("max REL error |Δ|/max(1,|eig|) over status==0 = %.3e  (gate ≤ 1e-6)  [worst k=%d]\n", maxrel, argk)
@printf("GATE: %s\n", maxrel <= 1e-6 ? "PASS" : "FAIL")

# also report flagged blocks' agreement with LAPACK (sanity — kernel still computes a value)
if nflag > 0
    fr = 0.0
    for k in 1:nb
        ok[k] && continue
        e = max(abs(rmin[k] - lap[1, k]), abs(rmax[k] - lap[2, k])) / max(1.0, abs(lap[2, k]))
        global fr = max(fr, e)
    end
    @printf("(flagged blocks: max rel err vs LAPACK = %.3e — caller would fall back to LAPACK)\n", fr)
end

# ---------------------------------------------------------------------------
# 2. Random non-symmetric stress
# ---------------------------------------------------------------------------
Random.seed!(2024)
Br = 200_000
Ar = Matrix{Float64}(undef, 16, Br)
refr = Matrix{Float64}(undef, 2, Br)
for k in 1:Br
    M = randn(4, 4)
    Ar[:, k] = vec(M)
    ev = real(eigvals(M)); refr[1, k] = minimum(ev); refr[2, k] = maximum(ev)
end
rm1, rm2, sr = Schur4GPU.schur4_batched(Ar)
mr = 0.0
for k in 1:Br
    sr[k] == 0 || continue
    e = max(abs(rm1[k] - refr[1, k]), abs(rm2[k] - refr[2, k])) / max(1.0, abs(refr[2, k]), abs(refr[1, k]))
    global mr = max(mr, e)
end
@printf("\nrandom 4×4 (n=%d): flagged=%d  max rel err(status0)=%.3e\n", Br, count(==(Int32(1)), sr), mr)

# ---------------------------------------------------------------------------
# 3. Companion-form stress matrix (ill-conditioned — the fp32-killer)
# ---------------------------------------------------------------------------
# companion of p(λ)=λ^4 + p2 λ^3 + p3 λ^2 + p4 λ + p5 with chosen real roots
roots = [1.0, 2.0, 50.0, 100.0]
function build_companion(roots)
    p = [1.0]
    for r in roots
        p = [p; 0.0] .- [0.0; r .* p]   # monic coeffs, high→low degree
    end
    C = zeros(4, 4)
    for i in 2:4; C[i, i-1] = 1.0; end
    C[1, 4] = -p[5]; C[2, 4] = -p[4]; C[3, 4] = -p[3]; C[4, 4] = -p[2]
    C
end
C = build_companion(roots)
Ac = reshape(vec(C), 16, 1)
cmin, cmax, cs = Schur4GPU.schur4_batched(Ac)
evc = real(eigvals(C))
@printf("\ncompanion(roots=%s): GPU=(%.10g,%.10g) status=%d  LAPACK=(%.10g,%.10g)\n",
        string(roots), cmin[1], cmax[1], cs[1], minimum(evc), maximum(evc))

# ---------------------------------------------------------------------------
# 4. Benchmark — batch ~2e6 (128^3)
# ---------------------------------------------------------------------------
Bbench = 2_097_152
@printf("\n=== BENCHMARK (B=%d) ===\n", Bbench)
Random.seed!(7)
# build benchmark battery by tiling the real blocks (representative conditioning)
Ab = Matrix{Float64}(undef, 16, Bbench)
@inbounds for k in 1:Bbench
    src = ((k - 1) % nb) + 1
    for m in 1:16; Ab[m, k] = blocks[m, src]; end
end

# CPU single-thread baseline: LAPACK eigvals on 4×4 in a loop (non-symmetric)
function cpu_baseline(A, rng)
    buf = Matrix{Float64}(undef, 4, 4)
    lo = Vector{Float64}(undef, length(rng)); hi = similar(lo)
    @inbounds for (o, k) in enumerate(rng)
        for m in 1:16; buf[m] = A[m, k]; end
        ev = real(eigvals(buf))
        lo[o] = minimum(ev); hi[o] = maximum(ev)
    end
    lo, hi
end
cpu_baseline(Ab, 1:1000)  # warmup
ncpu = 50_000             # subset for CPU timing (full 2e6 is slow); scale to Mmat/s
t_cpu = @elapsed cpu_baseline(Ab, 1:ncpu)
cpu_rate = ncpu / t_cpu / 1e6
@printf("CPU 1-thread LAPACK eigvals 4×4: %d mats in %.3f s  -> %.4f Mmat/s\n", ncpu, t_cpu, cpu_rate)

# GPU end-to-end (incl H2D) and solve-only (data resident)
Ad = CuArray(Ab)
gmin = CUDA.zeros(Float64, Bbench); gmax = CUDA.zeros(Float64, Bbench); gst = CUDA.zeros(Int32, Bbench)
Schur4GPU.schur4_batched!(gmin, gmax, gst, Ad); CUDA.synchronize()  # warmup

t_solve = CUDA.@elapsed begin
    Schur4GPU.schur4_batched!(gmin, gmax, gst, Ad)
end
solve_rate = Bbench / t_solve / 1e6

t_e2e = CUDA.@elapsed begin
    Ad2 = CuArray(Ab)
    rmn = CUDA.zeros(Float64, Bbench); rmx = CUDA.zeros(Float64, Bbench); sts = CUDA.zeros(Int32, Bbench)
    Schur4GPU.schur4_batched!(rmn, rmx, sts, Ad2)
    h1 = Array(rmn); h2 = Array(rmx); h3 = Array(sts)
end
e2e_rate = Bbench / t_e2e / 1e6

@printf("GPU solve-only (resident):  %d mats in %.4f s  -> %.2f Mmat/s   (%.1f× vs CPU)\n",
        Bbench, t_solve, solve_rate, solve_rate / cpu_rate)
@printf("GPU end-to-end (incl H2D):  %d mats in %.4f s  -> %.2f Mmat/s   (%.1f× vs CPU)\n",
        Bbench, t_e2e, e2e_rate, e2e_rate / cpu_rate)

@printf("\nDONE\n")
