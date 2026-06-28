#!/usr/bin/env julia
# validate_residual_gpu.jl
#
# CAPSTONE validation: end-to-end first-order 1D HLL residual on GPU
# (`gpu/residual1d_gpu.jl`, module `Residual1DGPU`) vs the CPU reference
# `residual_1d(Mline, dx, Ma; order=1, bc=:outflow)`.
#
#  * HEADLINE gate: max ABS and REL error over all N x 35 of R_gpu vs the on-disk
#    CPU reference. GATE: rel <= 1e-6 (expect ~1e-8). Reports worst cell/moment and
#    whether any residual mismatch is attributable to the skipped realizable_3D_M4
#    projection vs a composition bug.
#
# ENV: home is OVER QUOTA — read inputs from /storage/scratch1/6/sbryngelson3/gpudata,
# write nothing under home. Run with gpuenv2, depot on scratch.

import Pkg
Pkg.activate(joinpath(@__DIR__, "gpuenv2"))

using CUDA, Printf
include(joinpath(@__DIR__, "residual1d_gpu.jl"))
using .Residual1DGPU

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const DATA = "/storage/scratch1/6/sbryngelson3/gpudata"

# ---------------------------------------------------------------------------
# Load reference (cell-major on disk: cell i's 35 contiguous -> reshape (35,N))
# ---------------------------------------------------------------------------
meta = split(strip(read(joinpath(DATA, "resid.meta"), String)), '\n')
N  = parse(Int,     strip(meta[1]))
dx = parse(Float64, strip(meta[2]))
Ma = parse(Float64, strip(meta[3]))

M    = reshape(reinterpret(Float64, read(joinpath(DATA, "resid_M.f64"))), 35, N)  # col = cell
Rref = reshape(reinterpret(Float64, read(joinpath(DATA, "resid_R.f64"))), 35, N)
@printf("loaded N=%d  dx=%.6g  Ma=%.4g   (M in [%.3g, %.3g])\n", N, dx, Ma, extrema(M)...)

Mh = Matrix{Float64}(collect(M))

# ---------------------------------------------------------------------------
# GPU residual
# ---------------------------------------------------------------------------
Rgpu = residual1d_gpu(Mh, dx, Ma)
@assert all(isfinite, Rgpu) "GPU residual produced non-finite values"

# ---------------------------------------------------------------------------
# Compare (all N x 35)
# ---------------------------------------------------------------------------
maxabs = 0.0; maxrel = 0.0; argcell = 0; argmom = 0; ndiv = 0
refscale = maximum(abs, Rref)            # global residual scale for rel-error denom
for cell in 1:N
    for m in 1:35
        a = abs(Rgpu[m, cell] - Rref[m, cell])
        e = a / max(1.0, abs(Rref[m, cell]))
        if e > maxrel
            global maxrel = e; global argcell = cell; global argmom = m
        end
        global maxabs = max(maxabs, a)
        e > 1e-6 && (global ndiv += 1)
    end
end

@printf("\n=== HEADLINE (GPU first-order 1D HLL residual vs CPU residual_1d) ===\n")
@printf("N=%d  moments=35  (total comparisons=%d)\n", N, N*35)
@printf("max REL error |dR|/max(1,|Rref|) = %.3e  (gate <= 1e-6)  [cell %d, moment %d]\n",
        maxrel, argcell, argmom)
@printf("max ABS error = %.3e   (# comparisons > 1e-6: %d / %d)\n", maxabs, ndiv, N*35)
@printf("Rref scale (max|Rref|) = %.3e\n", refscale)
@printf("worst cell %d, moment %d:  Rgpu=%.10e  Rref=%.10e\n",
        argcell, argmom, Rgpu[argmom, argcell], Rref[argmom, argcell])
@printf("GATE: %s\n", maxrel <= 1e-6 ? "PASS" : "FAIL")

# realizable_3D_M4-skip attribution: the boundary cells are forced to 0 (outflow BC),
# so any mismatch is in the interior face-flux composition. Report where the worst
# cell sits and the local input realizability slack (M[1] density) for context.
if maxrel > 1e-6
    @printf("\n[attribution] worst cell %d is %s. Input was pre-passed through realizable_3D_M4,\n",
            argcell, (argcell == 1 || argcell == N) ? "a BOUNDARY cell (should be exactly 0)" : "an INTERIOR cell")
    @printf("              so a >1e-6 mismatch indicates either the realizable_3D_M4 skip is\n")
    @printf("              NOT identity here, or a composition bug. abs=%.3e rel=%.3e.\n", maxabs, maxrel)
else
    @printf("\nrealizable_3D_M4 skip impact: NONE at gate level (input pre-realized -> projection ~identity).\n")
end

@printf("\nDONE\n")
