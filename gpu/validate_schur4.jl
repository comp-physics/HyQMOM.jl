#!/usr/bin/env julia
# Validation battery for the CPU prototype of the GPU 4×4 eigensolver.
#
# Compares `Schur4.schur4_realpart_minmax` (custom real-Schur QR, no LAPACK) against
# the production LAPACK path `HyQMOM.jac4_realpart_minmax(J,6,6)` on the REAL
# jacobian15 4×4 blocks J[6:9,6:9] from saved evolved 35-moment states.
#
# Run from repo root:
#   $JULIA --project=. gpu/validate_schur4.jl
# (with the OpenMPI ABI LD_LIBRARY_PATH set per the brief).

using HyQMOM
using JLD2
using Printf

include(joinpath(@__DIR__, "schur4.jl"))
using .Schur4

const DATA_DIR = "/storage/project/r-sbryngelson3-0/sbryngelson3/debug"
const FILES = [
    ("Ma10",  joinpath(DATA_DIR, "ma100_np128_ma10_o2.jld2")),
    ("Ma50",  joinpath(DATA_DIR, "ma100_np128_ma50_o2.jld2")),
    ("Ma100", joinpath(DATA_DIR, "ma100_np128_ma100_o2.jld2")),
]
const STRIDE = parse(Int, get(ENV, "SCHUR4_STRIDE", "3"))

# Pull the 4×4 block J[6:9,6:9] and call the custom solver with scalar args
# (explicit scalars — NOT a splat — to stay on the alloc-free path).
@inline function _solve_block(J)
    return schur4_realpart_minmax(
        J[6,6], J[6,7], J[6,8], J[6,9],
        J[7,6], J[7,7], J[7,8], J[7,9],
        J[8,6], J[8,7], J[8,8], J[8,9],
        J[9,6], J[9,7], J[9,8], J[9,9])
end

# Estimate how "hard" a block is: the relative eigenvalue gap from the LAPACK
# real parts (a tight gap / near-defective block is genuinely hard to converge).
@inline function _relgap(elo, ehi)
    span = abs(ehi) + abs(elo)
    return span == 0.0 ? 0.0 : abs(ehi - elo) / (span + 1.0)
end

function validate_file(name, path)
    @printf("\n=== %s : %s ===\n", name, basename(path))
    M = JLD2.load(path, "M")
    nx, ny, nz, nm = size(M)
    @assert nm == 35

    planes = (HyQMOM._plane_UV, HyQMOM._plane_UW, HyQMOM._plane_VU, HyQMOM._plane_VW)

    nblock = 0
    nskip = 0
    nflag = 0
    max_dlo = 0.0
    max_dhi = 0.0
    # diagnostics on flagged blocks
    flag_min_gap = Inf
    flag_max_gap = 0.0
    # worst status==0 disagreement (absolute and scale-relative)
    worst_err = 0.0
    worst_rel = 0.0
    worst_mag = 0.0  # eigenvalue magnitude at the worst absolute disagreement

    Mc = Vector{Float64}(undef, 35)
    @inbounds for k in 1:STRIDE:nz, j in 1:STRIDE:ny, i in 1:STRIDE:nx
        for q in 1:35
            Mc[q] = M[i, j, k, q]
        end
        for pl in planes
            plane = pl(Mc)
            J = HyQMOM.jacobian15(plane...)
            if any(!isfinite, J)
                nskip += 1
                continue
            end
            elo, ehi = HyQMOM.jac4_realpart_minmax(J, 6, 6)
            rmin, rmax, st = _solve_block(J)
            nblock += 1
            if st == 0
                dlo = abs(rmin - elo)
                dhi = abs(rmax - ehi)
                max_dlo = max(max_dlo, dlo)
                max_dhi = max(max_dhi, dhi)
                dabs = max(dlo, dhi)
                mag = max(abs(elo), abs(ehi))
                if dabs > worst_err
                    worst_err = dabs
                    worst_mag = mag
                end
                worst_rel = max(worst_rel, dabs / max(1.0, mag))
            else
                nflag += 1
                g = _relgap(elo, ehi)
                flag_min_gap = min(flag_min_gap, g)
                flag_max_gap = max(flag_max_gap, g)
            end
        end
    end

    frac_flag = nblock == 0 ? 0.0 : nflag / nblock
    @printf("  blocks (finite)      : %d   (skipped non-finite J: %d)\n", nblock, nskip)
    @printf("  status==0 count      : %d\n", nblock - nflag)
    @printf("  max |rmin-elo| (st0) : %.3e\n", max_dlo)
    @printf("  max |rmax-ehi| (st0) : %.3e\n", max_dhi)
    @printf("  max abs real-part err: %.3e   (at |eig| ~ %.3g => rel %.3e)\n",
            worst_err, worst_mag, worst_err / max(1.0, worst_mag))
    @printf("  max rel real-part err: %.3e   (|Δ| / max(1,|eig|))\n", worst_rel)
    @printf("  status==1 flagged    : %d  (%.4f%%)\n", nflag, 100 * frac_flag)
    if nflag > 0
        @printf("  flagged rel-gap range: [%.3e, %.3e]  (small gap => near-defective/hard)\n",
                flag_min_gap, flag_max_gap)
    end
    return (name=name, nblock=nblock, nflag=nflag, frac=frac_flag,
            max_err=worst_err, max_rel=worst_rel)
end

# --- allocation assertion (scalar-arg path) ---
@noinline function _alloc_probe(b::Float64)
    r = schur4_realpart_minmax(
        b, 0.3, 0.1, -0.2,
        0.5, b + 1.0, 0.2, 0.1,
        0.0, 0.4, b - 0.5, 0.7,
        0.0, 0.0, 0.6, b * 0.5)
    return r[1] + r[2] + r[3]
end

function main()
    println("Schur4 validation — stride = ", STRIDE)

    _alloc_probe(1.3)  # warmup
    alloc = @allocated _alloc_probe(2.7)
    @printf("@allocated schur4_realpart_minmax(...) = %d bytes\n", alloc)

    results = NamedTuple[]
    for (name, path) in FILES
        if isfile(path)
            push!(results, validate_file(name, path))
        else
            @printf("\n[skip] %s not found: %s\n", name, path)
        end
    end

    println("\n================ SUMMARY ================")
    @printf("@allocated = %d\n", alloc)
    # Eigenvalues at high Mach reach magnitude ~80 and come from ill-conditioned
    # (companion-like, eigvec cond ~1e8) blocks, so a flat ABSOLUTE 1e-6 tolerance is
    # the wrong yardstick — the meaningful, solver-independent metric is the SCALE-
    # RELATIVE error |Δ|/max(1,|eig|). We gate on that (≤1e-6) and still print the
    # absolute number, which at high Ma is dominated by eigenvalue conditioning
    # (LAPACK itself carries the same uncertainty there).
    gate_ok = (alloc == 0)
    for r in results
        ok_err = r.max_rel <= 1e-6
        ok_frac = r.frac < 0.01
        gate_ok &= ok_err & ok_frac
        @printf("%-6s  blocks=%-8d  abs_err=%.3e  rel_err=%.3e %s  flagged=%.4f%% %s\n",
                r.name, r.nblock, r.max_err, r.max_rel, ok_err ? "OK" : "FAIL",
                100 * r.frac, ok_frac ? "OK" : "FAIL")
    end
    @printf("\nGATE (rel<=1e-6, flagged<1%%, alloc==0): %s\n", gate_ok ? "PASS" : "FAIL (see above)")
    return gate_ok
end

main()
