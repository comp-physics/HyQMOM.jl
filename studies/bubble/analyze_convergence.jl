#!/usr/bin/env julia
# Convergence analysis for the bubble refinement runs.
#
# Reads studies/bubble/out/bubble_Kn<Kn>_N<N>.bin for a ladder of N (each 2x the
# previous), restricts the finer grid to the coarser by conservative 2x2 block
# averaging, and reports L1/L2 successive-difference norms plus the observed
# order p = log2(e_k / e_{k+1}).
#
# Usage: julia --project=. studies/bubble/analyze_convergence.jl [Kn] [N1 N2 ...]

using Printf

Kn = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
Ns = length(ARGS) >= 2 ? parse.(Int, ARGS[2:end]) : [64, 128, 256]
sort!(Ns)

const OUTDIR = joinpath(@__DIR__, "out")

function load_run(Kn, N)
    fname = joinpath(OUTDIR, @sprintf("bubble_Kn%g_N%d.bin", Kn, N))
    isfile(fname) || error("missing $fname")
    open(fname, "r") do io
        nx = read(io, Int64); ny = read(io, Int64); nm = read(io, Int64)
        ft = read(io, Float64); steps = read(io, Int64)
        slice = Array{Float64}(undef, nx, ny, nm)
        read!(io, slice)
        (nx=Int(nx), ny=Int(ny), nm=Int(nm), final_time=ft, steps=Int(steps), slice=slice)
    end
end

# Conservative restriction: average each 2x2 block -> coarse grid (factor f=2^k).
function restrict_to(field::AbstractMatrix, f::Int)
    f == 1 && return copy(field)
    nx, ny = size(field)
    cx, cy = nx ÷ f, ny ÷ f
    out = zeros(eltype(field), cx, cy)
    @inbounds for j in 1:cy, i in 1:cx
        s = 0.0
        for jj in 1:f, ii in 1:f
            s += field[(i-1)*f+ii, (j-1)*f+jj]
        end
        out[i, j] = s / (f*f)
    end
    out
end

normdiff(a, b, dx) = (L1 = sum(abs.(a .- b)) * dx^2,
                      L2 = sqrt(sum((a .- b).^2) * dx^2))

# pick a quantity from the (nx,ny,35) slice: density (moment 1) is index 1
qty(slice, idx) = slice[:, :, idx]

println("="^72)
@printf("Bubble convergence  Kn=%g   grids: %s\n", Kn, join(Ns, ", "))
println("="^72)

runs = Dict(N => load_run(Kn, N) for N in Ns)
for N in Ns
    r = runs[N]
    @printf("  N=%-5d steps=%-5d t=%.5g\n", N, r.steps, r.final_time)
end

# density convergence
function density_convergence(runs, Ns)
    println("\nDensity (moment index 1):")
    @printf("  %-14s %-14s %-14s %-8s %-8s\n", "pair", "L1 diff", "L2 diff", "p(L1)", "p(L2)")
    prevL1 = NaN; prevL2 = NaN
    for k in 1:length(Ns)-1
        Nc, Nf = Ns[k], Ns[k+1]
        @assert Nf == 2Nc "ladder must be successive doublings (got $Nc -> $Nf)"
        coarse = qty(runs[Nc].slice, 1)
        fine_r = restrict_to(qty(runs[Nf].slice, 1), 2)
        dx = 1.0 / Nc
        nd = normdiff(coarse, fine_r, dx)
        pL1 = isnan(prevL1) ? NaN : log2(prevL1 / nd.L1)
        pL2 = isnan(prevL2) ? NaN : log2(prevL2 / nd.L2)
        @printf("  %-14s %-14.4e %-14.4e %-8s %-8s\n",
                "$(Nc)-$(Nf)", nd.L1, nd.L2,
                isnan(pL1) ? "-" : @sprintf("%.3f", pL1),
                isnan(pL2) ? "-" : @sprintf("%.3f", pL2))
        prevL1 = nd.L1; prevL2 = nd.L2
    end
end
density_convergence(runs, Ns)

println("\n(Observed order p from successive Richardson differences; p≈1 expected")
println(" for a discontinuous solution, higher in smooth regions.)")
