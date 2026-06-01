#!/usr/bin/env julia
# Bubble grid-refinement driver.
#
# Runs the 2D discontinuous bubble case (Rice et al., JCP 2026, Sec. 5.2) using
# the 3D HyQMOM solver at Nz=1, over a list of resolutions, and writes the final
# moment field (slice) to a raw binary per run for offline convergence analysis.
#
# Serial:   julia --project=. studies/bubble/run_refinement.jl [Kn] [tmax] [N1 N2 ...]
# MPI:      mpiexec -n P julia --project=. studies/bubble/run_refinement.jl ...
#
# Defaults: Kn=1.0, tmax=0.06, N in {64,128,256}.

using HyQMOM
using MPI
using Printf

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROC = MPI.Comm_size(COMM)

# ---- parse args -------------------------------------------------------------
args = copy(ARGS)
Kn   = length(args) >= 1 ? parse(Float64, args[1]) : 1.0
tmax = length(args) >= 2 ? parse(Float64, args[2]) : 0.06
Ns   = length(args) >= 3 ? parse.(Int, args[3:end]) : [64, 128, 256]

const OUTDIR = joinpath(@__DIR__, "out")
RANK == 0 && mkpath(OUTDIR)
MPI.Barrier(COMM)

function make_params(N, Kn, tmax)
    Nz = 1
    CFL = 0.5
    dx = 1.0 / N
    dy = 1.0 / N
    dz = 1.0 / Nz
    dtmax = CFL * min(dx, dy, dz)
    nnmax = ceil(Int, tmax / dtmax) + 100000
    return (
        Nx = N, Ny = N, Nz = Nz,
        tmax = tmax, Kn = Kn, Ma = 0.0,
        flag2D = 0, CFL = CFL,
        dx = dx, dy = dy, dz = dz,
        Nmom = 35, nnmax = nnmax, dtmax = dtmax,
        # IC: isothermal bubble, density/pressure ratio 2
        ic_type = :bubble,
        rho_in = 2.0, rho_out = 1.0, bubble_radius = 0.25,
        bubble_xc = 0.0, bubble_yc = 0.0,
        # unused-by-bubble jet params still read by the runner
        rhol = 1.0, rhor = 1.0, T = 1.0,
        r110 = 0.0, r101 = 0.0, r011 = 0.0,
        symmetry_check_interval = 1000,   # symmetric IC; keep cheap
        homogeneous_z = true,
        enable_memory_tracking = false,
        debug_output = false,
        track_corrections = true,         # Reviewer #3 diagnostics
    )
end

function dump_corrections(io, N, Kn)
    d = HyQMOM.CORRECTION_DIAG[]
    d === nothing && return
    @printf(io, "N=%d Kn=%g steps=%d  frac_real=%.4e frac_hyp=%.4e frac_any=%.4e  mean_dM=%.4e max_dM=%.4e  max_dCons/corr=%.3e\n",
            N, Kn, d.steps, d.frac_realizability, d.frac_hyperbolicity, d.frac_any,
            d.mean_dM_corrected, d.max_dM, d.max_dconserved_per_correction)
    @printf(io, "    global conserved rel-drift: ")
    for (nm, rd) in zip(d.conserved_names, d.conserved_rel_drift)
        @printf(io, "%s=%.2e ", nm, rd)
    end
    @printf(io, "\n")
    flush(io)
end

for N in Ns
    params = make_params(N, Kn, tmax)
    RANK == 0 && @printf("\n=== bubble N=%d Kn=%g tmax=%g (ranks=%d) ===\n", N, Kn, tmax, NPROC)
    t0 = time()
    M, final_time, steps, grid = simulation_runner(params)
    wall = time() - t0

    if RANK == 0
        # M is (N, N, 1, 35); store the z=1 slice
        slice = M[:, :, 1, :]                      # (N, N, 35)
        rho = slice[:, :, 1]
        fname = joinpath(OUTDIR, @sprintf("bubble_Kn%g_N%d.bin", Kn, N))
        open(fname, "w") do io
            write(io, Int64(N)); write(io, Int64(N)); write(io, Int64(35))
            write(io, Float64(final_time)); write(io, Int64(steps))
            write(io, Array{Float64}(slice))
        end
        @printf("  steps=%d  t=%.5g  wall=%.1fs  rho[min,max]=[%.4f,%.4f]  edge<rho>=%.5f\n",
                steps, final_time, wall, minimum(rho), maximum(rho),
                (sum(@view rho[1, :]) + sum(@view rho[end, :]) +
                 sum(@view rho[:, 1]) + sum(@view rho[:, end])) / (4N))
        @printf("  wrote %s\n", fname)
        open(joinpath(OUTDIR, @sprintf("corrections_Kn%g.txt", Kn)), "a") do io
            dump_corrections(io, N, Kn)
        end
        dump_corrections(stdout, N, Kn)
    end
    MPI.Barrier(COMM)
end

RANK == 0 && println("\nDone.")
