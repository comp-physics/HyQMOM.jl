#!/usr/bin/env julia
# Mach-number sweep for correction-activity diagnostics (Reviewer #3: correction
# stats "as a function of the Mach number"). Uses the crossing-jets IC (the
# paper's standard 2D case) at fixed resolution, varying Ma, with the same
# correction instrumentation as the bubble runs.
#
# MPI: mpiexec -n P julia --project=. studies/bubble/run_ma_jets.jl [N] [Kn] [tmax] [Ma1 Ma2 ...]

using HyQMOM, MPI, Printf

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)

args = copy(ARGS)
N    = length(args) >= 1 ? parse(Int, args[1])     : 256
Kn   = length(args) >= 2 ? parse(Float64, args[2]) : 1.0
tmax = length(args) >= 3 ? parse(Float64, args[3]) : 0.05
Mas  = length(args) >= 4 ? parse.(Float64, args[4:end]) : [0.0, 2.0, 4.0]

const OUTDIR = joinpath(@__DIR__, "out")
RANK == 0 && mkpath(OUTDIR)
MPI.Barrier(COMM)

function jets_params(N, Kn, Ma, tmax)
    CFL = 0.5; dx = 1.0/N; dz = 1.0
    dtmax = CFL * min(dx, dx, dz)
    return (
        Nx=N, Ny=N, Nz=1, tmax=tmax, Kn=Kn, Ma=Ma, flag2D=0, CFL=CFL,
        dx=dx, dy=dx, dz=dz, Nmom=35, nnmax=ceil(Int, tmax/dtmax)+100000, dtmax=dtmax,
        # default crossing-jets IC (no ic_type) uses rhol/rhor/T below
        rhol=1.0, rhor=0.01, T=1.0, r110=0.0, r101=0.0, r011=0.0,
        symmetry_check_interval=1000, homogeneous_z=true,
        enable_memory_tracking=false, debug_output=false, track_corrections=true,
    )
end

for Ma in Mas
    RANK == 0 && @printf("\n=== crossing jets N=%d Kn=%g Ma=%g tmax=%g ===\n", N, Kn, Ma, tmax)
    M, ft, steps, grid = simulation_runner(jets_params(N, Kn, Ma, tmax))
    if RANK == 0
        d = HyQMOM.CORRECTION_DIAG[]
        open(joinpath(OUTDIR, @sprintf("corrections_jets_Kn%g.txt", Kn)), "a") do io
            @printf(io, "Ma=%g N=%d steps=%d  frac_real=%.4e frac_hyp=%.4e frac_any=%.4e  mean_dM=%.4e max_dM=%.4e  max_dCons/corr=%.3e  massdrift=%.3e\n",
                    Ma, N, d.steps, d.frac_realizability, d.frac_hyperbolicity, d.frac_any,
                    d.mean_dM_corrected, d.max_dM, d.max_dconserved_per_correction, d.conserved_rel_drift[1])
        end
        @printf("  Ma=%g: frac_real=%.3e frac_hyp=%.3e frac_any=%.3e  max_dM=%.3e  max_dCons/corr=%.3e\n",
                Ma, d.frac_realizability, d.frac_hyperbolicity, d.frac_any, d.max_dM, d.max_dconserved_per_correction)
    end
    MPI.Barrier(COMM)
end

RANK == 0 && println("\nDone (Ma sweep).")
