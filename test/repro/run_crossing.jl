# Reproduce the MATLAB jet-crossing case (main_crossing_3DHyQMOM35.m) in Julia.
#
#   REPRO_MA=0.0 scripts/pace_mpi.sh 16 test/repro/run_crossing.jl
#   REPRO_MA=2.0 scripts/pace_mpi.sh 16 test/repro/run_crossing.jl
#
# On rank 0 saves M_final to  <repo>/../jl_M_Ma<Ma>.jld2  for comparison with the
# reference riemann_full3D_..._Ma<Ma>.mat (see RUNNING.md §7-8).
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using HyQMOM, MPI, JLD2, Printf
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

Ma = parse(Float64, get(ENV, "REPRO_MA", "0.0"))
Np = parse(Int, get(ENV, "REPRO_NP", "128"))

params = (
    Nx=Np, Ny=Np, Nz=Np, Nmom=35,
    tmax=0.008, Kn=1000.0, Ma=Ma, flag2D=0, CFL=1/3,
    nnmax=100000, dtmax=1000.0,
    rhol=1.0, rhor=0.001, T=1.0, r110=0.0, r101=0.0, r011=0.0,
    symmetry_check_interval=1000, homogeneous_z=false, debug_output=false,
    snapshot_interval=0,
    ic_type=:crossing_matlab,
)

M_final, final_time, steps, grid = simulation_runner(params)

if rank == 0
    @printf("DONE Ma=%g Np=%d: steps=%d t=%.6f size=%s totmass=%.10e\n",
            Ma, Np, steps, final_time, string(size(M_final)), sum(M_final[:,:,:,1]))
    out = joinpath(@__DIR__, "..", "..", "..", "jl_M_Ma$(Int(Ma)).jld2")
    jldsave(out; M=M_final, t=final_time, steps=steps)
    println("saved ", abspath(out))
end
MPI.Finalize()
