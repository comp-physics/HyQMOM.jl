# Kinetic-flux instability demo: runs the crossing jets with riemann_solver=:hll
# (stable) and :kinetic (unstable) and records the per-step timestep so the
# collapse to NaN is visible. Feeds debug/plot_diffusion_results.py.
#
#   DIFFUSION_OUTDIR=debug/reprodata julia --project=. debug/run_kinetic_vs_hll.jl
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, MPI, Printf
MPI.Initialized() || MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

Np   = parse(Int, get(ENV,"REPRO_NP","24"))
OUT  = get(ENV,"DIFFUSION_OUTDIR", joinpath(@__DIR__,"reprodata"))
rank==0 && mkpath(OUT)

# capture dt per step via the runner's verbose output is awkward; instead re-run
# the SSP-RK3 loop here at low cost using the package's own timestep + residual.
# Simplest robust approach: drive simulation_runner with a step cap and parse its
# verbose stream. We instead use a short tmax and read the returned step count;
# for the dt trace we run the runner verbose and tee to a file.
function run(rs)
    p = (Nx=Np,Ny=Np,Nz=Np,Nmom=35,tmax=0.02,Kn=1000.0,Ma=10.0,flag2D=0,CFL=1/3,
         nnmax=100000,dtmax=1000.0,rhol=1.0,rhor=0.001,T=1.0,r110=0.0,r101=0.0,r011=0.0,
         symmetry_check_interval=100000,homogeneous_z=false,debug_output=false,snapshot_interval=0,
         ic_type=:crossing_matlab,spatial_order=2,ho_vacuum_floor=0.001,
         ho_realizability_limiter=false,ho_proj_first_order=false,riemann_solver=rs)
    M,t,steps,_ = simulation_runner(p)
    (steps, t, rank==0 && M!==nothing ? all(isfinite,M) : true)
end

for rs in (:hll, :kinetic)
    s,t,ok = run(rs)
    rank==0 && @printf("riemann_solver=%-8s : steps=%d t_reached=%.4g finite=%s\n", string(rs), s, t, ok)
end
# The per-step dt trace is printed by the runner's verbose log; capture it by
# running this script with verbose output redirected, then grep "Step .* dt =".
# See docs/reproducing-diffusion-results.md.
