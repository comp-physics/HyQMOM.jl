# Mach-ladder diffusion study: first-order vs high-order reconstruction on the
# 3D 35-moment crossing jets. For each (Ma, order) it runs the crossing-jet IC,
# then records peak density, max |grad rho|, min density and total mass, plus a
# max-over-z density projection (the peak structure, since the jets move along
# the box diagonal). Outputs feed debug/plot_diffusion_results.py.
#
# Usage (see docs/reproducing-diffusion-results.md for the MPI environment):
#   REPRO_NP=128 srun --mpi=pmix -n 16 julia --project=. debug/run_mach_ladder.jl
#   REPRO_NP=48  julia --project=. debug/run_mach_ladder.jl      # quick single-rank check
#
# Env knobs:  REPRO_NP (grid, default 64), DIFFUSION_OUTDIR (output dir, default debug/reprodata),
#             REPRO_MAS ("10,25,50,100"), REPRO_TMAX (default 0.002)
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, MPI, Printf, DelimitedFiles
MPI.Initialized() || MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

Np    = parse(Int,     get(ENV,"REPRO_NP","64"))
tmax  = parse(Float64, get(ENV,"REPRO_TMAX","0.002"))
mas   = parse.(Int, split(get(ENV,"REPRO_MAS","10,25,50,100"),","))
OUT   = get(ENV,"DIFFUSION_OUTDIR", joinpath(@__DIR__,"reprodata"))
rank==0 && mkpath(OUT)

maxgrad(r) = begin
    nx,ny,nz=size(r); g=0.0; dx=1.0/nx
    @inbounds for k in 2:nz-1,j in 2:ny-1,i in 2:nx-1
        gx=(r[i+1,j,k]-r[i-1,j,k])/(2dx); gy=(r[i,j+1,k]-r[i,j-1,k])/(2dx); gz=(r[i,j,k+1]-r[i,j,k-1])/(2dx)
        g=max(g,sqrt(gx^2+gy^2+gz^2))
    end; g
end

function runcase(Ma, order)
    p = (Nx=Np,Ny=Np,Nz=Np,Nmom=35,tmax=tmax,Kn=1000.0,Ma=Float64(Ma),flag2D=0,CFL=1/3,
         nnmax=100000,dtmax=1000.0,rhol=1.0,rhor=0.001,T=1.0,r110=0.0,r101=0.0,r011=0.0,
         symmetry_check_interval=100000,homogeneous_z=false,debug_output=false,snapshot_interval=0,
         ic_type=:crossing_matlab,spatial_order=order,ho_vacuum_floor=0.001,
         ho_realizability_limiter=false,ho_proj_first_order=false,riemann_solver=:hll)
    M,t,steps,_ = simulation_runner(p)
    M  # rank 0 holds the gathered field; other ranks get nothing meaningful
end

metrics = String["Ma,order,peak_rho,maxgrad,min_rho,totmass"]
for Ma in mas, order in (1,2)
    M = runcase(Ma, order)
    if rank == 0 && M !== nothing
        rho = @view M[:,:,:,1]
        push!(metrics, @sprintf("%d,%d,%.6f,%.6f,%.6e,%.6e",Ma,order,maximum(rho),maxgrad(rho),minimum(rho),sum(rho)))
        proj = dropdims(maximum(rho,dims=3),dims=3)
        writedlm(joinpath(OUT,"proj_ma$(Ma)_o$(order).txt"), proj)
        @printf("Ma=%d o%d: peak=%.4f maxgrad=%.2f\n",Ma,order,maximum(rho),maxgrad(rho)); flush(stdout)
    end
end
if rank == 0
    write(joinpath(OUT,"ladder_metrics.csv"), join(metrics,"\n")*"\n")
    println("wrote $(joinpath(OUT,"ladder_metrics.csv")) and proj_*.txt")
end
