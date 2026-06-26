ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, MPI, JLD2, Printf
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

Np    = parse(Int,     get(ENV,"REPRO_NP","128"))
Ma    = parse(Float64, get(ENV,"REPRO_MA","100.0"))
tmax  = parse(Float64, get(ENV,"REPRO_TMAX","0.002"))
order = parse(Int,     get(ENV,"REPRO_ORDER","2"))
# near-vacuum high-order floor; default to the background density rhor
vacfloor = parse(Float64, get(ENV,"REPRO_VACFLOOR","0.001"))
# realizability scaling limiter (REPRO_LIMITER=1 to enable, default off)
use_limiter = parse(Int, get(ENV,"REPRO_LIMITER","0")) != 0
# Rodney's projection-triggered first-order recon (REPRO_PROJREC=1, default off)
use_projrec = parse(Int, get(ENV,"REPRO_PROJREC","0")) != 0

params = (
    Nx=Np, Ny=Np, Nz=Np, Nmom=35,
    tmax=tmax, Kn=1000.0, Ma=Ma, flag2D=0, CFL=1/3,
    nnmax=100000, dtmax=1000.0,
    rhol=1.0, rhor=0.001, T=1.0, r110=0.0, r101=0.0, r011=0.0,
    symmetry_check_interval=100000, homogeneous_z=false, debug_output=false,
    snapshot_interval=0,
    ic_type=:crossing_matlab,
    spatial_order=order,
    ho_vacuum_floor=vacfloor,
    ho_realizability_limiter=use_limiter,
    ho_proj_first_order=use_projrec,
)

t0 = time()
M_final, final_time, steps, grid = simulation_runner(params)
wall = time() - t0

if rank == 0
    rho = M_final[:,:,:,1]
    # max density gradient magnitude (interior centered diffs)
    function maxgrad(r)
        nx,ny,nz = size(r); g=0.0; dx=1.0/nx
        for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
            gx=(r[i+1,j,k]-r[i-1,j,k])/(2dx); gy=(r[i,j+1,k]-r[i,j-1,k])/(2dx); gz=(r[i,j,k+1]-r[i,j,k-1])/(2dx)
            g=max(g, sqrt(gx^2+gy^2+gz^2))
        end
        g
    end
    @printf("ORDER=%d Np=%d Ma=%g tmax=%g : steps=%d t=%.6f wall=%.1fs\n", order,Np,Ma,tmax,steps,final_time,wall)
    @printf("  density min/max = %.5f / %.5f   totmass=%.8e   max|grad rho|=%.4f\n",
            minimum(rho), maximum(rho), sum(rho), maxgrad(rho))
    out = "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np$(Np)_ma$(Int(Ma))_o$(order).jld2"
    jldsave(out; M=M_final, t=final_time, steps=steps, wall=wall)
    println("  saved $out")
end
MPI.Finalize()
