ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, MPI, JLD2, Printf
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# Grid-convergence driver for the crossing problem. Saves DENSITY ONLY (M000) so
# fine grids stay tractable: 1024^3 full 35-moment state is ~300 GB, but density
# alone is ~8 GB. Rodney's guidance: first-order reference solutions on fine grids,
# convergence judged on density (numerical diffusion is worst for M000).
#
# Env: CONV_NP, CONV_ORDER (1=ref, 2=high-order), CONV_MA, CONV_TMAX, CONV_VACFLOOR.

Np    = parse(Int,     get(ENV,"CONV_NP","128"))
order = parse(Int,     get(ENV,"CONV_ORDER","1"))
Ma    = parse(Float64, get(ENV,"CONV_MA","10.0"))
tmax  = parse(Float64, get(ENV,"CONV_TMAX","0.015"))
vacf  = parse(Float64, get(ENV,"CONV_VACFLOOR","0.001"))

params = (
    Nx=Np, Ny=Np, Nz=Np, Nmom=35,
    tmax=tmax, Kn=1000.0, Ma=Ma, flag2D=0, CFL=1/3,
    nnmax=1000000, dtmax=1000.0,
    rhol=1.0, rhor=0.001, T=1.0, r110=0.0, r101=0.0, r011=0.0,
    symmetry_check_interval=1000000, homogeneous_z=false, debug_output=false,
    snapshot_interval=0, ic_type=:crossing_matlab,
    spatial_order=order, ho_vacuum_floor=vacf,
)

t0 = time()
M_final, final_time, steps, grid = simulation_runner(params)
wall = time() - t0

if rank == 0
    rho = Array{Float64,3}(M_final[:,:,:,1])   # density only
    out = @sprintf("/storage/project/r-sbryngelson3-0/sbryngelson3/debug/conv_o%d_np%d_ma%d.jld2",
                   order, Np, Int(Ma))
    jldsave(out; rho=rho, Np=Np, order=order, Ma=Ma, tmax=final_time, steps=steps, wall=wall)
    @printf("CONV o%d Np=%d Ma=%g t=%.6f steps=%d wall=%.1fs  rho[min,max]=[%.5f,%.5f]  saved %s\n",
            order, Np, Ma, final_time, steps, wall, minimum(rho), maximum(rho), out)
end
MPI.Finalize()
