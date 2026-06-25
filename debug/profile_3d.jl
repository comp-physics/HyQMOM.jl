ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, MPI, Profile, Printf
MPI.Init()

mkparams(tmax) = (
    Nx=48, Ny=48, Nz=48, Nmom=35,
    tmax=tmax, Kn=1000.0, Ma=10.0, flag2D=0, CFL=1/3,
    nnmax=100000, dtmax=1000.0,
    rhol=1.0, rhor=0.001, T=1.0, r110=0.0, r101=0.0, r011=0.0,
    symmetry_check_interval=100000, homogeneous_z=false, debug_output=false,
    snapshot_interval=0, ic_type=:crossing_matlab, spatial_order=2, ho_vacuum_floor=0.001,
)

# JIT warmup: ~1 step
simulation_runner(mkparams(4.0e-4))

Profile.clear(); Profile.init(n=200_000_000, delay=0.0005)
@profile simulation_runner(mkparams(3.0e-3))   # ~7 steps

open("/storage/project/r-sbryngelson3-0/sbryngelson3/debug/profile_3d_flat.txt", "w") do io
    Profile.print(IOContext(io, :displaysize=>(10000,300)); format=:flat, sortedby=:count, mincount=30)
end
println("profile written")
MPI.Finalize()
