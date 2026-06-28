using MPI, CUDA, Printf
H="/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/gpu"
include(joinpath(H,"timestep3d_gpu.jl")); using .Timestep3DGPU
DATA="/storage/scratch1/6/sbryngelson3/gpudata"

MPI.Init(); comm=MPI.COMM_WORLD; rank=MPI.Comm_rank(comm); nranks=MPI.Comm_size(comm)
CUDA.device!(rank % CUDA.ndevices())
n=24; halo=2; Ma=2.0; dx=1.0/n; nstep=5
@assert n % nranks == 0; nzloc=div(n,nranks)

nb=parse(Int,strip(read(joinpath(DATA,"proj.meta"),String)))
src=reshape(reinterpret(Float64,read(joinpath(DATA,"proj_M.f64"))),35,nb)
Mfull=Array(reshape(src[:,1:n^3],35,n,n,n))

# single-GPU reference (rank 0), on-device CFL
usedref=Float64[]; Mref_host=zeros(0)
if rank==0
    Mref=CuArray(copy(Mfull))
    usedref=Timestep3DGPU.march3d_gpu!(Mref,dx,Ma,nstep)
    Mref_host=Array(Mref)
end

# multi-GPU z-slab march, on-device global CFL
z0=rank*nzloc
Mslab=CuArray(Array(@view Mfull[:,:,:,z0+1:z0+nzloc]))
used=Timestep3DGPU.march3d_slab_gpu!(Mslab,dx,Ma,nstep,comm)

# gather final field (rank order == z order)
sb=vec(Array(Mslab)); counts=fill(35*n*n*nzloc,nranks)
rbuf= rank==0 ? Vector{Float64}(undef,35*n^3) : Float64[]
MPI.Gatherv!(sb, rank==0 ? MPI.VBuffer(rbuf,counts) : nothing, comm; root=0)

if rank==0
    Mmulti=reshape(rbuf,35,n,n,n)
    df=maximum(abs.(Mmulti.-Mref_host))
    ddt=maximum(abs.(used.-usedref))
    @printf("nranks=%d n=%d nstep=%d  multi-GPU timestep vs single-GPU:\n",nranks,n,nstep)
    @printf("  dt-sequence max abs diff = %.3e\n", ddt)
    @printf("  final field  max abs diff = %.3e   %s\n", df,
            (df==0.0 && ddt==0.0) ? "BIT-IDENTICAL PASS" : "MISMATCH")
    @printf("  (final field finite: %s, range [%.3g, %.3g])\n", all(isfinite,Mmulti), extrema(Mmulti)...)
end
MPI.Barrier(comm); MPI.Finalize()
