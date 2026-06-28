using MPI, CUDA, Printf
include("/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/gpu/realize_gpu.jl"); using .RealizeGPU

MPI.Init(); comm=MPI.COMM_WORLD; rank=MPI.Comm_rank(comm); nranks=MPI.Comm_size(comm)
CUDA.device!(rank % CUDA.ndevices())
n=256; halo=2; nsteps=20; Ma=1.0
@assert n % nranks == 0; nz_loc=div(n,nranks)

M  = CUDA.rand(Float64,35,n,n,nz_loc).+0.5      # resident for whole run
Mo = similar(M)
Mmat=reshape(M,35,:); Momat=reshape(Mo,35,:)

# preallocated, PINNED host halo buffers + reused GPU ghost buffers
mk()= (h=Array{Float64}(undef,35,n,n,halo); CUDA.pin(h); h)
hs_top=mk(); hr_top=mk(); hs_bot=mk(); hr_bot=mk()
g_top=CUDA.zeros(Float64,35,n,n,halo); g_bot=CUDA.zeros(Float64,35,n,n,halo)
left=(rank-1+nranks)%nranks; right=(rank+1)%nranks
halo_t=Ref(0.0); comp_t=Ref(0.0)

function step!()
    if nranks>1
        t0=time_ns()
        copyto!(hs_bot, @view M[:,:,:,1:halo])                       # D2H contiguous
        copyto!(hs_top, @view M[:,:,:,nz_loc-halo+1:nz_loc])
        CUDA.synchronize()
        MPI.Sendrecv!(hs_top,hr_top,comm; dest=right,sendtag=1,source=left, recvtag=1)
        MPI.Sendrecv!(hs_bot,hr_bot,comm; dest=left, sendtag=2,source=right,recvtag=2)
        copyto!(g_top,hr_top); copyto!(g_bot,hr_bot)                 # H2D into reused ghosts
        CUDA.synchronize()
        halo_t[]+=(time_ns()-t0)/1e9
    end
    t1=time_ns()
    RealizeGPU.realizable_batched!(Momat,Mmat,Ma); CUDA.synchronize()
    comp_t[]+=(time_ns()-t1)/1e9
end

step!(); halo_t[]=0.0; comp_t[]=0.0
t=time_ns(); for _ in 1:nsteps; step!(); end; wall=(time_ns()-t)/1e9
cells=n*n*n
wmax=MPI.Allreduce(wall,max,comm); hmax=MPI.Allreduce(halo_t[],max,comm); cmax=MPI.Allreduce(comp_t[],max,comm)
if rank==0
    @printf("nranks=%d cube n=%d cells=%d (%d/GPU) steps=%d\n",nranks,n,cells,div(cells,nranks),nsteps)
    @printf("  wall %.3f s  (%.1f Mcells/s)\n", wmax, cells*nsteps/wmax/1e6)
    @printf("  compute %.3f s (%.0f%%)   halo %.3f s (%.0f%%)\n", cmax,100*cmax/wmax, hmax,100*hmax/wmax)
end
MPI.Barrier(comm); MPI.Finalize()
