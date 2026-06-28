using MPI, CUDA, Printf
include("/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/gpu/residual3d_gpu.jl"); using .Residual3DGPU
DATA="/storage/scratch1/6/sbryngelson3/gpudata"

MPI.Init(); comm=MPI.COMM_WORLD; rank=MPI.Comm_rank(comm); nranks=MPI.Comm_size(comm)
CUDA.device!(rank % CUDA.ndevices())
n=24; halo=2; Ma=2.0; dx=1.0/n
@assert n % nranks == 0; nzloc=div(n,nranks); nz_ext=nzloc+2halo

# identical full field on all ranks (realizable states from proj_M)
nb=parse(Int,strip(read(joinpath(DATA,"proj.meta"),String)))
src=reshape(reinterpret(Float64,read(joinpath(DATA,"proj_M.f64"))),35,nb)
Mfull=Array(reshape(src[:,1:n^3],35,n,n,n))

# single-GPU reference (rank 0)
Rref = rank==0 ? residual3d_box_gpu(Mfull,n,n,n,dx,Ma) : zeros(0)

# --- build extended slab; ghosts via host-staged halo exchange ---
z0 = rank*nzloc
Mext = zeros(Float64,35,n,n,nz_ext)
Mext[:,:,:,halo+1:halo+nzloc] .= @view Mfull[:,:,:, z0+1:z0+nzloc]   # interior
left  = rank>0        ? rank-1 : MPI.PROC_NULL
right = rank<nranks-1 ? rank+1 : MPI.PROC_NULL
sendT = vec(Mext[:,:,:, halo+nzloc-halo+1:halo+nzloc])   # my top interior halo planes
sendB = vec(Mext[:,:,:, halo+1:halo+halo])               # my bottom interior halo planes
recvB = fill(NaN, length(sendB)); recvT = fill(NaN, length(sendT))
MPI.Sendrecv!(sendT, recvB, comm; dest=right, source=left,  sendtag=1, recvtag=1)  # up: recv bottom ghost from left
MPI.Sendrecv!(sendB, recvT, comm; dest=left,  source=right, sendtag=2, recvtag=2)  # down: recv top ghost from right
# bottom ghost
if left==MPI.PROC_NULL
    for g in 1:halo; Mext[:,:,:,g] .= @view Mext[:,:,:,halo+1]; end          # outflow replicate plane 1
else
    Mext[:,:,:,1:halo] .= reshape(recvB,35,n,n,halo)
end
# top ghost
if right==MPI.PROC_NULL
    for g in 1:halo; Mext[:,:,:,halo+nzloc+g] .= @view Mext[:,:,:,halo+nzloc]; end
else
    Mext[:,:,:,halo+nzloc+1:nz_ext] .= reshape(recvT,35,n,n,halo)
end

# residual on extended slab (resident on this GPU), keep interior
Rext = residual3d_box_gpu(Mext,n,n,nz_ext,dx,Ma)
Rint = Rext[:,:,:,halo+1:halo+nzloc]

# gather interior residuals (rank order == z order)
sb=vec(Rint); counts=fill(35*n*n*nzloc,nranks)
rbuf = rank==0 ? Vector{Float64}(undef,35*n^3) : Float64[]
MPI.Gatherv!(sb, rank==0 ? MPI.VBuffer(rbuf,counts) : nothing, comm; root=0)

if rank==0
    Rmulti=reshape(rbuf,35,n,n,n)
    d=maximum(abs.(Rmulti.-Rref))
    @printf("nranks=%d  n=%d  z-slab residual vs single-GPU full domain:\n",nranks,n)
    @printf("  max abs diff = %.3e   %s\n", d, d==0.0 ? "BIT-IDENTICAL PASS" : "MISMATCH")
end
MPI.Barrier(comm); MPI.Finalize()
