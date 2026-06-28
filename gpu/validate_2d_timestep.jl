# validate_2d_timestep.jl — GPU 2D (nz=1) SSP-RK3 timestep self-consistency:
# nz=1 must equal the interior plane of a z-homogeneous 3D run (Lz=0). Needs proj_M.f64.
# Run: srun --mpi=pmix -n1 --gpus=1 julia --project=gpu/gpuenv2 gpu/validate_2d_timestep.jl
using CUDA, Printf
include("/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/gpu/timestep3d_gpu.jl"); using .Timestep3DGPU
DATA="/storage/scratch1/6/sbryngelson3/gpudata"
nb=parse(Int,strip(read(joinpath(DATA,"proj.meta"),String)))
src=reshape(reinterpret(Float64,read(joinpath(DATA,"proj_M.f64"))),35,nb)
n=24; dx=1.0/n; Ma=2.0; nstep=4
idx=round.(Int, range(1,nb,length=n*n))
M2=Array{Float64}(undef,35,n,n,1); for c in 1:n*n; M2[:,(c-1)%n+1,(c-1)÷n+1,1].=src[:,idx[c]]; end
# 2D timestep (nz=1) with fixed dts (shared, so dt isn't a variable)
dts = fill(1.0e-6, nstep)
A=CuArray(copy(M2)); u2=march3d_gpu!(A,dx,Ma,nstep; dts=dts); R2=Array(A)
@printf("2D timestep (nz=1) runs: finite=%s range=[%.3g,%.3g]\n", all(isfinite,R2), extrema(R2)...)
# z-homogeneous 3D (nz=4), same dts; middle plane must equal the 2D result
nz=4; M3=Array{Float64}(undef,35,n,n,nz); for k in 1:nz; M3[:,:,:,k].=M2[:,:,:,1]; end
B=CuArray(M3); march3d_gpu!(B,dx,Ma,nstep; dts=dts); R3=Array(B)
d=maximum(abs.(R3[:,:,:,2].-R2[:,:,:,1]))
@printf("2D(nz=1) timestep vs interior plane of z-homogeneous 3D timestep: max abs diff = %.3e  %s\n",
        d, d==0.0 ? "BIT-IDENTICAL" : (d<1e-10 ? "MATCH(<1e-10)" : "DIFFER"))
