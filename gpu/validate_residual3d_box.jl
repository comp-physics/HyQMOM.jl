using CUDA, Printf
include("/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/gpu/residual3d_gpu.jl"); using .Residual3DGPU
DATA="/storage/scratch1/6/sbryngelson3/gpudata"
nb=parse(Int,strip(read(joinpath(DATA,"proj.meta"),String)))
src=reshape(reinterpret(Float64,read(joinpath(DATA,"proj_M.f64"))),35,nb)
for n in (16,24)
    dx=1.0/n; Ma=2.0
    M=Array(reshape(src[:,1:n^3],35,n,n,n))
    Rc=residual3d_gpu(M,n,dx,Ma)                 # cubic kernel
    Rb=residual3d_box_gpu(M,n,n,n,dx,Ma)         # rectangular generalization, cube case
    d=maximum(abs.(Rc.-Rb))
    @printf("n=%d  box-vs-cubic max abs diff = %.3e   %s\n", n, d, d==0.0 ? "BIT-IDENTICAL" : "MISMATCH")
end
