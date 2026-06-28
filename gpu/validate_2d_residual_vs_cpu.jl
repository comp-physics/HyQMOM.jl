# validate_2d_residual_vs_cpu.jl — GPU 2D residual vs the MATLAB-ported CPU residual_ho_3d! (nz=1).
# Prereq: dump_r2d_cpu.jl in the MAIN project (CPU) writes r2d_M.f64 / r2d_R.f64. Expect rel <=1e-6 (GATE).
# (the ~e-7 floor is schur4(GPU) vs LAPACK(CPU) in the wave-speed eig). CPU residual_ho_3d! is the MATLAB port.
using CUDA, Printf
include("/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/gpu/residual3d_gpu.jl"); using .Residual3DGPU
DATA="/storage/scratch1/6/sbryngelson3/gpudata"
meta=split(strip(read("$DATA/r2d.meta",String)),'\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3])
M=reshape(collect(reinterpret(Float64,read("$DATA/r2d_M.f64"))),35,n,n,1)
Rref=reshape(collect(reinterpret(Float64,read("$DATA/r2d_R.f64"))),35,n,n,1)
Rgpu=residual3d_box_gpu(Array(M),n,n,1,dx,Ma; vacuum_floor=0.001, project_faces=true)
d=maximum(abs.(Rgpu.-Rref)); sc=max(maximum(abs.(Rref)),1.0)
r=maximum(abs.(Rgpu.-Rref)./max.(abs.(Rref),1.0))
nbad=count(>(1e-6), abs.(Rgpu.-Rref)./max.(abs.(Rref),1.0))
@printf("GPU 2D residual vs MATLAB-ported CPU residual_ho_3d! (nz=1, %d^2 cells):\n", n)
@printf("  max abs err = %.3e   (Rref scale %.3e)\n", d, sc)
@printf("  max rel err = %.3e   (# rel>1e-6: %d / %d)   GATE(rel<=1e-6): %s\n",
        r, nbad, length(Rref), r<=1e-6 ? "PASS" : "FAIL")
