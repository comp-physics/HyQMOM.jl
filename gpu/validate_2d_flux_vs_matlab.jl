# validate_2d_flux_vs_matlab.jl — DIRECT GPU vs MATLAB 2D golden (flag2D=1, Ma=0.5).
# Prereq: dump the golden in an MPI-free env that can load MAT/HDF5 (system OpenMPI off LD_LIBRARY_PATH):
#   julia --project=<matenv> dump_golden.jl  (reads legacy/3D_MATLAB/tests/goldenfiles/test_flux_eigenvalues_golden.mat
#   -> flxg_in.f64 / flxg_out.f64). Then run this under gpuenv2. Expect max rel ~4e-16 (machine precision).
using CUDA, Printf
H="/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/gpu"
include(joinpath(H,"realize_gpu.jl")); using .RealizeGPU
include(joinpath(H,"..","src","numerics","flux_closure_dev.jl")); using .FluxClosureDev
DATA="/storage/scratch1/6/sbryngelson3/gpudata"
input=collect(reinterpret(Float64, read("$DATA/flxg_in.f64")))
g=collect(reinterpret(Float64, read("$DATA/flxg_out.f64")))
Fxm=g[1:35]; Fym=g[36:70]; Fzm=g[71:105]; Mrm=g[106:140]; Ma=0.5
# GPU-launched realizability projection (the kernel used in the 2D solver)
Mr = vec(RealizeGPU.realizable_batched(reshape(input,35,1), [Ma]))
# flux device function (the exact code inlined in the 2D residual kernels)
F = collect(FluxClosureDev.flux_closure35_dev(Mr...)); Fx=F[1:35]; Fy=F[36:70]; Fz=F[71:105]
rel(a,b)=maximum(abs.(a.-b)./max.(abs.(b),1.0))
@printf("DIRECT GPU vs MATLAB 2D golden (flag2D=1, Ma=0.5):\n")
@printf("  realizable_3D_M4 (GPU kernel) : max abs %.3e  max rel %.3e\n", maximum(abs.(Mr.-Mrm)), rel(Mr,Mrm))
@printf("  flux Fx (GPU device fn)       : max abs %.3e  max rel %.3e\n", maximum(abs.(Fx.-Fxm)), rel(Fx,Fxm))
@printf("  flux Fy                       : max abs %.3e  max rel %.3e\n", maximum(abs.(Fy.-Fym)), rel(Fy,Fym))
@printf("  flux Fz                       : max abs %.3e  max rel %.3e\n", maximum(abs.(Fz.-Fzm)), rel(Fz,Fzm))
mx=maximum([rel(Mr,Mrm),rel(Fx,Fxm),rel(Fy,Fym),rel(Fz,Fzm)])
@printf("  => max rel = %.3e  %s\n", mx, mx<1e-10 ? "MATCHES MATLAB (<1e-10)" : "CHECK")
