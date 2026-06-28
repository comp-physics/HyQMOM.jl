"""
    realize_gpu.jl — batched CUDA port of the realizability projection device
    function `RealizeDev.realizable_3D_M4_dev` (`src/realizability/realize_dev.jl`).

One GPU thread corrects ONE cell: it reads the 35 raw moments of that cell and runs
the alloc-free scalar chain

    M2CS4_35 -> univariate floors / skewness cap -> realizability_S2 ->
    realizability_S220 -> projection35 (6x6 symmetric min-eig via in-kernel cyclic
    Jacobi) -> standardized_to_M4

writing 35 corrected raw moments back. fp64 throughout, no heap allocation, no
cuSOLVER (the eigensolver is per-thread, register/local).

LAYOUT (matches the on-disk `proj_M.f64` / `proj_ref.f64` and the other gpu/ kernels):
  * `Min`,`Mout` :: `CuMatrix{Float64}` (35, B) — column k = the 35 raw moments of cell k.
  * `Ma` is either a scalar `Float64` (same Mach number for every cell) or a
    `CuVector{Float64}` of length B (per-cell Mach number).

Pure addition under `gpu/`; not wired into production.
"""
module RealizeGPU

using CUDA

# realize_dev.jl (module RealizeDev) references ReconDev as a sibling via `using
# ..ReconDev`, so recon_dev.jl must be included into THIS module first.
include(joinpath(@__DIR__, "..", "src", "numerics", "recon_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "realizability", "realize_dev.jl"))
using .RealizeDev: realizable_3D_M4_dev

export realizable_batched!, realizable_batched

@inline function _store!(Mout, k, r)
    @inbounds begin
        Mout[1,k]=r[1];   Mout[2,k]=r[2];   Mout[3,k]=r[3];   Mout[4,k]=r[4]
        Mout[5,k]=r[5];   Mout[6,k]=r[6];   Mout[7,k]=r[7];   Mout[8,k]=r[8]
        Mout[9,k]=r[9];   Mout[10,k]=r[10]; Mout[11,k]=r[11]; Mout[12,k]=r[12]
        Mout[13,k]=r[13]; Mout[14,k]=r[14]; Mout[15,k]=r[15]; Mout[16,k]=r[16]
        Mout[17,k]=r[17]; Mout[18,k]=r[18]; Mout[19,k]=r[19]; Mout[20,k]=r[20]
        Mout[21,k]=r[21]; Mout[22,k]=r[22]; Mout[23,k]=r[23]; Mout[24,k]=r[24]
        Mout[25,k]=r[25]; Mout[26,k]=r[26]; Mout[27,k]=r[27]; Mout[28,k]=r[28]
        Mout[29,k]=r[29]; Mout[30,k]=r[30]; Mout[31,k]=r[31]; Mout[32,k]=r[32]
        Mout[33,k]=r[33]; Mout[34,k]=r[34]; Mout[35,k]=r[35]
    end
    return nothing
end

@inline function _realize_one(Min, k, Ma)
    @inbounds realizable_3D_M4_dev(
        Min[1,k],  Min[2,k],  Min[3,k],  Min[4,k],  Min[5,k],  Min[6,k],  Min[7,k],
        Min[8,k],  Min[9,k],  Min[10,k], Min[11,k], Min[12,k], Min[13,k], Min[14,k],
        Min[15,k], Min[16,k], Min[17,k], Min[18,k], Min[19,k], Min[20,k], Min[21,k],
        Min[22,k], Min[23,k], Min[24,k], Min[25,k], Min[26,k], Min[27,k], Min[28,k],
        Min[29,k], Min[30,k], Min[31,k], Min[32,k], Min[33,k], Min[34,k], Min[35,k],
        Ma)
end

# scalar-Ma kernel
function _realize_kernel_scalar!(Mout, Min, Ma::Float64, B::Int)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= B
        r = _realize_one(Min, k, Ma)
        _store!(Mout, k, r)
    end
    return nothing
end

# per-cell-Ma kernel
function _realize_kernel_vec!(Mout, Min, Ma, B::Int)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= B
        @inbounds mak = Ma[k]
        r = _realize_one(Min, k, mak)
        _store!(Mout, k, r)
    end
    return nothing
end

"""
    realizable_batched!(Mout, Min, Ma; threads=128)

In-place batched realizability projection. `Min`,`Mout`::`CuMatrix{Float64}` are
`(35, B)` (column k = the 35 raw moments of cell k). `Ma` is a scalar `Real` (same
Mach number for every cell) or a `CuVector{Float64}` of length B (per-cell). One
thread per cell.
"""
function realizable_batched!(Mout::CuMatrix{Float64}, Min::CuMatrix{Float64}, Ma::Real;
                             threads::Int=128)
    B = size(Min, 2)
    @assert size(Min, 1) == 35 "Min must be (35, B)"
    @assert size(Mout) == size(Min) "Mout must match Min"
    nblocks = cld(B, threads)
    @cuda threads=threads blocks=nblocks _realize_kernel_scalar!(Mout, Min, Float64(Ma), B)
    return nothing
end

function realizable_batched!(Mout::CuMatrix{Float64}, Min::CuMatrix{Float64},
                             Ma::CuVector{Float64}; threads::Int=128)
    B = size(Min, 2)
    @assert size(Min, 1) == 35 "Min must be (35, B)"
    @assert size(Mout) == size(Min) "Mout must match Min"
    @assert length(Ma) == B "Ma vector must be length B"
    nblocks = cld(B, threads)
    @cuda threads=threads blocks=nblocks _realize_kernel_vec!(Mout, Min, Ma, B)
    return nothing
end

"""
    realizable_batched(M_host, Ma; threads=128) -> Mout_host

Host convenience: upload `(35, B)` host matrix, project, return `(35, B)` host
matrix. `Ma` is a scalar `Real` or a length-B host `Vector{Float64}`.
"""
function realizable_batched(M_host::AbstractMatrix{Float64}, Ma; threads::Int=128)
    @assert size(M_host, 1) == 35 "M_host must be (35, B)"
    B = size(M_host, 2)
    Md = CuArray(M_host)
    Mo = similar(Md)
    if Ma isa AbstractVector
        realizable_batched!(Mo, Md, CuArray(Float64.(Ma)); threads=threads)
    else
        realizable_batched!(Mo, Md, Ma; threads=threads)
    end
    CUDA.synchronize()
    return Array(Mo)
end

end # module
