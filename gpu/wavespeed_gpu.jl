"""
    wavespeed_gpu.jl — batched CUDA port of the device wave-speed path
    `WavespeedDev.realize_and_speed_dev` (`gpu/wavespeed_dev.jl`).

One GPU thread computes the combined wave speeds (vmin, vmax) for ONE cell and ONE
axis: it reads the 35 raw moments of that cell and runs the alloc-free scalar chain

    eigenvalues6{,z}_hyperbolic_3D (2 planes -> jacobian15 3x3+4x4 block eigs,
      + correct_moments on the complex branch)  ->  closure_and_eigenvalues(marginal)

The body is `realize_and_speed_dev` (plain fp64 scalar arithmetic + `@fastmath`, no
allocation, reusing the validated Schur4 4x4 solver), so it GPU-compiles directly.

LAYOUT (matches `gpu/flux_closure_gpu.jl` / `gpu/schur4_gpu.jl` and the on-disk
`ws_M.f64`):
  * `M`    :: `CuMatrix{Float64}` (35, B) — column k = the 35 raw moments of cell k.
  * `vmin`,`vmax` :: `CuVector{Float64}` (B) — per-cell wave speeds for the chosen axis.

`axis` (1,2,3) is a compile-time-specialized scalar kernel argument; call once per
axis. `Ma` is accepted but unused in the wave-speed path. fp64 throughout. Pure
addition under `gpu/`; not wired into production.
"""
module WavespeedGPU

using CUDA

include(joinpath(@__DIR__, "wavespeed_dev.jl"))
using .WavespeedDev: realize_and_speed_dev

export wave_speeds_batched!, wave_speeds_batched

function _wave_speeds_kernel!(vmin, vmax, M, axis::Int, Ma::Float64, B::Int)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= B
        @inbounds begin
            a, b = realize_and_speed_dev(
                M[1,k],  M[2,k],  M[3,k],  M[4,k],  M[5,k],  M[6,k],  M[7,k],
                M[8,k],  M[9,k],  M[10,k], M[11,k], M[12,k], M[13,k], M[14,k],
                M[15,k], M[16,k], M[17,k], M[18,k], M[19,k], M[20,k], M[21,k],
                M[22,k], M[23,k], M[24,k], M[25,k], M[26,k], M[27,k], M[28,k],
                M[29,k], M[30,k], M[31,k], M[32,k], M[33,k], M[34,k], M[35,k],
                axis, Ma)
            vmin[k] = a
            vmax[k] = b
        end
    end
    return nothing
end

"""
    wave_speeds_batched!(vmin, vmax, M, axis, Ma; threads=128)

In-place batched wave speeds. `M::CuMatrix{Float64}` is `(35, B)` (column k = the 35
raw moments of cell k); `vmin`,`vmax`::`CuVector{Float64}` are length `B`. One thread
per cell. `axis ∈ (1,2,3)`.
"""
function wave_speeds_batched!(vmin::CuVector{Float64}, vmax::CuVector{Float64},
                              M::CuMatrix{Float64}, axis::Int, Ma::Real=0.0;
                              threads::Int=128)
    B = size(M, 2)
    @assert size(M, 1) == 35 "M must be (35, B)"
    @assert length(vmin) == B && length(vmax) == B "vmin/vmax must be length B"
    @assert axis in (1, 2, 3) "axis must be 1, 2, or 3"
    nblocks = cld(B, threads)
    @cuda threads=threads blocks=nblocks _wave_speeds_kernel!(vmin, vmax, M, axis, Float64(Ma), B)
    return nothing
end

"""
    wave_speeds_batched(M_host, axis, Ma; threads=128) -> (vmin, vmax)

Host convenience: upload `(35, B)` host matrix, solve for one axis, return the two
length-B host vectors.
"""
function wave_speeds_batched(M_host::AbstractMatrix{Float64}, axis::Int, Ma::Real=0.0;
                             threads::Int=128)
    @assert size(M_host, 1) == 35 "M_host must be (35, B)"
    B = size(M_host, 2)
    Md = CuArray(M_host)
    vmn = CUDA.zeros(Float64, B)
    vmx = CUDA.zeros(Float64, B)
    wave_speeds_batched!(vmn, vmx, Md, axis, Ma; threads=threads)
    CUDA.synchronize()
    return Array(vmn), Array(vmx)
end

end # module
