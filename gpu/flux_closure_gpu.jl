"""
    flux_closure_gpu.jl — batched CUDA port of the validated device flux closure
    `FluxClosureDev.flux_closure35_dev` (`src/numerics/flux_closure_dev.jl`).

One GPU thread computes the full analytic flux closure for ONE cell: it reads the
35 raw moments of that cell, runs the alloc-free scalar chain

    M2CS4_35 (M4toC4 + standardize) -> hyqmom_3D (5th-order closures) ->
    S_to_C_batch -> C5toM5_3D -> assemble  ->  (Fx[1:35] | Fy[1:35] | Fz[1:35])

and writes the 105 outputs. The body is `FluxClosureDev.flux_closure35_dev` inlined
verbatim (it is plain fp64 scalar arithmetic + `@fastmath`, no allocation, no eig,
no quadrature, no dynamic dispatch), so it GPU-compiles directly.

LAYOUT (documented choice):
  * `M` is a `CuMatrix{Float64}` of size `(35, B)` — COLUMN-PER-CELL: column k holds
    the 35 raw moments of cell k in canonical M4 order (M000,M100,...,M022). This
    matches the on-disk `flux_M.f64` (`reshape(raw, 35, nb)`) and the `gpu/schur4_gpu.jl`
    precedent, so NO host-side transpose is needed.
  * `F` is a `CuMatrix{Float64}` of size `(105, B)` — column k holds Fx[1:35] (1..35),
    Fy[1:35] (36..70), Fz[1:35] (71..105) of cell k.

Note on coalescing: with (35,B) / (105,B) each thread reads/writes a contiguous column
(AoS across threads), so the 35-element read of adjacent threads is strided by 35.
A fully coalesced SoA layout would be (B,35)/(B,105), but the per-cell arithmetic is
heavily compute-bound (hundreds of FLOPs per loaded word), so global-memory coalescing
is not the bottleneck; AoS is chosen to avoid an extra host transpose and to match the
on-disk dump and the existing eig kernels. (The host wrapper can accept either via the
transpose helper below.)

fp64 throughout. Pure addition under `gpu/` — NOT wired into production, CUDA NOT added
to the main HyQMOM Project.
"""
module FluxClosureGPU

using CUDA

include(joinpath(@__DIR__, "..", "src", "numerics", "flux_closure_dev.jl"))
using .FluxClosureDev: flux_closure35_dev

export flux_closure35_batched!, flux_closure35_batched

# ---------------------------------------------------------------------------
# CUDA kernel: one thread per cell. M is (35, B), F is (105, B), column-per-cell.
# ---------------------------------------------------------------------------
function _flux_closure35_kernel!(F, M, B::Int)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= B
        @inbounds begin
            Fc = flux_closure35_dev(
                M[1,k],  M[2,k],  M[3,k],  M[4,k],  M[5,k],  M[6,k],  M[7,k],
                M[8,k],  M[9,k],  M[10,k], M[11,k], M[12,k], M[13,k], M[14,k],
                M[15,k], M[16,k], M[17,k], M[18,k], M[19,k], M[20,k], M[21,k],
                M[22,k], M[23,k], M[24,k], M[25,k], M[26,k], M[27,k], M[28,k],
                M[29,k], M[30,k], M[31,k], M[32,k], M[33,k], M[34,k], M[35,k])
            for n in 1:105
                F[n,k] = Fc[n]
            end
        end
    end
    return nothing
end

"""
    flux_closure35_batched!(F, M; threads=128)

In-place batched flux closure. `M::CuMatrix{Float64}` is `(35, B)` (column k = the 35
raw moments of cell k), `F::CuMatrix{Float64}` is `(105, B)` (column k = Fx|Fy|Fz).
One thread per cell.
"""
function flux_closure35_batched!(F::CuMatrix{Float64}, M::CuMatrix{Float64};
                                 threads::Int=128)
    B = size(M, 2)
    @assert size(M, 1) == 35  "M must be (35, B)"
    @assert size(F, 1) == 105 "F must be (105, B)"
    @assert size(F, 2) == B   "F and M batch dims must match"
    nblocks = cld(B, threads)
    @cuda threads=threads blocks=nblocks _flux_closure35_kernel!(F, M, B)
    return nothing
end

"""
    flux_closure35_batched(M_host::AbstractMatrix{Float64}; threads=128) -> Matrix{Float64}

End-to-end host convenience: upload `(35, B)` host matrix, solve, return the `(105, B)`
host result (Fx|Fy|Fz per column).
"""
function flux_closure35_batched(M_host::AbstractMatrix{Float64}; threads::Int=128)
    @assert size(M_host, 1) == 35 "M_host must be (35, B)"
    B = size(M_host, 2)
    Md = CuArray(M_host)
    Fd = CUDA.zeros(Float64, 105, B)
    flux_closure35_batched!(Fd, Md; threads=threads)
    CUDA.synchronize()
    return Array(Fd)
end

end # module
