"""
    residual1d_gpu.jl — end-to-end first-order 1D HLL residual on GPU.

CAPSTONE composition of the already-ported, individually-validated device kernels
(`gpu/flux_closure_dev.jl`, `gpu/wavespeed_dev.jl`, `gpu/schur4.jl`) into a complete
on-device first-order spatial residual matching the CPU
`residual_1d(Mline, dx, Ma; order=1, bc=:outflow)` (`src/numerics/highorder_flux.jl`).

Two kernel launches over a row of N 35-moment cells (column-per-cell layout, (35,N)):

  1. `_face_flux_kernel!`  — one thread per interface i+1/2 (i=1..N-1). For each side
     it runs `realize_and_speed_Mr_dev` (hyperbolicity-corrected state `Mr` + wave
     speeds vmin/vmax), takes the physical flux `Fx = first 35 of flux_closure35_dev(Mr)`,
     and HLL-combines with `sL=min(lminL,lminR)`, `sR=max(lmaxL,lmaxR)` using the
     CORRECTED states for both the fluxes and the `(MRr-MLr)` diffusion term — exactly
     `face_flux_1d` (axis=1). Writes Fhat[:, i] (35, N-1).

  2. `_residual_kernel!` — one thread per cell i. Interior cells (2..N-1) get
     `R[:,i] = -(Fhat[:,i] - Fhat[:,i-1]) / dx`; boundary cells i=1,N stay 0 (outflow BC).

CAVEAT: the CPU `face_flux_1d` first applies `realizable_3D_M4` to each cell's moments.
That projection is NOT ported here (separate realizability projection, next port). The
validation input was already passed through `realizable_3D_M4`, so it is ~identity here.

`@fastmath` stays OFF in the wave-speed path (rsqrt flips the hyperbolicity discriminant).
fp64 throughout. Pure addition under `gpu/`; not wired into production.
"""
module Residual1DGPU

using CUDA

include(joinpath(@__DIR__, "wavespeed_dev.jl"))
include(joinpath(@__DIR__, "flux_closure_dev.jl"))
using .WavespeedDev: realize_and_speed_Mr_dev
using .FluxClosureDev: flux_closure35_dev

export residual1d_gpu!, residual1d_gpu

# ---------------------------------------------------------------------------
# Kernel 1: per-interface HLL face flux. M is (35,N), Fhat is (35,N-1).
# axis is fixed to 1 (matches residual_1d). Ma carried for parity (unused in speeds).
# ---------------------------------------------------------------------------
function _face_flux_kernel!(Fhat, M, Ma::Float64, Nf::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nf
        @inbounds begin
            # left state = cell i, right state = cell i+1
            MLr, lminL, lmaxL = realize_and_speed_Mr_dev(
                M[1,i],  M[2,i],  M[3,i],  M[4,i],  M[5,i],  M[6,i],  M[7,i],
                M[8,i],  M[9,i],  M[10,i], M[11,i], M[12,i], M[13,i], M[14,i],
                M[15,i], M[16,i], M[17,i], M[18,i], M[19,i], M[20,i], M[21,i],
                M[22,i], M[23,i], M[24,i], M[25,i], M[26,i], M[27,i], M[28,i],
                M[29,i], M[30,i], M[31,i], M[32,i], M[33,i], M[34,i], M[35,i],
                1, Ma)
            MRr, lminR, lmaxR = realize_and_speed_Mr_dev(
                M[1,i+1],  M[2,i+1],  M[3,i+1],  M[4,i+1],  M[5,i+1],  M[6,i+1],  M[7,i+1],
                M[8,i+1],  M[9,i+1],  M[10,i+1], M[11,i+1], M[12,i+1], M[13,i+1], M[14,i+1],
                M[15,i+1], M[16,i+1], M[17,i+1], M[18,i+1], M[19,i+1], M[20,i+1], M[21,i+1],
                M[22,i+1], M[23,i+1], M[24,i+1], M[25,i+1], M[26,i+1], M[27,i+1], M[28,i+1],
                M[29,i+1], M[30,i+1], M[31,i+1], M[32,i+1], M[33,i+1], M[34,i+1], M[35,i+1],
                1, Ma)

            # physical flux Fx = first 35 of the 105-tuple (Fx|Fy|Fz).
            # Explicit element passing (no splat: `...` lowers to _apply_iterate,
            # which is unsupported in GPU kernels).
            FL = flux_closure35_dev(
                MLr[1],  MLr[2],  MLr[3],  MLr[4],  MLr[5],  MLr[6],  MLr[7],
                MLr[8],  MLr[9],  MLr[10], MLr[11], MLr[12], MLr[13], MLr[14],
                MLr[15], MLr[16], MLr[17], MLr[18], MLr[19], MLr[20], MLr[21],
                MLr[22], MLr[23], MLr[24], MLr[25], MLr[26], MLr[27], MLr[28],
                MLr[29], MLr[30], MLr[31], MLr[32], MLr[33], MLr[34], MLr[35])
            FR = flux_closure35_dev(
                MRr[1],  MRr[2],  MRr[3],  MRr[4],  MRr[5],  MRr[6],  MRr[7],
                MRr[8],  MRr[9],  MRr[10], MRr[11], MRr[12], MRr[13], MRr[14],
                MRr[15], MRr[16], MRr[17], MRr[18], MRr[19], MRr[20], MRr[21],
                MRr[22], MRr[23], MRr[24], MRr[25], MRr[26], MRr[27], MRr[28],
                MRr[29], MRr[30], MRr[31], MRr[32], MRr[33], MRr[34], MRr[35])

            sL = min(lminL, lminR)
            sR = max(lmaxL, lmaxR)

            if sL >= 0.0
                for j in 1:35
                    Fhat[j, i] = FL[j]
                end
            elseif sR <= 0.0
                for j in 1:35
                    Fhat[j, i] = FR[j]
                end
            else
                den = sR - sL
                ss  = sL * sR
                for j in 1:35
                    Fhat[j, i] = (sR * FL[j] - sL * FR[j] + ss * (MRr[j] - MLr[j])) / den
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Kernel 2: first-order stencil with outflow BC. R is (35,N), Fhat is (35,N-1).
# Interior i=2..N-1: R[:,i] = -(Fhat[:,i] - Fhat[:,i-1]) / dx. Boundaries stay 0.
# ---------------------------------------------------------------------------
function _residual_kernel!(R, Fhat, dx::Float64, N::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 2 <= i <= N - 1
        @inbounds for j in 1:35
            R[j, i] = -(Fhat[j, i] - Fhat[j, i-1]) / dx
        end
    end
    return nothing
end

"""
    residual1d_gpu!(R, Fhat, M, dx, Ma; threads=128)

In-place GPU first-order residual. `M::CuMatrix{Float64}` is (35,N) (column = cell);
`Fhat::CuMatrix{Float64}` is (35,N-1) scratch; `R::CuMatrix{Float64}` is (35,N) and
MUST be pre-zeroed (boundary cells are left untouched). axis fixed to 1.
"""
function residual1d_gpu!(R::CuMatrix{Float64}, Fhat::CuMatrix{Float64},
                         M::CuMatrix{Float64}, dx::Real, Ma::Real=0.0;
                         threads::Int=128)
    N = size(M, 2)
    @assert size(M, 1) == 35 "M must be (35, N)"
    @assert size(Fhat) == (35, N - 1) "Fhat must be (35, N-1)"
    @assert size(R) == (35, N) "R must be (35, N)"
    Nf = N - 1
    @cuda threads=threads blocks=cld(Nf, threads) _face_flux_kernel!(Fhat, M, Float64(Ma), Nf)
    @cuda threads=threads blocks=cld(N, threads)  _residual_kernel!(R, Fhat, Float64(dx), N)
    return nothing
end

"""
    residual1d_gpu(M_host::AbstractMatrix{Float64}, dx, Ma; threads=128) -> Matrix{Float64}

Host convenience: upload (35,N) host matrix, compute the first-order residual, return
the (35,N) host result (column = cell, outflow BC).
"""
function residual1d_gpu(M_host::AbstractMatrix{Float64}, dx::Real, Ma::Real=0.0;
                        threads::Int=128)
    @assert size(M_host, 1) == 35 "M_host must be (35, N)"
    N = size(M_host, 2)
    Md   = CuArray(M_host)
    Fhat = CUDA.zeros(Float64, 35, N - 1)
    R    = CUDA.zeros(Float64, 35, N)
    residual1d_gpu!(R, Fhat, Md, dx, Ma; threads=threads)
    CUDA.synchronize()
    return Array(R)
end

end # module
