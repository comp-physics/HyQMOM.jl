"""
    residual2_gpu.jl — end-to-end SECOND-ORDER (MUSCL) 1D HLL residual on GPU.

Composes the ported device kernels (`src/numerics/recon_dev.jl`, `src/numerics/flux_closure_dev.jl`,
`gpu/wavespeed_dev.jl`, `gpu/schur4.jl`) into a complete on-device order-2 spatial
residual matching the CPU
`residual_1d(Mline, dx, Ma; order=2, bc=:outflow, use_limiter=false)`
(`src/numerics/highorder_flux.jl` + `src/numerics/reconstruction.jl`).

Four kernel launches over a row of N 35-moment cells (column-per-cell, (35,N)):

  1. `_recon_vars_kernel!` — one thread per cell i. Vc[:,i] = to_recon_vars(M[:,i]).

  2. `_facepair_kernel!` — one thread per interface i+1/2 (i=1..N-1). Builds the MUSCL
     faces (minmod slope, zero-gradient BC via clamped neighbor indices):
       Vplus_i      = muscl_plus (cell i,   neighbors max(i-1,1), min(i+1,N))
       Vminus_{i+1} = muscl_minus(cell i+1, neighbors i,          min(i+2,N))
     then the DEFAULT recon_face_pair gate (HO_VACUUM_FLOOR + recon_vars_ok + finite
     reconstruction) to produce ML[:,i], MR[:,i]. Boundary cells get zero slope
     (minmod with the duplicated neighbor), exactly as the CPU clamp.

  3. `_face_flux_kernel!` — one thread per interface. realize_and_speed (hyperbolicity
     correction + wave speeds) on ML/MR, physical flux, HLL combine -> Fhat[:,i] — the
     SAME face flux as the first-order residual (residual1d_gpu), just on the
     reconstructed states.

  4. `_residual_kernel!` — one thread per cell i. Interior (2..N-1):
     R[:,i] = -(Fhat[:,i]-Fhat[:,i-1])/dx; boundary cells stay 0 (outflow BC).

This is the use_limiter=false path: the cheap recon_face_pair gate, NOT the per-face
is_realizable-eig scaling limiter.

CAVEAT (same as first-order port): CPU `face_flux_1d` first applies `realizable_3D_M4`
to each face state; that projection is NOT ported here. The reconstructed faces are
~realizable for pre-realized input, so it is ~identity (validation gate ~1e-8).

`@fastmath` stays OFF in the wave-speed path. fp64 throughout. Pure addition under
`gpu/`; not wired into production.
"""
module Residual2GPU

using CUDA

include(joinpath(@__DIR__, "wavespeed_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "flux_closure_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "recon_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "realizability", "realize_dev.jl"))
using .WavespeedDev: realize_and_speed_Mr_dev
using .FluxClosureDev: flux_closure35_dev
using .ReconDev: to_recon_vars_tup, from_recon_vars_tup, recon_vars_ok_tup, minmod
using .RealizeDev: realizable_3D_M4_dev

export residual2_gpu!, residual2_gpu

# ---------------------------------------------------------------------------
# Kernel 1: per-cell recon variables. M is (35,N), Vc is (35,N).
# ---------------------------------------------------------------------------
function _recon_vars_kernel!(Vc, M, N::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds begin
            Mt = (M[1,i],M[2,i],M[3,i],M[4,i],M[5,i],M[6,i],M[7,i],M[8,i],M[9,i],M[10,i],
                  M[11,i],M[12,i],M[13,i],M[14,i],M[15,i],M[16,i],M[17,i],M[18,i],M[19,i],M[20,i],
                  M[21,i],M[22,i],M[23,i],M[24,i],M[25,i],M[26,i],M[27,i],M[28,i],M[29,i],M[30,i],
                  M[31,i],M[32,i],M[33,i],M[34,i],M[35,i])
            V = to_recon_vars_tup(Mt)
            for k in 1:35
                Vc[k,i] = V[k]
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Kernel 2: per-interface MUSCL faces + recon_face_pair gate.
# Vc is (35,N), M is (35,N); ML,MR are (35,N-1). vacf = HO_VACUUM_FLOOR.
# ---------------------------------------------------------------------------
function _facepair_kernel!(ML, MR, Vc, M, vacf::Float64, Nf::Int, N::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nf
        @inbounds begin
            im1 = max(i-1, 1)      # left neighbor of cell i
            ipa = min(i+1, N)      # right neighbor of cell i
            ip2 = min(i+2, N)      # right neighbor of cell i+1

            # MUSCL right face of cell i:  Vplus_i = V0 + 0.5*minmod(V0-Vm, Vp-V0)
            Vp = ntuple(Val(35)) do k
                v0 = Vc[k, i]
                s  = minmod(v0 - Vc[k, im1], Vc[k, ipa] - v0)
                v0 + 0.5 * s
            end
            # MUSCL left face of cell i+1: Vminus_{i+1} = V0 - 0.5*minmod(V0-Vm, Vp-V0)
            Vm = ntuple(Val(35)) do k
                v0 = Vc[k, i+1]
                s  = minmod(v0 - Vc[k, i], Vc[k, ip2] - v0)
                v0 - 0.5 * s
            end

            ML0_1 = M[1, i]
            MR0_1 = M[1, i+1]

            use_recon = false
            local Li::NTuple{35,Float64}
            local Ri::NTuple{35,Float64}
            # near-vacuum gate: below the floor, fall back to first-order cell states.
            if !(vacf > 0.0 && (ML0_1 < vacf || MR0_1 < vacf))
                if recon_vars_ok_tup(Vp) && recon_vars_ok_tup(Vm)
                    Li = from_recon_vars_tup(Vp)
                    Ri = from_recon_vars_tup(Vm)
                    finL = true; finR = true
                    for k in 1:35
                        finL &= isfinite(Li[k])
                        finR &= isfinite(Ri[k])
                    end
                    if Li[1] > 0.0 && Ri[1] > 0.0 && finL && finR
                        use_recon = true
                    end
                end
            end

            if use_recon
                for k in 1:35
                    ML[k, i] = Li[k]
                    MR[k, i] = Ri[k]
                end
            else
                for k in 1:35
                    ML[k, i] = M[k, i]
                    MR[k, i] = M[k, i+1]
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Kernel 3: per-interface HLL face flux on the reconstructed states ML/MR.
# Identical formula to residual1d_gpu's _face_flux_kernel!, reading ML/MR (35,N-1).
# ---------------------------------------------------------------------------
function _face_flux_kernel!(Fhat, ML, MR, Ma::Float64, Nf::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nf
        @inbounds begin
            MLr, lminL, lmaxL = realize_and_speed_Mr_dev(
                ML[1,i],  ML[2,i],  ML[3,i],  ML[4,i],  ML[5,i],  ML[6,i],  ML[7,i],
                ML[8,i],  ML[9,i],  ML[10,i], ML[11,i], ML[12,i], ML[13,i], ML[14,i],
                ML[15,i], ML[16,i], ML[17,i], ML[18,i], ML[19,i], ML[20,i], ML[21,i],
                ML[22,i], ML[23,i], ML[24,i], ML[25,i], ML[26,i], ML[27,i], ML[28,i],
                ML[29,i], ML[30,i], ML[31,i], ML[32,i], ML[33,i], ML[34,i], ML[35,i],
                1, Ma)
            MRr, lminR, lmaxR = realize_and_speed_Mr_dev(
                MR[1,i],  MR[2,i],  MR[3,i],  MR[4,i],  MR[5,i],  MR[6,i],  MR[7,i],
                MR[8,i],  MR[9,i],  MR[10,i], MR[11,i], MR[12,i], MR[13,i], MR[14,i],
                MR[15,i], MR[16,i], MR[17,i], MR[18,i], MR[19,i], MR[20,i], MR[21,i],
                MR[22,i], MR[23,i], MR[24,i], MR[25,i], MR[26,i], MR[27,i], MR[28,i],
                MR[29,i], MR[30,i], MR[31,i], MR[32,i], MR[33,i], MR[34,i], MR[35,i],
                1, Ma)

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
# Kernel 4: first-order stencil over the high-order faces, outflow BC.
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

# ---------------------------------------------------------------------------
# Optional face-state projection (opt-in): apply realizable_3D_M4 to each face
# state ML[:,i], MR[:,i] in place, EXACTLY as CPU `face_flux_1d` does before the
# flux. Default-off keeps the historical residual2 behavior byte-identical; the
# time-march turns it ON for faithfulness over many steps at high Ma (where the
# reconstructed faces can leave the realizable set).
# ---------------------------------------------------------------------------
@inline function _project_col!(A, i, Ma::Float64)
    @inbounds begin
        r = realizable_3D_M4_dev(
            A[1,i],  A[2,i],  A[3,i],  A[4,i],  A[5,i],  A[6,i],  A[7,i],
            A[8,i],  A[9,i],  A[10,i], A[11,i], A[12,i], A[13,i], A[14,i],
            A[15,i], A[16,i], A[17,i], A[18,i], A[19,i], A[20,i], A[21,i],
            A[22,i], A[23,i], A[24,i], A[25,i], A[26,i], A[27,i], A[28,i],
            A[29,i], A[30,i], A[31,i], A[32,i], A[33,i], A[34,i], A[35,i], Ma)
        for k in 1:35
            A[k,i] = r[k]
        end
    end
    return nothing
end

function _project_faces_kernel!(ML, MR, Ma::Float64, Nf::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nf
        _project_col!(ML, i, Ma)
        _project_col!(MR, i, Ma)
    end
    return nothing
end

"""
    residual2_gpu!(R, Fhat, ML, MR, Vc, M, dx, Ma; vacuum_floor=0.0, threads=128)

In-place GPU order-2 (MUSCL) residual. `M::CuMatrix{Float64}` (35,N) (column=cell);
`Vc` (35,N), `ML`,`MR` (35,N-1), `Fhat` (35,N-1) scratch; `R` (35,N) MUST be pre-zeroed
(boundary cells are left untouched). `vacuum_floor` is HO_VACUUM_FLOOR. axis fixed to 1.
"""
function residual2_gpu!(R::CuMatrix{Float64}, Fhat::CuMatrix{Float64},
                        ML::CuMatrix{Float64}, MR::CuMatrix{Float64},
                        Vc::CuMatrix{Float64}, M::CuMatrix{Float64},
                        dx::Real, Ma::Real=0.0;
                        vacuum_floor::Real=0.0, project_faces::Bool=false,
                        threads::Int=128)
    N = size(M, 2)
    @assert size(M, 1) == 35 "M must be (35, N)"
    @assert size(Vc) == (35, N) "Vc must be (35, N)"
    @assert size(ML) == (35, N - 1) "ML must be (35, N-1)"
    @assert size(MR) == (35, N - 1) "MR must be (35, N-1)"
    @assert size(Fhat) == (35, N - 1) "Fhat must be (35, N-1)"
    @assert size(R) == (35, N) "R must be (35, N)"
    Nf = N - 1
    @cuda threads=threads blocks=cld(N, threads)  _recon_vars_kernel!(Vc, M, N)
    @cuda threads=threads blocks=cld(Nf, threads) _facepair_kernel!(ML, MR, Vc, M, Float64(vacuum_floor), Nf, N)
    if project_faces
        @cuda threads=threads blocks=cld(Nf, threads) _project_faces_kernel!(ML, MR, Float64(Ma), Nf)
    end
    @cuda threads=threads blocks=cld(Nf, threads) _face_flux_kernel!(Fhat, ML, MR, Float64(Ma), Nf)
    @cuda threads=threads blocks=cld(N, threads)  _residual_kernel!(R, Fhat, Float64(dx), N)
    return nothing
end

"""
    residual2_gpu(M_host, dx, Ma; vacuum_floor=0.0, threads=128) -> Matrix{Float64}

Host convenience: upload (35,N) host matrix, compute the order-2 residual, return the
(35,N) host result (column=cell, outflow BC).
"""
function residual2_gpu(M_host::AbstractMatrix{Float64}, dx::Real, Ma::Real=0.0;
                       vacuum_floor::Real=0.0, threads::Int=128)
    @assert size(M_host, 1) == 35 "M_host must be (35, N)"
    N = size(M_host, 2)
    Md   = CuArray(M_host)
    Vc   = CUDA.zeros(Float64, 35, N)
    ML   = CUDA.zeros(Float64, 35, N - 1)
    MR   = CUDA.zeros(Float64, 35, N - 1)
    Fhat = CUDA.zeros(Float64, 35, N - 1)
    R    = CUDA.zeros(Float64, 35, N)
    residual2_gpu!(R, Fhat, ML, MR, Vc, Md, dx, Ma; vacuum_floor=vacuum_floor, threads=threads)
    CUDA.synchronize()
    return Array(R)
end

end # module
