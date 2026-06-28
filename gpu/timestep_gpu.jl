"""
    timestep_gpu.jl — full on-device 1D time-advance loop (order-2 SSP-RK3).

The CAPSTONE composition: combines the already-ported & validated GPU kernels —
the order-2 (MUSCL) HLL residual (`gpu/residual2_gpu.jl`, module `Residual2GPU`)
and the realizability projection (`gpu/realize_gpu.jl`, module `RealizeGPU`,
`realizable_3D_M4` per cell) — into a complete time-march that keeps the field
RESIDENT on the GPU. Only the final state is copied to the host.

Faithful to the CPU reference loop (1D, order=2, SSP-RK3, outflow BC):

  for s in 1:NSTEP
      dt = (1/3)*dx / max_i( |u_i| + 4*2.334*sqrt(c2_i + 1e-12) )      # r>0 only
            u_i = M[2,i]/M[1,i],  c2_i = max(M[3,i]/M[1,i] - u_i^2, 0)
      M1 = M        + dt*L(M);              proj!(M1)
      M2 = 0.75*M   + 0.25*(M1 + dt*L(M1)); proj!(M2)
      M3 = (1/3)*M  + (2/3)*(M2 + dt*L(M2));proj!(M3);   M = M3
  end

where `L` = the GPU order-2 residual (`vacuum_floor = HO_VACUUM_FLOOR = 0.001`)
and `proj!` = the GPU realizability projection. The RK3 combines are elementwise
`CuArray` broadcasts; the dt reduction is a per-cell speed kernel + a CUDA
max-reduction. The closure layout is `(35, N)` (column = cell), matching the
other `gpu/` kernels and the on-disk dumps.

`@fastmath` stays OFF (inherited from the wave-speed path). fp64 throughout.
Pure addition under `gpu/`; not wired into production.
"""
module TimestepGPU

using CUDA

include(joinpath(@__DIR__, "residual2_gpu.jl"))
include(joinpath(@__DIR__, "realize_gpu.jl"))
using .Residual2GPU: residual2_gpu!
using .RealizeGPU: realizable_batched!

export march_gpu!, HO_VACUUM_FLOOR_DEFAULT

const HO_VACUUM_FLOOR_DEFAULT = 0.001

# ---------------------------------------------------------------------------
# Per-cell CFL speed kernel. svec[i] = |u_i| + 4*2.334*sqrt(c2_i+1e-12) (r>0),
# else 0.0 (matches the CPU `r>0 || continue` skip). M is (35,N).
# ---------------------------------------------------------------------------
function _speed_kernel!(svec, M, N::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds begin
            r = M[1, i]
            if r > 0.0
                u  = M[2, i] / r
                c2 = M[3, i] / r - u * u
                if c2 < 0.0
                    c2 = 0.0
                end
                svec[i] = abs(u) + 4.0 * 2.334 * sqrt(c2 + 1e-12)
            else
                svec[i] = 0.0
            end
        end
    end
    return nothing
end

"""
    _cfl_dt(M, dx, svec; threads) -> Float64

On-device CFL dt: launch the per-cell speed kernel into `svec`, max-reduce,
`dt = (1/3)*dx / max(maxspeed, 1e-12)`.
"""
function _cfl_dt(M::CuMatrix{Float64}, dx::Float64, svec::CuVector{Float64}; threads::Int=128)
    N = size(M, 2)
    @cuda threads=threads blocks=cld(N, threads) _speed_kernel!(svec, M, N)
    vmax = CUDA.@allowscalar maximum(svec)   # CUDA reduction; scalar result fetch
    return (1.0 / 3.0) * dx / max(vmax, 1e-12)
end

"""
    march_gpu!(M_dev, dx, Ma, nstep; dts=nothing, vacuum_floor=0.001, threads=128)
        -> Vector{Float64}

Advance `M_dev::CuMatrix{Float64}` (35, N) (column = cell) for `nstep` order-2
SSP-RK3 steps, fully on device. `M_dev` is updated IN PLACE with the final state.

If `dts` is supplied (host or device vector of length `nstep`), those dt values
are USED verbatim (the dt reduction is skipped) — for clean validation against a
CPU run with an identical dt sequence. Otherwise dt is computed on device each
step via the CFL rule.

Returns the host `Vector{Float64}` of the `nstep` dt values actually used.
"""
function march_gpu!(M_dev::CuMatrix{Float64}, dx::Real, Ma::Real, nstep::Integer;
                    dts=nothing, vacuum_floor::Real=HO_VACUUM_FLOOR_DEFAULT,
                    threads::Int=128)
    @assert size(M_dev, 1) == 35 "M_dev must be (35, N)"
    N  = size(M_dev, 2)
    Nf = N - 1
    dxf  = Float64(dx)
    Maf  = Float64(Ma)
    vacf = Float64(vacuum_floor)

    # dt source
    dts_host = dts === nothing ? nothing : Float64.(collect(dts))
    if dts_host !== nothing
        @assert length(dts_host) >= nstep "dts must have at least nstep entries"
    end

    # resident scratch (all on device)
    Vc   = CUDA.zeros(Float64, 35, N)
    ML   = CUDA.zeros(Float64, 35, Nf)
    MR   = CUDA.zeros(Float64, 35, Nf)
    Fhat = CUDA.zeros(Float64, 35, Nf)
    R    = CUDA.zeros(Float64, 35, N)   # boundary rows stay 0 (outflow); interior overwritten each call
    M1   = CUDA.zeros(Float64, 35, N)
    M2   = CUDA.zeros(Float64, 35, N)
    M3   = CUDA.zeros(Float64, 35, N)
    Pbuf = CUDA.zeros(Float64, 35, N)   # projection ping-pong scratch
    svec = CUDA.zeros(Float64, N)

    used = Vector{Float64}(undef, nstep)

    M = M_dev   # alias: state lives in M_dev throughout

    for s in 1:nstep
        dt = dts_host === nothing ? _cfl_dt(M, dxf, svec; threads=threads) : dts_host[s]
        used[s] = dt

        # --- Stage 1: M1 = M + dt*L(M); proj!(M1) ---
        residual2_gpu!(R, Fhat, ML, MR, Vc, M, dxf, Maf; vacuum_floor=vacf, project_faces=true, threads=threads)
        @. M1 = M + dt * R
        realizable_batched!(Pbuf, M1, Maf; threads=threads)
        M1, Pbuf = Pbuf, M1

        # --- Stage 2: M2 = 0.75*M + 0.25*(M1 + dt*L(M1)); proj!(M2) ---
        residual2_gpu!(R, Fhat, ML, MR, Vc, M1, dxf, Maf; vacuum_floor=vacf, project_faces=true, threads=threads)
        @. M2 = 0.75 * M + 0.25 * (M1 + dt * R)
        realizable_batched!(Pbuf, M2, Maf; threads=threads)
        M2, Pbuf = Pbuf, M2

        # --- Stage 3: M3 = (1/3)*M + (2/3)*(M2 + dt*L(M2)); proj!(M3) ---
        residual2_gpu!(R, Fhat, ML, MR, Vc, M2, dxf, Maf; vacuum_floor=vacf, project_faces=true, threads=threads)
        @. M3 = (1.0/3.0) * M + (2.0/3.0) * (M2 + dt * R)
        realizable_batched!(Pbuf, M3, Maf; threads=threads)
        M3, Pbuf = Pbuf, M3

        # --- commit: M = M3 (copy into resident state) ---
        @. M = M3
    end

    CUDA.synchronize()
    return used
end

end # module
