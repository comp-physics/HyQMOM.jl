"""
    timestep3d_gpu.jl — full on-device 3D time-advance loop (order-2 SSP-RK3).

The PRODUCTION CAPSTONE: composes the already-ported & validated GPU kernels —
the order-2 (MUSCL) HLL 3D residual (`gpu/residual3d_gpu.jl`, module
`Residual3DGPU`, `residual3d_gpu!`) and the realizability projection
(`gpu/realize_gpu.jl`, module `RealizeGPU`, `realizable_3D_M4` per cell) — into a
complete 3D time-march that keeps the whole field RESIDENT on the GPU. Only the
final state is copied to the host.

This is the SAME SSP-RK3 / projection / dt structure as the validated 1D
`march_gpu!` (`gpu/timestep_gpu.jl`), but: (a) `L` = `residual3d_gpu!` (3D),
(b) a 3D CFL dt reduction (max over the whole 3D field of the per-cell 3D speed),
(c) the RK3 combines are elementwise broadcasts on the 3D `(35,n,n,n)` field.

Faithful to the CPU reference loop (3D, order=2, SSP-RK3, outflow BC,
`HO_VACUUM_FLOOR = 0.001`), as in `dump_step3d.jl`:

  for s in 1:NSTEP
      dt = (1/3)*dx / max_cell( max(|u|,|v|,|w|) + 4*2.334*sqrt(max(cx,cy,cz)+1e-12) )  # r>0 only
            u=M[2]/r, v=M[6]/r, w=M[16]/r
            cx=max(M[3]/r-u^2,0), cy=max(M[10]/r-v^2,0), cz=max(M[20]/r-w^2,0)
      M1 = M        + dt*L3d(M);              proj!(M1)
      M2 = 0.75*M   + 0.25*(M1 + dt*L3d(M1)); proj!(M2)
      M3 = (1/3)*M  + (2/3)*(M2 + dt*L3d(M2));proj!(M3);   M = M3
  end

DEVICE LAYOUT: `M_dev::CuArray{Float64,4}` is `(35, n, n, n)` — 35 moments
contiguous per cell, then i, j, k (matches `residual3d_gpu!` and the on-disk
`step3d_*.f64` dumps). The projection kernel `realizable_batched!` wants a
`(35, B)` matrix; we reshape the `(35,n,n,n)` field to `(35, n^3)` (zero-copy
`reshape` on a contiguous CuArray) since the 35-block of each cell is contiguous.

`@fastmath` stays OFF in the wave-speed path (inherited). fp64 throughout. No
tuple-splat `f(x...)` on the device. Per-face `realizable_3D_M4` projection is ON
in the residual (`project_faces=true`). Pure addition under `gpu/`.
"""
module Timestep3DGPU

using CUDA

include(joinpath(@__DIR__, "residual3d_gpu.jl"))
include(joinpath(@__DIR__, "realize_gpu.jl"))
using .Residual3DGPU: residual3d_gpu!
using .RealizeGPU: realizable_batched!

export march3d_gpu!, HO_VACUUM_FLOOR_DEFAULT

const HO_VACUUM_FLOOR_DEFAULT = 0.001

# ---------------------------------------------------------------------------
# Per-cell 3D CFL speed kernel. svec[idx] = max(|u|,|v|,|w|) +
# 4*2.334*sqrt(max(cx,cy,cz)+1e-12) for cells with r>0, else 0.0 (matches the
# CPU `r>0 || continue` skip). M is (35,n,n,n); svec is flattened length n^3.
# u=M[2]/r, v=M[6]/r, w=M[16]/r; cx=max(M[3]/r-u^2,0), cy=max(M[10]/r-v^2,0),
# cz=max(M[20]/r-w^2,0).
# ---------------------------------------------------------------------------
function _speed3d_kernel!(svec, M, n::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * n
        @inbounds begin
            i = (idx - 1) % n + 1
            r0 = (idx - 1) ÷ n
            j = r0 % n + 1
            k = r0 ÷ n + 1
            r = M[1, i, j, k]
            if r > 0.0
                u = M[2,  i, j, k] / r
                v = M[6,  i, j, k] / r
                w = M[16, i, j, k] / r
                cx = M[3,  i, j, k] / r - u * u
                cy = M[10, i, j, k] / r - v * v
                cz = M[20, i, j, k] / r - w * w
                if cx < 0.0; cx = 0.0; end
                if cy < 0.0; cy = 0.0; end
                if cz < 0.0; cz = 0.0; end
                au = abs(u); av = abs(v); aw = abs(w)
                amax = au > av ? au : av
                amax = amax > aw ? amax : aw
                cmax = cx > cy ? cx : cy
                cmax = cmax > cz ? cmax : cz
                svec[idx] = amax + 4.0 * 2.334 * sqrt(cmax + 1e-12)
            else
                svec[idx] = 0.0
            end
        end
    end
    return nothing
end

"""
    _cfl_dt3d(M, dx, svec, n; threads) -> Float64

On-device 3D CFL dt: launch the per-cell speed kernel into `svec`, max-reduce,
`dt = (1/3)*dx / max(maxspeed, 1e-12)`.
"""
function _cfl_dt3d(M::CuArray{Float64,4}, dx::Float64, svec::CuVector{Float64},
                   n::Int; threads::Int=128)
    ncells = n * n * n
    @cuda threads=threads blocks=cld(ncells, threads) _speed3d_kernel!(svec, M, n)
    vmax = CUDA.@allowscalar maximum(svec)
    return (1.0 / 3.0) * dx / max(vmax, 1e-12)
end

"""
    march3d_gpu!(M_dev, dx, Ma, nstep; dts=nothing, vacuum_floor=0.001, threads=128)
        -> Vector{Float64}

Advance `M_dev::CuArray{Float64,4}` (35, n, n, n) for `nstep` order-2 SSP-RK3
steps, fully on device. `M_dev` is updated IN PLACE with the final state; only it
needs to leave the GPU (copy via `Array(M_dev)`).

If `dts` is supplied (host or device vector of length >= nstep), those dt values
are USED verbatim (the dt reduction is skipped) — for clean validation against a
CPU run with an identical dt sequence. Otherwise dt is computed on device each
step via the 3D CFL rule.

Returns the host `Vector{Float64}` of the `nstep` dt values actually used.
"""
function march3d_gpu!(M_dev::CuArray{Float64,4}, dx::Real, Ma::Real, nstep::Integer;
                      dts=nothing, vacuum_floor::Real=HO_VACUUM_FLOOR_DEFAULT,
                      threads::Int=128)
    @assert size(M_dev, 1) == 35 "M_dev must be (35, n, n, n)"
    n = size(M_dev, 2)
    @assert size(M_dev) == (35, n, n, n) "M_dev must be a cubic (35, n, n, n) field"
    dxf  = Float64(dx)
    Maf  = Float64(Ma)
    vacf = Float64(vacuum_floor)

    dts_host = dts === nothing ? nothing : Float64.(collect(dts))
    if dts_host !== nothing
        @assert length(dts_host) >= nstep "dts must have at least nstep entries"
    end

    ncells = n * n * n

    # resident scratch (all on device)
    R    = CUDA.zeros(Float64, 35, n, n, n)
    Fbuf = CUDA.zeros(Float64, 35, n + 1, n, n)   # residual face-flux buffer
    M1   = CUDA.zeros(Float64, 35, n, n, n)
    M2   = CUDA.zeros(Float64, 35, n, n, n)
    M3   = CUDA.zeros(Float64, 35, n, n, n)
    Pbuf = CUDA.zeros(Float64, 35, n, n, n)       # projection ping-pong scratch
    svec = CUDA.zeros(Float64, ncells)

    # zero-copy (35, n^3) views for the batched per-cell projection kernel
    M1m   = reshape(M1,   35, ncells)
    M2m   = reshape(M2,   35, ncells)
    M3m   = reshape(M3,   35, ncells)
    Pbufm = reshape(Pbuf, 35, ncells)

    used = Vector{Float64}(undef, nstep)
    M = M_dev   # alias: state lives in M_dev throughout

    for s in 1:nstep
        dt = dts_host === nothing ? _cfl_dt3d(M, dxf, svec, n; threads=threads) : dts_host[s]
        used[s] = dt

        # --- Stage 1: M1 = M + dt*L(M); proj!(M1) ---
        residual3d_gpu!(R, Fbuf, M, n, dxf, Maf; vacuum_floor=vacf, project_faces=true, threads=threads)
        @. M1 = M + dt * R
        realizable_batched!(Pbufm, M1m, Maf; threads=threads)
        M1, Pbuf = Pbuf, M1
        M1m, Pbufm = Pbufm, M1m

        # --- Stage 2: M2 = 0.75*M + 0.25*(M1 + dt*L(M1)); proj!(M2) ---
        residual3d_gpu!(R, Fbuf, M1, n, dxf, Maf; vacuum_floor=vacf, project_faces=true, threads=threads)
        @. M2 = 0.75 * M + 0.25 * (M1 + dt * R)
        realizable_batched!(Pbufm, M2m, Maf; threads=threads)
        M2, Pbuf = Pbuf, M2
        M2m, Pbufm = Pbufm, M2m

        # --- Stage 3: M3 = (1/3)*M + (2/3)*(M2 + dt*L(M2)); proj!(M3) ---
        residual3d_gpu!(R, Fbuf, M2, n, dxf, Maf; vacuum_floor=vacf, project_faces=true, threads=threads)
        @. M3 = (1.0/3.0) * M + (2.0/3.0) * (M2 + dt * R)
        realizable_batched!(Pbufm, M3m, Maf; threads=threads)
        M3, Pbuf = Pbuf, M3
        M3m, Pbufm = Pbufm, M3m

        # --- commit: M = M3 (copy into resident state) ---
        @. M = M3
    end

    CUDA.synchronize()
    return used
end

end # module
