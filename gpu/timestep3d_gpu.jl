"""
    timestep3d_gpu.jl — on-device 3D order-2 SSP-RK3 time loop (single- and multi-GPU).

One module, one RK3 body. Both entry points share `_rk3_step!` (the SSP-RK3 stage
sequence + per-cell realizability projection) and `_speed_box_kernel!` (the per-cell
CFL speed). They differ ONLY in their residual operator `L!` and their `dt`:

  * `march3d_gpu!(M, dx, Ma, nstep; ...)` — single GPU, cubic field `(35,n,n,n)`.
    `L!` = `residual3d_gpu!` (which is itself `residual3d_box_gpu!` for a cube);
    `dt` = local CFL.

  * `march3d_slab_gpu!(M, dx, Ma, nstep, comm; halo=2, ...)` — multi-GPU z-slab.
    Each rank keeps its interior `(35,n,n,nz_loc)` resident; per stage `L!` refreshes
    the `halo` z-planes via host-staged `MPI.Sendrecv!` (system OpenMPI is built
    `--without-cuda`, so no CUDA-aware path), runs `residual3d_box_gpu!` on the
    EXTENDED slab, and slices the interior. `dt` = `Allreduce(max)` of per-rank max
    speed (`max` is exact → `dt`, and the whole march, are bit-identical to single-GPU;
    validated in `validate_timestep3d_mpi.jl` for 1/2/4 ranks).

Loop (SSP-RK3, order 2, projection ON each stage):
    dt = (1/3)*dx / max_cell( max(|u|,|v|,|w|) + 4*2.334*sqrt(max(cx,cy,cz)+1e-12) )
    M1 = M        + dt*L(M);              proj!(M1)
    M2 = 0.75*M   + 0.25*(M1 + dt*L(M1)); proj!(M2)
    M3 = (1/3)*M  + (2/3)*(M2 + dt*L(M2));proj!(M3);   M = M3

Layout `(35, n, n, nz)` (35 contiguous per cell). Pure addition under `gpu/`.
"""
module Timestep3DGPU

using CUDA, MPI

include(joinpath(@__DIR__, "residual3d_gpu.jl"))
include(joinpath(@__DIR__, "realize_gpu.jl"))
using .Residual3DGPU: residual3d_gpu!, residual3d_box_gpu!
using .RealizeGPU: realizable_batched!

export march3d_gpu!, march3d_slab_gpu!, HO_VACUUM_FLOOR_DEFAULT

const HO_VACUUM_FLOOR_DEFAULT = 0.001

# ---------------------------------------------------------------------------
# Per-cell 3D CFL speed for a rectangular (nx,ny,nz) field. svec flattened length
# nx*ny*nz. speed = max(|u|,|v|,|w|) + 4*2.334*sqrt(max(cx,cy,cz)+1e-12) for cells
# with r>0, else 0 (matches the CPU `r>0||continue`). The ONE speed kernel — the
# cubic case is just nx==ny==nz.
# ---------------------------------------------------------------------------
function _speed_box_kernel!(svec, M, nx::Int, ny::Int, nz::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r0 = (idx - 1) ÷ nx
            j = r0 % ny + 1;        k = r0 ÷ ny + 1
            r = M[1, i, j, k]
            if r > 0.0
                u = M[2,i,j,k]/r; v = M[6,i,j,k]/r; w = M[16,i,j,k]/r
                cx = M[3,i,j,k]/r - u*u;  cy = M[10,i,j,k]/r - v*v;  cz = M[20,i,j,k]/r - w*w
                if cx < 0.0; cx = 0.0; end
                if cy < 0.0; cy = 0.0; end
                if cz < 0.0; cz = 0.0; end
                au = abs(u); av = abs(v); aw = abs(w)
                amax = au > av ? au : av;  amax = amax > aw ? amax : aw
                cmax = cx > cy ? cx : cy;  cmax = cmax > cz ? cmax : cz
                svec[idx] = amax + 4.0 * 2.334 * sqrt(cmax + 1e-12)
            else
                svec[idx] = 0.0
            end
        end
    end
    return nothing
end

# local max wave speed over a rectangular field
function _local_vmax(M::CuArray{Float64,4}, svec::CuVector{Float64},
                     nx::Int, ny::Int, nz::Int; threads::Int=128)
    @cuda threads=threads blocks=cld(nx*ny*nz, threads) _speed_box_kernel!(svec, M, nx, ny, nz)
    return CUDA.@allowscalar maximum(svec)
end

_cfl_from_vmax(vmax, dx) = (1.0/3.0) * dx / max(vmax, 1e-12)

# ---------------------------------------------------------------------------
# The ONE SSP-RK3 step. `L!(Rint, state)` writes the residual of `state` into
# `Rint`; `proj!` is the per-cell realizability projection (out-of-place batched
# kernel + writeback). All buffers are `(35,n,n,nz)`; the `*m` are their
# `reshape(_, 35, n*n*nz)` views for the batched projection.
# ---------------------------------------------------------------------------
@inline function _rk3_step!(M, M1, M2, M3, M1m, M2m, M3m, Pbufm, Rint, dt, Maf, threads, L!)
    proj!(Xm) = (realizable_batched!(Pbufm, Xm, Maf; threads=threads); copyto!(Xm, Pbufm))
    L!(Rint, M);  @. M1 = M + dt * Rint;                              proj!(M1m)
    L!(Rint, M1); @. M2 = 0.75*M + 0.25*(M1 + dt*Rint);               proj!(M2m)
    L!(Rint, M2); @. M3 = (1.0/3.0)*M + (2.0/3.0)*(M2 + dt*Rint);     proj!(M3m)
    @. M = M3
    return nothing
end

# allocate the shared RK3 scratch for a (35,nx,ny,nz) state
function _rk3_buffers(nx::Int, ny::Int, nz::Int)
    ncl = nx * ny * nz
    M1 = CUDA.zeros(Float64, 35, nx, ny, nz); M2 = similar(M1); M3 = similar(M1)
    Pbuf = similar(M1); svec = CUDA.zeros(Float64, ncl)
    return (M1, M2, M3, Pbuf,
            reshape(M1,35,ncl), reshape(M2,35,ncl), reshape(M3,35,ncl), reshape(Pbuf,35,ncl),
            svec)
end

"""
    march3d_gpu!(M_dev, dx, Ma, nstep; dts=nothing, vacuum_floor=…, threads=128) -> Vector{Float64}

Advance a single-GPU field `M_dev (35,nx,ny,nz)` for `nstep` SSP-RK3 steps. Any
rectangular extent — cubic `(35,n,n,n)` OR a 2D spatial grid `(35,nx,ny,1)` (the
35-moment velocity space is always 3D; a 2D run is just `nz=1`, giving `Lz=0` on
z-uniform data). Outflow BC on all 6 faces (the single-GPU full domain has no
neighbors). If `dts` is given those dt are used verbatim; else local CFL each step.
Returns the dt vector used.
"""
function march3d_gpu!(M_dev::CuArray{Float64,4}, dx::Real, Ma::Real, nstep::Integer;
                      dts=nothing, vacuum_floor::Real=HO_VACUUM_FLOOR_DEFAULT, threads::Int=128)
    @assert size(M_dev, 1) == 35 "M_dev must be (35,nx,ny,nz)"
    nx = size(M_dev, 2); ny = size(M_dev, 3); nz = size(M_dev, 4)
    dxf = Float64(dx); Maf = Float64(Ma); vacf = Float64(vacuum_floor)
    dts_host = dts === nothing ? nothing : Float64.(collect(dts))

    R = CUDA.zeros(Float64, 35, nx, ny, nz)
    fmax = max((nx+1)*ny*nz, (ny+1)*nx*nz, (nz+1)*nx*ny)
    flat = CUDA.zeros(Float64, 35, fmax)            # box face-scratch (alloc-free reuse)
    M1, M2, M3, Pbuf, M1m, M2m, M3m, Pbufm, svec = _rk3_buffers(nx, ny, nz)
    M = M_dev
    L! = (Rint, st) -> residual3d_box_gpu!(Rint, st, nx, ny, nz, dxf, Maf;
                                           vacuum_floor=vacf, project_faces=true,
                                           threads=threads, flat=flat)

    used = Vector{Float64}(undef, nstep)
    for s in 1:nstep
        dt = dts_host === nothing ? _cfl_from_vmax(_local_vmax(M, svec, nx, ny, nz; threads=threads), dxf) : dts_host[s]
        used[s] = dt
        _rk3_step!(M, M1, M2, M3, M1m, M2m, M3m, Pbufm, R, dt, Maf, threads, L!)
    end
    CUDA.synchronize()
    return used
end

"""
    march3d_slab_gpu!(M, dx, Ma, nstep, comm; halo=2, dts=nothing, vacuum_floor=…, threads=128)

Advance this rank's resident z-slab interior `M (35,n,n,nz_loc)` for `nstep` SSP-RK3
steps. Halo planes are host-staged for `MPI.Sendrecv!`; global CFL via `Allreduce(max)`
unless `dts` is supplied. Returns the dt vector used.
"""
function march3d_slab_gpu!(M::CuArray{Float64,4}, dx::Real, Ma::Real, nstep::Integer, comm;
                           halo::Int=2, dts=nothing, vacuum_floor::Real=HO_VACUUM_FLOOR_DEFAULT,
                           threads::Int=128)
    rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)
    @assert size(M, 1) == 35
    n = size(M, 2); nzloc = size(M, 4)
    @assert size(M) == (35, n, n, nzloc) "M must be (35,n,n,nz_loc)"
    @assert nzloc >= halo "z-slab decomposition is 3D-only (need nz_loc >= halo); for 2D (nz=1) use single-GPU march3d_gpu!"
    nz_ext = nzloc + 2*halo
    left  = rank > 0          ? rank - 1 : MPI.PROC_NULL
    right = rank < nranks - 1 ? rank + 1 : MPI.PROC_NULL
    dxf = Float64(dx); Maf = Float64(Ma); vacf = Float64(vacuum_floor)
    dts_host = dts === nothing ? nothing : Float64.(collect(dts))

    Mext = CUDA.zeros(Float64, 35, n, n, nz_ext)
    Rext = CUDA.zeros(Float64, 35, n, n, nz_ext)
    fmax = max((n+1)*n*nz_ext, (nz_ext+1)*n*n)
    flat = CUDA.zeros(Float64, 35, fmax)                       # box face-scratch (alloc-free reuse)
    M1, M2, M3, Pbuf, M1m, M2m, M3m, Pbufm, svec = _rk3_buffers(n, n, nzloc)
    Rint = CUDA.zeros(Float64, 35, n, n, nzloc)
    pin() = (h = Array{Float64}(undef, 35, n, n, halo); CUDA.pin(h); h)
    hsT = pin(); hsB = pin(); hrT = pin(); hrB = pin()
    itop = nzloc + 1; gtop = halo + nzloc + 1                   # top interior / top ghost first ext plane

    L! = function (Rout, state)
        @inbounds Mext[:, :, :, halo+1:halo+nzloc] .= state
        copyto!(hsB, @view Mext[:, :, :, halo+1:halo+halo])     # my bottom interior halo planes
        copyto!(hsT, @view Mext[:, :, :, itop:halo+nzloc])      # my top interior halo planes
        CUDA.synchronize()
        MPI.Sendrecv!(hsT, hrB, comm; dest=right, source=left,  sendtag=1, recvtag=1)
        MPI.Sendrecv!(hsB, hrT, comm; dest=left,  source=right, sendtag=2, recvtag=2)
        if left == MPI.PROC_NULL
            @inbounds for g in 1:halo; Mext[:, :, :, g] .= @view Mext[:, :, :, halo+1]; end
        else
            copyto!(@view(Mext[:, :, :, 1:halo]), reshape(hrB, 35, n, n, halo))
        end
        if right == MPI.PROC_NULL
            @inbounds for g in 1:halo; Mext[:, :, :, gtop+g-1] .= @view Mext[:, :, :, halo+nzloc]; end
        else
            copyto!(@view(Mext[:, :, :, gtop:nz_ext]), reshape(hrT, 35, n, n, halo))
        end
        residual3d_box_gpu!(Rext, Mext, n, n, nz_ext, dxf, Maf;
                            vacuum_floor=vacf, project_faces=true, threads=threads, flat=flat)
        @inbounds Rout .= @view Rext[:, :, :, halo+1:halo+nzloc]
        return nothing
    end

    used = Vector{Float64}(undef, nstep)
    for s in 1:nstep
        if dts_host === nothing
            gmax = MPI.Allreduce(_local_vmax(M, svec, n, n, nzloc; threads=threads), max, comm)
            dt = _cfl_from_vmax(gmax, dxf)
        else
            dt = dts_host[s]
        end
        used[s] = dt
        _rk3_step!(M, M1, M2, M3, M1m, M2m, M3m, Pbufm, Rint, dt, Maf, threads, L!)
    end
    CUDA.synchronize()
    return used
end

end # module
