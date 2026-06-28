"""
    timestep3d_mpi.jl — multi-GPU (z-slab) on-device order-2 SSP-RK3 time loop.

Each MPI rank owns a z-slab interior `(35, n, n, nz_loc)` RESIDENT on its GPU and
advances it with the SAME SSP-RK3 / projection / CFL structure as the single-GPU
`march3d_gpu!` (`gpu/timestep3d_gpu.jl`). The only inter-GPU traffic is the thin
`halo` z-planes, host-staged for the MPI `Sendrecv!` (system OpenMPI is built
`--without-cuda`, so no CUDA-aware path is needed).

Per stage the residual is `residual3d_box_gpu!` run on the rank's EXTENDED slab
`(35, n, n, nz_loc + 2*halo)` whose ghost z-planes are refreshed by the halo
exchange (neighbor interior planes; outflow replicas at the global z-boundary,
matching the cubic index-clamp). Interior cells never reach the extended z-edges,
so the interior residual is bit-identical to the single-GPU full-domain residual
(validated in `validate_slab_residual_mpi.jl`). RK combines + per-cell projection
act on the contiguous interior buffers. The CFL `dt` is the global `Allreduce(max)`
of the per-rank max wave speed — `max` is exact, so `dt` matches single-GPU
bit-for-bit, hence the whole march matches bit-for-bit.

Pure addition under `gpu/`; not wired into production.
"""
module Timestep3DMPI

using CUDA, MPI

include(joinpath(@__DIR__, "residual3d_gpu.jl"))
include(joinpath(@__DIR__, "realize_gpu.jl"))
using .Residual3DGPU: residual3d_box_gpu!
using .RealizeGPU: realizable_batched!

export march3d_slab_gpu!

# Per-cell 3D CFL speed kernel for a rectangular (nx,ny,nz) field (generalizes
# Timestep3DGPU._speed3d_kernel!). svec flattened length nx*ny*nz.
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

"""
    march3d_slab_gpu!(M, dx, Ma, nstep, comm; halo=2, dts=nothing,
                      vacuum_floor=0.001, threads=128) -> Vector{Float64}

Advance this rank's resident z-slab interior `M::CuArray (35,n,n,nz_loc)` for
`nstep` order-2 SSP-RK3 steps. Global CFL `dt` via `Allreduce(max)` unless `dts`
is supplied (then used verbatim). Returns the host `nstep` dt vector.
"""
function march3d_slab_gpu!(M::CuArray{Float64,4}, dx::Real, Ma::Real, nstep::Integer, comm;
                           halo::Int=2, dts=nothing, vacuum_floor::Real=0.001, threads::Int=128)
    rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)
    @assert size(M, 1) == 35
    n = size(M, 2); nzloc = size(M, 4)
    @assert size(M) == (35, n, n, nzloc) "M must be (35,n,n,nz_loc)"
    nz_ext = nzloc + 2 * halo
    left  = rank > 0          ? rank - 1 : MPI.PROC_NULL
    right = rank < nranks - 1 ? rank + 1 : MPI.PROC_NULL
    dxf = Float64(dx); Maf = Float64(Ma); vacf = Float64(vacuum_floor)
    dts_host = dts === nothing ? nothing : Float64.(collect(dts))
    ncl = n * n * nzloc

    # resident scratch
    Mext = CUDA.zeros(Float64, 35, n, n, nz_ext)
    Rext = CUDA.zeros(Float64, 35, n, n, nz_ext)
    M1 = CUDA.zeros(Float64, 35, n, n, nzloc); M2 = similar(M1); M3 = similar(M1)
    Rint = similar(M1); Pbuf = similar(M1)
    svec = CUDA.zeros(Float64, ncl)
    M1m = reshape(M1, 35, ncl); M2m = reshape(M2, 35, ncl)
    M3m = reshape(M3, 35, ncl); Pbufm = reshape(Pbuf, 35, ncl)

    # pinned host halo buffers
    pin() = (h = Array{Float64}(undef, 35, n, n, halo); CUDA.pin(h); h)
    hsT = pin(); hsB = pin(); hrT = pin(); hrB = pin()
    itop = halo + nzloc - halo + 1   # first plane of top interior halo
    gtop = halo + nzloc + 1          # first plane of top ghost

    # L(state) -> Rint : copy state into extended interior, exchange ghosts, box residual, slice
    function L!(state)
        @inbounds Mext[:, :, :, halo+1:halo+nzloc] .= state
        copyto!(hsB, @view Mext[:, :, :, halo+1:halo+halo])      # my bottom interior planes
        copyto!(hsT, @view Mext[:, :, :, itop:halo+nzloc])       # my top interior planes
        CUDA.synchronize()
        MPI.Sendrecv!(hsT, hrB, comm; dest=right, source=left,  sendtag=1, recvtag=1)  # recv bottom ghost from left
        MPI.Sendrecv!(hsB, hrT, comm; dest=left,  source=right, sendtag=2, recvtag=2)  # recv top ghost from right
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
                            vacuum_floor=vacf, project_faces=true, threads=threads)
        @inbounds Rint .= @view Rext[:, :, :, halo+1:halo+nzloc]
        return nothing
    end

    proj!(Xm) = (realizable_batched!(Pbufm, Xm, Maf; threads=threads);
                 copyto!(Xm, Pbufm))   # write projected result back into X

    used = Vector{Float64}(undef, nstep)
    for s in 1:nstep
        if dts_host === nothing
            @cuda threads=threads blocks=cld(ncl, threads) _speed_box_kernel!(svec, M, n, n, nzloc)
            lmax = CUDA.@allowscalar maximum(svec)
            gmax = MPI.Allreduce(lmax, max, comm)
            dt = (1.0/3.0) * dxf / max(gmax, 1e-12)
        else
            dt = dts_host[s]
        end
        used[s] = dt

        L!(M);  @. M1 = M + dt * Rint;                         proj!(M1m)
        L!(M1); @. M2 = 0.75*M + 0.25*(M1 + dt * Rint);        proj!(M2m)
        L!(M2); @. M3 = (1.0/3.0)*M + (2.0/3.0)*(M2 + dt*Rint); proj!(M3m)
        @. M = M3
    end
    CUDA.synchronize()
    return used
end

end # module
