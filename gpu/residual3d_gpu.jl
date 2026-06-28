"""
    residual3d_gpu.jl — on-device 3D order-2 (MUSCL) HLL residual.

Composes the validated per-cell / per-line device kernels
(`src/numerics/recon_dev.jl`, `src/numerics/flux_closure_dev.jl`, `gpu/wavespeed_dev.jl`,
`src/realizability/realize_dev.jl`, `gpu/schur4.jl`) into the 3D unsplit residual, the GPU
analogue of the CPU `residual_ho_3d!` (`src/numerics/highorder_3d.jl`) at
`order=2`. The residual is the SUM of the per-line order-2 1D HLL residual along
the three axes:

    R[:,i,j,k] = Lx(M)[:,i,j,k] + Ly(M)[:,i,j,k] + Lz(M)[:,i,j,k]

where each `L_axis` reconstructs MUSCL faces along that axis, applies the
per-face `realizable_3D_M4` projection (`project_faces=true`, as the CPU
`face_flux_1d` always does), forms the HLL flux with the axis-appropriate
physical flux (`flux_closure35_dev` returns Fx|Fy|Fz; axis a uses block a) and
wave speeds (`realize_and_speed_Mr_dev(..., axis, Ma)`), then differences.

DEVICE LAYOUT (documented):
  `M_dev` and `R_dev` are `(35, nx, ny, nz)` CuArrays — 35 moments CONTIGUOUS
  per cell (fastest), then i, j, k. This matches the on-disk `r3d_M.f64`
  flatten (`permutedims(interior,(4,1,2,3))` then `vec`) and the column-per-cell
  `(35,N)` convention of the 1D kernels, so a line along ANY axis reads the
  35-block of each cell as a unit; threads within a warp stream consecutive
  cells along the line.

BOUNDARY (outflow): the interior is the real n^3 field with NO stored halo. Each
axis treats the line of n real cells as bounded by `halo` outflow ghosts (= copy
of the edge cell). This is realized by CLAMPing the cell index to [1,n] when
gathering the 4 MUSCL stencil cells of each face — EXACTLY reproducing the CPU
`residual_ho_3d!` which pads x/y with outflow halos and pads the z column with
`repeat(edge)`. (n+1 faces per line: face f between cells f and f+1, f=0..n,
ghost cells 0 and n+1 clamped to 1 and n.) All n interior cells get a real
residual — nothing is zeroed (the CPU 3D path computes every interior cell from
its ghost-backed fluxes; a standalone 1D "zero the boundary" rule does not apply
to the 3D composition).

Per axis: kernel 1 computes the face flux `Fhat` at every face into a reusable
`(35, n+1, p1, p2)` buffer (each face computed ONCE); kernel 2 differences and
ACCUMULATES `-(Fhat[f] - Fhat[f-1])/ds` into `R`. `R` is zeroed once up front,
then the three axes add into it.

`@fastmath` stays OFF in the wave-speed path (inherited). fp64 throughout. No
tuple-splat `f(x...)` anywhere on the device. Pure addition under `gpu/`; not
wired into production.
"""
module Residual3DGPU

using CUDA

include(joinpath(@__DIR__, "wavespeed_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "flux_closure_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "recon_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "realizability", "realize_dev.jl"))
using .WavespeedDev: realize_and_speed_Mr_dev
using .FluxClosureDev: flux_closure35_dev
using .ReconDev: to_recon_vars_tup, from_recon_vars_tup, recon_vars_ok_tup, minmod
using .RealizeDev: realizable_3D_M4_dev

export residual3d_gpu!, residual3d_gpu

# ---------------------------------------------------------------------------
# Per-face core: given the four MUSCL-stencil cells (35-tuples) C_{f-1}, C_f,
# C_{f+1}, C_{f+2} along an axis, return the order-2 HLL face flux Fhat (35-tuple)
# for the face between C_f and C_{f+1}. Faithful to CPU `face_flux_1d` composed
# with the MUSCL `recon_face_pair` gate (default, vacuum-floored). project=true
# applies `realizable_3D_M4` to both face states first (CPU always does).
# ---------------------------------------------------------------------------
@inline function _face_flux_core(Cfm1::NTuple{35,Float64}, Cf::NTuple{35,Float64},
                                 Cfp1::NTuple{35,Float64}, Cfp2::NTuple{35,Float64},
                                 axis::Int, Ma::Float64, vacf::Float64, project::Bool)
    Vfm1 = to_recon_vars_tup(Cfm1)
    Vf   = to_recon_vars_tup(Cf)
    Vfp1 = to_recon_vars_tup(Cfp1)
    Vfp2 = to_recon_vars_tup(Cfp2)

    # MUSCL right face of cell f:  Vplus = V0 + 0.5*minmod(V0-Vm, Vp-V0)
    Vp = ntuple(Val(35)) do k
        v0 = Vf[k]
        s  = minmod(v0 - Vfm1[k], Vfp1[k] - v0)
        v0 + 0.5 * s
    end
    # MUSCL left face of cell f+1: Vminus = V0 - 0.5*minmod(V0-Vm, Vp-V0)
    Vm = ntuple(Val(35)) do k
        v0 = Vfp1[k]
        s  = minmod(v0 - Vf[k], Vfp2[k] - v0)
        v0 - 0.5 * s
    end

    ML0 = Cf[1]
    MR0 = Cfp1[1]

    use_recon = false
    Li = Cf       # first-order fallback = cell means
    Ri = Cfp1
    if !(vacf > 0.0 && (ML0 < vacf || MR0 < vacf))
        if recon_vars_ok_tup(Vp) && recon_vars_ok_tup(Vm)
            Lc = from_recon_vars_tup(Vp)
            Rc = from_recon_vars_tup(Vm)
            finL = true; finR = true
            for k in 1:35
                finL &= isfinite(Lc[k])
                finR &= isfinite(Rc[k])
            end
            if Lc[1] > 0.0 && Rc[1] > 0.0 && finL && finR
                use_recon = true
                Li = Lc
                Ri = Rc
            end
        end
    end

    MLf = use_recon ? Li : Cf
    MRf = use_recon ? Ri : Cfp1

    # per-face realizability projection (CPU face_flux_1d always projects)
    if project
        MLf = realizable_3D_M4_dev(
            MLf[1],  MLf[2],  MLf[3],  MLf[4],  MLf[5],  MLf[6],  MLf[7],
            MLf[8],  MLf[9],  MLf[10], MLf[11], MLf[12], MLf[13], MLf[14],
            MLf[15], MLf[16], MLf[17], MLf[18], MLf[19], MLf[20], MLf[21],
            MLf[22], MLf[23], MLf[24], MLf[25], MLf[26], MLf[27], MLf[28],
            MLf[29], MLf[30], MLf[31], MLf[32], MLf[33], MLf[34], MLf[35], Ma)
        MRf = realizable_3D_M4_dev(
            MRf[1],  MRf[2],  MRf[3],  MRf[4],  MRf[5],  MRf[6],  MRf[7],
            MRf[8],  MRf[9],  MRf[10], MRf[11], MRf[12], MRf[13], MRf[14],
            MRf[15], MRf[16], MRf[17], MRf[18], MRf[19], MRf[20], MRf[21],
            MRf[22], MRf[23], MRf[24], MRf[25], MRf[26], MRf[27], MRf[28],
            MRf[29], MRf[30], MRf[31], MRf[32], MRf[33], MRf[34], MRf[35], Ma)
    end

    # hyperbolicity correction + wave speeds (axis-aware)
    MLr, lminL, lmaxL = realize_and_speed_Mr_dev(
        MLf[1],  MLf[2],  MLf[3],  MLf[4],  MLf[5],  MLf[6],  MLf[7],
        MLf[8],  MLf[9],  MLf[10], MLf[11], MLf[12], MLf[13], MLf[14],
        MLf[15], MLf[16], MLf[17], MLf[18], MLf[19], MLf[20], MLf[21],
        MLf[22], MLf[23], MLf[24], MLf[25], MLf[26], MLf[27], MLf[28],
        MLf[29], MLf[30], MLf[31], MLf[32], MLf[33], MLf[34], MLf[35], axis, Ma)
    MRr, lminR, lmaxR = realize_and_speed_Mr_dev(
        MRf[1],  MRf[2],  MRf[3],  MRf[4],  MRf[5],  MRf[6],  MRf[7],
        MRf[8],  MRf[9],  MRf[10], MRf[11], MRf[12], MRf[13], MRf[14],
        MRf[15], MRf[16], MRf[17], MRf[18], MRf[19], MRf[20], MRf[21],
        MRf[22], MRf[23], MRf[24], MRf[25], MRf[26], MRf[27], MRf[28],
        MRf[29], MRf[30], MRf[31], MRf[32], MRf[33], MRf[34], MRf[35], axis, Ma)

    # physical flux: flux_closure35_dev returns (Fx|Fy|Fz) flattened (105); axis a -> block a
    FLall = flux_closure35_dev(
        MLr[1],  MLr[2],  MLr[3],  MLr[4],  MLr[5],  MLr[6],  MLr[7],
        MLr[8],  MLr[9],  MLr[10], MLr[11], MLr[12], MLr[13], MLr[14],
        MLr[15], MLr[16], MLr[17], MLr[18], MLr[19], MLr[20], MLr[21],
        MLr[22], MLr[23], MLr[24], MLr[25], MLr[26], MLr[27], MLr[28],
        MLr[29], MLr[30], MLr[31], MLr[32], MLr[33], MLr[34], MLr[35])
    FRall = flux_closure35_dev(
        MRr[1],  MRr[2],  MRr[3],  MRr[4],  MRr[5],  MRr[6],  MRr[7],
        MRr[8],  MRr[9],  MRr[10], MRr[11], MRr[12], MRr[13], MRr[14],
        MRr[15], MRr[16], MRr[17], MRr[18], MRr[19], MRr[20], MRr[21],
        MRr[22], MRr[23], MRr[24], MRr[25], MRr[26], MRr[27], MRr[28],
        MRr[29], MRr[30], MRr[31], MRr[32], MRr[33], MRr[34], MRr[35])

    off = (axis - 1) * 35
    sL = min(lminL, lminR)
    sR = max(lmaxL, lmaxR)

    if sL >= 0.0
        return ntuple(j -> FLall[off + j], Val(35))
    elseif sR <= 0.0
        return ntuple(j -> FRall[off + j], Val(35))
    else
        den = sR - sL
        ss  = sL * sR
        return ntuple(Val(35)) do j
            (sR * FLall[off + j] - sL * FRall[off + j] + ss * (MRr[j] - MLr[j])) / den
        end
    end
end

@inline _cell(M, i::Int, j::Int, k::Int) =
    ntuple(m -> @inbounds(M[m, i, j, k]), Val(35))

@inline _clamp(a::Int, n::Int) = a < 1 ? 1 : (a > n ? n : a)

"""
    residual3d_gpu!(R, Fbuf, M, n, dx, Ma; vacuum_floor=0.001, project_faces=true, threads=128)

In-place 3D order-2 HLL residual. `M::CuArray{Float64,4}` (35,n,n,n) interior
field (no stored halo). `R::CuArray{Float64,4}` (35,n,n,n) is OVERWRITTEN
(zeroed then summed over the 3 axes). `Fbuf::CuArray{Float64,4}` is a
(35, n+1, n, n) scratch reused for each axis. `vacuum_floor` = HO_VACUUM_FLOOR.
Cubic grid (dx=dy=dz). project_faces=true matches the CPU 3D path.
"""
function residual3d_gpu!(R::CuArray{Float64,4}, Fbuf::CuArray{Float64,4},
                         M::CuArray{Float64,4}, n::Int, dx::Real, Ma::Real;
                         vacuum_floor::Real=0.001, project_faces::Bool=true,
                         threads::Int=128)
    @assert size(Fbuf) == (35, n + 1, n, n) "Fbuf must be (35,n+1,n,n)"
    # The cubic residual is exactly the nx==ny==nz case of `residual3d_box_gpu!`
    # (same kernels). Reuse the caller's Fbuf as the box face-scratch — its element
    # count (n+1)*n*n equals the box `fmax` for a cube — so this stays alloc-free.
    flat = reshape(Fbuf, 35, (n + 1) * n * n)
    residual3d_box_gpu!(R, M, n, n, n, dx, Ma;
                        vacuum_floor=vacuum_floor, project_faces=project_faces,
                        threads=threads, flat=flat)
    return nothing
end

"""
    residual3d_gpu(M_host, n, dx, Ma; vacuum_floor=0.001, project_faces=true, threads=128)
        -> Array{Float64,4}

Host convenience: upload (35,n,n,n), compute the 3D residual, return (35,n,n,n).
"""
function residual3d_gpu(M_host::Array{Float64,4}, n::Int, dx::Real, Ma::Real;
                        vacuum_floor::Real=0.001, project_faces::Bool=true,
                        threads::Int=128)
    @assert size(M_host) == (35, n, n, n) "M_host must be (35,n,n,n)"
    Md   = CuArray(M_host)
    R    = CUDA.zeros(Float64, 35, n, n, n)
    Fbuf = CUDA.zeros(Float64, 35, n + 1, n, n)
    residual3d_gpu!(R, Fbuf, Md, n, dx, Ma;
                    vacuum_floor=vacuum_floor, project_faces=project_faces, threads=threads)
    CUDA.synchronize()
    return Array(R)
end

# ===========================================================================
# RECTANGULAR-BOX generalization (nx,ny,nz independent), outflow BC on all 6
# faces. This is a STRICT generalization of the cubic kernels above: with
# nx==ny==nz==n the indexing and arithmetic are identical, so results are
# bit-for-bit the same. Used for multi-GPU z-slab decomposition: each rank runs
# this on its EXTENDED slab (35, nx, ny, nz_loc+2*halo) whose halo z-planes are
# ghosts (neighbor data, or outflow copies at the global z-boundary); the
# interior nz_loc planes' residuals are then bit-identical to the single-GPU
# full-domain result because every interior cell sees its real +/-2 neighbors.
# Same `_face_flux_core` / `_cell` / `_clamp` as the cubic path -> bit parity.
# ===========================================================================
function _fhat_x_g!(Fbuf, M, nx::Int, ny::Int, nz::Int, Ma::Float64, vacf::Float64, project::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nx + 1
    if idx <= nf * ny * nz
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            j = r % ny + 1;         k = r ÷ ny + 1
            f = t - 1
            cm1 = _cell(M, _clamp(f - 1, nx), j, k); c0  = _cell(M, _clamp(f, nx), j, k)
            cp1 = _cell(M, _clamp(f + 1, nx), j, k); cp2 = _cell(M, _clamp(f + 2, nx), j, k)
            Fh = _face_flux_core(cm1, c0, cp1, cp2, 1, Ma, vacf, project)
            for m in 1:35; Fbuf[m, t, j, k] = Fh[m]; end
        end
    end
    return nothing
end

function _fhat_y_g!(Fbuf, M, nx::Int, ny::Int, nz::Int, Ma::Float64, vacf::Float64, project::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = ny + 1
    if idx <= nf * nx * nz
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         k = r ÷ nx + 1
            f = t - 1
            cm1 = _cell(M, i, _clamp(f - 1, ny), k); c0  = _cell(M, i, _clamp(f, ny), k)
            cp1 = _cell(M, i, _clamp(f + 1, ny), k); cp2 = _cell(M, i, _clamp(f + 2, ny), k)
            Fh = _face_flux_core(cm1, c0, cp1, cp2, 2, Ma, vacf, project)
            for m in 1:35; Fbuf[m, t, i, k] = Fh[m]; end
        end
    end
    return nothing
end

function _fhat_z_g!(Fbuf, M, nx::Int, ny::Int, nz::Int, Ma::Float64, vacf::Float64, project::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nz + 1
    if idx <= nf * nx * ny
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         j = r ÷ nx + 1
            f = t - 1
            cm1 = _cell(M, i, j, _clamp(f - 1, nz)); c0  = _cell(M, i, j, _clamp(f, nz))
            cp1 = _cell(M, i, j, _clamp(f + 1, nz)); cp2 = _cell(M, i, j, _clamp(f + 2, nz))
            Fh = _face_flux_core(cm1, c0, cp1, cp2, 3, Ma, vacf, project)
            for m in 1:35; Fbuf[m, t, i, j] = Fh[m]; end
        end
    end
    return nothing
end

function _diff_x_g!(R, Fbuf, nx::Int, ny::Int, nz::Int, ds::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            for m in 1:35; R[m, i, j, k] += -(Fbuf[m, i + 1, j, k] - Fbuf[m, i, j, k]) / ds; end
        end
    end
    return nothing
end

function _diff_y_g!(R, Fbuf, nx::Int, ny::Int, nz::Int, ds::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            for m in 1:35; R[m, i, j, k] += -(Fbuf[m, j + 1, i, k] - Fbuf[m, j, i, k]) / ds; end
        end
    end
    return nothing
end

function _diff_z_g!(R, Fbuf, nx::Int, ny::Int, nz::Int, ds::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            for m in 1:35; R[m, i, j, k] += -(Fbuf[m, k + 1, i, j] - Fbuf[m, k, i, j]) / ds; end
        end
    end
    return nothing
end

export residual3d_box_gpu!, residual3d_box_gpu

"""
    residual3d_box_gpu!(R, M, nx, ny, nz, dx, Ma; vacuum_floor=0.001, project_faces=true, threads=128)

Rectangular generalization of `residual3d_gpu!` with outflow BC on all 6 faces.
`M`,`R` are `(35,nx,ny,nz)`. Allocates a reused face buffer internally. With
nx==ny==nz this is bit-identical to `residual3d_gpu!`.
"""
function residual3d_box_gpu!(R::CuArray{Float64,4}, M::CuArray{Float64,4},
                             nx::Int, ny::Int, nz::Int, dx::Real, Ma::Real;
                             vacuum_floor::Real=0.001, project_faces::Bool=true,
                             threads::Int=128, flat::Union{Nothing,CuMatrix{Float64}}=nothing)
    @assert size(M) == (35, nx, ny, nz) "M must be (35,nx,ny,nz)"
    @assert size(R) == (35, nx, ny, nz) "R must be (35,nx,ny,nz)"
    Maf = Float64(Ma); dxf = Float64(dx); vacf = Float64(vacuum_floor)
    fx = (nx + 1) * ny * nz; fy = (ny + 1) * nx * nz; fz = (nz + 1) * nx * ny
    fmax = max(fx, fy, fz)
    # reusable face buffer: caller may supply a (35, >=fmax) scratch to avoid per-call alloc
    if flat === nothing
        flat = CUDA.zeros(Float64, 35, fmax)
    else
        @assert size(flat, 1) == 35 && size(flat, 2) >= fmax "flat must be (35, >=fmax)"
    end
    Bx = reshape(view(flat, :, 1:fx), 35, nx + 1, ny, nz)
    By = reshape(view(flat, :, 1:fy), 35, ny + 1, nx, nz)
    Bz = reshape(view(flat, :, 1:fz), 35, nz + 1, nx, ny)
    bc = cld(nx * ny * nz, threads)

    fill!(R, 0.0)
    @cuda threads=threads blocks=cld(fx, threads) _fhat_x_g!(Bx, M, nx, ny, nz, Maf, vacf, project_faces)
    @cuda threads=threads blocks=bc               _diff_x_g!(R, Bx, nx, ny, nz, dxf)
    @cuda threads=threads blocks=cld(fy, threads) _fhat_y_g!(By, M, nx, ny, nz, Maf, vacf, project_faces)
    @cuda threads=threads blocks=bc               _diff_y_g!(R, By, nx, ny, nz, dxf)
    @cuda threads=threads blocks=cld(fz, threads) _fhat_z_g!(Bz, M, nx, ny, nz, Maf, vacf, project_faces)
    @cuda threads=threads blocks=bc               _diff_z_g!(R, Bz, nx, ny, nz, dxf)
    return nothing
end

"Host convenience: upload `(35,nx,ny,nz)`, compute the box residual, return `(35,nx,ny,nz)`."
function residual3d_box_gpu(M_host::Array{Float64,4}, nx::Int, ny::Int, nz::Int, dx::Real, Ma::Real;
                            vacuum_floor::Real=0.001, project_faces::Bool=true, threads::Int=128)
    @assert size(M_host) == (35, nx, ny, nz) "M_host must be (35,nx,ny,nz)"
    Md = CuArray(M_host); R = CUDA.zeros(Float64, 35, nx, ny, nz)
    residual3d_box_gpu!(R, Md, nx, ny, nz, dx, Ma;
                        vacuum_floor=vacuum_floor, project_faces=project_faces, threads=threads)
    CUDA.synchronize()
    return Array(R)
end

end # module
