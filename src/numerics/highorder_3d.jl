"""
    residual_line(Mext, ds, axis, Ma; order=2, g=2)

1D method-of-lines residual for a line of 35-moment cells that already includes
`g` ghost cells at each end (filled externally with neighbor / BC data). Returns
the interior residual (Ni, 35) = -(Fhat[i+1/2] - Fhat[i-1/2])/ds, reconstructing
face states from the ghosts (no boundary condition applied here). This is the
shared primitive for the 3D unsplit residual (x, y via MPI halos; z via padded
ghosts). order=1 uses cell-centered states; order=2 uses MUSCL with per-interface
fallback to first order on nonpositive reconstructed density.
"""
function residual_line(Mext::AbstractMatrix, ds::Real, axis::Int, Ma::Real;
                       order::Int=2, g::Int=2, use_limiter::Bool=false,
                       use_proj_recon::Bool=false)
    Ntot = size(Mext, 1)
    Ni = Ntot - 2g
    # interface fluxes at i+1/2 for interior interfaces: need faces at indices
    # spanning g..Ntot-g. Compute Fhat at every interface that bounds an interior cell.
    # Interior cells are rows g+1 .. g+Ni; their bounding interfaces are g+1/2 .. g+Ni+1/2.
    Fhat = Vector{Vector{Float64}}(undef, g + Ni)
    function face_states(iL)  # interface between cell iL and iL+1
        if order == 1
            return Mext[iL, :], Mext[iL+1, :]
        else
            Vl = muscl_faces(to_recon_vars(Mext[iL-1,:]), to_recon_vars(Mext[iL,:]), to_recon_vars(Mext[iL+1,:]))[2]
            Vr = muscl_faces(to_recon_vars(Mext[iL,:]),   to_recon_vars(Mext[iL+1,:]), to_recon_vars(Mext[iL+2,:]))[1]
            recon_face_pair(Vl, Vr, Mext[iL,:], Mext[iL+1,:])
        end
    end
    if use_proj_recon && order == 2
        # Rodney's projection-triggered control: a cell whose mean is flagged for
        # the realizability projection (smallest Delta_2 eigenvalue < 0, i.e.
        # `realizability_margin < 0`) reconstructs FIRST-ORDER (its face = cell mean);
        # all other cells get full MUSCL. recon_face_pair still guards the MUSCL
        # faces against nonpositive reconstructed density. Per-cell, local, and uses
        # the same realizability signal as the projection itself.
        Vc = [to_recon_vars(@view Mext[i, :]) for i in axes(Mext, 1)]
        flagged = [realizability_margin(@view Mext[i, :]) < 0 for i in axes(Mext, 1)]
        function face_states_proj(iL)   # interface between cell iL and iL+1
            VplusL  = flagged[iL]   ? Vc[iL]   : muscl_faces(Vc[iL-1], Vc[iL],   Vc[iL+1])[2]
            VminusR = flagged[iL+1] ? Vc[iL+1] : muscl_faces(Vc[iL],   Vc[iL+1], Vc[iL+2])[1]
            return recon_face_pair(VplusL, VminusR, Mext[iL, :], Mext[iL+1, :])
        end
        for iface in g:(g+Ni)
            ML, MR = face_states_proj(iface)
            Fhat[iface] = face_flux_1d(ML, MR, axis, Ma)
        end
    elseif use_limiter && order == 2
        # Realizability scaling limiter branch.
        # Precompute reconstruction variables for all rows (g >= 2 guarantees
        # indices iL-1 .. iL+2 are in range for iL in g:(g+Ni)).
        Vc = [to_recon_vars(@view Mext[i, :]) for i in axes(Mext, 1)]
        function face_states_lim(iL)   # interface between cell iL and iL+1
            # right face of cell iL: Vplus = V0 + 0.5*theta*slope
            _, VplusL, _  = scaling_limited_faces(Vc[iL-1], Vc[iL],   Vc[iL+1])
            # left face of cell iL+1: Vminus = V0 - 0.5*theta*slope
            VminusR, _, _ = scaling_limited_faces(Vc[iL],   Vc[iL+1], Vc[iL+2])
            return from_recon_vars(VplusL), from_recon_vars(VminusR)
        end
        for iface in g:(g+Ni)
            ML, MR = face_states_lim(iface)
            Fhat[iface] = face_flux_1d(ML, MR, axis, Ma)
        end
    else
        for iface in g:(g+Ni)            # interfaces bounding interior cells
            ML, MR = face_states(iface)
            Fhat[iface] = face_flux_1d(ML, MR, axis, Ma)
        end
    end
    R = zeros(Ni, 35)
    for ii in 1:Ni
        c = g + ii                   # cell row in Mext
        R[ii, :] = -(Fhat[c] .- Fhat[c-1]) ./ ds
    end
    return R
end

function residual_ho_3d!(R::Array{Float64,4}, M::Array{Float64,4},
                         nx::Int, ny::Int, nz::Int, halo::Int,
                         dx::Real, dy::Real, dz::Real, Ma::Real;
                         order::Int=2, use_limiter::Bool=false, use_proj_recon::Bool=false)
    fill!(R, 0.0)
    g = halo
    # X: lines along i (have halos), for each interior (jh,k)
    for k in 1:nz, j in 1:ny
        jh = j + halo
        Mext = @view M[:, jh, k, :]                 # (nx+2halo, 35)
        Rl = residual_line(Mext, dx, 1, Ma; order=order, g=g, use_limiter=use_limiter, use_proj_recon=use_proj_recon)   # (nx,35)
        for i in 1:nx; R[i+halo, jh, k, :] .+= Rl[i, :]; end
    end
    # Y: lines along j, for each interior (ih,k)
    for k in 1:nz, i in 1:nx
        ih = i + halo
        Mext = @view M[ih, :, k, :]
        Rl = residual_line(Mext, dy, 2, Ma; order=order, g=g, use_limiter=use_limiter, use_proj_recon=use_proj_recon)
        for j in 1:ny; R[ih, j+halo, k, :] .+= Rl[j, :]; end
    end
    # Z: no halo in z -> pad with outflow ghosts (copy edge), for each interior (ih,jh)
    for i in 1:nx, j in 1:ny
        ih = i + halo; jh = j + halo
        col = M[ih, jh, :, :]                        # (nz,35)
        Mext = vcat(repeat(col[1:1,:], g, 1), col, repeat(col[nz:nz,:], g, 1))  # outflow pad
        Rl = residual_line(Mext, dz, 3, Ma; order=order, g=g, use_limiter=use_limiter, use_proj_recon=use_proj_recon)   # (nz,35)
        for k in 1:nz; R[ih, jh, k, :] .+= Rl[k, :]; end
    end
    return R
end

# ---------------------------------------------------------------------------
# Projection-activation diagnostic (env-gated, zero-allocation, single cheap
# branch per cell when off).
# Set HYQMOM_PROJ_COUNT=1 to count how many cells _project_interior! actually
# corrects each step.  When the env var is unset (default) the Ref holds false
# and the per-cell `if _PROJ_COUNT_ENABLED[]` costs one Ref dereference + a
# predictably-false branch — zero allocation, no extra function calls.
# ---------------------------------------------------------------------------
const _PROJ_COUNT_ENABLED = Ref{Bool}(
    get(ENV, "HYQMOM_PROJ_COUNT", "0") != "0"
)
const _PROJ_CORRECTIONS   = Ref{Int}(0)

"""
    reset_proj_counter!()

Reset the per-step projection-correction counter to zero. Always resets the counter,
regardless of `HYQMOM_PROJ_COUNT`; the counter is only *incremented* when that env var
is set (otherwise it stays 0). Exported for use by the diagnostic drivers.
"""
reset_proj_counter!() = (_PROJ_CORRECTIONS[] = 0; nothing)

"""
    proj_correction_count() -> Int

Return the current projection-correction count (cells that were unrealizable
before `_project_interior!` corrected them).  Only meaningful when
`HYQMOM_PROJ_COUNT=1`.
"""
proj_correction_count() = _PROJ_CORRECTIONS[]

function _project_interior!(M, nx,ny,nz,halo, Ma)
    if _PROJ_COUNT_ENABLED[]
        for k in 1:nz, j in 1:ny, i in 1:nx
            ih=i+halo; jh=j+halo
            # count cells that are unrealizable before projection
            if realizability_margin(@view M[ih,jh,k,:]) < 0
                _PROJ_CORRECTIONS[] += 1
            end
            M[ih,jh,k,:] = realizable_3D_M4(M[ih,jh,k,:], Ma)
        end
    else
        for k in 1:nz, j in 1:ny, i in 1:nx
            ih=i+halo; jh=j+halo
            M[ih,jh,k,:] = realizable_3D_M4(M[ih,jh,k,:], Ma)
        end
    end
end

function step_highorder_3d!(M::Array{Float64,4}, dt::Real, decomp, bc::Symbol,
                            nx,ny,nz,halo, dx,dy,dz, Ma;
                            order::Int=2, use_limiter::Bool=false, use_proj_recon::Bool=false)
    R = similar(M)
    int = (halo+1:halo+nx, halo+1:halo+ny, 1:nz, :)
    # stage helper: M_in (with halos) -> returns updated interior-only array (full M-shape, halos zero)
    function L!(Mwork)
        halo_exchange_3d!(Mwork, decomp, bc)
        residual_ho_3d!(R, Mwork, nx,ny,nz,halo, dx,dy,dz, Ma; order=order, use_limiter=use_limiter, use_proj_recon=use_proj_recon)
        return R
    end
    M0 = copy(M)
    # stage 1: M1 = M + dt*L(M)
    L!(M); @views M[int...] .= M0[int...] .+ dt .* R[int...]; _project_interior!(M,nx,ny,nz,halo,Ma)
    # stage 2: M2 = 3/4 M0 + 1/4 (M1 + dt L(M1))
    L!(M); @views M[int...] .= (3/4).*M0[int...] .+ (1/4).*(M[int...] .+ dt .* R[int...]); _project_interior!(M,nx,ny,nz,halo,Ma)
    # stage 3: M = 1/3 M0 + 2/3 (M2 + dt L(M2))
    L!(M); @views M[int...] .= (1/3).*M0[int...] .+ (2/3).*(M[int...] .+ dt .* R[int...]); _project_interior!(M,nx,ny,nz,halo,Ma)
    halo_exchange_3d!(M, decomp, bc)
    return nothing
end
