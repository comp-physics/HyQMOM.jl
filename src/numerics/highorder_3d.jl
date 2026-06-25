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
function residual_line(Mext::AbstractMatrix, ds::Real, axis::Int, Ma::Real; order::Int=2, g::Int=2)
    Ntot = size(Mext, 1)
    Ni = Ntot - 2g
    # interface fluxes at i+1/2 for interior interfaces: need faces at indices
    # spanning g..Ntot-g. Compute Fhat at every interface that bounds an interior cell.
    # Interior cells are rows g+1 .. g+Ni; their bounding interfaces are g+1/2 .. g+Ni+1/2.
    Fhat = Dict{Int,Vector{Float64}}()
    function face_states(iL)  # interface between cell iL and iL+1
        if order == 1
            return Mext[iL, :], Mext[iL+1, :]
        else
            Vl = muscl_faces(to_recon_vars(Mext[iL-1,:]), to_recon_vars(Mext[iL,:]), to_recon_vars(Mext[iL+1,:]))[2]
            Vr = muscl_faces(to_recon_vars(Mext[iL,:]),   to_recon_vars(Mext[iL+1,:]), to_recon_vars(Mext[iL+2,:]))[1]
            Li = from_recon_vars(Vl); Ri = from_recon_vars(Vr)
            (Li[1] > 0 && Ri[1] > 0) ? (Li, Ri) : (Mext[iL,:], Mext[iL+1,:])
        end
    end
    for iface in g:(g+Ni)            # interfaces bounding interior cells
        ML, MR = face_states(iface)
        Fhat[iface] = face_flux_1d(ML, MR, axis, Ma)
    end
    R = zeros(Ni, 35)
    for ii in 1:Ni
        c = g + ii                   # cell row in Mext
        R[ii, :] = -(Fhat[c] .- Fhat[c-1]) ./ ds
    end
    return R
end
