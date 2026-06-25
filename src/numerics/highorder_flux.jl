"""
    realize_and_speed(M, axis, Ma)

Hyperbolicity-correct M for the given axis and return (Mr, vpmin, vpmax) with the
combined 6x6 + 1D-closure wave speeds, matching the interior flux path.
"""
function realize_and_speed(M::AbstractVector, axis::Int, Ma::Real)
    if axis == 1
        v6min, v6max, Mr = eigenvalues6_hyperbolic_3D(M, 1, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[[1,2,3,4,5]])
    elseif axis == 2
        v6min, v6max, Mr = eigenvalues6_hyperbolic_3D(M, 2, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[[1,6,10,13,15]])
    else
        v6min, v6max, Mr = eigenvalues6z_hyperbolic_3D(M, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[[1,16,20,23,25]])
    end
    return Mr, min(v5min, v6min), max(v5max, v6max)
end

"Physical flux (length 35) of moment vector M in the given axis direction."
function _phys_flux(M::AbstractVector, axis::Int)
    Fx, Fy, Fz = Flux_closure35_3D(M)
    return axis == 1 ? Fx : (axis == 2 ? Fy : Fz)
end

"""
    face_flux_1d(M_L, M_R, axis, Ma)

HLL interface flux from left/right face moment states. Each side is projected
(realizable_3D_M4) and hyperbolicity-corrected before fluxing.
"""
function face_flux_1d(M_L::AbstractVector, M_R::AbstractVector, axis::Int, Ma::Real)
    ML = realizable_3D_M4(M_L, Ma)
    MR = realizable_3D_M4(M_R, Ma)
    MLr, lminL, lmaxL = realize_and_speed(ML, axis, Ma)
    MRr, lminR, lmaxR = realize_and_speed(MR, axis, Ma)
    FL = _phys_flux(MLr, axis)
    FR = _phys_flux(MRr, axis)
    sL = min(lminL, lminR)
    sR = max(lmaxL, lmaxR)
    if sL >= 0
        return FL
    elseif sR <= 0
        return FR
    else
        return (sR .* FL .- sL .* FR .+ (sL*sR) .* (MRr .- MLr)) ./ (sR - sL)
    end
end

"""
    residual_1d(Mline, dx, Ma; order=2)

Method-of-lines spatial residual for a 1D row of 35-moment cells (Ncell x 35) in
the x-direction. order=1: first-order (cell-centered). order=2: MUSCL on the
bounded reconstruction variables, with local fallback to first order if a
reconstructed face has nonpositive density.
"""
function residual_1d(Mline::AbstractMatrix, dx::Real, Ma::Real; order::Int=2)
    Nc = size(Mline, 1)
    axis = 1
    # Right-face L/R moment states at each interface i+1/2, i=1..Nc-1
    ML = [zeros(35) for _ in 1:Nc-1]   # left state at interface i+1/2 (from cell i)
    MR = [zeros(35) for _ in 1:Nc-1]   # right state at interface i+1/2 (from cell i+1)
    if order == 1
        for i in 1:Nc-1
            ML[i] = Mline[i, :]; MR[i] = Mline[i+1, :]
        end
    else
        V = [to_recon_vars(Mline[i, :]) for i in 1:Nc]
        # per-cell left/right face recon-vars with zero-gradient BC
        Vminus = [zeros(35) for _ in 1:Nc]; Vplus = [zeros(35) for _ in 1:Nc]
        for i in 1:Nc
            vm = V[max(i-1,1)]; v0 = V[i]; vp = V[min(i+1,Nc)]
            Vminus[i], Vplus[i] = muscl_faces(vm, v0, vp)
        end
        for i in 1:Nc-1
            Li = from_recon_vars(Vplus[i])     # right face of cell i
            Ri = from_recon_vars(Vminus[i+1])  # left face of cell i+1
            # local order degradation: fall back to 1st order if EITHER face has bad density
            if Li[1] > 0 && Ri[1] > 0
                ML[i] = Li; MR[i] = Ri
            else
                ML[i] = Mline[i, :]; MR[i] = Mline[i+1, :]
            end
        end
    end
    Fhat = [face_flux_1d(ML[i], MR[i], axis, Ma) for i in 1:Nc-1]
    R = zeros(Nc, 35)
    for i in 2:Nc-1
        R[i, :] = -(Fhat[i] .- Fhat[i-1]) ./ dx
    end
    # zero-gradient BC: no net flux at the physical boundary cells
    return R
end
