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
    residual_1d(Mline, dx, Ma; order=2, bc=:outflow, use_limiter=false)

Method-of-lines spatial residual for a 1D row of 35-moment cells (Ncell x 35) in
the x-direction. order=1: first-order (cell-centered). order=2: MUSCL on the
bounded reconstruction variables, with local fallback to first order if a
reconstructed face has nonpositive density.

bc=:outflow (default): zero-gradient boundary conditions — boundary cells i=1 and
  i=Nc receive zero residual (no net flux through the domain walls).
bc=:periodic: wrap neighbor indices so the domain is periodic. All Nc interfaces
  i+1/2 (i=1..Nc, with i+1 wrapping) are computed and every cell gets a residual.

use_limiter=false (default): existing muscl_faces + recon_face_pair path (byte-identical
  to the pre-existing behavior). use_limiter=true: order==2 faces built with
  scaling_limited_faces instead; faces are realizable by construction so no fallback
  is needed. The order==1 path is unaffected by this flag.
"""
function residual_1d(Mline::AbstractMatrix, dx::Real, Ma::Real;
                     order::Int=2, bc::Symbol=:outflow, use_limiter::Bool=false)
    Nc = size(Mline, 1)
    axis = 1
    R = zeros(Nc, 35)

    if bc == :periodic
        wrap(i) = mod(i-1, Nc) + 1
        # Face states at interface i+1/2 for i=1..Nc (i+1 wraps)
        ML = [zeros(35) for _ in 1:Nc]
        MR = [zeros(35) for _ in 1:Nc]
        if order == 1
            for i in 1:Nc
                ML[i] = Mline[i, :]; MR[i] = Mline[wrap(i+1), :]
            end
        elseif use_limiter
            Vc = [to_recon_vars(@view Mline[i, :]) for i in 1:Nc]
            for i in 1:Nc
                ip1 = wrap(i+1)
                _, Vplus_i, _     = scaling_limited_faces(Vc[wrap(i-1)], Vc[i],   Vc[ip1])
                Vminus_ip1, _, _  = scaling_limited_faces(Vc[i],         Vc[ip1], Vc[wrap(i+2)])
                ML[i] = from_recon_vars(Vplus_i)
                MR[i] = from_recon_vars(Vminus_ip1)
            end
        else
            V = [to_recon_vars(Mline[i, :]) for i in 1:Nc]
            Vminus = [zeros(35) for _ in 1:Nc]; Vplus = [zeros(35) for _ in 1:Nc]
            for i in 1:Nc
                Vminus[i], Vplus[i] = muscl_faces(V[wrap(i-1)], V[i], V[wrap(i+1)])
            end
            for i in 1:Nc
                ML[i], MR[i] = recon_face_pair(Vplus[i], Vminus[wrap(i+1)],
                                               Mline[i, :], Mline[wrap(i+1), :])
            end
        end
        Fhat = [face_flux_1d(ML[i], MR[i], axis, Ma) for i in 1:Nc]
        for i in 1:Nc
            R[i, :] = -(Fhat[i] .- Fhat[wrap(i-1)]) ./ dx
        end
    elseif bc == :outflow  # zero-gradient BCs
        # Right-face L/R moment states at each interface i+1/2, i=1..Nc-1
        ML = [zeros(35) for _ in 1:Nc-1]   # left state at interface i+1/2 (from cell i)
        MR = [zeros(35) for _ in 1:Nc-1]   # right state at interface i+1/2 (from cell i+1)
        if order == 1
            for i in 1:Nc-1
                ML[i] = Mline[i, :]; MR[i] = Mline[i+1, :]
            end
        elseif use_limiter
            Vc = [to_recon_vars(@view Mline[i, :]) for i in 1:Nc]
            for i in 1:Nc-1
                _, Vplus_i, _     = scaling_limited_faces(Vc[max(i-1,1)], Vc[i],   Vc[min(i+1,Nc)])
                Vminus_ip1, _, _  = scaling_limited_faces(Vc[i],          Vc[i+1], Vc[min(i+2,Nc)])
                ML[i] = from_recon_vars(Vplus_i)
                MR[i] = from_recon_vars(Vminus_ip1)
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
                # local order degradation: fall back to 1st order if either face is
                # unrealizable (bad density OR variance OR non-finite reconstruction)
                ML[i], MR[i] = recon_face_pair(Vplus[i], Vminus[i+1],
                                               Mline[i, :], Mline[i+1, :])
            end
        end
        Fhat = [face_flux_1d(ML[i], MR[i], axis, Ma) for i in 1:Nc-1]
        for i in 2:Nc-1
            R[i, :] = -(Fhat[i] .- Fhat[i-1]) ./ dx
        end
        # zero-gradient BC: no net flux at the physical boundary cells (i=1, i=Nc remain zero)
    else
        throw(ArgumentError("residual_1d: unknown bc=$bc (use :outflow or :periodic)"))
    end
    return R
end
