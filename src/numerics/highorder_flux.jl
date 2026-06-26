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

# Normal-momentum index per axis: M[_NMOM[axis]] is the normal momentum component.
# Density is always M[1]. Same index is valid for the flux vector F (F[1]=mass flux=M[m],
# F[m]=normal-momentum flux=normal stress). Verified against Flux_closure35_3D output:
#   Fx[1]=M100=M[2], Fx[2]=M200;  Fy[1]=M010=M[6], Fy[6]=M020;  Fz[1]=M001=M[16], Fz[16]=M002.
const _NMOM = (2, 6, 16)

"""
    _flux_jacobian(Mr, axis) -> A (35x35)

Per-axis 35-moment flux Jacobian `A = ∂F_axis/∂M` at state `Mr`, assembled by
second-order central finite differences of [`_phys_flux`](@ref) column-by-column.
The HyQMOM closure is not exposed in closed analytic-Jacobian form for the full
35-moment system, so the Jacobian is built numerically (correctness-first; B1).
"""
function _flux_jacobian(Mr::AbstractVector, axis::Int)
    n = length(Mr)
    A = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n
        h = 1e-6 * max(abs(Mr[j]), 1.0)
        Mp = collect(float.(Mr)); Mp[j] += h
        Mm = collect(float.(Mr)); Mm[j] -= h
        Fp = _phys_flux(Mp, axis)
        Fm = _phys_flux(Mm, axis)
        @views A[:, j] .= (Fp .- Fm) ./ (2h)
    end
    return A
end

"""
    ld_eigvecs(Mr, axis, Ma) -> (R_inner, L_inner, λ_inner)

Right (`R_inner`, columns) and left (`L_inner`, rows) eigenvectors and eigenvalues
(`λ_inner`) of the **linearly-degenerate (LD)** fields of the per-axis 35-moment flux
Jacobian `A = ∂F_axis/∂M` at state `Mr` — i.e. the contact and shear fields advected
at the normal (material) velocity `u_n = Mr[_NMOM[axis]]/Mr[1]`.

The Jacobian (assembled in [`_flux_jacobian`](@ref)) is hyperbolic: all 35 eigenvalues
are real and `A` is diagonalizable. The LD subspace is the eigenspace with eigenvalue
`u_n`; for the realizable 3D 35-moment state it has dimension 9 (one contact + the
shear/cross-moment modes). LD modes are identified as the eigenvalues within `tol` of
`u_n`, where `tol` scales with the spectral radius. The left eigenvectors are taken as
the matching rows of `inv(V)` (the dual basis of the full eigenbasis `V`), so
`L_inner * R_inner ≈ I` holds for any basis choice within the degenerate subspace.

`Ma` is accepted for API stability with the HLLEM caller (Task B2); the flux Jacobian
is evaluated directly at the supplied `Mr`, so `Ma` is not used here.

HLLEM (Task B2) anti-diffuses exactly these LD fields.
"""
function ld_eigvecs(Mr::AbstractVector, axis::Int, Ma::Real)
    A = _flux_jacobian(Mr, axis)
    un = Mr[_NMOM[axis]] / Mr[1]

    E = eigen(A)
    vals = E.values
    V = E.vectors

    # Hyperbolic system: eigenvalues are real. Drop any spurious imaginary parts.
    specrad = maximum(abs.(real.(vals)))
    tol = 1e-6 * max(1.0, specrad)
    idx = findall(k -> abs(real(vals[k]) - un) <= tol && abs(imag(vals[k])) <= tol,
                  eachindex(vals))
    isempty(idx) && error("ld_eigvecs: no linearly-degenerate (u_n) modes found for axis $axis")

    Vinv = inv(V)
    R_inner = real.(V[:, idx])
    L_inner = real.(Vinv[idx, :])
    λ_inner = real.(vals[idx])
    return R_inner, L_inner, λ_inner
end

"Contact (material) wave speed S_M = normal velocity of the HLL star state, clamped to [sL,sR]."
function hllc_contact_speed(MLr::AbstractVector, MRr::AbstractVector, sL::Real, sR::Real, axis::Int)
    m = _NMOM[axis]
    FL = _phys_flux(MLr, axis); FR = _phys_flux(MRr, axis)
    Uden = (sR*MRr[1] - sL*MLr[1] - (FR[1] - FL[1])) / (sR - sL)        # HLL density
    Umom = (sR*MRr[m] - sL*MLr[m] - (FR[m] - FL[m])) / (sR - sL)        # HLL normal momentum
    # Guard: degenerate (non-positive) HLL density or non-finite quotient → wave midpoint.
    # Realizable inputs give Uden > 0, so this branch never fires on valid states.
    (!(Uden > 0)) && return 0.5*(sL + sR)
    sm = Umom / Uden
    return isfinite(sm) ? clamp(sm, sL, sR) : 0.5*(sL + sR)
end

"""
    hllc_star(MKr, sK, S_M, axis) -> U*_K  (length 35)

Per-side **kinetic** HLLC star state for side `K`. It is built so that the mass-flux
Rankine–Hugoniot across the `sK` wave holds with contact speed `S_M`, namely the
density is rescaled `ρ* = ρ_K (sK − u_K)/(sK − S_M)` (`u_K` = normal mean velocity),
the **normal** mean velocity is shifted to `S_M`, while the tangential mean
velocities and **all** central (and hence standardized) moments are preserved.

Because the standardized-moment structure is unchanged and the density stays
positive, `hllc_star(MKr,…)` is realizable whenever `MKr` is. This is the physically
correct *per-side* contact-region state and supplies the contact-jump direction.

NOTE (key derivation result): a purely per-side star cannot satisfy the full
35-component HLL-consistency identity for the **nonlinear** HyQMOM closure (it does
hold exactly for mass and the three momenta, but fails on the higher even normal
moments — the central→raw map is nonlinear in the mean-velocity shift). The
consistency-exact star pair is assembled by [`hllc_star_pair`](@ref), which couples
both sides through the HLL average. See `docs/riemann-solver-scope.md`.
"""
function hllc_star(MKr::AbstractVector, sK::Real, S_M::Real, axis::Int)
    rho = MKr[1]
    u = MKr[2]/rho; v = MKr[6]/rho; w = MKr[16]/rho
    un = axis == 1 ? u : (axis == 2 ? v : w)
    den = sK - S_M
    # density rescale from the mass-flux RH; guard a vanishing star region (S_M→sK)
    rstar = abs(den) > 1e-14 ? rho*(sK - un)/den : rho
    C4, _ = M2CS4_35(MKr)
    C200=C4[3];  C300=C4[4];  C400=C4[5];  C110=C4[7];  C210=C4[8];  C310=C4[9]
    C020=C4[10]; C120=C4[11]; C220=C4[12]; C030=C4[13]; C130=C4[14]; C040=C4[15]
    C101=C4[17]; C201=C4[18]; C301=C4[19]; C002=C4[20]; C102=C4[21]; C202=C4[22]
    C003=C4[23]; C103=C4[24]; C004=C4[25]; C011=C4[26]; C111=C4[27]; C211=C4[28]
    C021=C4[29]; C121=C4[30]; C031=C4[31]; C012=C4[32]; C112=C4[33]; C013=C4[34]; C022=C4[35]
    um = axis == 1 ? S_M : u
    vm = axis == 2 ? S_M : v
    wm = axis == 3 ? S_M : w
    Marr = C4toM4_3D(rstar, um, vm, wm,
                     C200, C110, C101, C020, C011, C002,
                     C300, C210, C201, C120, C111, C102, C030, C021, C012, C003,
                     C400, C310, C301, C220, C211, C202, C130, C121, C112, C103,
                     C040, C031, C022, C013, C004)
    return Marr[_M2CS4_IDX]
end

"""
    hllc_star_pair(MLr, MRr, sL, sR, S_M, axis) -> (U*_L, U*_R)

The **consistency-exact** HLLC star pair. The two star states are the unique pair
that simultaneously satisfies

  * HLL-consistency (the integral constraint over the fan):
    `((S_M−sL)·U*_L + (sR−S_M)·U*_R)/(sR−sL) = U_HLL`, and
  * the kinetic contact jump: `U*_R − U*_L = hllc_star(R) − hllc_star(L)`,

solved by anchoring on the HLL average `U_HLL`:

    U*_L = U_HLL − (sR−S_M)/(sR−sL) · (g_R − g_L)
    U*_R = U_HLL + (S_M−sL)/(sR−sL) · (g_R − g_L)

with `g_K = hllc_star(M_K, sK, S_M, axis)`. By construction this satisfies the
Rankine–Hugoniot condition across **each** acoustic wave AND across the contact
(`F*_R − F*_L = S_M(U*_R − U*_L)`) for any jump direction; the kinetic jump fixes the
physical contact closure (normal velocity = `S_M`, central-moment structure carried
across). HLL-consistency holds to machine precision. Realizability is NOT guaranteed
for every input (strong colliding streams can push a star state out of R — the
documented hard case A3 handles by falling back to HLL).
"""
function hllc_star_pair(MLr::AbstractVector, MRr::AbstractVector,
                        sL::Real, sR::Real, S_M::Real, axis::Int)
    FL = _phys_flux(MLr, axis); FR = _phys_flux(MRr, axis)
    Uhll = (sR .* MRr .- sL .* MLr .- (FR .- FL)) ./ (sR - sL)
    gL = hllc_star(MLr, sL, S_M, axis)
    gR = hllc_star(MRr, sR, S_M, axis)
    K = gR .- gL                       # kinetic contact-jump direction
    UsL = Uhll .- ((sR - S_M)/(sR - sL)) .* K
    UsR = Uhll .+ ((S_M - sL)/(sR - sL)) .* K
    return UsL, UsR
end

"""
    hllc_flux(MLr, MRr, sL, sR, S_M, axis) -> length-35 interface flux

Four-region HLLC numerical flux. Uses the consistency-exact star pair
([`hllc_star_pair`](@ref)) so the star fluxes satisfy Rankine–Hugoniot across both
acoustic waves and the contact, and the construction reduces to HLL when integrated
over the fan. As a safety net (A3 formalizes the fallback policy) the contact-region
star state is checked: if it is non-finite or leaves the realizable set, the flux
falls back to the two-wave HLL flux.
"""
function hllc_flux(MLr::AbstractVector, MRr::AbstractVector,
                   sL::Real, sR::Real, S_M::Real, axis::Int)
    FL = _phys_flux(MLr, axis); FR = _phys_flux(MRr, axis)
    if sL >= 0
        return FL
    elseif sR <= 0
        return FR
    end
    UsL, UsR = hllc_star_pair(MLr, MRr, sL, sR, S_M, axis)
    Us = S_M >= 0 ? UsL : UsR
    # One-sided realizability check is sufficient: the returned flux uses only the
    # contacted-side star via its RH relation (F* = F_K + sK*(U*_K − M_K)), so only
    # that side's star state needs to be realizable for the result to be valid.
    if !all(isfinite, Us) || !is_realizable(Us)
        return (sR .* FL .- sL .* FR .+ (sL*sR) .* (MRr .- MLr)) ./ (sR - sL)
    end
    return S_M >= 0 ? (FL .+ sL .* (UsL .- MLr)) : (FR .+ sR .* (UsR .- MRr))
end

"""
    hllem_flux(MLr, MRr, sL, sR, FL, FR, axis, Ma; φ=1.0) -> length-35 interface flux

HLLEM numerical flux (Dumbser & Balsara 2016). HLLEM keeps the two-wave HLL envelope
(so it stays inside the HLL positivity cone) but **anti-diffuses the linearly-degenerate
(contact + shear) fields** that plain HLL smears. Only the genuinely two-sided case
`sL<0<sR` is anti-diffused; `sL>=0` -> `FL`, `sR<=0` -> `FR` (identical to HLL).

The anti-diffusion is built from the LD eigenstructure ([`ld_eigvecs`](@ref)) of the
per-axis flux Jacobian, linearized at the **HLL average state**
`U_HLL = (sR·MRr − sL·MLr − (FR−FL))/(sR−sL)` (projected back to R via
`realizable_3D_M4`; the natural single intermediate state of the two-wave solver):

    δ*_k = clamp(1 − max(λ_k,0)/sR − min(λ_k,0)/sL, 0, 1)        # per LD mode k, in [0,1]
    f = f_HLL − φ · (sL·sR)/(sR−sL) · R_inner · diag(δ*) · L_inner · (MRr − MLr)

This is the standard Dumbser–Balsara formula. `δ*_k → 1` as `u_n → 0` (both terms vanish);
for `u_n ≠ 0` the contacted-side ratio is non-zero and `δ*_k = 1 − u_n/sR < 1` (u_n>0 case).
It tapers to 0 as `λ_k` approaches a wave speed.

**Coalescing-eigenvalue / vacuum guard.** Near vacuum the finite-difference Jacobian and
its eigenbasis become ill-conditioned. The anti-diffusion is DROPPED (plain HLL is
returned) when any of: the linearization state is non-finite/non-realizable;
`ld_eigvecs` errors or returns non-finite values; the biorthonormality residual
`‖L_inner·R_inner − I‖` exceeds a tolerance (degenerate basis); or the basis norm is
huge. Finally, the two HLLEM intermediate states implied by the anti-diffused flux are
checked with the realizability oracle — if either is non-finite or non-realizable, the
flux falls back to HLL.
"""
function hllem_flux(MLr::AbstractVector, MRr::AbstractVector,
                    sL::Real, sR::Real, FL::AbstractVector, FR::AbstractVector,
                    axis::Int, Ma::Real; φ::Real=1.0)
    # One-sided fans: identical to HLL (no LD field crosses the interface).
    if sL >= 0
        return FL
    elseif sR <= 0
        return FR
    end
    f_hll = (sR .* FL .- sL .* FR .+ (sL*sR) .* (MRr .- MLr)) ./ (sR - sL)

    # Linearization point: the HLL average state, projected back onto R.
    U_HLL = (sR .* MRr .- sL .* MLr .- (FR .- FL)) ./ (sR - sL)
    if !all(isfinite, U_HLL)
        return f_hll
    end
    Ulin = realizable_3D_M4(U_HLL, Ma)
    if !all(isfinite, Ulin) || !is_realizable(Ulin)
        return f_hll
    end

    # LD eigenstructure; guard against errors / coalescence / degenerate basis.
    local R_inner, L_inner, λ_inner
    try
        R_inner, L_inner, λ_inner = ld_eigvecs(Ulin, axis, Ma)
    catch
        return f_hll
    end
    if !(all(isfinite, R_inner) && all(isfinite, L_inner) && all(isfinite, λ_inner))
        return f_hll
    end
    k = length(λ_inner)
    k == 0 && return f_hll
    LR = L_inner * R_inner
    if norm(LR - Matrix{Float64}(I, k, k)) > 1e-6 || norm(R_inner) > 1e6 || norm(L_inner) > 1e6
        return f_hll                       # degenerate / ill-conditioned basis (near vacuum)
    end

    # Anti-diffusion coefficients δ* in [0,1] (=1 for pure contact λ=u_n in (sL,sR)).
    δ = [clamp(1.0 - max(λ, 0.0)/sR - min(λ, 0.0)/sL, 0.0, 1.0) for λ in λ_inner]
    jump = MRr .- MLr
    anti = R_inner * (δ .* (L_inner * jump))          # R·diag(δ*)·L·(MR−ML)
    f = f_hll .- φ .* (sL*sR/(sR - sL)) .* anti
    if !all(isfinite, f)
        return f_hll
    end

    # Realizability gate on the two HLLEM intermediate states implied by f.
    # f = FL + sL(U*_L − MLr) = FR + sR(U*_R − MRr), giving
    #   U*_L = U_HLL − (sR/(sR−sL))·anti,  U*_R = U_HLL − (sL/(sR−sL))·anti.
    UstarL = U_HLL .- φ .* (sR/(sR - sL)) .* anti
    UstarR = U_HLL .- φ .* (sL/(sR - sL)) .* anti
    if !(all(isfinite, UstarL) && all(isfinite, UstarR) &&
         is_realizable(UstarL) && is_realizable(UstarR))
        return f_hll
    end
    return f
end

"""
Interface-flux (Riemann-solver) selector. Default `:hll` is the original, validated
two-wave HLL flux (byte-identical). `:rusanov` is a robust local Lax–Friedrichs
fallback. `:hllc` is the four-region HLLC flux with consistency-exact star pair and
automatic realizability fallback to HLL. `:hllem` is the HLLEM flux
([`hllem_flux`](@ref)): it anti-diffuses the linearly-degenerate (contact/shear) fields
while staying inside the HLL positivity cone, with a coalescing/vacuum guard and a
realizability fallback to HLL. Set from `simulation_runner` via the
`riemann_solver` param, or directly (`HyQMOM.RIEMANN_SOLVER[] = :hllc`). Future
solvers (`:kinetic`) plug into `face_flux_1d`'s branch — see
`docs/riemann-solver-scope.md`. OPT-IN: anything other than `:hll` must be requested
explicitly.
"""
const RIEMANN_SOLVER = Ref{Symbol}(:hll)

"""
    face_flux_1d(M_L, M_R, axis, Ma)

Interface flux from left/right face moment states. Each side is projected
(realizable_3D_M4) and hyperbolicity-corrected before fluxing. The flux formula is
chosen by `RIEMANN_SOLVER[]` (default `:hll`, byte-identical to the original scheme).
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
    rs = RIEMANN_SOLVER[]
    if rs === :hll
        if sL >= 0
            return FL
        elseif sR <= 0
            return FR
        else
            return (sR .* FL .- sL .* FR .+ (sL*sR) .* (MRr .- MLr)) ./ (sR - sL)
        end
    elseif rs === :rusanov
        # local Lax–Friedrichs (Rusanov): robust, more diffusive than HLL.
        a = max(abs(sL), abs(sR))
        return 0.5 .* (FL .+ FR) .- 0.5a .* (MRr .- MLr)
    elseif rs === :hllc
        return hllc_flux(MLr, MRr, sL, sR, hllc_contact_speed(MLr, MRr, sL, sR, axis), axis)
    elseif rs === :hllem
        return hllem_flux(MLr, MRr, sL, sR, FL, FR, axis, Ma)
    else
        throw(ArgumentError("unknown riemann_solver=$(rs); available: :hll (default), :rusanov, :hllc, :hllem"))
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
