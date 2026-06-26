using Test
using HyQMOM

@testset "riemann_solver selector" begin
    ML = InitializeM4_35(1.0,  0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    MR = InitializeM4_35(0.5, -0.2, 0.0, 0.0, 1.2, 0.0, 0.0, 1.0, 0.0, 1.0)

    @test HyQMOM.RIEMANN_SOLVER[] === :hll          # default is HLL

    HyQMOM.RIEMANN_SOLVER[] = :hll
    Fh = face_flux_1d(ML, MR, 1, 2.0)
    HyQMOM.RIEMANN_SOLVER[] = :rusanov
    Fr = face_flux_1d(ML, MR, 1, 2.0)
    @test all(isfinite, Fh) && all(isfinite, Fr)
    @test !isapprox(Fh, Fr)                         # genuinely different flux on a jump

    # Consistency: on a uniform state (L == R) every flux returns the physical flux.
    Mu = InitializeM4_35(1.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    HyQMOM.RIEMANN_SOLVER[] = :hll
    Fu_h = face_flux_1d(Mu, Mu, 1, 0.0)
    HyQMOM.RIEMANN_SOLVER[] = :rusanov
    Fu_r = face_flux_1d(Mu, Mu, 1, 0.0)
    @test isapprox(Fu_h, Fu_r; atol=1e-12)

    # Unknown selector is a hard error.
    HyQMOM.RIEMANN_SOLVER[] = :bogus
    @test_throws ArgumentError face_flux_1d(ML, MR, 1, 2.0)

    HyQMOM.RIEMANN_SOLVER[] = :hll                   # reset; don't leak global state
end

@testset "hllc contact speed" begin
    using HyQMOM: hllc_contact_speed, realize_and_speed, realizable_3D_M4
    Mu = InitializeM4_35(1.0, 0.37, 0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    Mr,sL,sR = realize_and_speed(Mu, 1, 0.0)
    # uniform state: contact speed == the bulk normal velocity
    @test isapprox(hllc_contact_speed(Mr, Mr, sL, sR, 1), 0.37; atol=1e-10)
    # bracketed by the HLL wave speeds
    ML = realizable_3D_M4(InitializeM4_35(1.0, 0.5,0,0,1.0,0,0,1,0,1), 2.0)
    MR = realizable_3D_M4(InitializeM4_35(0.3,-0.4,0,0,1.2,0,0,1,0,1), 2.0)
    MLr,lL,_ = realize_and_speed(ML,1,2.0); MRr,_,lR = realize_and_speed(MR,1,2.0)
    s = hllc_contact_speed(MLr, MRr, min(lL,lR), max(lL,lR), 1)
    @test min(lL,lR) <= s <= max(lL,lR)
end

@testset "hllc flux branch (face_flux_1d)" begin
    HyQMOM.RIEMANN_SOLVER[] = :hllc
    # Consistency: uniform state returns the physical flux (atol 1e-10), all axes.
    # Axes 2 and 3 use hard-coded indices 6 and 16 in hllc_star/_NMOM; test all three
    # so a per-axis index transcription bug cannot pass silently.
    Mu = InitializeM4_35(1.0, 0.25, 0, 0, 1.0, 0, 0, 1, 0, 1)
    for axis in (1,2,3)
        @test isapprox(face_flux_1d(Mu, Mu, axis, 0.0),
                       HyQMOM._phys_flux(HyQMOM.realizable_3D_M4(Mu, 0.0), axis); atol=1e-10)
    end
    # Finite on a generic jump state.
    ML = InitializeM4_35(1.0,  0.5, 0, 0, 1.0, 0, 0, 1, 0, 1)
    MR = InitializeM4_35(0.3, -0.4, 0, 0, 1.2, 0, 0, 1, 0, 1)
    @test all(isfinite, face_flux_1d(ML, MR, 1, 2.0))
    # Near-vacuum Ma=100 pair: realizability fallback must keep flux finite.
    MLv = InitializeM4_35(1.0,   60.0, 0, 0, 1.0, 0, 0, 1, 0, 1)
    MRv = InitializeM4_35(1e-5, -60.0, 0, 0, 1.0, 0, 0, 1, 0, 1)
    @test all(isfinite, face_flux_1d(MLv, MRv, 1, 100.0))
    HyQMOM.RIEMANN_SOLVER[] = :hll   # reset — don't leak global state
end

@testset "hllc star states" begin
    using HyQMOM: hllc_star, hllc_star_pair, hllc_flux, hllc_contact_speed,
                  realize_and_speed, realizable_3D_M4, _phys_flux, is_realizable
    ML = realizable_3D_M4(InitializeM4_35(1.0, 0.5,0,0,1.0,0,0,1,0,1), 2.0)
    MR = realizable_3D_M4(InitializeM4_35(0.3,-0.4,0,0,1.2,0,0,1,0,1), 2.0)
    MLr,lL,_ = realize_and_speed(ML,1,2.0); MRr,_,lR = realize_and_speed(MR,1,2.0)
    sL=min(lL,lR); sR=max(lL,lR); SM=hllc_contact_speed(MLr,MRr,sL,sR,1)

    # Per-side kinetic star states preserve the central-moment structure (density
    # rescale + normal velocity -> S_M), hence are realizable whenever the input is.
    UsL=hllc_star(MLr,sL,SM,1); UsR=hllc_star(MRr,sR,SM,1)
    @test is_realizable(UsL) && is_realizable(UsR)
    # the kinetic star moves the normal mean velocity onto the contact speed
    @test isapprox(UsL[2]/UsL[1], SM; rtol=1e-10)
    @test isapprox(UsR[2]/UsR[1], SM; rtol=1e-10)

    # Consistency-exact star PAIR: couples both sides through the HLL average.
    UpL, UpR = hllc_star_pair(MLr,MRr,sL,sR,SM,1)
    Uhll = (sR.*MRr .- sL.*MLr .- (_phys_flux(MRr,1).-_phys_flux(MLr,1)))./(sR-sL)
    # (2) HLL-consistency -- the binding integral constraint, to machine precision.
    @test isapprox(((SM-sL).*UpL .+ (sR-SM).*UpR)./(sR-sL), Uhll; rtol=1e-8)
    # (1) Rankine-Hugoniot across each acoustic wave: F*_K = F_K + sK (U*_K - M_K).
    FsL = _phys_flux(MLr,1) .+ sL.*(UpL .- MLr)
    FsR = _phys_flux(MRr,1) .+ sR.*(UpR .- MRr)
    # (3) contact closure / linearly-degenerate field: F*_R - F*_L = S_M (U*_R - U*_L).
    @test isapprox(FsR .- FsL, SM.*(UpR .- UpL); rtol=1e-8, atol=1e-10)
    # consistent pair is realizable for this physical input (else A3 falls back to HLL)
    @test is_realizable(UpL) && is_realizable(UpR)

    # hllc_flux: uniform state returns the physical flux (consistency of the solver).
    Mu = realizable_3D_M4(InitializeM4_35(1.0,0.25,0,0,1.0,0,0,1,0,1), 0.0)
    Mur,sLu,sRu = realize_and_speed(Mu,1,0.0)
    SMu = hllc_contact_speed(Mur,Mur,sLu,sRu,1)
    @test isapprox(hllc_flux(Mur,Mur,sLu,sRu,SMu,1), _phys_flux(Mur,1); atol=1e-10)
    @test all(isfinite, hllc_flux(MLr,MRr,sL,sR,SM,1))
    # the star flux used by hllc_flux matches the RH star flux of the contacted side
    Fh = hllc_flux(MLr,MRr,sL,sR,SM,1)
    @test isapprox(Fh, SM>=0 ? FsL : FsR; rtol=1e-8, atol=1e-10)

    # HLL-consistency for all three axes using a generic state with non-trivial v and w.
    # hllc_star_pair uses hard-coded per-axis indices (2,6,16); looping all three catches
    # any y/z transcription bug that the axis=1-only checks above would miss.
    MgL = realizable_3D_M4(InitializeM4_35(1.0, 0.4, 0.3, 0.2, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0), 2.0)
    MgR = realizable_3D_M4(InitializeM4_35(0.4,-0.3,-0.2,-0.1, 1.2, 0.0, 0.0, 1.0, 0.0, 1.0), 2.0)
    for axis in (1,2,3)
        MgLr, lgL, _ = realize_and_speed(MgL, axis, 2.0)
        MgRr, _, lgR = realize_and_speed(MgR, axis, 2.0)
        sgL = min(lgL, lgR); sgR = max(lgL, lgR)
        SMg = hllc_contact_speed(MgLr, MgRr, sgL, sgR, axis)
        UpgL, UpgR = hllc_star_pair(MgLr, MgRr, sgL, sgR, SMg, axis)
        Uhllg = (sgR.*MgRr .- sgL.*MgLr .-
                 (_phys_flux(MgRr,axis) .- _phys_flux(MgLr,axis))) ./ (sgR - sgL)
        @test isapprox(((SMg-sgL).*UpgL .+ (sgR-SMg).*UpgR) ./ (sgR-sgL),
                       Uhllg; rtol=1e-8)
    end
end

@testset "ld_eigvecs (B1: linearly-degenerate eigenvectors)" begin
    using HyQMOM: ld_eigvecs, _phys_flux, _NMOM, realizable_3D_M4
    using LinearAlgebra

    Mr = realizable_3D_M4(InitializeM4_35(1.0,0.3,-0.1,0.05,1.2,0.0,0.0,1.1,0.0,0.9), 2.0)

    # Reference Jacobian via central finite differences of the physical flux.
    function _ref_A(M, axis)
        n = length(M); F0 = _phys_flux(M, axis); A = zeros(n, n)
        for j in 1:n
            h = 1e-6 * max(abs(M[j]), 1.0)
            Mp = copy(M); Mp[j] += h; Mm = copy(M); Mm[j] -= h
            A[:, j] = (_phys_flux(Mp, axis) .- _phys_flux(Mm, axis)) ./ (2h)
        end
        return A
    end

    for axis in 1:3
        R, L, lam = ld_eigvecs(Mr, axis, 2.0)
        A = _ref_A(Mr, axis)
        un = Mr[_NMOM[axis]] / Mr[1]
        k = length(lam)
        @test k >= 1
        @test size(R, 1) == 35 && size(R, 2) == k
        @test size(L, 1) == k && size(L, 2) == 35

        # (1) eigenvector residual: A*R[:,j] ≈ lam[j]*R[:,j]
        for j in 1:k
            r = A * R[:, j] .- lam[j] .* R[:, j]
            @test norm(r) / norm(R[:, j]) < 1e-8
        end
        # (2) biorthonormality on the LD subspace
        @test isapprox(L * R, Matrix{Float64}(I, k, k); atol=1e-8)
        # (3) contact: at least one eigenvalue ≈ u_n
        @test any(x -> isapprox(x, un; atol=1e-6), lam)
        # all returned LD eigenvalues are the material speed u_n
        @test all(x -> isapprox(x, un; atol=1e-6), lam)
    end
end

@testset "hllem flux (B2: anti-diffusion + :hllem branch)" begin
    using HyQMOM: realize_and_speed, realizable_3D_M4, _phys_flux

    # --- Uniform state (L == R): HLLEM == physical flux == HLL (anti-diffusion -> 0)
    Mu = InitializeM4_35(1.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    Mur = realizable_3D_M4(Mu, 0.0)
    Fphys = _phys_flux(realize_and_speed(Mur, 1, 0.0)[1], 1)
    HyQMOM.RIEMANN_SOLVER[] = :hll
    Fu_hll = face_flux_1d(Mu, Mu, 1, 0.0)
    HyQMOM.RIEMANN_SOLVER[] = :hllem
    Fu_em = face_flux_1d(Mu, Mu, 1, 0.0)
    @test isapprox(Fu_em, Fu_hll; atol=1e-10)
    @test isapprox(Fu_em, Fphys; atol=1e-10)

    # --- Reduces to HLL on one-sided fans (sL>=0 or sR<=0): supersonic right-going
    # Choose a strongly right-moving pair so both wave speeds are positive (sL>=0).
    MLs = InitializeM4_35(1.0, 5.0, 0,0, 0.05, 0,0, 0.05, 0, 0.05)
    MRs = InitializeM4_35(0.9, 5.0, 0,0, 0.05, 0,0, 0.05, 0, 0.05)
    MLr_s, lL_s, _ = realize_and_speed(realizable_3D_M4(MLs, 2.0), 1, 2.0)
    MRr_s, lL2_s, lR_s = realize_and_speed(realizable_3D_M4(MRs, 2.0), 1, 2.0)
    if min(lL_s, lL2_s) >= 0      # genuinely supersonic case
        HyQMOM.RIEMANN_SOLVER[] = :hll
        Fhll_s = face_flux_1d(MLs, MRs, 1, 2.0)
        HyQMOM.RIEMANN_SOLVER[] = :hllem
        Fem_s = face_flux_1d(MLs, MRs, 1, 2.0)
        @test isapprox(Fem_s, Fhll_s; atol=1e-12)
    end

    # --- Genuinely anti-diffusive: contact/jump with sL<0<sR -> HLLEM != HLL.
    ML = InitializeM4_35(1.0,  0.3, 0.1, -0.05, 1.0, 0.0, 0.0, 1.1, 0.0, 0.9)
    MR = InitializeM4_35(0.6, -0.2, -0.1, 0.05, 1.2, 0.0, 0.0, 0.9, 0.0, 1.1)
    HyQMOM.RIEMANN_SOLVER[] = :hll
    F_hll = face_flux_1d(ML, MR, 1, 2.0)
    HyQMOM.RIEMANN_SOLVER[] = :hllem
    F_em = face_flux_1d(ML, MR, 1, 2.0)
    @test all(isfinite, F_em)
    @test !isapprox(F_em, F_hll)        # anti-diffusion term is nonzero (THE point)

    # --- Finite + realizable in near-vacuum Ma=100 collision (guard/fallback holds).
    C200=1e-4
    MLv = InitializeM4_35(1.0,   60.0, 0,0, C200, 0,0, C200, 0, C200)
    MRv = InitializeM4_35(1e-5, -60.0, 0,0, C200, 0,0, C200, 0, C200)
    HyQMOM.RIEMANN_SOLVER[] = :hllem
    for ax in 1:3
        Fv = face_flux_1d(MLv, MRv, ax, 100.0)
        @test all(isfinite, Fv)
    end

    HyQMOM.RIEMANN_SOLVER[] = :hll          # reset; don't leak global state
end
