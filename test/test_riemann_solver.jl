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
    # Consistency: uniform state returns the physical flux (atol 1e-10).
    Mu = InitializeM4_35(1.0, 0.25, 0, 0, 1.0, 0, 0, 1, 0, 1)
    @test isapprox(face_flux_1d(Mu, Mu, 1, 0.0),
                   HyQMOM._phys_flux(HyQMOM.realizable_3D_M4(Mu, 0.0), 1); atol=1e-10)
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
end
