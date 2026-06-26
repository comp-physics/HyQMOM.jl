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
