# test/test_realizability_oracle.jl
using Test
using HyQMOM

@testset "realizability oracle" begin
    # An equilibrium Maxwellian moment vector is strictly realizable.
    M = InitializeM4_35(1.0, 0.2, -0.1, 0.05, 1.3, 0.0, 0.0, 1.1, 0.0, 0.9)
    @test is_realizable(M)
    @test realizability_margin(M) > 0

    # Non-finite / nonpositive density / negative variance are rejected.
    Mbad = copy(M); Mbad[1] = -1.0
    @test !is_realizable(Mbad)
    @test realizability_margin(Mbad) == -Inf
    Mnan = copy(M); Mnan[5] = NaN
    @test !is_realizable(Mnan)

    # Oracle agrees with the shipped projection: a grossly unrealizable state is
    # flagged, and the projection restores realizability. NOTE: the Appendix B
    # projection lands the state ON the realizable boundary (its target has
    # |<p2 p2'>| = 0, so the smallest eigenvalue is ~0), where the eigenvalue's
    # sign is LAPACK/platform-dependent. So assert the margin is restored to within
    # a small FP tolerance of the boundary (not strict >= 0) and that the projection
    # strictly improved it -- both platform-robust. (Julia 1.9's LAPACK returns a
    # tiny negative margin here; a strict is_realizable check is not portable.)
    Mu = copy(M); Mu[12] *= 5.0      # grossly inflate an M220-type cross moment
    @test !is_realizable(Mu)         # oracle detects the unrealizable state
    Mr = realizable_3D_M4(Mu, 2.0)   # shipped Appendix B projection
    @test realizability_margin(Mr) > realizability_margin(Mu)   # projection improved it
    @test realizability_margin(Mr) > -1e-8                       # restored to the boundary
end
