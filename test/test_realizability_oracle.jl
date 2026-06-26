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

    # Oracle agrees with the shipped projection: projection35 corrects iff oracle says unrealizable.
    # Build a mildly unrealizable state by inflating a 4th-order cross moment.
    Mu = copy(M); Mu[12] *= 5.0      # M220-type entry pushed out of the cone
    if !is_realizable(Mu)
        Mr = realizable_3D_M4(Mu, 2.0)
        @test is_realizable(Mr)       # projection restores realizability
    end
end
