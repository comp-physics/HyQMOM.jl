using Test
using HyQMOM
using LinearAlgebra

@testset "recon-vars bijection" begin
    # realizable moment vectors from particle samples
    function sample_M(seed)
        # deterministic pseudo-particles (no RNG: fixed lattice + shift)
        rho = 0.7 + 0.1*seed
        u0, v0, w0 = 0.1*seed, -0.05*seed, 0.02*seed
        T = 1.0 + 0.1*seed
        return InitializeM4_35(rho, u0, v0, w0, T, 0.0, 0.0, T, 0.0, T)
    end
    for s in 1:5
        M = sample_M(s)
        V = to_recon_vars(M)
        @test length(V) == 35
        M2 = from_recon_vars(V)
        @test M2 ≈ M atol=1e-10 rtol=1e-10
    end
end
