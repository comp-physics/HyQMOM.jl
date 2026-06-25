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

@testset "MUSCL limiter + faces" begin
    @test minmod(2.0, 3.0) == 2.0
    @test minmod(-2.0, 3.0) == 0.0
    @test minmod(-2.0, -5.0) == -2.0
    # On a LINEAR field, minmod returns the exact slope (2nd-order, no clamping)
    Vm1 = fill(1.0, 35); V0 = fill(2.0, 35); Vp1 = fill(3.0, 35)
    s = muscl_slopes(Vm1, V0, Vp1)
    @test all(s .≈ 1.0)
    Vminus, Vplus = muscl_faces(Vm1, V0, Vp1)
    @test all(Vminus .≈ 1.5) && all(Vplus .≈ 2.5)
    # At a local MAX, limiter clamps slope to 0 (1st-order, TVD)
    s2 = muscl_slopes(fill(1.0,35), fill(3.0,35), fill(1.0,35))
    @test all(s2 .== 0.0)
end
