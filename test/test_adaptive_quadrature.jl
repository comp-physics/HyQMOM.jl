using Test
using HyQMOM
using LinearAlgebra

# Build 1D raw moments [M0..M4] of a Gaussian N(u, T) scaled by density rho.
gaussian_moments(rho, u, T) = [
    rho,
    rho * u,
    rho * (u^2 + T),
    rho * (u^3 + 3u*T),
    rho * (u^4 + 6u^2*T + 3T^2),
]

# Reconstruct raw moments up to order kmax from a quadrature (w, u).
recover(w, u, kmax) = [sum(w .* (u .^ k)) for k in 0:kmax]

@testset "adaptive 1D HyQMOM quadrature" begin

    @testset "1. N=3 moment recovery (Gaussian), w>0" begin
        for (rho, u, T) in ((1.0, 0.0, 1.0), (2.3, -0.7, 0.5), (0.4, 1.2, 2.0))
            m = gaussian_moments(rho, u, T)
            w, x, N = hyqmom_quadrature_1d(m)
            @test N == 3
            @test length(w) == 3 && length(x) == 3
            @test all(w .> 0)
            rec = recover(w, x, 4)
            @test rec ≈ m rtol=1e-10 atol=1e-10
        end
    end

    @testset "2. adaptive reduction on non-realizable N=3" begin
        # Start from a Gaussian, then drag M4 below the realizability bound
        # (eta < q^2 + 1) so the full 3-node quadrature would have w<0.
        m = gaussian_moments(1.0, 0.0, 1.0)
        m[5] *= 0.3   # deflate M4 -> kurtosis below bound, N=3 not realizable
        w, x, N = hyqmom_quadrature_1d(m)
        @test N < 3
        @test all(w .>= 0)
        # Must still recover moments up to the order N supports:
        # N=2 -> M0..M3, N=1 -> M0..M1
        kmax = N == 2 ? 3 : 1
        rec = recover(w, x, kmax)
        @test rec ≈ m[1:kmax+1] rtol=1e-10 atol=1e-10
    end

    @testset "3. deep vacuum / cold -> N=1 monokinetic" begin
        rho = 1.5; u = 0.3
        # variance -> 0 : M2 = M1^2/M0 (sigma^2 = 0)
        m = [rho, rho*u, rho*u^2, rho*u^3, rho*u^4]
        w, x, N = hyqmom_quadrature_1d(m)
        @test N == 1
        @test w ≈ [rho] atol=1e-12
        @test x ≈ [u] atol=1e-12
    end

    @testset "4. non-negativity over random realizable sweep" begin
        # deterministic LCG, no RNG dependency
        seed = UInt64(12345)
        nextrand() = (seed = (seed*6364136223846793005 + 1442695040888963407) % UInt64(2)^63;
                      Float64(seed) / Float64(UInt64(2)^63))
        nbad = 0
        for _ in 1:2000
            rho = 0.1 + 2.0*nextrand()
            u   = -2.0 + 4.0*nextrand()
            T   = 1e-3 + 3.0*nextrand()
            m = gaussian_moments(rho, u, T)
            w, x, N = hyqmom_quadrature_1d(m)
            all(w .>= -1e-14) || (nbad += 1)
            # recovery to supported order
            kmax = N == 3 ? 4 : (N == 2 ? 3 : 1)
            rec = recover(w, x, kmax)
            isapprox(rec, m[1:kmax+1]; rtol=1e-8, atol=1e-8) || (nbad += 1)
        end
        @test nbad == 0
    end
end
