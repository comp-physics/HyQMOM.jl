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

    @testset "4. non-negativity and recovery over skewed realizable sweep" begin
        # Build raw moments from standardized (q, η), mean μ, variance σ², density ρ.
        # η ≥ q²+1 is the Hamburger realizability bound for a 5-moment sequence.
        # M0=ρ; M1=ρμ; M2=ρ(μ²+σ²); M3=ρ(μ³+3μσ²+q·σ³); M4=ρ(μ⁴+6μ²σ²+4μqσ³+ησ⁴)
        function skewed_moments(ρ, μ, σ², q, η)
            σ = sqrt(max(σ², 0.0))
            M0 = ρ
            M1 = ρ * μ
            M2 = ρ * (μ^2 + σ²)
            M3 = ρ * (μ^3 + 3*μ*σ² + q*σ^3)
            M4 = ρ * (μ^4 + 6*μ^2*σ² + 4*μ*q*σ^3 + η*σ^4)
            return [M0, M1, M2, M3, M4]
        end

        # Deterministic LCG — no RNG dependency
        seed = UInt64(12345)
        nextrand() = (seed = (seed * 6364136223846793005 + 1442695040888963407) % UInt64(2)^63;
                      Float64(seed) / Float64(UInt64(2)^63))

        nbad    = 0
        Ns_seen = Set{Int}()

        # 300 cases: 200 fully realizable (N=3 expected), 60 below the N=3
        # realizability bound with σ²>0 (N=2 expected), 40 cold/vacuum (N=1 expected).
        for i in 1:300
            if i <= 200
                # Fully realizable: η well above q²+1 → expect N=3
                ρ  = 0.1 + 2.9 * nextrand()
                μ  = -3.0 + 6.0 * nextrand()
                σ² = 1e-2 + 3.0 * nextrand()
                q  = -2.0 + 4.0 * nextrand()
                η  = q^2 + 1.1 + 5.0 * nextrand()
                m  = skewed_moments(ρ, μ, σ², q, η)
            elseif i <= 260
                # η below q²+1 but σ²>0 → N=3 rejected by negative Vandermonde
                # weights, N=2 fallback expected
                ρ  = 0.1 + 1.9 * nextrand()
                μ  = -2.0 + 4.0 * nextrand()
                σ² = 1e-3 + 1.0 * nextrand()
                q  = -3.0 + 6.0 * nextrand()
                η  = max(1.01, q^2 + 1.0 - 0.5 - 2.0 * nextrand())
                m  = skewed_moments(ρ, μ, σ², q, η)
            else
                # Cold / vacuum: σ²→0 → N=1 monokinetic
                ρ  = 0.1 + 1.9 * nextrand()
                μ  = -2.0 + 4.0 * nextrand()
                σ² = 1e-16 * nextrand()
                q  = -1.0 + 2.0 * nextrand()
                η  = q^2 + 1.0 + nextrand()
                m  = skewed_moments(ρ, μ, σ², q, η)
            end

            w, x, N = hyqmom_quadrature_1d(m)
            push!(Ns_seen, N)

            # (a) all weights ≥ -1e-12
            all(wi -> wi >= -1e-12, w) || (nbad += 1)

            # (b) moment recovery to supported order (N=3→M0..M4, N=2→M0..M3, N=1→M0,M1)
            kmax = N == 3 ? 4 : (N == 2 ? 3 : 1)
            rec  = recover(w, x, kmax)
            isapprox(rec, m[1:kmax+1]; rtol=1e-9, atol=1e-9) || (nbad += 1)
        end

        @test nbad == 0
        # (c) sweep must exercise the full adaptive ladder
        @test 1 ∈ Ns_seen   # N=1 (cold/vacuum) hit at least once
        @test 2 ∈ Ns_seen   # N=2 (reduction from N=3) hit at least once
        @test 3 ∈ Ns_seen   # N=3 (full rule) hit at least once
    end
end
