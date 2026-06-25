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

@testset "HLL face flux consistency" begin
    M = InitializeM4_35(1.0, 0.3, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    # uniform L==R: HLL flux must equal the physical x-flux of M
    Fhat = face_flux_1d(copy(M), copy(M), 1, 0.0)
    Fx, _, _ = Flux_closure35_3D(M)
    @test Fhat ≈ Fx atol=1e-10 rtol=1e-10
    @test length(Fhat) == 35
end

@testset "1D residual" begin
    Ncell = 16
    M0 = InitializeM4_35(1.0, 0.2, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    # uniform field -> zero residual (interior)
    Mline = repeat(reshape(M0,1,35), Ncell, 1)
    R = residual_1d(Mline, 0.1, 0.0; order=2)
    @test maximum(abs.(R[3:Ncell-2, :])) < 1e-9
    @test size(R) == (Ncell, 35)

    # order=1 path: uniform field also gives zero interior residual
    R1 = residual_1d(Mline, 0.1, 0.0; order=1)
    @test maximum(abs.(R1[3:Ncell-2, :])) < 1e-9

    # gradient-field test: smooth density ramp exercises MUSCL (order=2)
    N = 16
    dx = 1.0 / N
    Mgrad = zeros(N, 35)
    for i in 1:N
        rho_i = 1.0 + 0.3*(i-1)/(N-1)
        Mgrad[i, :] = InitializeM4_35(rho_i, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    end
    Rg = residual_1d(Mgrad, dx, 0.0; order=2)
    # (a) all values must be finite
    @test all(isfinite, Rg)
    # (b) interior residual is NOT near zero — scheme responds to the gradient
    @test maximum(abs.(Rg[3:N-2, :])) > 1e-6
    # (c) density residual in the interior is finite and nonzero (transport of gradient)
    @test all(isfinite, Rg[3:N-2, 1])
    @test maximum(abs.(Rg[3:N-2, 1])) > 1e-6
end
