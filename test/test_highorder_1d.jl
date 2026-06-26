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

@testset "SSP-RK3 order" begin
    # scalar ODE dy/dt = -y, y(0)=1, exact y(T)=exp(-T)
    L(y) = -y
    T = 1.0
    err(n) = (dt = T/n; y = 1.0; for _ in 1:n; y = ssp_rk3_step(y, dt, L); end; abs(y - exp(-T)))
    e1 = err(10); e2 = err(20)
    @test e2 < e1
    @test log2(e1/e2) > 2.7   # ~3rd-order convergence
end

# advance a 1D moment field with outflow (zero-gradient) BCs; helper used by the tests below
function _advance_1d(Mline, dx, dt, nsteps, Ma)
    L(M) = residual_1d(M, dx, Ma; order=2)
    for _ in 1:nsteps
        Mline = ssp_rk3_step(Mline, dt, L)
    end
    return Mline
end

@testset "1D smooth order-of-accuracy" begin
    # Smooth sinusoidal density on a periodic domain; measure L1 self-convergence.
    # Periodic BC is required so boundary errors don't pollute the order study.
    # minmod limiter clips at the sine extrema, so the measured L1 rate sits slightly
    # below the formal 2.0; this verifies the scheme is genuinely 2nd-order-convergent.
    Ma = 0.0; tfinal = 0.05

    function run_periodic(N; order=2)
        dx = 1.0/N
        Mline = zeros(N, 35)
        for i in 1:N
            x = (i-0.5)*dx
            rho = 1.0 + 0.2*sin(2pi*x)
            Mline[i, :] = InitializeM4_35(rho, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        end
        dt = 0.15*dx/4.5
        nsteps = ceil(Int, tfinal/dt); dt = tfinal/nsteps
        L(M) = residual_1d(M, dx, Ma; order=order, bc=:periodic)
        for _ in 1:nsteps
            Mline = ssp_rk3_step(Mline, dt, L)
        end
        return Mline
    end

    # L1 self-convergence: compare fine grid coarsened to coarse grid
    coarsen(a) = (a[1:2:end] .+ a[2:2:end]) ./ 2
    d2(N; order=2) = run_periodic(N; order=order)[:, 1]
    e(Nc; order=2) = sum(abs.(coarsen(d2(2*Nc; order=order)) .- d2(Nc; order=order))) / Nc

    # order=2: expected rate ~1.86 (measured); assert > 1.6
    e32_2 = e(32; order=2); e64_2 = e(64; order=2)
    rate2 = log2(e32_2 / e64_2)
    @test rate2 > 1.6   # 2nd-order MUSCL with periodic BC

    # order=1: expected rate ~1.0; assert < 1.2 to confirm it does NOT match 2nd order
    e32_1 = e(32; order=1); e64_1 = e(64; order=1)
    rate1 = log2(e32_1 / e64_1)
    @test rate1 < 1.2   # 1st-order upwind

    # order=2 must be strictly higher order than order=1
    @test rate2 > rate1
end

@testset "1D realizability + conservation (shock tube)" begin
    Ma = 0.0; N = 100; dx = 1.0/N
    Ml = InitializeM4_35(1.0,   0.0,0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    Mr = InitializeM4_35(0.125, 0.0,0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    Mline = zeros(N, 35)
    for i in 1:N
        Mline[i, :] = (i <= N÷2) ? Ml : Mr
    end
    mass0 = sum(Mline[:, 1])
    dt = 0.2*dx/3.0; nsteps = 40
    Mline = _advance_1d(Mline, dx, dt, nsteps, Ma)
    @test all(isfinite, Mline)
    @test minimum(Mline[:, 1]) > 0                     # density positive
    # realizable: variances positive everywhere
    for i in 1:N
        _, S4 = M2CS4_35(Mline[i, :])
        @test (S4[5]-S4[4]^2-1) > -1e-8                # H200 >= 0 (x)
    end
    @test abs(sum(Mline[:, 1]) - mass0) / mass0 < 1e-12  # mass conserved (no through-flow @ walls)
end

@testset "1D smooth order-of-accuracy with limiter (theta=1 everywhere)" begin
    # On a smooth periodic density sinusoid there are no realizability violations, so
    # scaling_limited_faces returns theta=1 for every cell. The limiter-on path must
    # therefore produce the same ~2nd-order convergence as the standard MUSCL path.
    # We verify: (a) the limiter is inactive (theta<1 fraction ≈ 0) on smooth data,
    # and (b) the observed L1 self-convergence rate is > 1.8 with use_limiter=true.
    Ma = 0.0; tfinal = 0.05

    function run_lim(N; use_limiter=true)
        dx = 1.0/N
        Mline = zeros(N, 35)
        for i in 1:N
            x = (i-0.5)*dx
            rho = 1.0 + 0.2*sin(2pi*x)
            Mline[i, :] = InitializeM4_35(rho, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        end
        dt = 0.15*dx/10.5     # conservative CFL: wave speed ≲ 10.5 for T=1, u=1
        nsteps = ceil(Int, tfinal/dt); dt = tfinal/nsteps
        L(M) = residual_1d(M, dx, Ma; order=2, bc=:periodic, use_limiter=use_limiter)
        for _ in 1:nsteps
            Mline = ssp_rk3_step(Mline, dt, L)
        end
        return Mline
    end

    coarsen(a) = [(a[2i-1]+a[2i])/2 for i in 1:length(a)÷2]

    # (a) theta-locality: on the smooth IC at t=0, theta must be 1 everywhere
    Nc_diag = 128
    Msmooth = zeros(Nc_diag, 35)
    for i in 1:Nc_diag
        x = (i-0.5)/Nc_diag
        rho = 1.0 + 0.2*sin(2pi*x)
        Msmooth[i, :] = InitializeM4_35(rho, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    end
    Vc = [to_recon_vars(@view Msmooth[i, :]) for i in 1:Nc_diag]
    wrap(i) = mod(i-1, Nc_diag) + 1
    thetas  = [begin _, _, θ = scaling_limited_faces(Vc[wrap(i-1)], Vc[i], Vc[wrap(i+1)]); θ end
               for i in 1:Nc_diag]
    frac_below1 = count(θ -> θ < 1.0 - 1e-14, thetas) / Nc_diag
    @test frac_below1 < 0.01   # limiter must be inactive on smooth data

    # (b) L1 self-convergence rate > 1.8 with use_limiter=true
    d32  = run_lim(32)[:, 1]; d64  = run_lim(64)[:, 1]
    d128 = run_lim(128)[:, 1]
    e1 = sum(abs.(coarsen(d64)  .- d32))  / 32
    e2 = sum(abs.(coarsen(d128) .- d64))  / 64
    rate = log2(e1 / e2)
    @test rate > 1.8   # smooth 2nd-order: limiter must NOT clip theta on smooth data
end

@testset "1D high-order stays realizable through transport (Ma=100 analog)" begin
    Nc = 256
    Mline = zeros(Nc, 35)
    for i in 1:Nc                       # two dense slabs streaming into a near-vacuum gap
        x = (i - 0.5)/Nc
        if x < 0.4
            Mline[i,:] = InitializeM4_35(1.0,  50.0, 0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
        elseif x > 0.6
            Mline[i,:] = InitializeM4_35(1.0, -50.0, 0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
        else
            Mline[i,:] = InitializeM4_35(1e-4, 0.0, 0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
        end
    end
    # OPT-IN limiter path stays finite through the vacuum band:
    R = residual_1d(Mline, 1.0/Nc, 100.0; order=2, bc=:outflow, use_limiter=true)
    @test all(isfinite, R)              # no Inf/NaN escapes when the limiter is on
    # DEFAULT path (use_limiter=false) is unchanged: identical to the call without the kw.
    R_def  = residual_1d(Mline, 1.0/Nc, 100.0; order=2, bc=:outflow, use_limiter=false)
    R_base = residual_1d(Mline, 1.0/Nc, 100.0; order=2, bc=:outflow)
    @test isequal(R_def, R_base)        # default == pre-existing behavior, byte-identical
end
