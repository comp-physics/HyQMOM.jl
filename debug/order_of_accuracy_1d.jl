# Order-of-accuracy validation for the 1D high-order scheme with realizability limiter.
#
# Measures L1 self-convergence on a smooth periodic density sinusoid (no vacuum) and
# verifies the Zhang--Shu limiter is inactive (theta=1 everywhere) on smooth data but
# active (theta<1) in the vacuum band of a colliding-slab problem.
#
#   HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. debug/order_of_accuracy_1d.jl
#
# Expected outputs:
#   - Smooth L1 self-convergence rate ~1.86 (order=2 with limiter ON)
#   - theta<1 fraction ~0% on smooth IC
#   - theta<1 fraction confined to vacuum band on colliding-slab IC

ENV["HYQMOM_SKIP_PLOTTING"] = "true"
ENV["CI"] = "true"

using HyQMOM, Printf

# ---------------------------------------------------------------------------
# Smooth periodic IC: ρ = 1 + 0.2 sin(2πx), u=1, isotropic T=1
# ---------------------------------------------------------------------------
function smooth_ic(Nc)
    dx = 1.0 / Nc
    M  = zeros(Nc, 35)
    for i in 1:Nc
        x     = (i - 0.5) * dx
        rho   = 1.0 + 0.2 * sinpi(2x)
        M[i, :] = InitializeM4_35(rho, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    end
    return M
end

# Colliding slabs + near-vacuum (1D analog of the 3D Ma-100 problem)
function colliding_ic(Nc; Ma=50.0)
    dx = 1.0 / Nc
    M  = zeros(Nc, 35)
    Uc = Ma / sqrt(2.0)
    for i in 1:Nc
        x = (i - 0.5) * dx
        if x < 0.4
            M[i, :] = InitializeM4_35(1.0,  Uc,  0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        elseif x > 0.6
            M[i, :] = InitializeM4_35(1.0, -Uc,  0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        else
            M[i, :] = InitializeM4_35(1e-4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        end
    end
    return M
end

# ---------------------------------------------------------------------------
# theta-locality diagnostic
# ---------------------------------------------------------------------------
# theta check uses outflow BC (wrap = clamp) to match colliding IC geometry
function theta_fraction_below1(M; periodic=false)
    Nc     = size(M, 1)
    clamp_bc(i) = clamp(i, 1, Nc)
    wrap_bc(i)  = mod(i - 1, Nc) + 1
    idx   = periodic ? wrap_bc : clamp_bc
    Vc    = [to_recon_vars(@view M[i, :]) for i in 1:Nc]
    count = 0
    for i in 1:Nc
        _, _, θ = scaling_limited_faces(Vc[idx(i-1)], Vc[i], Vc[idx(i+1)])
        count += (θ < 1.0 - 1e-14) ? 1 : 0
    end
    return count / Nc
end

# CFL-safe dt for colliding slabs
function cfl_dt_coll(M, dx)
    v = 0.0
    for i in 1:size(M, 1)
        r = M[i, 1]; r > 0 || continue
        u = M[i, 2] / r; c2 = max(M[i, 3] / r - u^2, 0.0)
        v = max(v, abs(u) + 4 * 2.334 * sqrt(c2 + 1e-12))
    end
    return 0.5 * dx / max(v, 1e-12)
end

println("=" ^ 60)
println("THETA-LOCALITY DIAGNOSTIC")
println("=" ^ 60)

Nc_diag = 256

println("\n[A] Smooth sinusoid IC (Nc=$Nc_diag, t=0)")
M_smooth = smooth_ic(Nc_diag)
frac_smooth = theta_fraction_below1(M_smooth; periodic=true)
@printf("    theta<1 fraction: %.4f  (expect = 0)\n", frac_smooth)

println("\n[B] Colliding slabs + near-vacuum (Nc=$Nc_diag, Ma=50)")
println("    NOTE: piecewise-constant IC has zero slopes everywhere (minmod clips to 0")
println("    at a discontinuity), so theta=1 trivially at t=0. Check after a few steps.")
frac_coll, n_vac = let
    Mc    = colliding_ic(Nc_diag; Ma=50.0)
    dx_c  = 1.0 / Nc_diag
    nw    = 10
    for _ in 1:nw
        dt_c = cfl_dt_coll(Mc, dx_c)
        Lc(U) = residual_1d(U, dx_c, 50.0; order=2, bc=:outflow, use_limiter=true)
        Mc    = ssp_rk3_step(Mc, dt_c, Lc)
        for i in 1:Nc_diag
            Mc[i, :] = realizable_3D_M4(Mc[i, :], 50.0)
        end
    end
    fc   = theta_fraction_below1(Mc; periodic=false)
    nv   = count(i -> Mc[i, 1] < 0.01, 1:Nc_diag)
    @printf("    theta<1 fraction (after %d steps): %.4f\n", nw, fc)
    @printf("    Low-density cells (rho<0.01): %d / %d  (%.1f%%)\n",
            nv, Nc_diag, 100.0*nv/Nc_diag)
    if fc > 0
        @printf("    Limiter fires in/near the vacuum band.\n")
    else
        @printf("    Limiter inactive (all faces still realizable after %d steps).\n", nw)
    end
    fc, nv
end

# ---------------------------------------------------------------------------
# Order-of-accuracy study: L1 self-convergence
# ---------------------------------------------------------------------------
println("\n" * "=" ^ 60)
println("ORDER-OF-ACCURACY STUDY (use_limiter=true, bc=:periodic)")
println("=" ^ 60)

const Ma_oa = 0.0
const tfinal = 0.05

function run_periodic(Nc; order=2, use_limiter=false)
    dx     = 1.0 / Nc
    M      = smooth_ic(Nc)
    # CFL: wave speed estimate ~ |u| + 4*max_quadrature_abscissa * sqrt(T) ≈ 1 + 4*2.334 ≈ 10.3
    cfl_fac = 0.15
    dt0    = cfl_fac * dx / 10.5
    nsteps = ceil(Int, tfinal / dt0)
    dt     = tfinal / nsteps
    L(U)   = residual_1d(U, dx, Ma_oa; order=order, bc=:periodic, use_limiter=use_limiter)
    for _ in 1:nsteps
        M = ssp_rk3_step(M, dt, L)
    end
    return M
end

# coarsen fine grid to coarse by local averaging
function coarsen_density(d_fine)
    Nc = length(d_fine) ÷ 2
    [(d_fine[2i-1] + d_fine[2i]) / 2 for i in 1:Nc]
end

# Run Nc = 32,64,128,256 (512 takes a few minutes; enable with env var if desired)
grids  = [32, 64, 128, 256]
if get(ENV, "OOA_FINE", "0") == "1"
    push!(grids, 512)
end

println("\nGrid sizes: ", grids)
println("\n  order=2, use_limiter=true:")

errors2_lim = Float64[]
for Nc in grids
    rho_c = run_periodic(Nc;   order=2, use_limiter=true)[:, 1]
    rho_f = run_periodic(2Nc;  order=2, use_limiter=true)[:, 1]
    err   = sum(abs.(coarsen_density(rho_f) .- rho_c)) / Nc
    push!(errors2_lim, err)
    @printf("    Nc=%4d  L1 err = %.4e\n", Nc, err)
end
println()
println("  Observed convergence rates (use_limiter=true):")
for k in 1:length(grids)-1
    r = log2(errors2_lim[k] / errors2_lim[k+1])
    @printf("    Nc=%d->%d : rate = %.3f\n", grids[k], grids[k+1], r)
end

println("\n  order=2, use_limiter=false (default MUSCL path):")
errors2_def = Float64[]
for Nc in grids
    rho_c = run_periodic(Nc;  order=2, use_limiter=false)[:, 1]
    rho_f = run_periodic(2Nc; order=2, use_limiter=false)[:, 1]
    err   = sum(abs.(coarsen_density(rho_f) .- rho_c)) / Nc
    push!(errors2_def, err)
    @printf("    Nc=%4d  L1 err = %.4e\n", Nc, err)
end
println()
println("  Observed convergence rates (use_limiter=false):")
for k in 1:length(grids)-1
    r = log2(errors2_def[k] / errors2_def[k+1])
    @printf("    Nc=%d->%d : rate = %.3f\n", grids[k], grids[k+1], r)
end

println("\n  order=1 (upwind reference):")
errors1 = Float64[]
for Nc in grids
    rho_c = run_periodic(Nc;  order=1)[:, 1]
    rho_f = run_periodic(2Nc; order=1)[:, 1]
    err   = sum(abs.(coarsen_density(rho_f) .- rho_c)) / Nc
    push!(errors1, err)
    @printf("    Nc=%4d  L1 err = %.4e\n", Nc, err)
end
println()
println("  Observed convergence rates (order=1):")
for k in 1:length(grids)-1
    r = log2(errors1[k] / errors1[k+1])
    @printf("    Nc=%d->%d : rate = %.3f\n", grids[k], grids[k+1], r)
end

println()
println("=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
rate_lim_32_64 = log2(errors2_lim[1] / errors2_lim[2])
rate_lim_64_128 = log2(errors2_lim[2] / errors2_lim[3])
rate1_lo = log2(errors1[end-1] / errors1[end])
@printf("  Limiter-ON  rate (32->64):   %.3f\n", rate_lim_32_64)
@printf("  Limiter-ON  rate (64->128):  %.3f\n", rate_lim_64_128)
@printf("  Upwind rate (%d->%d):  %.3f\n", grids[end-1], grids[end], rate1_lo)
@printf("  theta<1 fraction (smooth):   %.4f\n", frac_smooth)
@printf("  theta<1 fraction (colliding): %.4f\n", frac_coll)
println()
if min(rate_lim_32_64, rate_lim_64_128) > 1.8
    println("  PASS: limiter-on smooth order > 1.8 (theta=1 everywhere on smooth data).")
else
    println("  CONCERN: limiter-on smooth order < 1.8 — limiter may be over-firing on smooth data.")
    println("           Check: frac_smooth should be 0.0 (was $frac_smooth)")
end
if frac_smooth < 0.01
    println("  PASS: theta<1 fraction on smooth IC is ~0 (limiter correctly inactive).")
else
    println("  CONCERN: theta<1 fraction on smooth IC is $(round(frac_smooth*100,digits=2))% — over-firing.")
end
@printf("  theta<1 fraction (colliding, after warm-up): %.4f\n", frac_coll)

# ---------------------------------------------------------------------------
# 1D SHARPNESS VS FIRST-ORDER (proxy for 3D Mach-ladder; 3D requires MPI/SLURM)
# ---------------------------------------------------------------------------
println()
println("=" ^ 60)
println("1D SHARPNESS: high-order+limiter vs first-order (colliding slabs)")
println("  (3D MPI case requires SLURM; deferred to HPC run)")
println("=" ^ 60)
let
    Nc_sharp = 128; Ma_s = 10.0; dx_s = 1.0 / Nc_sharp
    # Use tmax past slab collision; t_collision ~ 0.1/Uc ~ 0.014; run to ~0.025
    tmax_s = 0.025

    function run_sharp(order, use_lim)
        Mc = colliding_ic(Nc_sharp; Ma=Ma_s)
        t  = 0.0; nst = 0
        while t < tmax_s
            dt_s = let v = 0.0
                for i in 1:Nc_sharp
                    r = Mc[i,1]; r > 0 || continue
                    u = Mc[i,2]/r; c2 = max(Mc[i,3]/r - u^2, 0.0)
                    v = max(v, abs(u) + 4*2.334*sqrt(c2 + 1e-12))
                end
                min((1/3)*dx_s/max(v, 1e-12), tmax_s - t)
            end
            Ls(U) = residual_1d(U, dx_s, Ma_s; order=order, bc=:outflow,
                                use_limiter=use_lim)
            Mc = ssp_rk3_step(Mc, dt_s, Ls)
            for i in 1:Nc_sharp
                Mc[i, :] = realizable_3D_M4(Mc[i, :], Ma_s)
            end
            t += dt_s; nst += 1
            all(isfinite, @view Mc[:, 1]) || (println("  NON-FINITE at step $nst"); break)
        end
        Mc, t, nst
    end

    Mho, t_ho, n_ho  = run_sharp(2, true)
    Mfo, t_fo, n_fo  = run_sharp(1, false)
    rho_ho = Mho[:, 1]; rho_fo = Mfo[:, 1]
    peak_ho = maximum(rho_ho); peak_fo = maximum(rho_fo)
    @printf("  After ~%.4f (Ma=%.0f, Nc=%d):\n", t_ho, Ma_s, Nc_sharp)
    @printf("    peak rho: high-order+limiter = %.4f  |  first-order = %.4f\n", peak_ho, peak_fo)
    @printf("    ratio (ho/fo): %.3f  (expect > 1 => higher peak => sharper)\n", peak_ho/peak_fo)
    if peak_ho > peak_fo
        println("  PASS: high-order+limiter produces sharper density peak than first-order.")
    else
        println("  NOTE: first-order peak >= high-order peak at this short run time.")
    end
end
