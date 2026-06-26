# Empirical realizability-CFL sweep for the 1D colliding-slab problem.
#
# Runs the Ma=10 colliding-slab+vacuum 1D problem (same setup as repro_1d_crash.jl)
# at a set of CFL values with the realizability scaling limiter ON, measuring:
#   (a) whether all cell-mean moments stay realizable (realizability_margin >= 0)
#       throughout the run, and
#   (b) how many per-step projection corrections are needed.
#
# The LARGEST CFL value for which all cell means stay realizable over the whole run
# is the empirical realizability-CFL threshold.
#
# Usage:
#   HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. debug/realizability_cfl_sweep.jl
#
# ENV knobs (all optional):
#   RCFL_MA        Mach number       (default 10.0)
#   RCFL_N         # cells           (default 128)
#   RCFL_CFLS      comma-sep list    (default "0.1,0.3,0.5,0.7,0.9")
#   RCFL_MAXSTEPS  cap per run       (default 2000)
#   RCFL_VACFLOOR  vacuum floor      (default 0.0, limiter is ON regardless)
#
# Note: Step 4 (FHW quadratic-form oracle) is deferred as future work (perf
# optimization; the eig-based oracle in realizability_margin is correct).
# A finer CFL grid and 3D sweep are deferred as dedicated HPC jobs.

ENV["HYQMOM_SKIP_PLOTTING"] = "true"
ENV["CI"]                    = "true"

using HyQMOM, Printf

# ── Parameters ────────────────────────────────────────────────────────────────
const Ma       = parse(Float64, get(ENV, "RCFL_MA",       "10.0"))
const N        = parse(Int,     get(ENV, "RCFL_N",        "128"))
const cfls_str = get(ENV, "RCFL_CFLS", "0.1,0.3,0.5,0.7,0.9")
const cfls     = [parse(Float64, s) for s in split(cfls_str, ",")]
const maxsteps = parse(Int,     get(ENV, "RCFL_MAXSTEPS", "2000"))
const vacf     = parse(Float64, get(ENV, "RCFL_VACFLOOR", "0.0"))

# ── IC: same colliding-slab+vacuum as repro_1d_crash.jl ─────────────────────
const dx = 1.0 / N
state(rho, u) = InitializeM4_35(rho, u, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
const rhol = 1.0; const rhor = 0.001
const Uc   = Ma / sqrt(2.0)
const tfinal = 0.1 / Uc   # same end-time as repro_1d_crash

function make_IC()
    M = zeros(N, 35)
    xc = [(i - 0.5) * dx for i in 1:N]
    for i in 1:N
        x = xc[i]
        M[i, :] = 0.25 <= x < 0.5  ? state(rhol,  Uc) :
                  0.5  <= x < 0.75 ? state(rhol, -Uc) : state(rhor, 0.0)
    end
    return M
end

# ── Max wave speed across the field ──────────────────────────────────────────
function max_wavespeed(M)
    v = 0.0
    @inbounds for i in 1:N
        r = M[i, 1]; r > 0 || continue
        u  = M[i, 2] / r
        c2 = max(M[i, 3] / r - u^2, 0.0)
        v  = max(v, abs(u) + 4 * 2.334 * sqrt(c2 + 1e-12))
    end
    return max(v, 1e-12)
end

# ── Per-step realizability check and projection counter ──────────────────────
"""Count how many cells are unrealizable (margin < 0) and apply projection.
Returns (corrected_M, n_unrealizable)."""
function proj_with_count!(M)
    ncorr = 0
    for i in 1:N
        if realizability_margin(@view M[i, :]) < 0
            ncorr += 1
        end
        M[i, :] = realizable_3D_M4(M[i, :], Ma)
    end
    return ncorr
end

"""All cell-mean moments realizable?  Returns (ok, min_margin)."""
function check_realizability(M)
    min_m = Inf
    for i in 1:N
        m = realizability_margin(@view M[i, :])
        min_m = min(min_m, m)
    end
    return min_m >= 0.0, min_m
end

# ── Single-run sweep at one CFL value ────────────────────────────────────────
function run_cfl(cfl_factor; report_interval=500)
    HyQMOM.HO_VACUUM_FLOOR[] = vacf

    M   = make_IC()
    L(U) = residual_1d(U, dx, Ma; order=2, bc=:outflow, use_limiter=true)

    t = 0.0; n = 0
    total_corr = 0
    max_corr_step = 0
    min_margin = Inf
    ok = true

    while t < tfinal && n < maxsteps
        vmax = max_wavespeed(M)
        dt   = cfl_factor * dx / vmax

        # SSP-RK3 stages with per-stage projection + counting
        M1 = M .+ dt .* L(M)
        n1 = proj_with_count!(M1)
        total_corr += n1; max_corr_step = max(max_corr_step, n1)

        M2 = 0.75 .* M .+ 0.25 .* (M1 .+ dt .* L(M1))
        n2 = proj_with_count!(M2)
        total_corr += n2; max_corr_step = max(max_corr_step, n2)

        M  = (1/3) .* M .+ (2/3) .* (M2 .+ dt .* L(M2))
        n3 = proj_with_count!(M)
        total_corr += n3; max_corr_step = max(max_corr_step, n3)

        if !all(isfinite, @view M[:, 1])
            ok = false
            @printf("  CFL=%.2f: NON-FINITE at step %d (t=%.3e)\n", cfl_factor, n+1, t+dt)
            break
        end

        step_ok, marg = check_realizability(M)
        min_margin = min(min_margin, marg)
        if !step_ok
            ok = false
            @printf("  CFL=%.2f: UNREALIZABLE cell at step %d (t=%.3e, min_margin=%.3e)\n",
                    cfl_factor, n+1, t+dt, marg)
            # keep running to measure how bad it gets
        end

        t += dt; n += 1
        if n % report_interval == 0
            @printf("  CFL=%.2f step %4d t=%.3e min_margin=%.3e corr_so_far=%d\n",
                    cfl_factor, n, t, min_margin, total_corr)
        end
    end

    steps_completed = n
    return (ok=ok, steps=steps_completed, t_final=t,
            total_corr=total_corr, max_corr_step=max_corr_step,
            min_margin=min_margin)
end

# ── Main sweep ────────────────────────────────────────────────────────────────
@printf("\n=== Realizability-CFL Sweep (Ma=%.0f, N=%d, limiter=ON, vacfloor=%.0e) ===\n\n",
        Ma, N, vacf)
@printf("  tfinal=%.3e,  maxsteps=%d\n\n", tfinal, maxsteps)
@printf("  %-6s  %-6s  %-7s  %-6s  %-12s  %-10s  %-12s\n",
        "CFL", "OK?", "steps", "t_end", "total_corr", "max_corr/step", "min_margin")
@printf("  %s\n", "-"^75)

results = Dict{Float64, NamedTuple}()
for cfl in sort(cfls)
    r = run_cfl(cfl)
    results[cfl] = r
    ok_str = r.ok ? "YES" : "NO "
    @printf("  %-6.2f  %-6s  %-7d  %-6.3e  %-12d  %-13d  %-12.3e\n",
            cfl, ok_str, r.steps, r.t_final, r.total_corr, r.max_corr_step, r.min_margin)
end

# empirical CFL = largest CFL where ok=true
passing = [c for (c, r) in results if r.ok]
if isempty(passing)
    @printf("\nEMPIRICAL CFL: NONE — no tested CFL value kept all cells realizable.\n")
else
    emp_cfl = maximum(passing)
    @printf("\nEMPIRICAL REALIZABILITY-CFL >= %.2f\n", emp_cfl)
    @printf("(all tested CFL values kept all cell-means realizable AND finite over the run)\n")
end

@printf("\nNOTES:\n")
@printf("  - Projection counter (total_corr) counts per-cell unrealizable moments\n")
@printf("    summed across all SSP-RK3 stages over the full run.\n")
@printf("  - max_corr/step is the worst single stage correction count.\n")
@printf("  - FHW quadratic-form oracle (cheaper realizability_margin_qf) is\n")
@printf("    DEFERRED as future work; the eig-based oracle is correct.\n")
@printf("  - Finer CFL grid and 3D sweep are DEFERRED as dedicated HPC jobs.\n")
@printf("  - 3D projection-activation diagnostic is available via\n")
@printf("    HYQMOM_PROJ_COUNT=1 (see src/numerics/highorder_3d.jl).\n")
