# Cheap 1D analog of the high-order 3D crossing: two dense Mach-`Ma` slabs collide
# through a near-vacuum background. Reproduces the near-vacuum behaviour of the 3D
# high-order scheme (step_highorder_3d!) serially in seconds, using the same
# kernels (residual_1d high-order + per-stage realizability projection) and an
# adaptive CFL timestep. Useful for studying the high-order vacuum instability
# without an MPI run. See docs/ma100-highorder-crash-analysis.md.
#
#   julia --project=. debug/repro_1d_crash.jl
#   R1D_MA=50 R1D_VACFLOOR=1e-2 julia --project=. debug/repro_1d_crash.jl
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using HyQMOM, Printf

const Ma         = parse(Float64, get(ENV, "R1D_MA",       "100.0"))
const N          = parse(Int,     get(ENV, "R1D_N",        "512"))
const order      = parse(Int,     get(ENV, "R1D_ORDER",    "2"))
const vacf       = parse(Float64, get(ENV, "R1D_VACFLOOR", "0.001"))
const use_lim    = get(ENV, "R1D_LIMITER", "0") != "0"
const rhol, rhor, T = 1.0, 0.001, 1.0
const dx = 1.0 / N

state(rho, u) = InitializeM4_35(rho, u, 0.0, 0.0, T, 0.0, 0.0, T, 0.0, T)

# adaptive CFL timestep from the current field (mirrors the solver's per-step dt)
function cfl_dt(M)
    v = 0.0
    @inbounds for i in 1:size(M, 1)
        r = M[i, 1]; r > 0 || continue
        u = M[i, 2] / r; c2 = max(M[i, 3] / r - u^2, 0.0)
        v = max(v, abs(u) + 4 * 2.334 * sqrt(c2 + 1e-12))
    end
    (1 / 3) * dx / max(v, 1e-12)
end

function run()
    HyQMOM.HO_VACUUM_FLOOR[] = vacf
    Uc = Ma / sqrt(2.0)
    xc = [(i - 0.5) * dx for i in 1:N]
    M = zeros(N, 35)
    for i in 1:N                       # two dense slabs (±Uc) in a near-vacuum
        x = xc[i]
        M[i, :] = 0.25 <= x < 0.5 ? state(rhol, Uc) :
                  0.5  <= x < 0.75 ? state(rhol, -Uc) : state(rhor, 0.0)
    end
    proj!(U) = (for i in 1:N; U[i, :] = realizable_3D_M4(U[i, :], Ma); end)
    L(U) = residual_1d(U, dx, Ma; order=order, bc=:outflow, use_limiter=use_lim)

    tfinal = 0.1 / Uc; t = 0.0; n = 0
    while t < tfinal
        dt = cfl_dt(M)
        M1 = M .+ dt .* L(M);                            proj!(M1)
        M2 = 0.75 .* M .+ 0.25 .* (M1 .+ dt .* L(M1));   proj!(M2)
        M  = (1/3) .* M .+ (2/3) .* (M2 .+ dt .* L(M2));  proj!(M)
        t += dt; n += 1
        all(isfinite, @view M[:, 1]) || (@printf("NON-FINITE at step %d (t=%.3e)\n", n, t); break)
    end
    @printf("Ma=%.0f N=%d order=%d floor=%.0e : %d steps, t=%.4e, rho[min,max]=[%.4e,%.4e]\n",
            Ma, N, order, vacf, n, t, minimum(M[:, 1]), maximum(M[:, 1]))
end

run()
