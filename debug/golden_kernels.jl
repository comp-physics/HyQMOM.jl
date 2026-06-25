ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, JLD2, Printf, Random, LinearAlgebra

# Golden-reference harness for the kernels touched by the performance pass.
#   julia --project=. golden_kernels.jl capture   -> writes golden_kernels.jld2
#   julia --project=. golden_kernels.jl compare   -> recomputes, prints max abs/rel diff
# Optimization is correct iff compare shows ~machine-precision diffs (0 for a pure
# StaticArrays refactor; <~1e-12 if a small-eig solver is swapped).

const GOLD = "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/golden_kernels.jld2"

# ---- deterministic battery of realizable input states across regimes ----
function build_states()
    rng = MersenneTwister(20260625)
    states = Vector{Tuple{Vector{Float64},Float64}}()   # (M, Ma)
    for Ma in (0.0, 2.0, 10.0, 50.0, 100.0)
        for rho in (1.0, 1.0e-1, 1.0e-3, 1.0e-5)
            for _ in 1:30
                # velocities scaled to Ma, temperatures/anisotropy randomized
                sc = max(Ma, 1.0)
                u = sc*(2rand(rng)-1); v = sc*(2rand(rng)-1); w = sc*(2rand(rng)-1)
                T1 = 0.2 + 2rand(rng); T2 = 0.2 + 2rand(rng); T3 = 0.2 + 2rand(rng)
                c110 = 0.3*(2rand(rng)-1)*sqrt(T1*T2)
                c101 = 0.3*(2rand(rng)-1)*sqrt(T1*T3)
                c011 = 0.3*(2rand(rng)-1)*sqrt(T2*T3)
                M = collect(InitializeM4_35(rho, u, v, w, T1, c110, c101, T2, c011, T3))
                # project to guarantee realizability, then keep BOTH raw and projected
                push!(states, (M, Ma))
                push!(states, (realizable_3D_M4(M, Ma), Ma))
            end
        end
    end
    states
end

# ---- evaluate all target kernels for one state, return a flat Float64 vector ----
function eval_kernels(M, Ma)
    out = Float64[]
    app!(x) = append!(out, vec(collect(Float64.(x))))
    C4, S4 = M2CS4_35(M);                          app!(C4); app!(S4)
    Mr = realizable_3D_M4(M, Ma);                  app!(Mr)
    Fx, Fy, Fz = Flux_closure35_3D(Mr);            app!(Fx); app!(Fy); app!(Fz)
    vx = eigenvalues6_hyperbolic_3D(Mr, 1, 0, Ma); app!(vx[1]); app!(vx[2]); app!(vx[3])
    vy = eigenvalues6_hyperbolic_3D(Mr, 2, 0, Ma); app!(vy[1]); app!(vy[2]); app!(vy[3])
    vz = eigenvalues6z_hyperbolic_3D(Mr, 0, Ma);   app!(vz[1]); app!(vz[2]); app!(vz[3])
    cx = closure_and_eigenvalues(Mr[[1,2,3,4,5]]); app!(cx[1]); app!(cx[2]); app!(cx[3])
    V = to_recon_vars(Mr);                         app!(V)
    Mb = from_recon_vars(V);                       app!(Mb)
    ff = face_flux_1d(Mr, Mr, 1, Ma);              app!(ff)
    return out
end

function evaluate_all(states)
    rows = Vector{Vector{Float64}}(undef, length(states))
    for (i,(M,Ma)) in enumerate(states)
        rows[i] = eval_kernels(M, Ma)
    end
    rows
end

mode = length(ARGS) >= 1 ? ARGS[1] : "compare"
states = build_states()
rows = evaluate_all(states)
@printf("battery: %d states, %d scalar outputs/state\n", length(states), length(rows[1]))

if mode == "capture"
    jldsave(GOLD; rows=rows)
    println("captured golden -> $GOLD")
else
    function compare(rows, gold)
        maxabs = 0.0; maxrel = 0.0; nbad = 0
        for i in 1:length(rows)
            a = rows[i]; b = gold[i]
            length(a) == length(b) || error("length mismatch at state $i")
            for k in 1:length(a)
                d = abs(a[k]-b[k]); r = d / max(abs(b[k]), 1e-300)
                d > maxabs && (maxabs = d)
                (isfinite(b[k]) && r > maxrel) && (maxrel = r)
                (d > 1e-10 && r > 1e-10) && (nbad += 1)
            end
        end
        return maxabs, maxrel, nbad
    end
    g = jldopen(GOLD); gold = g["rows"]; close(g)
    @assert length(gold) == length(rows) "state count mismatch"
    maxabs, maxrel, nbad = compare(rows, gold)
    @printf("max abs diff = %.3e   max rel diff = %.3e   (#entries failing 1e-10 = %d)\n", maxabs, maxrel, nbad)
    println(nbad == 0 ? "PASS (within 1e-10)" : "FAIL")
end
