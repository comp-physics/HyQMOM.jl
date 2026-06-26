"""
    chyqmom_nodes_3d(M::AbstractVector) -> (n::Vector{Float64}, U::Matrix{Float64})

3D conditional HyQMOM (CHyQMOM) joint velocity-node inversion.

Invert a length-35 raw-moment vector `M` (standard 35-moment ordering, see below)
into a non-negative 3D velocity quadrature: node weights `n` (all `≥ 0`,
`Σn ≈ M[1] = M000`) and abscissas `U` of size `(Nnodes, 3)` with
`U[k,:] = (Ux, Uy, Uz)` of node `k`.

This is a **pure addition** (nothing in the solver path calls it); it is the
unblocker for a future kinetic flux.

# Algorithm (conditional CHyQMOM: Yuan–Fox CQMOM conditioning + Fox–Laurent
# CHyQMOM closure of the conditional shape)

The joint quadrature is the factorized (conditional) form
`f = Σ_a ρ_a δ(x-Ux_a) [Σ_b ω_{b|a} δ(y-Uy_{ab})] [Σ_c ω_{c|ab} δ(z-Uz_{abc})]`.

1. **x:** invert the x-marginal raw moments `M_{i00}` (i=0..4) with the adaptive
   1D primitive [`hyqmom_quadrature_1d`](@ref) → x-nodes `(ρ_a, Ux_a)`, a=1..Nx
   (Nx∈{3,2,1}, adaptively reduced on realizability violation).
2. **y|x:** the conditional y-mean deviations `Vf_a` and conditional variances
   `σ²_a` are obtained from the (x-conditioned) central cross-moments `c_{i10}`
   (i=0..Nx-1) and `c_{i20}` (i=0..Nx-1) via a Gram (generalized Vandermonde)
   solve in the centered x-abscissas. The conditional standardized skewness `q`
   and kurtosis `η` are taken **shared across x-nodes** (Fox CHyQMOM closure),
   fixed to reproduce the pure y central moments `c_{030}`, `c_{040}`. Each
   per-x-node conditional y-moment sequence is then inverted with
   `hyqmom_quadrature_1d` → `(ω_{b|a}, Uy_{ab})`.
3. **z|x,y:** analogously, condition z on the (x,y) parent nodes — `Wf_p`, `τ²_p`
   from the cross-moments `c_{ij1}` (i+j≤3) and `c_{ij2}` (i+j≤2), shared `q_z`,
   `η_z` from `c_{003}`, `c_{004}` — and invert per parent node.

The **adaptive-N** of `hyqmom_quadrature_1d` at every conditional level is the
realizability mechanism: a level that returns N<3 simply spawns fewer children,
so near-vacuum / cold / boundary states yield finite, non-negative quadratures.

# Recovery
For a clean realizable Gaussian state (full 3×3×3 = 27 nodes) the construction
recovers **29 of the 35 moments to ~1e-8** — crucially ALL 15 x/y/z marginal
moments and every low/mid-order cross moment. Using a SHARED conditional
skewness/kurtosis (the Fox closure, chosen for realizability robustness) it can
only TRUNCATE a known set of high-order cross moments:
- `M_{310}, M_{130}` (`x³y, xy³`): with only 3 x-nodes the conditional y-mean is
  degree-2 in x, so these are determined (not free) by the lower constraints.
- z-mean cubics `M_{301}, M_{211}, M_{121}, M_{031}` (`x³z, x²yz, xy²z, y³z`):
  the z¹ staircase `c_{ij1}` (i+j≤3) has **10** cross constraints but there are
  **≤9** (x,y) parent nodes, so **at least one** of these four must be dropped;
  the condition-number cap typically drops two for a full 3×3×3 cloud.
- `M_{103}, M_{013}` (`x z³, y z³`): z³ cross moments need a per-(x,y) conditional
  skewness, but the skewness is shared.
For the reference Gaussian the actually-truncated set is the six moments
`M_{310}, M_{130}, M_{301}, M_{211}, M_{103}, M_{013}` (max abs error ≈ 1.7e-2 on
`M_{103}`); each truncated moment is still captured to small absolute or ≲1%
relative error. Reduced (near-vacuum / cold) states return fewer nodes and may
truncate additional high-order moments.
The conditional fits use a well-conditioned SPD Gram solve (`g = G⁻¹ t`, not a
pseudo-inverse) with lowest-total-degree-first *monomial* (column) selection: a
candidate monomial is admitted only if it keeps the design count `≤ Np` and the
design condition number below `condmax` (= 1e4; this cap is the lever that
governs how many z-mean cubics survive — see `_gram_fit`). Rank-deficient /
reduced (near-vacuum, cold) conditional systems therefore reproduce the
low-order (mass, mean, variance) constraints exactly and never crash.

# 35-moment ordering
`n = 1..35 ↔ (i,j,k)`, `M_n = ∫ vx^i vy^j vz^k f`:
```
(0,0,0)(1,0,0)(2,0,0)(3,0,0)(4,0,0)
(0,1,0)(1,1,0)(2,1,0)(3,1,0)(0,2,0)(1,2,0)(2,2,0)(0,3,0)(1,3,0)(0,4,0)
(0,0,1)(1,0,1)(2,0,1)(3,0,1)(0,0,2)(1,0,2)(2,0,2)(0,0,3)(1,0,3)(0,0,4)
(0,1,1)(1,1,1)(2,1,1)(0,2,1)(1,2,1)(0,3,1)(0,1,2)(1,1,2)(0,1,3)(0,2,2)
```

# Returns
- `n::Vector{Float64}` node weights (`≥ -1e-12`, `Σn ≈ M000`)
- `U::Matrix{Float64}` size `(Nnodes, 3)`, `U[k,:] = (Ux, Uy, Uz)`
"""
function chyqmom_nodes_3d(M::AbstractVector)
    length(M) >= 35 || throw(ArgumentError("chyqmom_nodes_3d expects a length-35 moment vector"))
    rho = float(M[1])
    rho > 0 || throw(ArgumentError("chyqmom_nodes_3d requires M000 = M[1] > 0 (got $rho)"))

    # Dense 5x5x5 raw-moment array (zeros where the 35-set has no entry). The
    # 35-set is downward-closed, so every sub-moment needed by the central-moment
    # expansions below is present.
    Mraw = zeros(5, 5, 5)
    @inbounds for n in 1:35
        i, j, k = _CHYQ_TRIPLES[n]
        Mraw[i+1, j+1, k+1] = float(M[n])
    end

    bu = Mraw[2,1,1] / rho
    bv = Mraw[1,2,1] / rho
    bw = Mraw[1,1,2] / rho

    # Normalized central moment c_{ijk} = <(x-bu)^i (y-bv)^j (z-bw)^k> (per-density)
    cm(i, j, k) = _chyq_central(Mraw, i, j, k, bu, bv, bw) / rho

    # ----------------------------------------------------------------- x level
    # Normalized x-marginal raw moments [1, bu, M200/ρ, M300/ρ, M400/ρ].
    mx = (1.0, bu, Mraw[3,1,1]/rho, Mraw[4,1,1]/rho, Mraw[5,1,1]/rho)
    wx, Ux, _Nx = hyqmom_quadrature_1d(collect(mx))
    Upx = Ux .- bu                       # centered x-abscissas

    # ----------------------------------------------------------------- y | x
    coordsX = reshape(Upx, length(Upx), 1)
    mean_t_y = [((0,), 0.0), ((1,), cm(1,1,0)), ((2,), cm(2,1,0))]
    var_t_y  = [((0,), cm(0,2,0)), ((1,), cm(1,2,0)), ((2,), cm(2,2,0))]
    Vf, sigy2, qY, etaY = _condition_direction(wx, coordsX, mean_t_y, var_t_y,
                                               cm(0,3,0), cm(0,4,0))

    # Build (x,y) parent nodes.
    wxy = Float64[]; Uxy_x = Float64[]; Uxy_y = Float64[]
    @inbounds for a in eachindex(wx)
        wx[a] > 0 || continue
        muy = bv + Vf[a]
        s2  = max(sigy2[a], 0.0)
        my  = _shape_moments(wx[a], muy, s2, qY, etaY)
        wsub, usub, _ = hyqmom_quadrature_1d(my)
        for b in eachindex(wsub)
            push!(wxy, wsub[b]); push!(Uxy_x, Ux[a]); push!(Uxy_y, usub[b])
        end
    end

    # ----------------------------------------------------------------- z | x,y
    Np = length(wxy)
    coordsXY = hcat(Uxy_x .- bu, Uxy_y .- bv)        # Np x 2 centered (x,y)
    # z-mean cross-moment targets c_{ij1}, staircase i+j ≤ 3
    mean_t_z = Tuple{NTuple{2,Int},Float64}[]
    for i in 0:3, j in 0:(3-i)
        push!(mean_t_z, ((i, j), cm(i, j, 1)))
    end
    # z-variance cross-moment targets c_{ij2}, staircase i+j ≤ 2
    var_t_z = Tuple{NTuple{2,Int},Float64}[]
    for i in 0:2, j in 0:(2-i)
        push!(var_t_z, ((i, j), cm(i, j, 2)))
    end
    Wf, tauz2, qZ, etaZ = _condition_direction(wxy, coordsXY, mean_t_z, var_t_z,
                                               cm(0,0,3), cm(0,0,4))

    # Build final (x,y,z) nodes.
    nx_list = Float64[]; ux_list = Float64[]; uy_list = Float64[]; uz_list = Float64[]
    @inbounds for p in 1:Np
        wxy[p] > 0 || continue
        muz = bw + Wf[p]
        t2  = max(tauz2[p], 0.0)
        mz  = _shape_moments(wxy[p], muz, t2, qZ, etaZ)
        wsub, usub, _ = hyqmom_quadrature_1d(mz)
        for c in eachindex(wsub)
            push!(nx_list, wsub[c])
            push!(ux_list, Uxy_x[p]); push!(uy_list, Uxy_y[p]); push!(uz_list, usub[c])
        end
    end

    # Scale weights back to physical density and assemble outputs.
    n = nx_list .* rho
    U = hcat(ux_list, uy_list, uz_list)
    return n, U
end

# 35-moment exponent triples (module-level constant).
const _CHYQ_TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),
 (0,2,0),(1,2,0),(2,2,0),
 (0,3,0),(1,3,0),
 (0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),
 (0,0,2),(1,0,2),(2,0,2),
 (0,0,3),(1,0,3),
 (0,0,4),
 (0,1,1),(1,1,1),(2,1,1),
 (0,2,1),(1,2,1),
 (0,3,1),
 (0,1,2),(1,1,2),
 (0,1,3),
 (0,2,2)]

# Un-normalized central moment Σ via binomial expansion of the raw-moment array.
function _chyq_central(Mraw, i, j, k, bu, bv, bw)
    s = 0.0
    @inbounds for a in 0:i, b in 0:j, d in 0:k
        s += binomial(i, a) * binomial(j, b) * binomial(k, d) *
             (-bu)^(i-a) * (-bv)^(j-b) * (-bw)^(k-d) * Mraw[a+1, b+1, d+1]
    end
    return s
end

# Raw moment sequence [m0..m4] (scaled by weight `w`) of a 1D distribution with
# mean `mu`, variance `s2 ≥ 0`, standardized skewness `q`, kurtosis `eta`.
function _shape_moments(w, mu, s2, q, eta)
    s2 = max(s2, 0.0)
    sig = sqrt(s2)
    s3 = sig^3; s4 = s2^2
    m0 = w
    m1 = w * mu
    m2 = w * (mu^2 + s2)
    m3 = w * (mu^3 + 3mu*s2 + q*s3)
    m4 = w * (mu^4 + 6mu^2*s2 + 4mu*q*s3 + eta*s4)
    return [m0, m1, m2, m3, m4]
end

# Generic conditional-direction fit. Given parent nodes (weights `pw`, centered
# conditioning-direction abscissas `coords`, Np x D), solve for the per-node
# conditional mean deviation `Wf` and variance `var` in a new direction that
# reproduce the central cross-moment `mean_targets` (new-direction power 1) and
# `var_targets` (power 2), plus a single shared standardized skewness `q` and
# kurtosis `eta` set to reproduce the pure new-direction central moments `cd3`,
# `cd4` (the Fox CHyQMOM closure of the conditional shape). A shared shape is
# what keeps the per-node conditional sequences realizable; letting the skewness
# vary per node was found to drive boundary nodes non-realizable (collapsing them
# to monokinetic and losing their conditional variance), so it is intentionally
# avoided.
#
# Targets are vectors of `(expo::NTuple{D,Int}, value)`. The mean / variance
# solves drop (least-squares) any constraints the parent-node count cannot
# support, so the routine never crashes for reduced parent quadratures.
function _condition_direction(pw, coords, mean_targets, var_targets, cd3, cd4)
    Wf = _gram_fit(pw, coords, mean_targets)
    s  = _gram_fit(pw, coords, var_targets)
    var = max.(s .- Wf .^ 2, 0.0)
    sig = sqrt.(var)

    den3 = sum(pw .* sig .^ 3)
    q = abs(den3) > 1e-14 ?
        (cd3 - sum(pw .* (Wf .^ 3 .+ 3 .* Wf .* var))) / den3 : 0.0
    den4 = sum(pw .* sig .^ 4)
    eta = abs(den4) > 1e-14 ?
        (cd4 - sum(pw .* (Wf .^ 4 .+ 6 .* Wf .^ 2 .* var .+ 4 .* Wf .* sig .^ 3 .* q))) / den4 :
        (q^2 + 1.0)
    return Wf, var, q, eta
end

# Fit the per-node conditional value `Wf_p = Σ_m g_m φ_m(node_p)` reproducing the
# moment constraints `Σ_p pw_p φ_e(node_p) Wf_p = target_e`. The monomial basis is
# selected from the target exponents, lowest total degree first, keeping a
# monomial only if it is linearly INDEPENDENT (in the pw-weighted inner product)
# of the already-selected ones over the actual parent-node geometry. This makes
# the resulting Gram system symmetric positive-definite and well-conditioned, so
# the low-order constraints (mass, mean, variance) are reproduced exactly and the
# solve is numerically stable; the targets whose monomials are linearly dependent
# over the nodes (e.g. a cubic over collinear abscissas) are the documented
# CHyQMOM cross-moment truncation and are left unconstrained.
function _gram_fit(pw, coords, targets)
    Np = length(pw)
    nt = length(targets)
    order = sortperm([sum(targets[m][1]) for m in 1:nt])   # low total degree first
    sw = sqrt.(max.(pw, 0.0))
    # Add monomial columns (low degree first) only while the weighted design
    # matrix `B = sqrt(pw) .* Phi` stays well-conditioned. This adaptively caps
    # the conditional-mean / variance polynomial degree to the level the actual
    # parent-node geometry supports: a near-collinear / near-degenerate node
    # cloud admits only low-degree terms, which keeps the fit BOUNDED (no
    # exploding abscissas) while a well-spread cloud admits the higher-degree
    # cross terms. Dropped (higher-degree) targets are the CHyQMOM truncation.
    condmax = 1e4
    sel = Int[]
    B = Matrix{Float64}(undef, Np, 0)
    @inbounds for m in order
        col = Float64[sw[p] * _monomial(coords, p, targets[m][1]) for p in 1:Np]
        Btry = hcat(B, col)
        all(iszero, col) && continue
        if size(Btry, 2) <= Np && _design_cond(Btry) < condmax
            B = Btry
            push!(sel, m)
            length(sel) == Np && break
        end
    end
    isempty(sel) && return zeros(Np)

    nb = length(sel)
    Phi = Matrix{Float64}(undef, Np, nb)
    t = Vector{Float64}(undef, nb)
    @inbounds for m in 1:nb
        e = targets[sel[m]][1]
        t[m] = targets[sel[m]][2]
        for p in 1:Np
            Phi[p, m] = _monomial(coords, p, e)
        end
    end
    G = Phi' * (pw .* Phi)                                  # nb x nb SPD, well-conditioned
    g = G \ t
    return Phi * g
end

# 2-norm condition number of a (tall) design matrix; Inf if rank-deficient.
function _design_cond(B)
    s = svdvals(B)
    (isempty(s) || s[end] <= 0) && return Inf
    return s[1] / s[end]
end

# Monomial Π_d coords[p,d]^e[d] for node p (0^0 = 1).
function _monomial(coords, p, e)
    v = 1.0
    @inbounds for d in eachindex(e)
        ed = e[d]
        ed == 0 || (v *= coords[p, d]^ed)
    end
    return v
end
