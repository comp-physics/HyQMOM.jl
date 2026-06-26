"""
    hyqmom_quadrature_1d(m::AbstractVector) -> (w, u, N)

Adaptive 1D HyQMOM quadrature inversion.

Given a 1D raw-moment sequence `m = [M0, M1, M2, M3, M4]` (length 5, `M0 > 0`),
return a non-negative quadrature `(w, u, N)` with `N` weights `w` (all `≥ 0`) and
`N` abscissas `u`, reproducing the raw moments up to the order the `N`-node rule
supports. The node count adapts DOWN when the higher-N rule would be
non-realizable (negative weights, coalescing abscissas, or non-positive
variance):

- `N = 3` (full 3-node HyQMOM): reproduces `M0..M4`. Accepted iff the variance
  is positive, the standardized 4th moment satisfies the realizability bound
  `eta ≥ q^2 + 1`, the abscissas are distinct, and all weights are `≥ 0`.
- `N = 2` (2-node Gauss): reproduces `M0..M3`. Used when `N=3` is rejected.
  Accepted iff the variance is positive (weights are then automatically `≥ 0`).
- `N = 1` (monokinetic): reproduces `M0, M1`. Used for vacuum/cold states
  (`σ² ≤ 0`). Always valid for `M0 > 0`.

# Construction (N=3)
The mean `ū = M1/M0` and the central moments `c2,c3,c4` give the standardized
skewness `q = c3/c2^{3/2}` and kurtosis `eta = c4/c2^2`. The standardized
abscissas are the three roots of the HyQMOM degree-3 orthogonal polynomial,

    up = [ (q - D)/2,  0,  (q + D)/2 ],   D = sqrt(4*eta - 3*q^2),

which (with weights from the moment system) reproduce the standardized moments
`1, 0, 1, q, eta` exactly. The physical abscissas are `u = ū + sqrt(c2)*up` and
the weights solve the 3×3 Vandermonde system `Σ_α w_α u_α^k = M_k`, `k = 0..2`.

# Returns
- `w::Vector{Float64}` length `N`, all `≥ 0`
- `u::Vector{Float64}` length `N`, abscissas
- `N::Int` in `{3, 2, 1}`
"""
function hyqmom_quadrature_1d(m::AbstractVector)
    length(m) >= 5 || throw(ArgumentError("hyqmom_quadrature_1d expects [M0..M4] (length >= 5)"))
    M0 = float(m[1]); M1 = float(m[2]); M2 = float(m[3]); M3 = float(m[4]); M4 = float(m[5])
    M0 > 0 || throw(ArgumentError("hyqmom_quadrature_1d requires M0 > 0 (got $M0)"))

    # Tolerances
    wtol   = -1e-12      # weights accepted if >= wtol (then clamped to 0)
    gaptol = 1e-9        # minimum standardized abscissa separation
    vartol = 1e-12       # minimum variance for a multi-node rule

    ubar = M1 / M0
    # Central moments
    c2 = M2/M0 - ubar^2
    c3 = M3/M0 - 3*ubar*(M2/M0) + 2*ubar^3
    c4 = M4/M0 - 4*ubar*(M3/M0) + 6*ubar^2*(M2/M0) - 3*ubar^4

    # ---- N = 1 : vacuum / cold (non-positive variance) ----
    if !(c2 > vartol)
        return [M0], [ubar], 1
    end

    # ---- N = 3 : full 3-node HyQMOM ----
    n3 = _try_hyqmom3(M0, M1, M2, ubar, c2, c3, c4, wtol, gaptol)
    n3 === nothing || return n3

    # ---- N = 2 : 2-node Gauss (reproduces M0..M3) ----
    n2 = _try_gauss2(M0, ubar, c2, c3, wtol)
    n2 === nothing || return n2

    # ---- N = 1 : monokinetic fallback ----
    return [M0], [ubar], 1
end

# Attempt the 3-node HyQMOM rule. Returns (w,u,3) or nothing if non-realizable.
function _try_hyqmom3(M0, M1, M2, ubar, c2, c3, c4, wtol, gaptol)
    scale = sqrt(c2)
    q   = c3 / (scale * c2)     # standardized skewness c3 / c2^{3/2}
    eta = c4 / (c2 * c2)        # standardized kurtosis  c4 / c2^2

    disc = 4*eta - 3*q^2        # discriminant of the standardized cubic
    disc > 0 || return nothing
    D = sqrt(disc)

    up = (q - D)/2, 0.0, (q + D)/2
    # distinct abscissas (standardized) ?
    (D > gaptol) || return nothing

    u = [ubar + scale*up[1], ubar + scale*up[2], ubar + scale*up[3]]

    # Vandermonde solve for weights from M0, M1, M2
    V = [1.0 1.0 1.0;
         u[1] u[2] u[3];
         u[1]^2 u[2]^2 u[3]^2]
    w = V \ [M0, M1, M2]

    all(>=(wtol), w) || return nothing
    w = max.(w, 0.0)
    return w, u, 3
end

# Attempt the 2-node Gauss rule (reproduces M0..M3). Returns (w,u,2) or nothing.
function _try_gauss2(M0, ubar, c2, c3, wtol)
    c2 > 0 || return nothing
    scale = sqrt(c2)
    q = c3 / (scale * c2)              # standardized skewness
    s = sqrt(1 + q^2/4)
    up = (q/2 - s, q/2 + s)           # standardized 2-node abscissas
    u = [ubar + scale*up[1], ubar + scale*up[2]]

    V = [1.0 1.0; u[1] u[2]]
    w = V \ [M0, M0*ubar]             # match M0, M1 (M2,M3 then follow)

    all(>=(wtol), w) || return nothing
    w = max.(w, 0.0)
    return w, u, 2
end
