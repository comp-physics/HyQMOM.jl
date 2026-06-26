"""
    realizability_margin(M) -> Float64

Signed distance of the 35-moment vector `M` from the boundary of the realizable moment set R,
measured as the smallest eigenvalue of the 6x6 matrix `delta2star3D(...)` (i.e. <p2 p2'>), which
is exactly the test the Appendix B projection (`projection35`) uses. Returns `-Inf` for
non-finite states, nonpositive density, or nonpositive directional variance (hard non-realizable).
`>= 0` <=> realizable (to the 4th-order cross-moment conditions); `> 0` <=> strictly interior.
"""
function realizability_margin(M::AbstractVector)
    (length(M) == 35 && all(isfinite, M)) || return -Inf
    M000 = M[1]
    (M000 > 0) || return -Inf
    C4, S4 = M2CS4_35(M)
    (C4[3] > 0 && C4[10] > 0 && C4[20] > 0) || return -Inf   # C200,C020,C002 directional variances
    E1 = delta2star3D(S4[_SIDX]...)
    all(isfinite, E1) || return -Inf
    return minimum(real(_geigvals(E1)))
end

"""
    is_realizable(M; lam_min=0.0) -> Bool

True if `realizability_margin(M) >= lam_min`. `lam_min > 0` keeps a strict interior margin
(useful to avoid sitting exactly on the cone boundary where hyperbolicity degenerates).
"""
@inline is_realizable(M::AbstractVector; lam_min::Real=0.0) = realizability_margin(M) >= lam_min
