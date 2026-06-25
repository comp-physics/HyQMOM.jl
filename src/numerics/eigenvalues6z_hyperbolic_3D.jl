using LinearAlgebra: eigvals

"""
    eigenvalues6z_hyperbolic_3D(M, flag2D, Ma)

Eigenvalues of the 3D flux Jacobian in the z direction, with hyperbolicity
correction. Direct port of `eigenvalues6z_hyperbolic_3D.m`. Uses the WU and WV
transverse planes via `jacobian15` (3x3 block 13:15 and 4x4 block 6:9); on a
complex 3x3 block the moments are projected to a hyperbolic set via
`correct_moments_hyperbolic_3D`. Returns `(v6min, v6max, Mr)`.

`flag2D`/`Ma` are accepted for caller compatibility (unused, matching MATLAB).
Helpers `_plane_*`, `_jac15_eig`, `_has_complex`, `correct_moments_hyperbolic_3D`
are defined in eigenvalues6_hyperbolic_3D.jl.
"""
function eigenvalues6z_hyperbolic_3D(M::AbstractVector, flag2D::Int, Ma::Real; debug_output=false)
    pa, pb = _plane_WU(M), _plane_WV(M)
    va_min, va_max, hca = _jac15_eig(pa)
    vb_min, vb_max, hcb = _jac15_eig(pb)
    v6min = min(va_min, vb_min)
    v6max = max(va_max, vb_max)
    Mr = M

    if hca || hcb
        Mr = correct_moments_hyperbolic_3D(M)
        a_min, a_max, _ = _jac15_eig(_plane_WU(Mr))
        b_min, b_max, _ = _jac15_eig(_plane_WV(Mr))
        v6min = min(a_min, b_min)
        v6max = max(a_max, b_max)
    end

    return v6min, v6max, Mr
end
