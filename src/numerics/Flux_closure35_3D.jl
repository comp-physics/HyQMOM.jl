"""
    Flux_closure35_3D(M4)

Compute the 3D HLL flux moments for all 35 moments via the HyQMOM closure, with
NO realizability correction. Direct port of `Flux_closure35_3D.m` from
Code_Riemann_3D_35mom_july2026 (the revised solver, where realizability is a
separate step applied after the spatial update).

The input `M4` is assumed already hyperbolicity-corrected (e.g. by
`eigenvalues6_hyperbolic_3D`). Returns `(Fx, Fy, Fz)`, each a 35-vector of flux
moments in the standard ordering.

This is the pure-flux companion to `realizable_3D_M4`; together they replace the
combined `Flux_closure35_and_realizable_3D` in the projection-based solver.

The per-cell math lives in the single-source, allocation-free device kernel
`flux_closure35_dev` (`src/numerics/flux_closure_dev.jl`), which is shared verbatim
with the GPU kernel. This CPU entry point simply delegates to it and reshapes the
flattened `NTuple{105}` result (1..35 = Fx, 36..70 = Fy, 71..105 = Fz) into the
three 35-element `(Fx, Fy, Fz)` vectors that callers expect. The 35 scalar inputs
are the raw moments in canonical `M4` order, so `flux_closure35_dev(M4...)` is the
correct call.
"""
function Flux_closure35_3D(M4::AbstractVector)
    F = flux_closure35_dev(M4...)
    Fx = collect(F[1:35])
    Fy = collect(F[36:70])
    Fz = collect(F[71:105])
    return Fx, Fy, Fz
end
