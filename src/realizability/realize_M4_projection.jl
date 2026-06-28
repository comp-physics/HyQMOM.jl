"""
    realizable_3D_M4(M4, Ma)

Check and correct 3D unrealizable moments using the revised moment-projection
method (Appendix B). Direct port of `realizable_3D.m` from
Code_Riemann_3D_35mom_july2026.

Takes the 35-moment vector `M4` (orders 0-4, standard layout) and the Mach number
`Ma`, and returns the corrected 35-moment vector `M4r`. Internally: convert to
central/standardized moments, enforce univariate realizability (with an
Ma-dependent skewness cap), correct the 2nd-order cross moments, bound
S220/S202/S022, apply `projection35`, then reconstruct the raw moments.

This is the projection-based replacement for the legacy minor-cascade
`realizable_3D` (28-argument standardized-moment corrector). It is provided
alongside the legacy path; wiring it into the solver is a separate step.

The realizability CORRECTION (M2CS4_35 -> univariate floors / skewness cap ->
`realizability_S2` -> `realizability_S220` -> `projection35`) lives in the
single-source, allocation-free device kernel `realizable_3D_M4_corr_dev`
(`src/realizability/realize_dev.jl`), shared verbatim with the GPU realizability
kernel. It returns the corrected reconstruction variables (means + floored variances
+ 28 corrected standardized moments) — exactly `standardized_to_M4`'s argument layout.
This CPU entry point delegates the whole correction to it and then reconstructs the raw
moments with the autogen `standardized_to_M4`, byte-identical to the legacy inline path
(and the golden battery).

Why reconstruct here rather than reuse the device kernel's `realizable_3D_M4_dev`
(which finishes with `from_recon_vars_dev`)? Every correction stage in the dev kernel is
byte-identical to the CPU sources (verified 0.0 over the 1200-state battery), but the
final S->C->M step differs by ~1 ULP between the alloc-free `_c4tom4_35` and the autogen
`C4toM4_3D` (an unavoidable @fastmath reassociation difference across the two code
shapes). On its own that 1 ULP is negligible, but `M2CS4_35(realizable_3D_M4(...))` at
deep vacuum (rho~1e-5) amplifies it past the 1e-10 golden gate. Reconstructing with the
autogen here keeps the CPU result bit-for-bit with the reference. The min-eig branch
inside `projection35_dev` uses an in-kernel 6x6 symmetric Jacobi sweep instead of LAPACK
`_geigvals`, but the (sign-only) branch decisions match, so the corrected standardized
moments are byte-identical. Public signature/return type unchanged.
"""
function realizable_3D_M4(M4::AbstractVector, Ma::Real)
    return standardized_to_M4(realizable_3D_M4_corr_dev(M4..., Ma)...)
end
