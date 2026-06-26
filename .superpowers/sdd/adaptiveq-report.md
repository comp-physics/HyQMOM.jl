# Adaptive 1D HyQMOM quadrature inversion — report

## Summary
New pure-addition primitive `hyqmom_quadrature_1d(m) -> (w, u, N)` that inverts a
1D raw-moment sequence `[M0..M4]` into a non-negative quadrature, adaptively
reducing the node count `N ∈ {3,2,1}` when the higher-N rule would be
non-realizable. Nothing in the flux/residual path calls it — golden unaffected.

## N=3 construction (abscissas + weights)
- Mean `ū = M1/M0`; central moments `c2,c3,c4` computed inline.
- Standardized skewness `q = c3/c2^{3/2}`, kurtosis `eta = c4/c2^2`.
- Standardized abscissas = roots of the HyQMOM degree-3 orthogonal polynomial:
  `up = [(q-D)/2, 0, (q+D)/2]`, `D = sqrt(4*eta - 3*q^2)`.
  Algebraically these reproduce the standardized moments `1,0,1,q,eta` exactly
  (verified analytically: m3→q, m4→eta).
- Physical abscissas `u = ū + sqrt(c2)*up`.
- Weights from the 3×3 Vandermonde solve `Σ_α w_α u_α^k = M_k`, k=0..2.
  Because the abscissas already encode the closure, this automatically recovers
  M3 and M4 as well.

## Reduction triggers (the adaptive mechanism)
- **N=3 rejected** when any of: variance `c2 ≤ 1e-12`; discriminant
  `4*eta-3*q^2 ≤ 0`; abscissa gap `D ≤ 1e-9` (coalescing); any weight `< -1e-12`.
  Negative weights occur exactly when `eta < q^2 + 1` (the H200 4th-moment
  realizability bound) — this is the documented N=3 rejection condition.
- **N=2** (Gauss, reproduces M0..M3): `u = ū + σ(q/2 ± sqrt(1+q²/4))`, weights
  from M0,M1. These 2-node weights are non-negative for any `c2 > 0`, so N=2
  succeeds whenever variance is positive.
- **N=1** (monokinetic): `u=[M1/M0]`, `w=[M0]`. Used when `c2 ≤ 1e-12`
  (vacuum/cold) or as final fallback. Always valid for M0>0.

## Moment-recovery results per N
- N=3 on Gaussian sets (3 cases): recovers M0..M4 to rtol/atol 1e-10. w>0.
- N=2 (deflated-M4 non-realizable case): recovers M0..M3 to 1e-10; w=[0.5,0.5].
- N=1 (zero-variance case): recovers M0,M1 exactly; w=[M0], u=[M1/M0].

## Random sweep non-negativity (strengthened — test 4 rev2, 2026-06-26)
Test 4 was replaced with a 300-case skewed sweep covering the full adaptive ladder.
Cases use raw moments built from standardized `(q, η)` via the exact formulas
`M0=ρ; M1=ρμ; M2=ρ(μ²+σ²); M3=ρ(μ³+3μσ²+qσ³); M4=ρ(μ⁴+6μ²σ²+4μqσ³+ησ⁴)`.

- i ∈ [1,200]: η = q²+1.1 + 5ε (fully realizable, varied q ∈ (-2,2)) → N=3.
- i ∈ [201,260]: η = max(1.01, q²+1−0.5−2ε), σ²>0, q ∈ (-3,3) → η < q²+1 for
  |q|≳0.72, so N=3 Vandermonde weights go negative → N=2 fallback.
- i ∈ [261,300]: σ² = 1e-16·ε < vartol=1e-12 → N=1 monokinetic.

Assertions: (a) all w ≥ −1e-12; (b) recover M0..M_{2N−2} to rtol 1e-9;
(c) Ns_seen ⊇ {1, 2, 3} (all three branches exercised).
Result: 22/22 pass (nbad==0, 1∈Ns_seen, 2∈Ns_seen, 3∈Ns_seen confirmed).

## TDD
- RED: wrote `test/test_adaptive_quadrature.jl` (4 testsets) → all 4 errored
  (`hyqmom_quadrature_1d` undefined) before implementation.
- GREEN: after implementing → 19/19 pass.

## Files changed
- `src/moments/hyqmom_quadrature_1d.jl` (new)
- `src/HyQMOM.jl` (export + include — inert addition)
- `test/test_adaptive_quadrature.jl` (new)
- `test/runtests.jl` (registers the new test)

## Concerns
- N=3 root-finding uses the closed-form HyQMOM cubic roots (not iterative), so it
  is numerically stable; rejection to N=2/N=1 is the realizability safety net.
- The primitive is inert (no caller). A future kinetic flux would consume it.
