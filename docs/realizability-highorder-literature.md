# Realizability-preserving high-order schemes for HyQMOM — the science behind the Ma=100 problem

A literature-grounded analysis of why our high-order scheme breaks in near-vacuum at
high Mach, and the established theory for fixing it. Companion to
`ma100-highorder-crash-analysis.md` (which documents the crash itself) and
`HIGHORDER.md` (status/usage). Written for the HyQMOM.jl collaborators (SHB, Fox, Posey).

---

## 1. The problem, stated precisely

At Ma=100 with a 1000:1 density ratio, the crossing jets carve a deep vacuum
(ρ → ~1e-5) behind the dense fronts. There the primitives we derive from the moments,

```
u    = M100 / M000
C200 = M200 / M000 − u²
```

are catastrophic-cancellation noise — two large, nearly equal numbers divided by a
near-zero density. Our high-order path then does the one thing the moment-method
literature warns against: it **reconstructs the moments themselves** (standardized
central moments) with MUSCL and recombines independently-limited slopes at the face.
The recombined face moment lands **outside the realizable moment set** — negative or
huge directional variance — and the downstream `eigvals` / `sqrt(C200)` produces
non-finite values that the RK update spreads as NaN. First-order HLL does not crash
only because it diffuses the vacuum away before the moment set degenerates.

This is **not a porting bug and not unique to us.** It is the central, known failure
mode of high-order quadrature-based moment methods (QBMM): high-order reconstruction
does not, by itself, keep the moment vector inside the realizable set.

---

## 2. Why it is fundamental: realizability is guaranteed only for first order

The **realizable moment set R** — moment vectors that correspond to a genuine
non-negative distribution — is **convex**. Realizability is equivalent to positive
semidefiniteness of the Hankel (moment) matrix, equivalently the canonical moments /
quadrature weights being non-negative. Convexity is the load-bearing property of the
entire theory.

For HyQMOM specifically:

- **Fox & Laurent, SIAM J. Appl. Math. 82 (2022)** (arXiv:2103.10138) — the 1D HyQMOM
  closure our solver implements. Globally hyperbolic, conservative, defined for any
  realizable even-order moment set; the characteristic polynomial factors into
  orthogonal polynomials with interlacing real roots. The closure is realizability-
  interior *by construction*. HyQMOM is **strictly a moment closure — it never
  reconstructs a VDF**, so the moment vector stays in R on its own.
- **Laurent & Fox, ESAIM: Proc. & Surveys 76 (2024)** — proves the **first-order HLL
  scheme is realizability-preserving under a CFL condition** (the updated cell mean is
  a convex combination of realizable states, so it stays in the convex set R). Test
  cases include the crossing / two-delta states that stress realizability.
- **3D lineage:** Fox, Laurent & Vié, JCP 365 (2018) (CHyQMOM); Patel, Desjardins &
  Fox, JCP:X 1 (2019) (3D CHyQMOM). Our 35-moment 3D solver is this family; QBMMlib
  (Bryngelson et al., SoftwareX 12, 2020) is our own implementation.

### How our own paper maintains realizability (the method the code implements)

Important: the multivariate HyQMOM paper (Bryngelson, Fox & Laurent, "Fourth-Order
HyQMOM Closures for Multidimensional Kinetic Equations," JCP revision) does **not** rely
on a realizability-*preserving* HLL. It uses a **post-free-transport moment-correction**,
which it calls "an integral part of the 2D/3D HyQMOM closure... Without it, simulation of
high-speed flows is not possible because the free-transport fluxes can generate
non-realizable moments." After the (first-order) flux, moments that fall outside R are
restored by either:

- the **main-text constraint cascade** — Sylvester's criterion on the leading principal
  minors of the moment matrices, enforcing the explicit per-moment bounds derived in §3.1
  / §5.1 (our `realizable_3D.jl` + `realizability_S*.jl`); plus the **hyperbolicity
  correction Eq (42)** when the 6×6 2D-moment block has complex eigenvalues; or
- the faster, "more robust" **Appendix B moment-projection**: find the smallest eigenvalue
  λ₁ of ⟨p₂p₂ᵀ⟩; if λ₁ < λ_min (=0), reset the order >2 cross-moments to a realizable
  *target distribution* and recheck (our `projection35.jl` + `realize_M4_projection.jl` —
  documented in-code as a direct port of Appendix B).

**With this correction, the paper achieves arbitrary Knudsen AND Mach number at first
order, including the Ma=100 crossing and impinging jets (Figs 7, 8) and Ma=3 3D jets
(Fig 13).** For hypersonic crossing jets even the second-order moments go non-realizable;
the correction handles that too. So realizability at first order is a **solved, published
problem** — and `src/` already implements both the cascade and the projection.

**The actual gap.** Our crash is the **high-order MUSCL step leaving R during
*reconstruction* — at the faces, before the flux** — so the recombined face moment
overflows to Inf in the flux and the cell mean is already NaN by the time the per-cell
correction/projection runs. The paper's correction acts on the cell mean *after* the flux;
it cannot reach a reconstruction that has already corrupted the flux. And the paper itself
lists, in its Future Work, "implement **high-order spatial reconstruction schemes to reduce
numerical diffusion in the HLL solver**" — i.e. **our high-order work IS the paper's own
stated future-work item, and it is not claimed solved in the paper.** No published HyQMOM
scheme applies a realizability *limiter* to the high-order *reconstruction* of standardized
velocity moments — that is the genuinely open piece (Jacob's territory), and it confirms
Rodney's "we most likely need a minimum density below which the scheme is limited to first
order." The paper also notes that exact multivariate orthogonal polynomials "could
eliminate the need for the moment-correction algorithm for high-speed flows" — a separate
future direction.

---

## 3. The established fix: the Zhang–Shu template transplanted to moment space

Every realizability-preserving high-order moment scheme is a transplant of the
**Zhang & Shu** positivity/bound-preserving framework (JCP 2010/2011; Proc. R. Soc. A
467, 2011), replacing the admissible set G = {ρ>0, p>0} with R = {realizable moments}.
The five ingredients:

1. **R is convex** (Hankel-PSD / canonical-moment positivity). This licenses everything.
2. **A first-order update that maps R → R under a CFL bound** — the updated mean is a
   convex combination of realizable Riemann/HLL states. *We already have this* (our
   first-order HLL, per Laurent & Fox 2024).
3. **A scaling ("squeezing") limiter** makes the high-order reconstruction realizable:
   replace each reconstructed face/quadrature value `w_f` by
   ```
   w̃_f = w̄ + θ (w_f − w̄),   θ ∈ [0,1] the largest value keeping w̃_f ∈ R
   ```
   Since R is convex and the cell mean `w̄ ∈ int R`, such θ exists; θ → 1 at the design
   rate in smooth regions, so accuracy is retained. This is the **direct fix for "MUSCL
   leaves R in near-vacuum."**
4. **SSP-RK time stepping.** SSP-RK stages are convex combinations of forward-Euler
   steps, so each stage stays in R. **Standard (non-SSP) RK does NOT preserve
   realizability** even with realizable stages (Vikas/Fox show this explicitly). We use
   SSP-RK3 — worth confirming it is a genuine Shu–Osher SSP scheme.
5. **Boundary-of-moment-space / vacuum fallback.** Where the limiter cannot find θ>0
   (mean itself on ∂R) or the closure loses hyperbolicity: drop locally to first order,
   apply a density/weight floor and rescale higher moments to stay interior, and/or use
   a realizability-*interior* closure.

The papers differ only in (a) how R is represented for cheap projection and (b) how the
boundary/vacuum is regularized.

---

## 4. Two concrete published recipes for our solver

### Option A — Vikas, Wang, Passalacqua & Fox, JCP 230 (2011) [the one Jacob uses]

The canonical QMOM realizable-FV recipe (full method also in AIAA-2010-1080):

- **Reconstruct quadrature *weights* at high order, but *abscissas* at first order
  (piecewise constant).** Holding abscissas constant is the central trick: it lets the
  weight update at each node be grouped so the positive term dominates the negative
  upwind contributions, keeping the effective VDF non-negative.
- **Per-node outgoing-flux positivity constraint on Δt:** for each quadrature node, the
  cell weight minus the net *outgoing* reconstructed-weight flux must stay positive
  (interior-side reconstructed abscissa only). Generalizes to a sum over faces in 3D.
- **Local first-order fallback:** wherever the positivity condition fails, drop that
  node's weight slope to zero. (A realizability slope limiter by another name.)
- **Cost / limitation:** "quasi-p-th order" — abscissas are only first-order, so formal
  order drops where abscissas vary strongly, which is exactly the high-Ma,
  large-velocity-gradient regime.

This is what **Posey, Fox & Houim (arXiv:2603.13697, 2026)** do in practice (see §5).

### Option B — Fan, Huang & Wu, arXiv:2510.18380 (Oct 2025) [provable, accuracy-preserving, HyQMOM-specific]

*"Provably realizability-preserving finite volume method for quadrature-based moment
models of kinetic equations"* — built **specifically for three-point HyQMOM and
two-node Gaussian-EQMOM five-moment systems**, i.e. our closure family. The
accuracy-preserving alternative to Vikas:

- Recasts realizability as a **non-negative quadratic form in the moment vector** (a
  bilinear reformulation of the Hankel-PSD condition) — realizability becomes a cheap
  algebraic check rather than ray–boundary geometry.
- **Realizability-embedding HLL:** rigorously derived wave speeds and intermediate
  ("star") states proven to stay in R, so the first-order building block maps R → R.
- A **scaling-type limiter** enforces strict realizability of reconstructed interface
  states **in moment space, without degrading accuracy** (no first-order abscissas).
- Explicit **CFL bounds**: collisionless, plus a BGK bound *uniform in the relaxation
  time* (handles stiff continuum → kinetic transitions), with a semi-implicit BGK
  variant inheriting the collisionless bound.
- **Limitation:** currently **1D / five-moment only.** Extension to 3D and 35 moments is
  not treated — that extension is the genuinely novel, paper-worthy piece.

### Closure-side complement — realizability-interior closures

- **Bandopadhyay, arXiv:2606.26032 (2026)** — "realizability-interior closures for
  odd-order kinetic moment systems." Choosing the closure margin D(U)=0 saturates the
  realizability bound and **loses hyperbolicity** — exactly our "leaves R → non-finite"
  path. A one-parameter family C_{η,n} stays strictly interior (D>0); the **Morin–
  McDonald closure (η=1)** is the maximal-margin endpoint with **zero failures on
  near-vacuum / extreme-anisotropy stress tests** (the geometric endpoint η=0 is ~1%
  more accurate but shows ~1% failures at extreme states).
- **McDonald & Torrilhon, JCP 251 (2013)** and the McDonald-group hyperbolicity work
  (our local ref: Rice, Plante-Sabourin & McDonald, JCP 562, 2026, "Robustly hyperbolic
  high-order moment-closures for multidimensional gases") attack the same pathology from
  the closure side: there exist realizable states where the closure degenerates / loses
  hyperbolicity; interior closures and closed-form fluxes restore robustness.

### Other transplants of the same template (for reference)

- Alldredge & Schneider, JCP 295 (2015) — realizability-preserving DG for entropy-based
  (M_N) closures; approximate R by a convex polytope (H-representation) for cheap ray
  projection. Documents that the entropy optimization blows up near ∂R — the
  entropy-closure analogue of our negative-variance blow-up.
- Schneider, Kall & Alldredge, JCP 322 (2016) — Kershaw closures, high-order
  realizability-preserving DG (closed-form, no per-cell optimization).
- Sabat, Larat, Vié & Massot (2014) — convex-state-preserving DG for disperse-phase
  flows on unstructured meshes.
- TVD variants: JCP 408 (2020); arXiv:2205.10974 — lighter-weight realizable
  second-order flux limiters, if retrofitting MUSCL minimally.

---

## 5. What Jacob's paper actually does (and the key nuance)

**Posey, Fox & Houim, arXiv:2603.13697 (Mar 2026)** — *"A robust high-resolution
algorithm for quadrature-based moment methods applied to high-speed polydisperse
multiphase flows."*

Important: his moments are **particle-size (mass) moments with one velocity per node
(monokinetic), closed with GQMOM** — **not** velocity-moment HyQMOM. So he never has to
keep a *velocity* standardized-moment vector inside R the way our 35-moment solver does.
His paper **sidesteps the high-order velocity-realizability problem rather than solving
it with a limiter.** The robustness comes from:

- **Reconstruct quadrature-node / primitive variables, not raw moments.** "High-order
  reconstruction of the mass moments and abscissae are prone to moment corruption and
  other unrealizability issues" → **abscissas reconstructed first-order** (= Option A);
  weights and conditional variables get 5th-order WENO + a TVD limiter.
- **Graduated, geometry-aware order reduction near vacuum — not a single global floor.**
  Detect vacuum interfaces / particle "islands" and "lakes" by counting sign changes of
  (α − α_min) across the stencil; if >1: drop 5th → 3rd-order WENO; if still bad: drop
  to **first order**. Also degrade on large abscissa variation across the stencil (>5%).
- **Most-diffusive flux on degraded edges:** Rusanov where reconstruction has fallen to
  first order; HLLC (gas) / AUSM⁺-up (granular) elsewhere.
- **Realizability repair:** check Θ ≥ 0; floor/remove cells whose moments go unrealizable
  during transport (`α_p < 1e-11` or `M_0 < 1e5` → remove particles, fill void with
  gas); dedicated moment-correction algorithms flagged as future work.
- SSP-RK + Strang splitting.

His PhD thesis (Posey, *High-Order Multiphase Modeling of Reactive Polydisperse
Particles*, Univ. Florida, 2024; local copy `~/Posey_J.pdf`) elaborates the same recipe
and frames the graduated order-reduction as a **TENO-like idea run backwards**: rather
than *growing* the stencil until a valid high-order one is found, it keeps the
highest-order method it was given and *shrinks* the stencil (WENO5 → WENO3 → first-order)
until the reconstruction is "safe from corruption." Abscissas are first-order "as is
standard for higher-order QMOM implementations."

The transferable lesson: **don't reconstruct standardized moments directly; reconstruct
realizability-safe variables, first-order the fragile ones, and reduce order by stencil
inspection (graduated), using the most diffusive flux on degraded edges** — strictly
more refined than a single hand-set density floor.

---

## 6. Diagnosis of our current code and the recommended path

**We already have the principled per-cell mechanism — the paper's correction/projection.**
First-order realizability is *not* the problem: `realize_M4_projection.jl` / `projection35.jl`
(Appendix B projection) and `realizable_3D.jl` / `realizability_S*.jl` (main-text Sylvester
cascade) restore the cell-mean moments to R after every flux, and that is exactly what lets
the published first-order scheme run at arbitrary Ma/Kn. The missing piece is purely at the
**high-order reconstruction** layer.

**Our `ho_vacuum_floor` is a crude, global stopgap for that reconstruction layer only** —
one density threshold that flips the *entire* high-order path back to first order. It does
**not** replace the per-cell projection (that still runs); it just avoids feeding the
projection a face state that has already overflowed. That is why it has an irreducible
robustness↔sharpness tradeoff: too low → crash, too high → smears the jet fringe. The
methods below replace this single threshold with a **local, graduated, realizability-aware**
reconstruction that keeps the *face* states in R before the flux — layered on top of the
paper's existing per-cell correction, not instead of it.

**Our reconstruction variable is the problem.** We reconstruct standardized central
moments and recombine independently-limited slopes — exactly the "reconstruct raw
moments → corruption" pattern Vikas and Posey warn against.

Recommended path, in increasing ambition:

1. **Minimal, closure-agnostic fix (Zhang–Shu / Vikas template):** add a **scaling
   limiter** to the MUSCL reconstruction. Build a realizability oracle for our moment
   vector (Hankel-PSD test, or the Fan–Huang–Wu quadratic form), then for each
   reconstructed face state compute the largest θ∈[0,1] with `w̄+θ(w_f−w̄) ∈ R`; fall
   back to first order locally where θ→0. Replaces the global floor with a principled,
   local limiter. Confirm SSP-RK3 and the realizability CFL bound.
2. **Graduated geometry-aware fallback (Posey-style):** detect near-vacuum stencils and
   degrade order locally (2 → 1), Rusanov/most-diffusive flux on degraded faces, instead
   of the global floor.
3. **Principled, accuracy-preserving, paper-worthy (Fan–Huang–Wu → 3D/35-moment):** port
   their realizability-embedding HLL + quadratic-form scaling limiter from 1D/5-moment to
   our 3D 35-moment system. Genuinely novel; squarely Jacob's expertise.
4. **Closure-side robustness:** consider a realizability-interior closure
   (Morin–McDonald / maximal-margin) so the closure never sits exactly on ∂R where
   hyperbolicity is lost.

**Honest caveats from the literature:** these proofs guarantee realizability of cell
means and reconstructed point values *under an explicit CFL* — they do **not** guarantee
accuracy near ∂R, where everything degrades to first order anyway (consistent with
Rodney's "minimum density below which the scheme is first order"). At extreme
anisotropy/Mach the *closure* (not just the scheme) can lose hyperbolicity; only interior
closures or floors fix that. Most rigorous proofs are 1D / structured; multi-D
realizability across faces and dimensional splitting needs care (convexity still helps).

### Implemented method (optional, `ho_realizability_limiter=true`)

The Zhang–Shu scaling limiter (§3 recipe 3) has been implemented as an **opt-in
alternative** to the default `recon_face_pair` binary fallback. The implementation
consists of three layered components:

**Realizability oracle.** `realizability_margin(m)` / `is_realizable(m)` in
`src/realizability/realizability_oracle.jl` test whether a 35-moment vector lies in the
realizable set R. They reuse the same `delta2star3D` smallest-eigenvalue test as the
shipped Appendix B projection (`projection35.jl`): compute the smallest eigenvalue
λ₁ of the 6×6 Hankel-like block; if λ₁ ≥ 0 the moment vector is realizable. This
is the same realizability criterion the paper already uses for the cell-mean
correction — it is not a new or inconsistent test.

**Cell-wise scaling limiter.** `scaling_limited_faces` replaces `muscl_faces +
recon_face_pair` at the face-reconstruction step. For each cell and each face
direction it performs a bisection search (or analytic bound) for the largest θ∈[0,1]
such that:
```
w̃_face = w̄_cell + θ*(w_face − w̄_cell)  ∈ R
```
Because R is convex and the cell mean lies in the interior of R (maintained by the
per-cell Appendix B projection), such θ always exists. θ=1 returns the unmodified
MUSCL reconstruction in smooth regions (full design accuracy); θ→0 degrades
continuously to the first-order cell-centered state at individual faces near vacuum.
The limiter is cell-local: it does not touch faces in well-resolved regions.

**SSP-RK3 time integration.** Each stage of SSP-RK3 is a convex combination of
forward-Euler steps. If each forward-Euler step maps R → R under a CFL bound (which
the first-order HLL does, per Laurent & Fox ESAIM 2024), then every SSP-RK3 stage
maps R → R by convexity. Combined with the scaling limiter — which guarantees
realizable face states entering the HLL flux — realizability of cell means is
preserved by construction through the entire high-order update.

**Comparison with the default path:**

| property | `ho_vacuum_floor` (default) | `ho_realizability_limiter` (opt-in) |
| --- | --- | --- |
| fallback granularity | global density threshold; entire cell drops to first order | per-face, continuous-θ; individual faces degrade independently |
| threshold | hand-set (e.g. 1e-3); robustness↔sharpness tradeoff | none; θ determined automatically by the realizability test |
| realizability guarantee | no (fallback heuristic avoids the problematic region) | yes, by construction (convexity of R + SSP-RK3) |
| vacuum penetration (observed) | limited by the floor (~1e-3) | deeper (~9.7e-6 ρ_min observed in 1D repro) |
| default | yes | no — `ho_vacuum_floor` retained as default |

**Activation:** set `ho_realizability_limiter=true` in the params named tuple passed
to `simulation_runner`. Demo environments: `REPRO_LIMITER=1` (3D demo) and
`R1D_LIMITER=1` (cheap 1D serial repro, no MPI). The `ho_vacuum_floor` param is NOT
removed; both can coexist.

**Framing note.** The limiter is **not a crash-fix**: the existing `recon_face_pair`
binary guard in the default path already prevents the original Ma=100 1D crash by
falling back to first order when the reconstructed face is non-realizable. The
scaling limiter's value is that it is a **principled, graduated alternative** — it
keeps more high-order accuracy near vacuum (continuous θ rather than all-or-nothing)
and provides a realizability guarantee by construction rather than by a hand-set
density.

### Implemented method + results summary

Measured results from RP-T6/RP-T7 are available; see
`docs/realizability-preserving-highorder-results.md` for the full collaborator-facing
write-up. Key findings:

- **Smooth accuracy preserved:** L1 self-convergence on density = 1.865 (32→64) and
  1.973 (64→128) with limiter ON — ~2nd order, identical to limiter-off. θ fraction
  below 1 = 0.000 on the smooth sinusoid.
- **Locality confirmed:** θ<1 in ≈4.7% of cells on the colliding-slab+vacuum case,
  confined to the low-density band.
- **Sharpness gain (1D Ma=10, Nc=128):** peak density 2.035 (limiter+HO) vs. 1.843
  (first-order); ratio ≈1.105.
- **Two-layer design quantified:** with projection in report-only mode, the limiter
  alone does NOT keep all cell means realizable (min margin ≈ −2.71, 102 unrealizable
  cell-stages at CFL 0.9). The Appendix B projection backstop remains required.
- **Stability CFL:** ≥ 0.90 in the tested 1D problem.
- **Golden-gate:** 0 entries failing at 1e-10; default path byte-identical.

Deferred (not done): full 3D Ma=10/25/50/100 Mach-ladder HPC run; finer-grid / 3D
CFL sweep; FHW quadratic-form oracle (perf optimization); formal realizability-
preserving HLL theorem for 3D/35-moment (the publishable extension — Jacob's
territory).

---

## 7. Key references

| ref | what it gives us |
| --- | --- |
| Fox & Laurent, SIAM J. Appl. Math. 82 (2022), arXiv:2103.10138 | the 1D HyQMOM closure we implement; realizability-interior by construction |
| Laurent & Fox, ESAIM Proc. Surv. 76 (2024) | first-order HLL is realizability-preserving under CFL |
| Fox, Laurent & Vié, JCP 365 (2018); Patel, Desjardins & Fox, JCP:X 1 (2019) | CHyQMOM / 3D CHyQMOM — our 35-moment lineage |
| **Vikas, Wang, Passalacqua & Fox, JCP 230 (2011)** | high-order weights + first-order abscissas + per-node positivity CFL + local fallback (Option A) |
| **Fan, Huang & Wu, arXiv:2510.18380 (2025)** | provable realizability-preserving FV *for HyQMOM*; realizable HLL + quadratic-form limiter; 1D/5-moment (Option B) |
| Posey, Fox & Houim, arXiv:2603.13697 (2026); Posey PhD thesis (UF, 2024, `~/Posey_J.pdf`) | applied recipe: reconstruct nodes/primitives, first-order abscissas, graduated (TENO-like, shrinking) vacuum fallback, Rusanov on degraded edges |
| Zhang & Shu, JCP 2010/2011; Proc. R. Soc. A 467 (2011) | the positivity/bound-preserving template all of the above transplant |
| Alldredge & Schneider, JCP 295 (2015); Schneider et al., JCP 322 (2016) | scaling-limiter DG for entropy-based / Kershaw closures (closure-agnostic confirmation) |
| Bandopadhyay, arXiv:2606.26032 (2026); McDonald & Torrilhon, JCP 251 (2013); Rice, Plante-Sabourin & McDonald, JCP 562 (2026) | realizability-interior / robustly-hyperbolic closures (closure-side fix) |
