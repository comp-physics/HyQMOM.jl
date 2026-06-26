# A research-grade Riemann solver for 3D 35-moment HyQMOM — scope & literature

Notes for Fox, Posey, and SHB. The reconstruction half of the high-order scheme is in place
(scaling limiter, projection-triggered first-order, realizability oracle, per-cell projection
backstop — see `realizability-highorder-literature.md` and `realizability-preserving-highorder-results.md`).
The **flux** underneath is still first-order **HLL**, and the Mach-ladder data below shows the flux is
now the dominant remaining diffusion source. This note scopes what a "clever," research-grade,
realizability-preserving Riemann solver for this system would look like, grounded in the literature.

**Everything here is planned as OPT-IN** (a `riemann_solver` selector defaulting to `:hll`, the
current behavior, byte-identical and golden-gated; see [[optin-and-documented-features]]).

---

## 1. What the data says the flux is costing us

3D crossing jets, Np=64, matched dynamical time `tmax=0.02·10/Ma`, `ho_vacuum_floor=0`, Granite Rapids,
three reconstruction controls all on the **same HLL flux**:

| Ma | first-order peak ρ | scaling-limiter | projection-triggered | first-order ∇ρ | limiter ∇ρ | projrec ∇ρ |
| --- | --- | --- | --- | --- | --- | --- |
| 10  | 0.454 | 0.596 (1.31×) | 0.655 (1.44×) | 6.16 | 9.51 | 13.2 |
| 25  | 0.456 | 0.537 (1.18×) | 0.639 (1.40×) | 5.75 | 8.14 | 11.9 |
| 50  | 0.440 | 0.510 (1.16×) | 0.656 (1.49×) | 5.14 | 7.47 | 10.2 |
| 100 | 0.442 | 0.488 (1.10×) | 0.628 (1.42×) | 4.95 | 6.42 | 8.86 |

Mass conserved to identical digits (9.47458e2) in all 12 runs; **no crashes at any Mach including
Ma=100 with no density floor.** Centro-symmetry error: first-order ~1e-13 (exact), both high-order
controls ~3–9% (shrinking with Ma).

Takeaways that set the Riemann-solver requirements:
1. **All the sharpening so far is from reconstruction; the flux is HLL.** Even projection-triggered
   high-order tops out at ~0.66 peak density vs a true jet value of 1.0 — the rest is HLL contact
   diffusion. The flux is the next lever.
2. **Realizability is solved at the control/reconstruction layer**, so the Riemann solver does not
   have to *fix* robustness — but it must **not break it**: its star state(s) must stay in the convex
   realizable cone R near vacuum.
3. **Symmetry is a separate, orthogonal issue** (a decomposition/reconstruction asymmetry — NOT the flux,
   and NOT operator splitting; the scheme is unsplit) — see §6.

---

## 2. Why HLL is the wrong solver here, precisely

HLL (Harten, Lax & van Leer, SIAM Rev. 25, 1983) keeps only the two extreme wave speeds and collapses
the **entire intermediate fan into one constant star state**. The 35-moment HyQMOM flux Jacobian has
~15 real, interlacing eigenvalues per direction (Fox & Laurent, SIAM J. Appl. Math. 82, 2022): one
contact-like field at the mean velocity, plus many linearly-degenerate (LD) intermediate/shear waves.
HLL smears **all** of them at the rate of the fastest waves — an O(1) error independent of mesh. In the
crossing jets the dominant smeared feature is exactly the **density contact** (jet ρ≈1 vs background
ρ≈0.001) and the **shear** at jet edges — the LD fields HLL diffuses worst.

So the goal is a solver that **restores the contact/shear (LD) structure** while **preserving
realizability near vacuum**.

---

## 3. The candidate families (with literature and fit)

### 3.1 Realizability-embedded HLL — the robust, proven baseline (low effort)
- **Fan, Huang & Wu 2025** (arXiv:2510.18380): a *provably* realizability-preserving HLL for HyQMOM.
  Wave speeds are the **abscissa bounds**: for 3-node HyQMOM `δ⁺ = max{v₁,v₂,v₃,0}`, `δ⁻ = min{v₁,v₂,v₃,0}`
  (no safety factor — the delta-node support bounds propagation exactly). Proves `δ⁺M−F(M)∈R` and
  `F(M)−δ⁻M∈R`, so the HLL star state is a convex combination of realizable states → realizable, under
  `Δt/Δx·max(δ⁺−δ⁻) ≤ 1/2`. Plus a scaling limiter for high-order interface states and a BGK-uniform CFL.
  **1D, 5-moment only.**
- **Laurent & Fox 2024** (ESAIM Proc. 76): proves the standard HLL on HyQMOM is realizable under CFL via
  an orthogonal-polynomial property; wave speeds = roots of the characteristic polynomial.
- **Fit:** this is essentially our current HLL with *tighter, realizability-aware* wave speeds — a cheap
  upgrade that reduces diffusion slightly and gives a realizability guarantee. But it is **two-wave**:
  it fixes realizability, **not** contact smearing. It is the baseline the cleverer solvers build on.

### 3.2 HLLC — restore the one contact (low effort, high value)
- **Toro–Spruce–Speares 1994; Batten et al. 1997** — insert a contact wave at `S*` between the acoustic
  waves; provably positively conservative under Einfeldt speeds.
- **Precedent on moment systems:** Sangam 2008 and Berthon–Dubroca–Sangam (ten-moment Gaussian closure,
  HLLC, proven positive-definite pressure tensor); Berthon–Charrier–Dubroca 2007 (M1 radiation, HLLC kept
  inside the realizability cone); Wang–Tang–Wu 2024 (DG ten-moment HLLC). HLLC for moment closures is
  established — **but not for 35-moment HyQMOM.**
- **Fit:** for HyQMOM we *know the contact speed* — CHyQMOM pins an eigenvalue at the mean normal
  velocity `u_n`, so `S* = u_n` is free. Restoring that one wave directly de-diffuses the density
  interface. Highest value-per-effort first step. Caveat: one contact under-resolves the ~13 LD
  intermediate waves; star-state realizability must be checked (use our oracle; fall back to HLL).

### 3.3 HLLEM / HLLI — anti-diffuse the whole LD sub-block (medium effort, best ROI) ★
- **Einfeldt–Munz–Roe–Sjögreen 1991** (HLLEM); **Dumbser & Balsara 2016** (JCP 304, general
  conservative + non-conservative); **Dumbser et al. 2018** (HLLI, multi-D), arXiv:1801.00450.
  Replaces HLL's constant star with a **linear profile**, anti-diffusing **only the linearly-degenerate
  fields**: `f_HLLEM = f_HLL − φ·(S_L S_R)/(S_R−S_L)·R* δ* L* (Q_R−Q_L)`, using only the **inner (LD)
  eigenvectors**, with `δ*` bounded in (0,1] and `φ∈[0,1]` blending HLL↔HLLEM.
- **Why it fits 35-moment HyQMOM specifically:** (a) needs only the **inner eigenvectors**, not the full
  35×35 eigendecomposition — and HyQMOM's orthogonal-polynomial/abscissa structure supplies them cheaply;
  (b) **path-conservative** form handles the non-conservative products that appear in moment closures;
  (c) **inherits HLL positivity** because anti-diffusion lives strictly in the LD subspace — no entropy
  fix, robust at low density where Roe fails; (d) falls back to HLL on NaN.
- **Precedent on a moment system:** Ben Nasr, Gerolymos, Vallet 2014 (arXiv:1307.2154) — HLLEM
  anti-diffusion for Reynolds-stress transport (the only HLLEM-on-a-moment-system in the literature).
- **Caveat that is actually a feature:** HLLEM drops the anti-diffusion at **coalescing eigenvalues** —
  which for HyQMOM happen on the **realizability boundary (vacuum)**. So it gracefully reverts to robust
  HLL exactly where robustness matters, and de-diffuses everywhere else.
- **Fit:** the best ROI — restores contact AND shear, cheap (inner eigenvectors only), keeps HLL
  positivity. This is the recommended centerpiece.

### 3.4 Kinetic flux (KFVS) — the native, realizable-by-construction flux (medium effort)
- **Vikas, Wang, Passalacqua & Fox 2011; Desjardins–Fox–Villedieu 2008.** Because the HyQMOM Jacobian
  eigenvalues *are* the quadrature abscissas, the abscissa-upwind flux `F_k = Σ_α n_α U_α^{k+1}` split by
  `sign(U_α)` is the characteristic-exact flux. **Realizable at first order by a convex-combination
  argument** (the half-fluxes use sub-sets of the non-negative quadrature). It is a multi-wave solver
  (one wave per node) and is *consistent with the closure itself* — no separate macroscopic wave model.
- **Honest caveat:** high-order kinetic realizability is **not proven** for QBMM (Fan–Huang–Wu use a
  tailored HLL precisely because of this); Vikas keeps abscissas first-order ("quasi-high-order").
- **Fit:** the most elegant "realizable by construction" option; first-order KFVS is still diffusive, so
  it needs the high-order reconstruction we already have. A strong alternative core to HLLEM.

### 3.5 Relaxation / Suliciu — exact contacts + provable positivity (high effort, high rigor)
- **Bouchut 2004 (book); Suliciu 1990; Chalons–Coulombel–Serre 2012** (the abstract template for any
  system with an entropy); **Bouchut–Klingenberg–Waagan 2007/2010** (7-wave MHD relaxation — the worked
  example that the framework scales to many LD waves with provable positivity + discrete entropy + exact
  contacts); **Coulombel–Goudon 2006** (the realizable cone is an invariant region for entropy-based
  moment closures — the entropy input the framework needs).
- **Fit:** the principled route to *exact* contact resolution with provable positivity/entropy, but you
  must design the relaxation system for 35 moments (choose relaxation variables, verify the
  subcharacteristic condition). High payoff, high cost. The "rigorous" alternative to HLLEM.

### 3.6 Demoted / dead ends
- **Roe / full characteristic:** dead end at 35 moments. Einfeldt's theorem — *no linearized solver is
  positively conservative* (fails near vacuum); no analytic eigenvectors; O(35³)/interface; the field
  abandons Roe above ~5 moments (Boccelli et al. 2024 use Rusanov for 14-moment; McDonald–Torrilhon 2013
  note no closed-form flux for ≥14-moment). **Middle ground:** PVM / Newton-Roe (Pimentel-García et al.
  2021) approximates `|A|` with a matrix polynomial needing only eigenvalue *bounds* (we have the
  abscissas), O(n) matvecs — but realizability near vacuum needs a limiter bolted on.
- **Gas-kinetic / UGKS (Xu):** poor fit at Kn=1000 — in the collisionless limit it *degenerates to the
  same free-transport upwinding as KFVS* while requiring a continuous-distribution interface model
  incompatible with Dirac-delta quadrature. No QBMM-GKS exists. Only earns its cost at small Kn
  (impinging jets), where a low-Mach fix (§6) may matter more.

---

## 4. Recommendation map

| Goal | Best fit | Realizability near vacuum | Effort |
| --- | --- | --- | --- |
| Robust baseline + realizability guarantee | HLL with FHW/Laurent–Fox wave speeds (`δ±=max/min{vᵢ,0}`) + scaling limiter | proven under CFL≤1/2 | low (largely have it) |
| Restore the density contact cheaply | **HLLC** (`S*=u_n` is free from the closure) | check star state w/ oracle, fall back to HLL | low |
| **Restore contact + shear, keep HLL positivity** | **HLLEM/HLLI** anti-diffusing the LD sub-block via HyQMOM inner eigenvectors | inherits HLL positivity; reverts to HLL at vacuum | **medium — best ROI** |
| Realizable-by-construction, closure-consistent | **KFVS** (abscissa-upwind kinetic flux) | first-order realizable by construction | medium |
| Exact contacts + provable positivity/entropy | Suliciu/relaxation (CCS framework; BKW many-wave template) | provable under subcharacteristic condition | high |
| Roe-quality upwind w/o eigendecomposition | PVM / Newton-Roe (abscissa bounds) | not guaranteed → add limiter | medium |
| Full Roe | — | fails (Einfeldt; no eigenvectors; vacuum blow-up) | not recommended |

---

## 5. Recommended research-grade design (staged, all opt-in)

A `riemann_solver` selector (`:hll` default, byte-identical; `:hllc`, `:hllem`, `:kinetic` opt-in),
golden-gated, documented per [[optin-and-documented-features]]. Each stage layered on the realizable
HLL base and on the reconstruction controls we already have:

- **Stage A — HLLC** (cheap win): restore the contact at `S*=u_n`; check star realizability with the
  oracle, fall back to HLL where it would exit R. Expect the biggest single drop in contact diffusion.
- **Stage B — HLLEM/HLLI** (centerpiece): anti-diffuse the LD sub-block using HyQMOM's inner
  eigenvectors; `φ`-blend to HLL; auto-revert at coalescing eigenvalues (vacuum). Restores shear too.
- **Stage C — the paper:** either (i) a **kinetic/KFVS** flux from the closure (realizable-by-
  construction, closure-consistent) or (ii) a **Suliciu/relaxation** solver (exact contacts, provable
  positivity) for the 35-moment system — benchmarked against HLLC/HLLEM. Both are novel at this order.

**The publishable opening:** no HLLC/HLLEM/relaxation solver exists for a 35-moment (or comparably
high-order) HyQMOM/Grad system. Combining **Dumbser–Balsara path-conservative HLLEM anti-diffusion**
with **Fan–Huang–Wu realizability-embedded wave speeds + a scaling limiter** is novel and squarely
targets "clever, low-diffusion, realizability-preserving." Most positivity proofs are first-order; high
order needs the convex-set scaling limiter we already built.

---

## 6. Orthogonal: the symmetry issue (NOT operator splitting — corrected)

**Correction (verified against the code 2026-06-26):** the 3D high-order scheme is **unsplit** —
`residual_ho_3d!` sums the x/y/z line residuals into one residual and a single SSP-RK3 advances it
(`step_highorder_3d!`). There is **no dimensional/operator splitting**, so the earlier "Strang splitting"
fix and the Roe-1991 split-commutator explanation do **not** apply. A diagnostic on the Ma=10 fields
(`debug/` ladder `.jld2`) shows the symmetry picture is two separate effects:

- **Swap-asymmetry at ALL orders (x vs y,z ≈ 0.11; y vs z ≈ 0.002):** even first order is *not*
  invariant under axis swaps that involve x, although it is centro-symmetric (reverse-all) to ~1e-13.
  The crossing IC is swap-symmetric in all three axes, so this is numerical — almost certainly the MPI
  x–y domain decomposition (with the rank counts used, x is split more than y; z is replicated, so y
  behaves like z). Needs confirming serial-vs-parallel and across decompositions (the prior
  "MPI-lossless" check may not have exercised an asymmetric `px≠py`).
- **High-order centro break (~0.08 at Ma=10, vs 1e-13 first order):** appears only with high-order
  reconstruction — consistent with low-dissipation amplification of a seed asymmetry (Fleischmann–Adami–
  Adams 2019), the seed here being the swap-asymmetry above and/or the nonlinear per-cell projection.

**Right fixes (a separate debugging effort, not a quick task):** (i) make the directional handling
symmetric — check the decomposition (`px=py`, halo handling) and the z-direction's outflow padding vs the
x/y halos; (ii) reflection-symmetric reductions in `residual_ho_3d!` (the `.+=` accumulation order); (iii)
if needed, genuinely multi-D HLL/HLLC (Balsara 2010/2012). These are independent of the Riemann-solver
choice. **Do NOT** add Strang splitting — there is no operator split to symmetrize.

---

## 6b. Result: HLLC is implemented but fallback-dominated on the crossing jets (A1–A4)

`riemann_solver=:hllc` is implemented (opt-in, default `:hll` byte-identical, golden-clean) and **verified
to be a genuine, conservative, contact-resolving HLLC** in isolation: on a mild-velocity jump its star
pair satisfies Rankine–Hugoniot across both acoustic waves and the contact (residuals ~1e-15), is
HLL-consistent to ~1e-16, and sharpens the HLL flux by ~23%. The per-side star cannot be 35-component
consistent for the nonlinear closure, so an **anchored coupled star pair** is used (see
`src/numerics/highorder_flux.jl`).

**However, on the high-Mach crossing jets it gives ~no benefit.** Measured at Ma=10, Np=32: first-order
`:hllc` is **byte-identical** to `:hll` (peak ρ 0.6055, `max|∇ρ|` 5.48); projection-triggered `:hllc`
differs only marginally (peak ρ 0.869 vs 0.856). Reason: the jets collide at relative velocity
≈ 2·Ma/√3 (~11 at Ma=10), and the HLLC star states **leave the realizable cone in that collision**, so
the built-in realizability fallback reverts to HLL exactly where the contact lives. The contact
restoration only engages in mild regions where it does not help.

**Implication (steers the next stage):** simple contact restoration cannot beat HLL here because
realizability is binding *in the flux*, not just the reconstruction. The fix must anti-diffuse the
contact/shear **without leaving R** — i.e. **HLLEM** (anti-diffusion confined to the linearly-degenerate
subspace, inheriting HLL positivity; §3.3) or the **realizable-by-construction kinetic flux** (§3.4).
This is the Stage-B/C work and is Jacob's domain. HLLC stays in the tree as an opt-in, validated
building block (its contact closure feeds HLLEM).

## 6c. Result: HLLEM is implemented + correct but NEAR-INERT for this closure (B1–B2)

`riemann_solver=:hllem` is implemented (opt-in, default `:hll` byte-identical, golden-clean). The code is
**mathematically correct** (Dumbser–Balsara form, sign/scaling/δ*/consistency/guard all verified): it
builds the per-axis flux Jacobian by finite differences (its extreme eigenvalues match `realize_and_speed`
exactly — `ld_eigvecs`, Task B1), extracts the 9 linearly-degenerate modes at λ=`u_n`, and applies
`f = f_HLL − φ·(sL sR)/(sR−sL)·R_inner·diag(δ*)·L_inner·ΔM`.

**But it provides essentially no anti-diffusion for this closure.** Measured `|f_hllem − f_hll|` (relative):
pure density contact **7.6e-9**, pure shear **3.9e-12**, the colliding-jet regime (u=±5.77) **1.7e-13** —
i.e. `:hllem ≈ :hll` exactly where sharpening is wanted. **Root cause:** a physical contact/shear/collision
jump has ~zero projection onto the λ=`u_n` LD eigenspace of the (FD) Jacobian (measured LD-subspace energy
of a pure-shear jump ≈ 5e-12). The anti-diffusion `R·δ*·L·ΔM` is therefore ≈ 0. This is either (i) the FD
Jacobian's **9-fold-degenerate** λ=`u_n` cluster yielding an ill-conditioned/arbitrary eigenbasis from
`eigen` (cond(V) up to ~2e4), so `R·L` is not the true spectral projector onto the invariant subspace, or
(ii) genuinely, in this closure, contact/shear moment-jumps couple to the acoustic fields (not purely
linearly degenerate). Distinguishing these requires the **analytic** Fox–Laurent eigenstructure (from the
orthogonal-polynomial factorization), not FD-`eigen` of a degenerate cluster — deep work, Jacob's domain.

**Bottom line (A–B):** both macroscopic Riemann solvers, implemented and adversarially verified, **fail to
beat HLL for the 35-moment HyQMOM closure** — HLLC because its star states leave the realizable cone in the
high-Mach collision (fallback to HLL), HLLEM because physical contact/shear jumps don't project onto the
computed LD eigenspace. The reasons are closure-structural, not coding errors. This **strongly indicates the
realizable-by-construction kinetic flux (§3.4, Stage C)** — native to the closure (eigenvalues = abscissas),
resolving waves through the quadrature nodes — as the right path, and/or that any HLLEM here needs the
analytic LD eigenstructure. Both `:hllc` and `:hllem` remain in the tree as opt-in, verified-correct
building blocks. (Performance note: the FD-Jacobian + `eigen` per face makes `:hllem` far too slow for
production as-is.)

## 7. Key references

HLL family: Harten–Lax–van Leer, SIAM Rev. 25 (1983); Einfeldt–Munz–Roe–Sjögreen, JCP 92 (1991);
Toro–Spruce–Speares, Shock Waves 4 (1994); Batten et al., SISC 18 (1997); **Dumbser & Balsara, JCP 304
(2016)** + HLLI arXiv:1801.00450; Ben Nasr et al. 2014 (arXiv:1307.2154, HLLEM for Reynolds stress);
Sangam 2008 / Berthon–Dubroca–Sangam (ten-moment HLLC); Berthon–Charrier–Dubroca, JSC 31 (2007, M1).
Realizable HLL for HyQMOM: **Fan–Huang–Wu, arXiv:2510.18380 (2025)**; **Laurent–Fox, ESAIM Proc. 76
(2024)**. Kinetic flux: **Vikas–Wang–Passalacqua–Fox, JCP 230 (2011)**; Desjardins–Fox–Villedieu, JCP
227 (2008). Relaxation: Bouchut (2004); Suliciu (1990); Chalons–Coulombel–Serre, M3AS 22 (2012);
Bouchut–Klingenberg–Waagan, Numer. Math. (2007, 2010); Coulombel–Goudon, JHDE 3 (2006). Roe limits /
middle ground: Boccelli et al., arXiv:2401.15233 (2024); McDonald–Torrilhon, JCP 251 (2013);
Pimentel-García et al., AMC 388 (2021). Closure/eigenstructure: **Fox–Laurent, SIAM J. Appl. Math. 82
(2022)**; Patel–Desjardins–Fox, JCP:X 4 (2019, 35-moment 3D CHyQMOM). Limiters/SSP: Zhang–Shu, JCP 229
(2010) & Proc. R. Soc. A 467 (2011); Gottlieb–Shu–Tadmor, SIAM Rev. 43 (2001); Wu–Shu, SIAM Rev. 65
(2023, GQL). Symmetry / multi-D: Roe, NMPDE 7 (1991); Fleischmann–Adami–Adams, Comput. Fluids 189
(2019); Colella, JCP 87 (1990, CTU); Balsara, JCP 229 (2010) & 231 (2012, multi-D HLL/HLLC). Low-Mach
(impinging regime): Thornber et al., JCP 227 (2008); Rieper, JCP 230 (2011); Minoshima–Miyoshi, JCP 446
(2021).
