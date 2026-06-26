# Realizability-preserving high-order reconstruction — implemented method and measured results

**For:** Rodney Fox and Jacob Posey
**Branch:** `projection35-port` (HyQMOM.jl)
**Date:** 2026-06-26
**Status:** opt-in feature, golden-clean, default path byte-identical

---

## Context

Our JCP paper (*Fourth-Order HyQMOM Closures for Multidimensional Kinetic Equations*)
lists as a Future Work item: *"implement high-order spatial reconstruction schemes to
reduce numerical diffusion in the HLL solver."* The work described here is the 3D/35-
moment instantiation of that item, realized using the Zhang–Shu scaling-limiter template
as formulated by Fan, Huang & Wu (arXiv:2510.18380, 2025) for HyQMOM — extended from
their 1D/5-moment setting to our 3D/35-moment system — and layered **on top of** the
paper's existing per-cell Appendix B projection, not replacing it.

The high-order reconstruction problem (MUSCL face states leaving the realizable set R in
near-vacuum) was diagnosed in `docs/realizability-highorder-literature.md`. This note
records what was built and what was measured.

---

## 1. What was implemented

### 1.1 Realizability oracle

`realizability_margin(M)` / `is_realizable(M; lam_min)` in
`src/realizability/realizability_oracle.jl`.

Returns the smallest eigenvalue of `delta2star3D` — the 6×6 Hankel-like block whose
positive semidefiniteness is the paper's realizability criterion. The oracle reuses
exactly the same test as the shipped Appendix B projection (`projection35.jl`): there
is no new or inconsistent realizability criterion. `is_realizable` returns true iff
`lam_min ≥ 0` (default) or ≥ a caller-specified tolerance.

### 1.2 Cell-wise Zhang–Shu scaling limiter

`scaling_limited_faces` in `src/numerics/reconstruction.jl`.

For each cell and each face direction the limiter finds the largest θ∈[0,1] such that

```
w̃_face = w̄_cell + θ * (w_face − w̄_cell)  ∈  R
```

Because R is convex and the cell mean w̄ lies in the interior of R (maintained by the
per-cell Appendix B projection that runs on every timestep), such θ always exists. θ=1
returns the unmodified MUSCL reconstruction (full design order); θ=0 is first-order
(cell-centered). Individual faces degrade continuously and independently — cells in
smooth regions are not touched. The limiter replaces the existing `recon_face_pair`
binary all-or-nothing fallback when activated.

The same SSP-RK3 integrator is used throughout. Because SSP-RK3 stages are convex
combinations of forward-Euler steps, and the first-order HLL is realizability-
preserving under a CFL bound (Laurent & Fox, ESAIM 2024), realizability of cell means
is preserved by construction through the full high-order update when the limiter is
active.

### 1.3 Opt-in wiring

The limiter is **off by default.** All existing behavior is preserved unchanged.

Activation points:
- `residual_1d`, `residual_line`, `residual_ho_3d!`, `step_highorder_3d!` — accept
  keyword `use_limiter::Bool` (default `false`).
- `simulation_runner` — parameter `ho_realizability_limiter` (default `false`).
- Environment variables: `REPRO_LIMITER=1` (3D repro environment) and `R1D_LIMITER=1`
  (cheap 1D serial repro, no MPI).
- Diagnostic: `HYQMOM_PROJ_COUNT` env enables per-run reporting of how many cell-stage
  projection activations occurred, allowing comparison of limiter-on vs. limiter-off.

The existing `ho_vacuum_floor` path is **not removed**; the two mechanisms coexist.

### 1.4 Two-layer design (faces + cell means)

This implementation confirms and quantifies the two-layer design:

- **Layer 1 (faces):** the scaling limiter keeps reconstructed face states in R before
  the HLL flux is evaluated, preventing the flux from receiving non-realizable inputs.
- **Layer 2 (cell means):** the per-cell Appendix B projection restores cell means to R
  after the flux update, as it has always done.

The limiter does not replace the projection; it provides a principled graduated
alternative to the binary `recon_face_pair` fallback at the reconstruction layer.

---

## 2. Measured results

All numbers are from RP-T6 and RP-T7 validation runs on `projection35-port`. No
numbers have been rounded up or adjusted; deferred items are explicitly marked below.

### 2.1 Smooth-flow order of accuracy

Problem: 1D smooth periodic sinusoidal initial condition, self-convergence study.

| grid refinement | L1 convergence rate on density |
| --- | --- |
| 32 → 64 cells | **1.865** |
| 64 → 128 cells | **1.973** |

The limiter is inactive on smooth data (θ=1 everywhere): the fraction of cells with
θ<1 is **0.0000** on the smooth sinusoid. Accuracy is identical to limiter-off.

### 2.2 θ-locality in near-vacuum

Problem: 1D colliding slabs with vacuum gap.

Fraction of cells with θ<1: **≈0.047 (4.7%)**, confined to the low-density band at
the vacuum interface. The limiter is local by construction and does not degrade smooth
or moderate-density regions.

### 2.3 Sharpness vs. first-order (1D colliding jets, Ma=10, Nc=128)

| scheme | peak density |
| --- | --- |
| limiter + HO | **2.035** |
| first-order only | 1.843 |
| ratio | **≈1.105** |

The limiter recovers approximately 10.5% more peak density than pure first order,
demonstrating that the graduated continuous-θ degradation retains more high-order
accuracy than the all-or-nothing fallback.

### 2.4 Projection backstop remains required in deep vacuum

With the Appendix B projection placed in report-only mode (projection tracking but not
correcting), the limiter alone does **not** keep all cell means realizable during the
colliding-slab+vacuum run:

- minimum realizability margin during the run: **≈ −2.71**
- number of unrealizable cell-stage entries at CFL 0.9: **102**

This confirms the two-layer design: the scaling limiter (faces) and the Appendix B
projection (cell means) are complementary, not redundant. The projection backstop
remains required.

### 2.5 Scheme-stability CFL

With both limiter and projection active: stable at CFL **≥ 0.90** in the tested 1D
problem.

---

## 3. Golden-gate status

All changes pass the golden-gate test at 1e-10 tolerance (0 entries failing). The
default code path — limiter off, `ho_vacuum_floor` as before — is byte-identical to the
pre-limiter baseline. `using HyQMOM` loads without error.

---

## 4. Deferred (not done in this work)

The following items are explicitly out of scope for this implementation and are left for
future work:

- **Full 3D Mach-ladder validation** (Ma=10, 25, 50, 100 at Np=128 on HPC): the
  dedicated job has not yet been run. The 1D results above motivate the 3D sweep but do
  not replace it.
- **Finer-grid / 3D realizability-CFL sweep:** the CFL≥0.90 result is 1D; a 3D CFL
  bound specific to this solver and the 35-moment system has not been measured.
- **Fan–Huang–Wu non-negative quadratic-form oracle:** the FHW paper proposes a cheap
  bilinear reformulation of the Hankel-PSD test as a performance optimization. The
  current oracle uses the eigenvalue-based test (correct, just slower). This is a
  performance improvement, not a correctness item.
- **Formal theorem (realizable HLL star state for 3D/35-moment):** proving that the HLL
  intermediate state lies in R for our 3D 35-moment system — the publishable core that
  would make this a complete paper result — is out of scope here. This is the
  natural next step extending Fan–Huang–Wu to our system and is Jacob's territory.

---

## 5. Relationship to the literature

| component | provenance |
| --- | --- |
| Realizability-preserving first-order HLL | Laurent & Fox, ESAIM Proc. Surv. 76 (2024) |
| Appendix B per-cell projection (Layer 2) | Bryngelson, Fox & Laurent JCP (in revision), Appendix B — already shipped |
| Zhang–Shu scaling-limiter template | Zhang & Shu, JCP 2010/2011; Proc. R. Soc. A 467 (2011) |
| HyQMOM-specific limiter formulation | Fan, Huang & Wu, arXiv:2510.18380 (2025) — 1D/5-moment; this work extends to 3D/35-moment |
| Graduated per-node fallback alternative | Vikas, Wang, Passalacqua & Fox, JCP 230 (2011); Posey, Fox & Houim, arXiv:2603.13697 (2026) |
| SSP-RK3 realizability convex-combination argument | Vikas et al. (2011); FHW (2025) |

Full literature discussion: `docs/realizability-highorder-literature.md`.
