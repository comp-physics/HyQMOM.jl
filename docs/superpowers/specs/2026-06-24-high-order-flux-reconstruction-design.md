# High-order spatial flux reconstruction for HyQMOM.jl — design

**Date:** 2026-06-24
**Status:** design (pending implementation plan)
**Goal (Rodney roadmap #2):** replace the first-order HLL spatial scheme with a
high-order, realizability-preserving Godunov scheme so the 3D jet-crossing problem
can be run at high Mach (target Ma≈100) **without numerical diffusion** smearing the
crossing.

**Target quality:** *match* the behavior/robustness of Jacob Posey's high-order QBMM
method (arXiv:2603.13697) on equivalent problems. Jacob is a collaborator pursuing
the same ideas; the aim is to be as good as his method first, not to one-up it. Any
HyQMOM-specific enhancements are later refinements, not the headline.

## 1. Background and motivation

The solver is currently **first-order in space and time**: first-order HLL
(`pas_HLL`/`flux_HLL`, cell-centered states, no reconstruction), dimension-split
(`Mnpx+Mnpy+Mnpz−2M`), forward Euler. It reproduces the reference MATLAB to ~5e-7
at Ma=0 and Ma=2 (Np=128, Kn=1000) after the `jacobian15` eigenvalue port and the
MPI halo-consistency fix. The remaining limitation is the O(Δx) numerical diffusion
of first-order HLL, which washes out sharp features at high Ma.

**Reference method.** Posey, Fox & Houim, *"A robust high-resolution algorithm for
quadrature-based moment methods applied to high-speed polydisperse multiphase
flows"* (arXiv:2603.13697, 2026) — "Jacob's" high-order QBMM method. Different model
(GQMOM over particle mass + compressible gas, AUSM nodal Riemann) but the numerical
recipe is the template:
- Reconstruct **nodal/primitive quadrature variables, not raw moments**, then
  re-assemble moments at the cell edge (raw-moment reconstruction corrupts
  realizability).
- **Abscissae kept first-order**; other variables WENO5 + TVD slope limiter.
- **Order degradation** near vacuum "islands/lakes" / large-abscissa-variation cells.
- **3-stage SSP-RK3** in time; 2nd-order Strang splitting for sources.
- They *delete* unrealizable cells; moment-correction (projection) is "future work."

**Our realizability tool.** HyQMOM.jl already has the validated moment-**projection**
(`realizable_3D_M4`) and **hyperbolicity correction** (`eigenvalues6{x,y,z}`,
`jacobian15`). For a single-phase kinetic gas we cannot "delete cells" the way Jacob
removes empty particle cells, so projection is our *natural* realizability safeguard
for reconstructed face states — the equivalent of the moment-correction Jacob defers,
used to reach his level of robustness, not to exceed it.

## 2. Design decisions (locked)

| Decision | Choice |
|---|---|
| Reconstruction | **MUSCL-2** (slope-limited) first; **WENO5** later (phase 3) |
| Realizability of faces | **Reconstruct standardized/central moments + project each face** with `realizable_3D_M4` + hyperbolicity correction |
| Development path | **1D prototype first**, then extend to the existing 3D x/y/z sweep |
| Time integration | **SSP-RK3** (3-stage), method of lines |

## 3. Architecture

Replace the cell-centered first-order flux with a high-order Godunov residual,
advanced by SSP-RK3. The current scheme stays available behind a
`spatial_order` switch (1 = legacy HLL, 2 = MUSCL) for A/B comparison and fallback.

Per RK stage, per direction (1D shown; 3D applies it along x, then y, then z):

```
cell-centered M
  → per cell: convert M → reconstruction variables  V = (rho, u, standardized moments)   [M2CS4_35-derived]
  → MUSCL reconstruct each component of V to faces:  V_L(i+1/2), V_R(i+1/2)               [slope limiter]
  → reassemble face moments:  M_L, M_R  from V_L, V_R
  → realize each face:  M_L ← realize+hyperbolicity(M_L, Ma);  M_R ← ...                   [our projection]
  → flux closure at faces:  F_L = Flux_closure35_3D(M_L), F_R = Flux_closure35_3D(M_R)
  → wave speeds at faces:    s_L, s_R via eigenvalues6{x,y,z} + closure_and_eigenvalues
  → HLL interface flux:      Fhat(i+1/2) = HLL(M_L, M_R, F_L, F_R, s_L, s_R)
  → residual:                L(M)_i = −(Fhat(i+1/2) − Fhat(i−1/2)) / Δx
SSP-RK3 combines three residual evaluations into the time update.
```

This reuses the validated kernels (`M2CS4_35`, `Flux_closure35_3D`,
`realizable_3D_M4`, `eigenvalues6{x,y,z}`, `closure_and_eigenvalues`) and replaces
only the *spatial assembly* and *time stepper*.

## 4. Components (new, each independently testable)

- **`src/numerics/reconstruction.jl`**
  - `recon_variables(M) -> V` and `assemble_moments(V) -> M`: bijection between the
    35-moment vector and the reconstruction variable set `V = (ρ, u, v, w, C200,
    C020, C002, and the standardized moments S…)`. Standardized/central moments are
    O(1) and bounded, which limits reconstruction corruption (Jacob's rationale).
  - `muscl_face_states(Vm1, V0, Vp1, Vp2; limiter) -> (V_Lface, V_Rface)`: 2nd-order
    slope-limited left/right states at the interface between cells 0 and +1.
  - `limiter` ∈ {minmod, MC, van Leer} (start minmod for robustness).
  - *Interface:* pure functions on small stencils; no global state.

- **`src/numerics/highorder_flux.jl`**
  - `face_flux(M_L, M_R, axis, Ma) -> Fhat`: realize L/R, compute `F_L/F_R` via
    `Flux_closure35_3D`, wave speeds via `eigenvalues6`, return HLL interface flux.
  - `spatial_residual_1d(M_line, dx, axis, Ma) -> dMdt_line`: assemble faces →
    residual for one grid line. Includes **order degradation** (fall back to
    first-order at cells flagged non-realizable / vacuum / extreme-variation).
  - *Interface:* operates on a 1D array of moment vectors + halos.

- **`src/numerics/ssp_rk.jl`**
  - `ssp_rk3_step!(M, dt, residual!)`: standard 3-stage SSP-RK3 wrapping any
    `residual!` operator. Direction-unsplit in 3D (sum x/y/z residuals per stage).

- **`examples/run_1d_highorder.jl`** — 1D prototype driver for validation.

3D integration (phase 2) reuses `apply_flux_update_3d!`’s sweep structure but calls
the new residual + SSP-RK3 instead of the Euler/`pas_HLL` step; the halo path
(`compute_halo_fluxes_and_wavespeeds_3d!`) must mirror the new face computation
exactly (the MPI-losslessness invariant we just established).

## 5. The realizability ↔ reconstruction coupling (the crux)

Three layers, weakest-to-strongest:
1. **Reconstruct bounded variables** (standardized/central moments + mean velocity),
   not raw moments — corruption is much smaller (Jacob).
2. **Slope limiter** makes the reconstruction TVD — no new extrema, so face states
   stay near the cell-average manifold.
3. **Project every face state** through `realizable_3D_M4` + hyperbolicity correction
   before it enters the flux — guarantees the flux sees only realizable, hyperbolic
   moments. This is our equivalent of Jacob's cell-removal / deferred moment
   correction; it is how we reach his robustness for a single-phase gas, not a
   claim to exceed it.
4. **Order degradation** (2nd → 1st) at cells flagged unrealizable before projection,
   adjacent to vacuum (ρ→0), or with large variation across the stencil — mirrors
   Jacob's island/lake/abscissa-variation handling.

Open risk: face projection could re-introduce diffusion (defeating the purpose). We
will **measure** how often/how strongly projection fires at faces in smooth regions
(should be rare) vs at shocks, and tune the variable set / limiter accordingly.

## 6. Testing & validation

**Phase 1 (1D):**
- *Order of accuracy:* smooth periodic moment field → confirm 2nd-order L1
  convergence (1st-order when limiter clamps at extrema, as expected).
- *1D Riemann / Sod-like moment shock tube:* show reduced diffusion vs first-order;
  realizability never violated (monitor min eigenvalue & H200/H020/H002 > 0).
- *Strong shock (high Ma):* robustness — runs to completion, stays realizable.
- *Conservation:* mass (M000) conserved to ~1e-14; quantify higher-moment
  conservation error introduced by face projection (Jacob reports this metric).

**Phase 2 (3D):**
- *Regression:* Ma=2 crossing should still match low-order physics at coarse res and
  converge faster under refinement.
- *MPI losslessness:* 1-rank vs N-rank bit-identical (extend the existing check to
  the high-order path).
- *Goal demonstration:* Ma=100 crossing — show the jets cross with sharp,
  un-smeared structure vs the first-order baseline.

**Phase 3:** WENO5 + full order-degradation; compare against Jacob's reported
behavior on shock-tube / high-Ma cases.

## 7. Phasing

- **Phase 1** — 1D MUSCL-2 + SSP-RK3 + face projection; validate order, Riemann,
  realizability, conservation. *Exit:* 2nd-order on smooth, robust + realizable on
  shocks, measurable diffusion reduction.
- **Phase 2** — extend to 3D sweeps + MPI (mirror halo path); validate crossing
  Ma=2 → Ma=100. *Exit:* Ma=100 crossing without numerical diffusion, lossless MPI.
- **Phase 3** — WENO5 + order degradation near vacuum/extreme cells; match Jacob.

## 8. Risks / open questions

- **Reconstruction variable set:** standardized vs central vs primitive — decide
  empirically in 1D (start: ρ, mean velocity, standardized moments).
- **Face projection vs diffusion:** measure; if it over-smears, keep the
  highest/"abscissa-like" moments lower-order (Jacob keeps abscissae 1st-order).
- **Conservation:** projecting faces preserves mass but perturbs higher moments;
  quantify and report (acceptable per Jacob if small).
- **CFL:** SSP-RK3 + MUSCL needs a stability-appropriate CFL (≈0.3, tune); 3D sum-CFL
  form (Jacob eq 79) vs current min form — revisit for high order.
- **Jacob's ref [28]** (high-order Godunov for monodisperse multiphase) has the
  reconstruction/Riemann specifics if WENO5 details are needed in Phase 3.

## 9. Out of scope

- Source-term high-order / Strang splitting (collisions) — current operator-split
  collision is adequate at Kn=1000 (collisionless) target cases; revisit later.
- The polydisperse/size-moment physics of Jacob's paper — we keep the velocity-moment
  HyQMOM model.
