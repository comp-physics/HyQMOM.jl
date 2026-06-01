# 2D Bubble Grid-Convergence Study — Design

**Date:** 2026-06-01
**Purpose:** Satisfy reviewer concerns on the HyQMOM JCP revision — Reviewer #1
(meshes too coarse / underresolved / "hard to know what to take from them") and,
secondarily, Reviewer #3 (accuracy, not just robustness). Demonstrate that the
2D solution **converges under grid refinement** to a resolution-independent limit
at a well-defined observed order.

## Decision summary (from brainstorming)
- **Deliverable:** self-convergence only (no external reference solution). Cheapest
  path that directly answers R1.
- **Engine:** the verified 3D solver run at `Nz=1` (matches MATLAB to ~1e-15,
  conserves to 13/13). No 2D legacy code touched. Per Rodney's suggestion.
- **Case:** the 2D discontinuous bubble of Rice, Plante-Sabourin & McDonald,
  *Robustly hyperbolic high-order moment-closures for multidimensional gases*,
  JCP 562 (2026) 115026, §5.2.

## Case definition (Rice §5.2)
- Domain `[-0.5, 0.5]^2`, `Nz=1`. Disk of radius `r = 0.25` centered at origin.
- Isothermal two-state, density/pressure ratio 2: inside `rho=2, T=1` (so `p=2`);
  outside `rho=1, T=1` (`p=1`). Zero bulk velocity. Discontinuous (sharp circle).
- Equilibrium (Maxwellian) moments via `InitializeM4_35`.
- `tmax` chosen so waves do not reach the wall (calibrated in pilot; ~0.05-0.08
  given max wavespeed ~2.45).

## Convergence metric (no truth solution needed)
Richardson / successive-grid differences on a ladder `h, h/2, h/4, ...`:
- Restrict the fine field to the coarse grid by conservative 2x2 block-averaging.
- `e_k = || u_h - restrict(u_{h/2}) ||` in L1 and L2.
- Observed order `p = log2(e_k / e_{k+1})`.
- Primary quantity: **density**. Secondary: one higher moment (temperature /
  4th-order) as a backup check.

## Execution
- **Phase 1 (pilot, serial):** add bubble IC; run `64^2, 128^2, 256^2` at `Kn=1,
  Ma=0, CFL=0.5`; validate harness; lock `tmax`; preliminary order.
- **Phase 2 (production, MPI/slurm):** extend ladder to `512^2, 1024^2 (, 2048^2)`;
  add `Kn in {0.1, 0.01}` for cross-regime robustness.
- **Bonus (near-free):** log corrected-cell fraction per run for a possible later
  R3 correction-impact answer.

## Code changes
1. `src/simulation_runner.jl`: add a `params.ic_type == :bubble` branch in the IC
   construction block. Radial fill using existing global-index/domain machinery
   (MPI-correct, serial-correct). Inputs: `rho_in, rho_out, bubble_radius,
   bubble_xc, bubble_yc` (with defaults), reuses `T, r110, r101, r011`.
2. `studies/bubble/run_refinement.jl`: driver — builds the param NamedTuple,
   runs a list of resolutions, saves final density (and one higher moment) to
   raw binary per N.
3. `studies/bubble/analyze_convergence.jl`: post-processing — block-restrict,
   compute L1/L2 successive differences, fit observed order, emit table + log-log
   plot data.

## Deliverables
Bubble IC in the solver; refinement runner + analyzer; convergence table and
log-log plot ready to drop into the revised manuscript.
