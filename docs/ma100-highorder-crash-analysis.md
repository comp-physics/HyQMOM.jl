# High-order crossing at high Mach — status, fixes, and robustness findings

Context: Rodney's roadmap step #2 is high-order spatial fluxes to eliminate
numerical diffusion in the Ma=100 jet crossing. His readme flagged the risk up
front — *"I'm not sure what difficulties we run into when mixing high-order
reconstruction and projection."* This documents exactly those difficulties, the
fixes applied, and what remains for the proper high-order treatment.

## The core difficulty

The crossing IC has a 1000:1 density ratio (jet ρ=1, background ρ=0.001) at Mach
`Ma`. As the dense jets move, they leave **deep near-vacuum** (ρ → ~1e-5) behind
them. There, the derived primitives are catastrophic-cancellation noise:
`u = M100/M000` and `C200 = M200/M000 − u²` divide/subtract vanishing quantities,
producing finite-but-unphysical states (captured: `u = −415`/`+747` vs physical
~70, `C200 = 2e5` or slightly negative). First-order HLL diffuses these away;
high-order reconstruction does not, so the noise grows and breaks the scheme.

This surfaced as several distinct failure modes, each fixed or mitigated:

## Fixes applied

1. **`projection35` eigvals port-fidelity guard.** MATLAB's `eig` returns `NaN` on
   a non-finite matrix; Julia's `eigvals` throws. Every eigen site guards its input
   to match MATLAB — `projection35` was the one site ported without it. Routed
   through the shared `_geigvals` guard. (Fixes the original
   `ArgumentError: matrix contains Infs or NaNs`.)

2. **`closure_and_eigenvalues` non-convergence guard.** Its 5×5 complex closure
   eigensolve can fail to *converge* (LAPACKException) — not just receive non-finite
   input — for extreme near-vacuum states. Now degrades to `NaN` on both, matching
   the other sites. (Fixes the Ma=100 high-order `LAPACKException` crash mode.)

3. **Near-vacuum density gate (`ho_vacuum_floor`).** The real mitigation. Below this
   density, cell moments are cancellation noise, so the interface uses the
   first-order cell-centered state — the vacuum then evolves like the robust
   first-order scheme while resolved cells keep full high-order. Exposed as the
   `ho_vacuum_floor` solver param (0 = off, default off). `recon_face_pair` also
   keeps cheap density/finiteness fallbacks. See `src/numerics/reconstruction.jl`.

## Robustness findings (3D Mach ladder, Np=128, matched dynamical time)

| Ma | first-order HLL | high-order HLL+MUSCL |
| --- | --- | --- |
| 10  | ✅ | ✅ |
| 25  | ✅ | ✅ (needs floor ≥ 1e-2; floor = 1e-3 → NaN) |
| 50  | ✅ | ✅ |
| 100 | ✅ | ⚠️ fragile — closure-eigensolve non-convergence; sensitive to floor/rank count |

Key conclusions:

- **High-order works and removes diffusion** for Ma ≤ 50 — peak density +32–76%
  over first-order, increasingly so with Mach. (See `debug/` figures.)
- **The floor is a robustness↔sharpness tradeoff, not a free fix.** A higher floor
  (1e-2) stabilizes Ma=25/50 but first-orders more of the jet fringe, eroding the
  high-order sharpness (Ma=10 peak: +76% at floor=1e-3 vs +32% at 1e-2). No single
  floor is both maximally sharp and robust across Mach.
- **Ma=100 high-order is not robust** with the floor stopgap — it has multiple
  distinct near-vacuum failure modes and is chaotically sensitive (it has both
  completed and crashed depending on floor/rank count). This is the regime that
  needs the proper treatment.

## What this means for the proper fix (Jacob's territory)

The durable solution is a **realizability-preserving high-order reconstruction**
plus the detailed Riemann solver Jacob is building — limiting that keeps cell means
physical in near-vacuum without a hand-tuned density floor, so the pathological
cells never form. The floor + guards here make the scheme usable for development at
Ma ≤ 50 and degrade gracefully (NaN, not crash) beyond.

## Reproduction

`debug/repro_1d_crash.jl` — a cheap serial 1D analog (colliding dense slabs through
near-vacuum) using the same kernels and an adaptive CFL timestep. Reproduces the
near-vacuum behaviour in seconds; sweep `R1D_MA`, `R1D_VACFLOOR` to see the floor
dependence. `debug/run_ma100_demo.jl` runs the full 3D crossing.

## Optional reconstruction-level fix (now available)

A principled alternative to `ho_vacuum_floor` is now available as an opt-in:
set `ho_realizability_limiter=true` in the simulation params.

This activates `scaling_limited_faces`, a Zhang–Shu scaling limiter that applies
to the face reconstruction step directly. For each face it finds the largest
θ∈[0,1] keeping the reconstructed state in the realizable set R (checked via the
same `delta2star3D` smallest-eigenvalue test as the Appendix B projection). θ=1
recovers full MUSCL accuracy in smooth regions; θ→0 at individual faces near vacuum,
without touching unaffected faces elsewhere in the domain.

Key distinction from `ho_vacuum_floor`: the limiter is **local, continuous, and
parameter-free** — no hand-set density threshold, realizability guaranteed by
construction. In the 1D repro it reaches ρ_min ~9.7e-6 while the floor's effective
cutoff is ~1e-3. It does **not** fix the underlying closure-eigensolve failure at
Ma=100 (that requires Jacob's proper Riemann solver work), but it removes the
reconstruction-level source of non-realizable face states without the
robustness↔sharpness tradeoff of the global floor.

`ho_vacuum_floor` remains the default (unchanged). Use `R1D_LIMITER=1` to test the
limiter in the 1D repro; `REPRO_LIMITER=1` for the full 3D demo. Full theory
background: `docs/realizability-highorder-literature.md` §6.
