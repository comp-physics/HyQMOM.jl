# Ma=100 high-order crossing crash — root-cause analysis

**Date:** 2026-06-25
**Symptom:** `ArgumentError: matrix contains Infs or NaNs` deep in an `eigvals`
call, when running the Ma=100 crossing-jets demo at **Np=128 with high-order
(`spatial_order=2`)**. The first-order run (`spatial_order=1`) at the same Np=128
completes (114 steps), and the high-order run at the coarser **Np=64 completes**
(57 steps). The crash is therefore *resolution- and order-dependent*, not a bug in
the base finite-volume scheme.

## TL;DR

High-order MUSCL reconstruction, applied in the deep near-vacuum region that the
Ma=100 / 1000:1-density-ratio crossing produces, generates **unrealizable
second-order moments** (negative or enormous directional variances) at cell faces.
The resulting interface flux overflows to `Inf`, the RK update spreads it to `NaN`,
and the very next realizability projection hits the **one `eigvals` call in the
codebase that is not guarded against non-finite input** — `projection35` — which
throws. Coarser grids and the first-order scheme keep the near-vacuum state smooth
enough that the overflow never forms.

## Exact crash site (by elimination *and* direct capture)

Every `eigvals`/eigen call in `src/` guards its input and returns `NaN` on a
non-finite matrix — deliberately matching MATLAB's `eig`, which returns `NaN`
eigenvalues rather than throwing:

| File | Guard |
| --- | --- |
| `numerics/eigenvalues6_hyperbolic_3D.jl:37` | `if any(!isfinite, J) return NaN…` |
| `numerics/closure_and_eigenvalues.jl:70` | `if any(!isfinite, z) return …NaN` |
| `numerics/compute_jacobian_eigenvalues.jl:25,46` | `if any(!isfinite, J6) … NaN` |
| **`realizability/projection35.jl:31` and `:80`** | **none** |

`projection35` is the only unguarded site, so the opaque error can *only* come from
there. A captured stacktrace confirms it directly:

```
projection35  (projection35.jl, eigvals(E1))
  └ realizable_3D_M4   (realize_M4_projection.jl:85)   # per-cell realizability projection
      └ step!  →  SSP-RK3 stage cell-projection
```

At the crash, all 28 standardized moments feeding `projection35` are `NaN`, so the
6×6 realizability matrix `E1 = delta2star3D(...)` is entirely `NaN`.

## Where the NaN is actually born (one stage earlier)

The all-`NaN` cell is a *symptom*; the non-finite value is born in the spatial
residual. Instrumented capture (1D analog, N=512, Ma=100) shows the first
non-finite **residual** at step 147 in the near-vacuum band (ρ ≈ 5e-5, five orders
below the jet density). The reconstruction stencil around the birth cell:

```
cell 128: rho=2.39e-04  u=  -1.26   C200= 5.41e+00   (ok)
cell 129: rho=1.18e-04  u=  -0.18   C200= 7.93e+00   (ok)
cell 130: rho=5.32e-05  u=-414.95   C200= 2.09e+05   <- |u| >> physical 70.7, huge variance
cell 131: rho=2.85e-05  u=+747.53   C200=-1.87e-11   <- NEGATIVE variance (unrealizable)
cell 132: rho=3.60e-05  u= -67.20   C200= 2.02e-12   (at the c2min floor)
```

## The causal chain

1. **Physics.** Ma=100 with rhol=1.0 / rhor=0.001 (1000:1) creates an expanding
   near-vacuum region (ρ → ~1e-5) around and between the colliding jets.
2. **Vacuum degeneracy.** Dividing momentum/energy moments by a vanishing density
   yields wild velocities (|u| up to ~750 vs the physical 70.7) and directional
   variances that are either enormous (~2e5) or **negative** (~−1.9e-11), i.e.
   unrealizable second-order moments.
3. **High-order amplification.** MUSCL reconstructs *standardized* variables and
   recombines independently-limited slopes at faces (e.g. `C400 = S400·C200²`).
   Across adjacent near-vacuum cells with huge/negative variances, the recombined
   face moments overflow to `Inf`; the HLL flux difference then produces a
   **non-finite residual**. First-order (no reconstruction) and coarser grids keep
   these gradients diffuse, so the overflow never forms.
4. **Propagation.** `Inf` residual → `Inf`/`NaN` moment after the RK update; the
   next operation spreads it (`Inf − Inf = NaN`) so the whole cell vector is `NaN`.
5. **Surfacing.** The next per-cell realizability projection standardizes the
   `NaN` cell (`M2CS4_35`) → all 28 standardized moments `NaN` → all-`NaN` `E1` →
   `projection35`'s unguarded `eigvals(E1)` throws.

Note: `realizable_3D_M4` floors `C200` at `c2min=1e-12`, but it does so on
line 32 — *after* `M2CS4_35` has already standardized using `sqrt(C2)` on line 31.
The floor cannot rescue a cell whose raw moments are already non-finite, and it
does not address a *negative* incoming variance before standardization.

## Reproduction (cheap, serial, deterministic)

`debug/repro_1d_crash.jl` is a 1D analog of `step_highorder_3d!` (SSP-RK3 +
`residual_1d(order)` + per-stage `realizable_3D_M4`), with two dense Ma=100 slabs
colliding through a near-vacuum background. It reproduces the identical crash in
seconds, with no MPI, and isolates the trigger:

| Config | Result | ρ_min reached |
| --- | --- | --- |
| **order=2, N=512** | **CRASH at step 147** (t≈1.19e-3) | 2.0e-5 |
| order=1, N=512 | completes 200 steps | 5.7e-6 (*deeper* vacuum, survives) |
| order=2, N=256 | completes 300 steps | 1.0e-3 |
| order=2, N=128 | completes 200 steps | 1.0e-3 |

This matches the 3D observations (Np=128 o2 crashes; Np=64 o2 and Np=128 o1
complete). order=1 reaches a *deeper* vacuum than the crashing case yet survives,
confirming the trigger is the **reconstruction**, not vacuum depth or the base
scheme.

Run it with:

```
HO_DEBUG=1 R1D_ORDER=2 R1D_N=512 julia --project=. debug/repro_1d_crash.jl
```

## A case that is "like this but doesn't crash"

- **Np=64, Ma=100, high-order** — completes 57 steps to t=1e-3
  (`debug/ma100_np64_ma100_o2.jld2`, ρ∈[3.7e-4, 1.93], mass conserved).
- **Np=128, Ma=100, first-order** — completes 114 steps
  (`debug/ma100_np128_ma100_o1.jld2`).
- 1D analogs: **order=2 at N≤256**, or **order=1 at any N** (above).

## Recommended fixes (not applied here — out of scope of this analysis)

1. **Immediate robustness (matches existing convention).** Guard
   `projection35`'s two `eigvals(E1)` calls the same way every other eigen site is
   guarded: if `any(!isfinite, E1)`, treat the moments as needing correction /
   return the MATLAB-equivalent `NaN` path rather than throwing. This converts an
   opaque crash into the same graceful degradation the rest of the code already
   uses — but it does **not** fix the underlying garbage moments.
2. **Real fix (reconstruction level).** The high-order face reconstruction already
   falls back to first order when a reconstructed face density is nonpositive
   (`Li[1] > 0 && Ri[1] > 0`). Extend that fallback to also trigger on a
   **nonpositive or non-finite reconstructed directional variance** (and on
   non-finite higher moments). That removes the source of the unrealizable face
   states in near-vacuum instead of patching the symptom downstream. This is the
   natural place for the more careful near-vacuum / realizability-preserving
   reconstruction that the high-order roadmap (and Jacob's Riemann-solver work)
   will need anyway.

## Investigation instrumentation left in place (ENV-gated, zero production cost)

All gated on `ENV["HO_DEBUG"]=="1"` (default off → behavior identical to before):

- `src/HyQMOM.jl`: `HO_DEBUG` switch + `_geigvals(A, label)` guarded-eigvals helper.
- `src/realizability/projection35.jl`: routes both `eigvals` through `_geigvals`
  and dumps the offending standardized moments.
- `src/realizability/realize_M4_projection.jl`: dumps raw density / directional
  variances when `M2CS4_35` produces non-finite standardized moments.
- `src/numerics/highorder_3d.jl`: per-interface face-state dump in `residual_line`.
- `debug/repro_1d_crash.jl`, `debug/probe_crash.jl`: the standalone reproducers.
