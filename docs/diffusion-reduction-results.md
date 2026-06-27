# Reducing numerical diffusion at high Mach — methods tried & results

**Problem.** The 3D 35-moment HyQMOM solver must capture Ma=100 crossing jets with a 1000:1
density ratio. First-order spatial discretization is robust but heavily diffusive: it smears the
dense jets and fills the vacuum behind them numerically. The goal of this effort was a scheme that
is **both low-diffusion and robust** at high Mach.

This note records every approach tried, what worked, what didn't, and why — with quantitative
results from the actual 3D crossing-jet runs (Np=128, Kn=1000).

---

## Headline result: high-order reconstruction + a realizability gate

The winner is **high-order (MUSCL) reconstruction made robust by a realizability gate on the
reconstructed faces** (`recon_face_pair` first-order fallback when a reconstructed face has
nonpositive / non-finite density *or directional variance*, plus a vacuum-density floor). This is
the scheme's analogue of Jacob/Posey's order-degradation + particle-removal, using our projection /
realizability machinery instead.

Two things had to be true at once, and now both are:

1. **It is sharp** — far less numerical diffusion than first order.
2. **It is robust at Ma=100** — the exact Np=128 / Ma=100 / order-2 / sharp-floor case that
   previously crashed (`ArgumentError: matrix contains Infs or NaNs`) now runs cleanly to the full
   matched dynamical time (185 steps, no NaN, mass conserved), with the timestep staying healthy
   *through* the crossing rather than collapsing.

![Ma=100 density: first-order vs high-order](../debug/fig_density_ma100.png)

*Ma=100, 128³, peak density (max along the diagonal jet axis). First-order (left) smears the two
crossing jets into a single diffuse blob (peak ρ=0.80). High-order (right) resolves them as two
distinct, sharp cores (peak ρ=1.07) on the identical grid.*

### Quantitative diffusion reduction (Np=128 Mach ladder, first-order → high-order)

| Ma  | peak ρ (1st → high) | gain | max \|∇ρ\| (1st → high) | ratio |
|-----|---------------------|------|--------------------------|-------|
| 10  | 0.84 → 1.01         | +21% | 18 → 52                  | 2.8×  |
| 25  | 0.81 → 1.13         | +39% | 19 → 54                  | 2.9×  |
| 50  | 0.82 → 1.19         | +45% | 14 → 50                  | 3.5×  |
| 100 | 0.80 → 1.07         | +34% | 14 → 51                  | 3.75× |

High-order retains 20–45% more peak density and produces ~3–3.75× sharper density gradients, with
total mass conserved identically. The gradient advantage *grows* with Ma (first-order diffuses more
at higher Mach; high-order holds its fronts).

![Methods summary](../debug/fig_methods_summary.png)

---

## Methods that did NOT pan out (and why)

A parallel effort explored "clever" Riemann fluxes to cut the interface diffusion of the baseline
two-wave HLL. None beat HLL for this closure; each failed for a distinct, now-understood reason.
(Full detail: `riemann-solver-scope.md` §6.)

- **HLLC (contact-restoring).** Implemented and verified genuine (coupled star pair, RH/consistency
  to 1e-15). But the star states leave the realizable moment cone in the high-Ma collision → it
  falls back to HLL → ≈ identical to HLL on the jets.

- **HLLEM (anti-diffusion).** Implemented and verified mathematically correct (Dumbser–Balsara, LD
  eigenstructure of the flux Jacobian). But physical contact/shear jumps have ≈0 projection onto the
  degenerate λ=uₙ eigenspace → the anti-diffusion term is inert (|HLLEM−HLL| ~ 1e-9).

- **Kinetic / KFVS flux (built in-house).** We built the missing machinery ourselves — an adaptive
  1D HyQMOM quadrature and a 3D CHyQMOM conditional velocity-node inversion (`chyqmom_nodes_3d`) —
  then an abscissa-upwind kinetic flux on top. The node inversion recovers only **29 of 35 moments**
  (six high-order cross moments are *structurally* truncated by the conditional 3-node construction).
  The resulting flux is therefore inconsistent with the moments the system transports (10–89% error
  on high-order moments even on smooth dense states), and the scheme is **numerically unstable**: the
  timestep collapses to NaN within ~6 steps, even at uniform density (so it is not a vacuum problem —
  it is the general high-order inconsistency). Realizable-by-construction ≠ stable.

The bottom-left panel above shows the kinetic flux's timestep collapsing to NaN while the
HLL-based high-order scheme runs stably on the same problem.

**Unifying lesson.** The diffusion bottleneck for this system is **not** the interface flux — it is
the *reconstruction/closure* layer. The win came from making high-order reconstruction realizable
and robust (cheap, closure-native), not from a fancier Riemann solver.

---

## Status of the code

All of the above live in-tree as **opt-in** options with the default path byte-identical
(`riemann_solver`, `spatial_order`, `ho_vacuum_floor`, `ho_realizability_limiter`,
`ho_proj_first_order`); the HLLC/HLLEM/kinetic building blocks are kept as documented,
verified-but-not-better artifacts. The high-order-reconstruction path is the production
recommendation for low diffusion at high Mach.

**Reproducing this:** see [`reproducing-diffusion-results.md`](reproducing-diffusion-results.md) —
`debug/run_mach_ladder.jl` (runs the ladder → metrics + density projections),
`debug/run_kinetic_vs_hll.jl` (the instability demo), and `debug/plot_diffusion_results.py`
(regenerates both figures).
