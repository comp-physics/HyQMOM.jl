# High-order spatial reconstruction — status & usage

Notes for Rodney Fox and Jacob Posey on the `projection35-port` branch: the
validated 3D port, the new high-order spatial scheme (roadmap step #2), the fixes
and their limits, and how to run it.

For the general package overview see `README.md`; for the GT PACE cluster recipe
(modules, MPI, precompile) see `RUNNING.md`.

---

## 1. What's on this branch (vs `master`)

1. **Validated 3D port.** The projection-method solver (`projection35` +
   `realizable_3D_M4`, the jacobian15 eigenvalue path, MPI domain decomposition)
   reproduces Rodney's MATLAB to **~5e-7** on the Ma=0 and Ma=2 crossing `.mat`
   references, and is MPI-lossless (1-rank vs N-rank bit-identical).

2. **High-order spatial fluxes (new).** Unsplit SSP-RK3 + MUSCL reconstruction of
   the bounded standardized moments + per-face/per-cell realizability projection,
   selectable via `spatial_order` (1 = first-order HLL, 2 = high-order). Still uses
   HLL — the Riemann solver is the part Jacob is replacing.

3. **Near-vacuum robustness fix** for high-order at high Mach (`ho_vacuum_floor`),
   plus graceful-degradation guards on the realizability/closure eigensolves. See
   §3 and `docs/ma100-highorder-crash-analysis.md`.

4. **Kernel performance** — ~2.1× faster high-order step, all numerics-preserving
   (analytic 3×3 eig, direct LAPACK 4×4, reused-buffer jacobian15/M2CS4).

5. **Cleanup** — removed investigation instrumentation and a dead eigenvalue path,
   curated `debug/` tooling, fixed a function-name typo. 301/301 tests pass.

---

## 2. How to run

The solver is one call, `simulation_runner(params)`. The two knobs that matter for
high-order:

| param | meaning |
| --- | --- |
| `spatial_order` | `1` = first-order HLL (diffusive), `2` = high-order HLL+MUSCL+SSP-RK3 |
| `ho_vacuum_floor` | below this density the high-order path falls back to first order (0 = off, default). Set to ~10× the background density for high-Ma robustness; see §3. **Default and unchanged.** |
| `ho_realizability_limiter` | **OPT-IN, default `false`.** When `true`, switches the high-order reconstruction from the binary `recon_face_pair` fallback to a continuous Zhang–Shu scaling limiter (`scaling_limited_faces`). For each cell face the limiter finds the largest θ∈[0,1] keeping the reconstructed face state in the realizable set R; θ=1 recovers full accuracy in smooth regions, θ→0 at individual faces near vacuum. This is a local, graduated alternative to the global density floor: no hand-set threshold, realizability guaranteed by construction, reaches deeper vacuum while preserving more high-order accuracy near the vacuum interface. `ho_vacuum_floor` remains the default path and is not removed. See `docs/realizability-highorder-literature.md` §6 for the underlying theory. |
| `ho_proj_first_order` | **OPT-IN, default `false`.** Rodney Fox's projection-triggered control: a cell whose mean is flagged for the realizability projection (smallest Δ₂ eigenvalue < 0, i.e. `realizability_margin < 0`) reconstructs **first-order**; all other cells get full MUSCL. One Δ₂ eigenvalue per cell (the same signal the projection uses), local by construction. In the 3D Mach-ladder (Np=64) it is **sharper *and* ~2.5× cheaper** than `ho_realizability_limiter` and retains sharpness at Ma=100 — see `docs/riemann-solver-scope.md` §1. Takes precedence over `ho_realizability_limiter` if both set. Demo env: `REPRO_PROJREC=1`. |
| `riemann_solver` | **OPT-IN, default `:hll`.** Interface flux for the high-order path. `:hll` = original two-wave HLL (byte-identical default); `:rusanov` = robust local Lax–Friedrichs; `:hllc` = contact-restoring HLLC (implemented, verified genuine); `:hllem` = HLLEM anti-diffusion (implemented, verified correct). **Important:** for this 35-moment closure **both `:hllc` and `:hllem` reduce to ≈`:hll`** on the crossing jets — `:hllc` because its star states leave the realizable cone in the collision (fallback), `:hllem` because physical contact/shear jumps don't project onto the computed λ=uₙ eigenspace. See `docs/riemann-solver-scope.md` §6b/§6c. `:kinetic` = realizable-by-construction abscissa-upwind/KFVS flux on the in-house `chyqmom_nodes_3d` velocity-node inversion — **EXPERIMENTAL, NOT for production: it is numerically UNSTABLE** (timestep collapses to NaN within a few steps, even at uniform density). Root cause: `chyqmom_nodes_3d` recovers only 29/35 moments (6 high-order cross moments structurally truncated), so the flux is inconsistent with the moments the system transports. Kept as an opt-in, documented research building block (default off, golden byte-identical). See `docs/riemann-solver-scope.md` §6d. The low-diffusion win needs a *non-truncating* node inversion (consistent kinetic flux) and/or the analytic LD eigenstructure — Jacob's domain. `:hllem` is also far too slow for production (per-face FD-Jacobian + `eigen`). Demo env: `REPRO_RS=hll|rusanov|hllc|hllem|kinetic`. Unknown values raise `ArgumentError`. |

### Quick demo (the crossing jets)

`debug/run_ma100_demo.jl` runs the 3D crossing and saves the moment field. It is
driven by env vars:

```bash
module load julia/1.11.3 openmpi/4.1.5        # GT PACE; see RUNNING.md
export UCX_TLS=sm,self HYQMOM_SKIP_PLOTTING=true CI=true

# high-order, Ma=10, 128^3, single node (pin ranks to the local node)
REPRO_NP=128 REPRO_MA=10 REPRO_TMAX=0.015 REPRO_ORDER=2 REPRO_VACFLOOR=0.001 \
  mpirun -np 64 --host $(hostname):192 --oversubscribe --bind-to none \
  julia --project=. debug/run_ma100_demo.jl

# first-order reference: REPRO_ORDER=1
```

It prints `steps`, wall time, `density min/max`, total mass, and `max|grad rho|`
(a sharpness proxy — high-order gives a larger value), and saves
`debug/ma100_np<Np>_ma<Ma>_o<order>.jld2`.

Notes on launching:
- **Single node:** `mpirun -np <N> --host $(hostname):<slots> ...`. In a multi-node
  Slurm allocation you MUST pin with `--host` (this OpenMPI build spans the whole
  allocation otherwise and the inter-node daemon launch fails). Add
  `--oversubscribe --bind-to none` to use more ranks than the Slurm slot count on
  an exclusive node.
- Use `scripts/pace_mpi.sh` for the standard single-node recipe (see `RUNNING.md`).

### Cheap 1D analog (no MPI)

`debug/repro_1d_crash.jl` — two dense slabs colliding through near-vacuum, same
kernels, serial, seconds. Good for studying the near-vacuum behaviour and the
`ho_vacuum_floor` dependence: `R1D_MA=50 R1D_VACFLOOR=1e-2 julia --project=. debug/repro_1d_crash.jl`.

---

## 3. Current status & limitations (important)

High-order **works and removes numerical diffusion** for **Ma ≤ 50** — peak density
+32–76% over first-order, increasingly so with Mach. The headline figures live in
`debug/` (e.g. the HLL-vs-MUSCL and Mach-ladder comparisons).

It is **not yet robust at Ma=100**. The deep near-vacuum the crossing produces
(ρ → ~1e-5 behind the jets) makes the derived primitives `u = M100/M000` and
`C200 = M200/M000 − u²` catastrophic-cancellation noise; high-order amplifies it.
This shows up as several failure modes (non-finite reconstruction, negative/huge
variance, closure-eigensolve non-convergence).

The `ho_vacuum_floor` stopgap helps but is a **robustness ↔ sharpness tradeoff**:
a higher floor stabilizes more Mach numbers but first-orders more of the jet
fringe, eroding the high-order benefit. There is no single floor that is both
robust and maximally sharp, and Ma=100 remains chaotically sensitive.

**An optional, principled alternative is now available** via
`ho_realizability_limiter=true`: the Zhang–Shu scaling limiter applied to the
HyQMOM moment set (see §2 table and `docs/realizability-highorder-literature.md`
§6). It is local and graduated — no hand-set density — and guarantees realizable
face states by construction. It is OPT-IN; `ho_vacuum_floor` remains the default.

**The durable fix is a realizability-preserving high-order reconstruction**
(limiting that keeps cell means physical in near-vacuum without a hand-set floor),
plus the detailed Riemann solver — i.e. Jacob's high-order work. The floor + guards
(and the optional scaling limiter) make the scheme usable for development at Ma ≤ 50
and degrade gracefully (NaN, not crash) beyond. Full analysis:
`docs/ma100-highorder-crash-analysis.md`.

Rodney's recommended development path: start at **Ma=10**, work up; reference
first-order convergence on fine grids (~1024³, judged on density). Convergence
scaffolding is ready in `debug/` (`convergence_run.jl`, `convergence_analysis.jl`,
`convergence_slurm.sbatch`) — needs a multi-node allocation.

---

## 4. Validation

- **MATLAB parity:** Ma=0/2 crossing reproduced to ~5e-7; kernel parity ≤1e-12
  (`test/matlab_parity/`, goldenfile tests in `test/`).
- **Bit-level regression:** `debug/golden_kernels.jl` (capture/compare) gates
  numerics-preserving changes; the perf and cleanup work is golden-clean.
- **Tests:** `julia --project=. -e 'using Pkg; Pkg.test()'` (the high-order suite
  is `test/test_highorder_1d.jl`, `test/test_highorder_3d.jl`).

---

## 5. Code map (high-order)

| file | role |
| --- | --- |
| `src/numerics/highorder_3d.jl` | unsplit 3D high-order residual + SSP-RK3 step |
| `src/numerics/highorder_flux.jl` | HLL face flux from reconstructed L/R states; 1D residual |
| `src/numerics/reconstruction.jl` | recon-var bijection, MUSCL, `recon_face_pair` (binary vacuum gate, default) and `scaling_limited_faces` (Zhang–Shu θ-limiter, opt-in) |
| `src/realizability/realizability_oracle.jl` | `realizability_margin` / `is_realizable` oracle (δ₂★ smallest-eigenvalue test, same criterion as Appendix B projection) |
| `src/numerics/ssp_rk.jl` | SSP-RK3 |
| `src/realizability/realize_M4_projection.jl`, `projection35.jl` | per-face/cell realizability projection (Appendix B method; always active) |
| `src/numerics/eigenvalues6_hyperbolic_3D.jl`, `small_eig.jl` | wave speeds (jacobian15 blocks; analytic 3×3 + direct 4×4) |
| `src/simulation_runner.jl` | time loop; `spatial_order` / `ho_vacuum_floor` / `ho_realizability_limiter` wiring |
