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
| `ho_vacuum_floor` | below this density the high-order path falls back to first order (0 = off). Set to ~10× the background density for high-Ma robustness; see §3. |

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

**The durable fix is a realizability-preserving high-order reconstruction**
(limiting that keeps cell means physical in near-vacuum without a hand-set floor),
plus the detailed Riemann solver — i.e. Jacob's high-order work. The floor + guards
make the scheme usable for development at Ma ≤ 50 and degrade gracefully (NaN, not
crash) beyond. Full analysis: `docs/ma100-highorder-crash-analysis.md`.

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
| `src/numerics/reconstruction.jl` | recon-var bijection, MUSCL, `recon_face_pair` (the vacuum gate) |
| `src/numerics/ssp_rk.jl` | SSP-RK3 |
| `src/realizability/realize_M4_projection.jl`, `projection35.jl` | per-face/cell realizability projection |
| `src/numerics/eigenvalues6_hyperbolic_3D.jl`, `small_eig.jl` | wave speeds (jacobian15 blocks; analytic 3×3 + direct 4×4) |
| `src/simulation_runner.jl` | time loop; `spatial_order` / `ho_vacuum_floor` wiring |
