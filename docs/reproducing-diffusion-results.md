# Reproducing the diffusion-reduction results

This reproduces the figures and numbers in [`diffusion-reduction-results.md`](diffusion-reduction-results.md):
the first-order vs high-order Mach-ladder comparison, the Ma=100 density figure, and the
kinetic-flux instability demo.

Scripts (all in `debug/`):
- `run_mach_ladder.jl` — runs the crossing jets for each (Ma, order); writes metrics + density projections.
- `run_kinetic_vs_hll.jl` — runs `:hll` (stable) vs `:kinetic` (unstable) and reports step counts / finiteness.
- `plot_diffusion_results.py` — builds `fig_density_ma100.png` and `fig_methods_summary.png`.

---

## 1. Environment

**Julia:** 1.11.x with the project instantiated (`julia --project=. -e 'using Pkg; Pkg.instantiate()'`).

**MPI:** the solver uses MPI even on one rank. Two working setups:

- **System OpenMPI (multi-rank, fastest).** Put a matching OpenMPI's `lib` first on
  `LD_LIBRARY_PATH` so MPI.jl binds it (the project's `LocalPreferences.toml` pins
  `binary="system"`, `abi="OpenMPI"`). Launch with `srun --mpi=pmix`, and on a single node
  **disable UCX** (it can fail with `ucx send failed: Destination is unreachable`):

  ```bash
  export LD_LIBRARY_PATH=/path/to/openmpi-4.1.x/lib:$LD_LIBRARY_PATH
  export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader OMPI_MCA_osc=pt2pt   # single-node: skip UCX
  ```
  On a SLURM cluster, grid sizes should divide the rank count (e.g. Np=128 with 8 or 16 ranks).

- **Bundled MPICH (single-rank, zero-config).** If the system MPI is troublesome, switch MPI.jl to
  its JLL binary — clean singleton init, no launcher needed:
  ```bash
  julia --project=. -e 'using MPIPreferences; MPIPreferences.use_jll_binary("MPICH_jll")'
  ```
  Then run scripts directly with `julia --project=. <script>` (no `srun`). Revert later with
  `MPIPreferences.use_system_binary()`.

> PACE (Georgia Tech) specifics used for the published runs:
> `JULIA=/usr/local/pace-apps/manual/packages/julia/1.11.3/bin/julia`,
> OpenMPI 4.1.8 at `…/openmpi-4.1.8-iit4xasl…/lib`, launched `srun --mpi=pmix -n 16` with UCX off.

**Plotting:** any Python with `numpy` + `matplotlib`. Use `PYTHONNOUSERSITE=1` if a broken user
site-packages shadows them.

---

## 2. Run the Mach ladder

Exact published result is Np=128 (≈3.5 h on 16 ranks); use a smaller `REPRO_NP` for a quick check.

```bash
export DIFFUSION_OUTDIR=$PWD/debug/reprodata

# full (published) — multi-rank:
REPRO_NP=128 srun --mpi=pmix -n 16 julia --project=. debug/run_mach_ladder.jl

# quick look — single rank (MPICH) or a few ranks:
REPRO_NP=48 julia --project=. debug/run_mach_ladder.jl
```

Writes `$DIFFUSION_OUTDIR/ladder_metrics.csv` and `proj_ma{10,25,50,100}_o{1,2}.txt`.

## 3. (Optional) kinetic-flux instability demo

```bash
# verbose run; tee to a log so the timestep trace can be plotted
REPRO_NP=24 julia --project=. debug/run_kinetic_vs_hll.jl 2>&1 | tee $DIFFUSION_OUTDIR/kinetic_vs_hll.log
```
Expect `:hll` to finish (`finite=true`) and `:kinetic` to abort within a few steps (`finite=false`).
If you skip this, the plot falls back to the recorded canonical timestep trace.

## 4. Make the figures

```bash
PYTHONNOUSERSITE=1 DIFFUSION_OUTDIR=$PWD/debug/reprodata FIGDIR=$PWD/debug \
  [KINETIC_LOG=$PWD/debug/reprodata/kinetic_vs_hll.log] \
  python debug/plot_diffusion_results.py
```
Produces `debug/fig_density_ma100.png` and `debug/fig_methods_summary.png`. Set `PEAK_MA=50`
(etc.) to render the density figure at a different Mach number.

---

## What you should see

- Peak density retained by high-order is ~20–45% above first-order across Ma=10→100, with
  ~3–3.75× sharper density gradients (`ladder_metrics.csv`).
- The Ma=100 high-order case completes to `t=0.002` with no NaN (the case that previously crashed).
- `:kinetic` is unstable — timestep collapses to NaN within ~6 steps.

For the underlying analysis see [`diffusion-reduction-results.md`](diffusion-reduction-results.md) and
[`riemann-solver-scope.md`](riemann-solver-scope.md) §6.
