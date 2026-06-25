# High-Order 3D Flux Reconstruction (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Extend the validated 1D high-order scheme (MUSCL-2 + SSP-RK3 + realizable face projection) to the full 3D MPI solver, selectable via `params.spatial_order=2`, leaving the first-order path intact; demonstrate the Ma=100 jet crossing without numerical diffusion.

**Architecture:** Unsplit method-of-lines: one spatial residual `L(M) = Lx + Ly + Lz` (each axis reconstructed independently with MUSCL on bounded reconstruction variables, HLL flux from projected face states), advanced by SSP-RK3. Each RK stage exchanges MPI halos, computes the residual, updates interior cells, then projects each interior cell to a realizable/hyperbolic state (`realizable_3D_M4`). Built as a new step path behind `spatial_order==2`, reusing the existing IC/MPI/halo/IO infrastructure.

**Tech Stack:** Julia 1.11, HyQMOM.jl, MPI (OpenMPI), `Test`. No new deps.

## Global Constraints

- Modules (PACE): `module load julia/1.11.3 openmpi/4.1.5` in the SAME shell as julia. Headless: `HYQMOM_SKIP_PLOTTING=true CI=true`. Multi-rank: `UCX_TLS=sm,self`. Precompile once before multi-rank. (See RUNNING.md.)
- M field layout: `M[ih, jh, k, 1:35]`, size `(nx+2*halo, ny+2*halo, nz, 35)`, `halo=2`. Interior x = `halo+1 : halo+nx`, interior y = `halo+1 : halo+ny`, z = `1:nz` (z is full domain on every rank, **no halo in z**). `decomp.local_size=(nx,ny,nz)`, `decomp.comm`, `halo_exchange_3d!(A, decomp, bc)` fills x–y halos from neighbors.
- Reuse validated kernels unchanged: `to_recon_vars`/`from_recon_vars`, `muscl_faces`, `face_flux_1d` (projects each face via `realizable_3D_M4` + hyperbolicity), `realize_and_speed`, `ssp_rk3_step`, `realizable_3D_M4`. Do NOT modify the first-order path.
- **MPI-losslessness invariant (non-negotiable):** rank-boundary face fluxes must be computed identically to a single-rank run. halo=2 supplies the 4-cell MUSCL stencil on both sides of every inter-rank interface; reconstruct + flux uniformly from halo data (exchanged BEFORE the residual), never apply a physical BC at an inter-rank boundary. Verified by a 1-rank-vs-N-rank bit-identical test (Task 4).
- 28-standardized-moment order and all conventions identical to Phase 1.
- TDD throughout. Mass (M000) conserved to ~1e-12; realizability maintained (min density > 0, H200/H020/H002 ≥ −1e-8).

---

## File structure

- Create `src/numerics/highorder_3d.jl` — `residual_line`, `residual_ho_3d!`, `step_highorder_3d!`.
- Modify `src/HyQMOM.jl` — include `highorder_3d.jl` after `ssp_rk.jl`; export the three functions.
- Modify `src/simulation_runner.jl` — branch on `params.spatial_order` (default 1 = existing path; 2 = new high-order step), inside the time loop.
- Create `test/test_highorder_3d.jl` — serial residual/RK/realizability tests + the 1-vs-N rank losslessness test (run under mpiexec).
- Modify `test/runtests.jl` — register the serial parts.
- Create `examples/run_3d_highorder_crossing.jl` — Ma=2 regression + Ma=100 diffusion demo driver.

---

### Task 1: Ghost-based 1D line residual (no internal BC)

The 3D residual reuses a single primitive that computes a line residual from an array that already includes ghost cells (so it works for MPI x/y halos and externally-padded z). Unlike `residual_1d` (which applies its own BC at array ends), this returns only the interior residual and treats all out-of-interior cells as valid neighbor/BC data.

**Files:**
- Create: `src/numerics/highorder_3d.jl`
- Modify: `src/HyQMOM.jl` (include after `include("numerics/ssp_rk.jl")`; export `residual_line`)
- Test: `test/test_highorder_3d.jl`

**Interfaces:**
- Consumes: `to_recon_vars`, `from_recon_vars`, `muscl_faces`, `face_flux_1d` (Phase 1).
- Produces: `residual_line(Mext::AbstractMatrix, ds::Real, axis::Int, Ma::Real; order::Int=2, g::Int=2)::Matrix{Float64}` — `Mext` is `(Ni + 2g, 35)` with `g` ghost rows each end; returns `(Ni, 35)` interior residual `-(Fhat[i+1/2] - Fhat[i-1/2])/ds`. Face states reconstructed from `Mext` neighbors (ghosts used directly; no BC applied). Local per-interface order degradation: if either reconstructed face has nonpositive density, both sides fall back to first order at that interface.

- [ ] **Step 1: Write the failing test**

Create `test/test_highorder_3d.jl`:
```julia
using Test
using HyQMOM
using LinearAlgebra

@testset "residual_line ghost-based" begin
    # uniform field with ghosts -> zero interior residual
    M0 = InitializeM4_35(1.0, 0.2, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    Ni = 8; g = 2
    Mext = repeat(reshape(M0,1,35), Ni+2g, 1)
    R = residual_line(Mext, 0.1, 1, 0.0; order=2, g=g)
    @test size(R) == (Ni, 35)
    @test maximum(abs.(R)) < 1e-9
    # equivalence to a periodic residual_1d in the interior:
    # build a periodic line of Np cells, pad with periodic ghosts, compare interior
    Np = 12; dx = 1.0/Np
    base = zeros(Np,35)
    for i in 1:Np
        x=(i-0.5)*dx; base[i,:]=InitializeM4_35(1.0+0.2*sin(2pi*x),1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0)
    end
    padded = vcat(base[Np-g+1:Np,:], base, base[1:g,:])   # periodic ghosts
    Rline = residual_line(padded, dx, 1, 0.0; order=2, g=g)
    Rperiodic = residual_1d(base, dx, 0.0; order=2, bc=:periodic)
    @test maximum(abs.(Rline .- Rperiodic)) < 1e-10
end
```

- [ ] **Step 2: Run to verify it fails**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using HyQMOM; @assert isdefined(HyQMOM,:residual_line)'`
Expected: FAIL — `residual_line` not defined.

- [ ] **Step 3: Implement**

Create `src/numerics/highorder_3d.jl`:
```julia
"""
    residual_line(Mext, ds, axis, Ma; order=2, g=2)

1D method-of-lines residual for a line of 35-moment cells that already includes
`g` ghost cells at each end (filled externally with neighbor / BC data). Returns
the interior residual (Ni, 35) = -(Fhat[i+1/2] - Fhat[i-1/2])/ds, reconstructing
face states from the ghosts (no boundary condition applied here). This is the
shared primitive for the 3D unsplit residual (x, y via MPI halos; z via padded
ghosts). order=1 uses cell-centered states; order=2 uses MUSCL with per-interface
fallback to first order on nonpositive reconstructed density.
"""
function residual_line(Mext::AbstractMatrix, ds::Real, axis::Int, Ma::Real; order::Int=2, g::Int=2)
    Ntot = size(Mext, 1)
    Ni = Ntot - 2g
    # interface fluxes at i+1/2 for interior interfaces: need faces at indices
    # spanning g..Ntot-g. Compute Fhat at every interface that bounds an interior cell.
    # Interior cells are rows g+1 .. g+Ni; their bounding interfaces are g+1/2 .. g+Ni+1/2.
    Fhat = Dict{Int,Vector{Float64}}()
    function face_states(iL)  # interface between cell iL and iL+1
        if order == 1
            return Mext[iL, :], Mext[iL+1, :]
        else
            Vl = muscl_faces(to_recon_vars(Mext[iL-1,:]), to_recon_vars(Mext[iL,:]), to_recon_vars(Mext[iL+1,:]))[2]
            Vr = muscl_faces(to_recon_vars(Mext[iL,:]),   to_recon_vars(Mext[iL+1,:]), to_recon_vars(Mext[iL+2,:]))[1]
            Li = from_recon_vars(Vl); Ri = from_recon_vars(Vr)
            (Li[1] > 0 && Ri[1] > 0) ? (Li, Ri) : (Mext[iL,:], Mext[iL+1,:])
        end
    end
    for iface in g:(g+Ni)            # interfaces bounding interior cells
        ML, MR = face_states(iface)
        Fhat[iface] = face_flux_1d(ML, MR, axis, Ma)
    end
    R = zeros(Ni, 35)
    for ii in 1:Ni
        c = g + ii                   # cell row in Mext
        R[ii, :] = -(Fhat[c] .- Fhat[c-1]) ./ ds
    end
    return R
end
```
Add to `src/HyQMOM.jl` after `include("numerics/ssp_rk.jl")`: `include("numerics/highorder_3d.jl")`; export `residual_line`.

- [ ] **Step 4: Run to verify pass**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using Test, HyQMOM, LinearAlgebra; include("test/test_highorder_3d.jl")'`
Expected: PASS — uniform→0 and interior matches `residual_1d(:periodic)` to 1e-10.

- [ ] **Step 5: Commit**
```bash
git add src/numerics/highorder_3d.jl src/HyQMOM.jl test/test_highorder_3d.jl
git commit -m "feat: ghost-based 1D line residual primitive for 3D high-order"
```

---

### Task 2: Unsplit 3D residual Lx+Ly+Lz

**Files:**
- Modify: `src/numerics/highorder_3d.jl`
- Modify: `src/HyQMOM.jl` (export `residual_ho_3d!`)
- Test: `test/test_highorder_3d.jl`

**Interfaces:**
- Produces: `residual_ho_3d!(R::Array{Float64,4}, M::Array{Float64,4}, nx,ny,nz,halo, dx,dy,dz, Ma; order=2)` — assumes `M` x–y halos are already exchanged. Fills `R` (same shape as M) interior with `Lx+Ly+Lz`; halos/boundary entries left zero. z uses outflow-padded ghosts (copy edge cells) since z has no halo.

- [ ] **Step 1: Failing test** — append to `test/test_highorder_3d.jl`:
```julia
@testset "residual_ho_3d uniform -> 0" begin
    halo=2; nx=6; ny=6; nz=6
    M0 = InitializeM4_35(1.0, 0.1, -0.1, 0.05, 1.0,0.0,0.0,1.0,0.0,1.0)
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for i in 1:nx+2halo, j in 1:ny+2halo, k in 1:nz; M[i,j,k,:]=M0; end
    R = zeros(size(M))
    residual_ho_3d!(R, M, nx,ny,nz,halo, 0.1,0.1,0.1, 0.0; order=2)
    @test maximum(abs.(R[halo+1:halo+nx, halo+1:halo+ny, :, :])) < 1e-9
end
```

- [ ] **Step 2: Run, expect FAIL** (`residual_ho_3d!` undefined).

- [ ] **Step 3: Implement** — append to `src/numerics/highorder_3d.jl`:
```julia
function residual_ho_3d!(R::Array{Float64,4}, M::Array{Float64,4},
                         nx::Int, ny::Int, nz::Int, halo::Int,
                         dx::Real, dy::Real, dz::Real, Ma::Real; order::Int=2)
    fill!(R, 0.0)
    g = halo
    # X: lines along i (have halos), for each interior (jh,k)
    for k in 1:nz, j in 1:ny
        jh = j + halo
        Mext = @view M[:, jh, k, :]                 # (nx+2halo, 35)
        Rl = residual_line(Mext, dx, 1, Ma; order=order, g=g)   # (nx,35)
        for i in 1:nx; R[i+halo, jh, k, :] .+= Rl[i, :]; end
    end
    # Y: lines along j, for each interior (ih,k)
    for k in 1:nz, i in 1:nx
        ih = i + halo
        Mext = @view M[ih, :, k, :]
        Rl = residual_line(Mext, dy, 2, Ma; order=order, g=g)
        for j in 1:ny; R[ih, j+halo, k, :] .+= Rl[j, :]; end
    end
    # Z: no halo in z -> pad with outflow ghosts (copy edge), for each interior (ih,jh)
    for i in 1:nx, j in 1:ny
        ih = i + halo; jh = j + halo
        col = M[ih, jh, :, :]                        # (nz,35)
        Mext = vcat(repeat(col[1:1,:], g, 1), col, repeat(col[nz:nz,:], g, 1))  # outflow pad
        Rl = residual_line(Mext, dz, 3, Ma; order=order, g=g)   # (nz,35)
        for k in 1:nz; R[ih, jh, k, :] .+= Rl[k, :]; end
    end
    return R
end
```
Export `residual_ho_3d!`.

- [ ] **Step 4: Run, expect PASS** (uniform interior residual < 1e-9).

- [ ] **Step 5: Commit**
```bash
git add src/numerics/highorder_3d.jl src/HyQMOM.jl test/test_highorder_3d.jl
git commit -m "feat: unsplit 3D high-order residual (Lx+Ly+Lz)"
```

---

### Task 3: SSP-RK3 3D step with per-stage halo + realizability projection

**Files:**
- Modify: `src/numerics/highorder_3d.jl`
- Modify: `src/HyQMOM.jl` (export `step_highorder_3d!`)
- Test: `test/test_highorder_3d.jl`

**Interfaces:**
- Consumes: `residual_ho_3d!`, `halo_exchange_3d!`, `realizable_3D_M4`.
- Produces: `step_highorder_3d!(M, dt, decomp, bc, nx,ny,nz,halo, dx,dy,dz, Ma; order=2)::Nothing` — advances `M` (with halos) one SSP-RK3 step. Each stage: `halo_exchange_3d!(M,decomp,bc)`; `residual_ho_3d!`; RK-combine interior; project every interior cell via `realizable_3D_M4`. Final `halo_exchange_3d!`.

- [ ] **Step 1: Failing test** — append:
```julia
@testset "step_highorder_3d serial conservation+realizability" begin
    halo=2; nx=8; ny=8; nz=8
    decomp = setup_mpi_cartesian_3d(nx,ny,nz,halo,MPI.COMM_WORLD)  # serial (1 rank)
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    # a smooth blob in density
    for k in 1:nz, j in 1:ny, i in 1:nx
        rho = 1.0 + 0.3*exp(-(((i-4.0))^2+((j-4.0))^2+((k-4.0))^2)/8)
        M[i+halo,j+halo,k,:] = InitializeM4_35(rho,0.1,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0)
    end
    mass0 = sum(M[halo+1:halo+nx, halo+1:halo+ny, :, 1])
    dt = 0.15*(1.0/nx)/4.5
    for _ in 1:5
        step_highorder_3d!(M, dt, decomp, :outflow, nx,ny,nz,halo, 1.0/nx,1.0/ny,1.0/nz, 0.0; order=2)
    end
    Min = M[halo+1:halo+nx, halo+1:halo+ny, :, :]
    @test all(isfinite, Min)
    @test minimum(Min[:,:,:,1]) > 0
    @test abs(sum(Min[:,:,:,1]) - mass0)/mass0 < 1e-10   # mass conserved (outflow, blob interior)
end
```
(Note: this testset uses MPI — run with `mpiexec -n 1`. `using MPI; MPI.Init()` at top of the test file guarded by `MPI.Initialized()`.)

- [ ] **Step 2: Run (mpiexec -n 1), expect FAIL** (`step_highorder_3d!` undefined).

- [ ] **Step 3: Implement** — append:
```julia
function _project_interior!(M, nx,ny,nz,halo, Ma)
    for k in 1:nz, j in 1:ny, i in 1:nx
        ih=i+halo; jh=j+halo
        M[ih,jh,k,:] = realizable_3D_M4(M[ih,jh,k,:], Ma)
    end
end

function step_highorder_3d!(M::Array{Float64,4}, dt::Real, decomp, bc::Symbol,
                            nx,ny,nz,halo, dx,dy,dz, Ma; order::Int=2)
    R = similar(M)
    int = (halo+1:halo+nx, halo+1:halo+ny, 1:nz, :)
    # stage helper: M_in (with halos) -> returns updated interior-only array (full M-shape, halos zero)
    function L!(Mwork)
        halo_exchange_3d!(Mwork, decomp, bc)
        residual_ho_3d!(R, Mwork, nx,ny,nz,halo, dx,dy,dz, Ma; order=order)
        return R
    end
    M0 = copy(M)
    # stage 1: M1 = M + dt*L(M)
    L!(M); @views M[int...] .= M0[int...] .+ dt .* R[int...]; _project_interior!(M,nx,ny,nz,halo,Ma)
    # stage 2: M2 = 3/4 M0 + 1/4 (M1 + dt L(M1))
    L!(M); @views M[int...] .= (3/4).*M0[int...] .+ (1/4).*(M[int...] .+ dt .* R[int...]); _project_interior!(M,nx,ny,nz,halo,Ma)
    # stage 3: M = 1/3 M0 + 2/3 (M2 + dt L(M2))
    L!(M); @views M[int...] .= (1/3).*M0[int...] .+ (2/3).*(M[int...] .+ dt .* R[int...]); _project_interior!(M,nx,ny,nz,halo,Ma)
    halo_exchange_3d!(M, decomp, bc)
    return nothing
end
```
Export `step_highorder_3d!`.

- [ ] **Step 4: Run (mpiexec -n 1), expect PASS** (finite, positive density, mass conserved < 1e-10).

- [ ] **Step 5: Commit**
```bash
git add src/numerics/highorder_3d.jl src/HyQMOM.jl test/test_highorder_3d.jl
git commit -m "feat: SSP-RK3 3D step with per-stage halo exchange + realizability projection"
```

---

### Task 4: MPI losslessness (1-rank vs N-rank bit-identical)

**Files:**
- Create: `test/test_highorder_3d_mpi.jl` (run under mpiexec; writes result to JLD2 for cross-rank-count comparison)
- Create helper script `test/repro/run_ho_3d.jl` (small crossing, configurable ranks/Np, saves M)

**Interfaces:** Consumes `step_highorder_3d!`, `simulation_runner` infra is NOT needed here — drive `step_highorder_3d!` directly on a small crossing IC built locally per rank.

- [ ] **Step 1:** Write `test/repro/run_ho_3d.jl`: build a small (Nx=Ny=Nz=24) crossing_matlab-style IC distributed across ranks via `setup_mpi_cartesian_3d` + the global-index IC fill (copy the pattern from simulation_runner's `:crossing_matlab` branch), run `nsteps=3` of `step_highorder_3d!`, gather to rank 0 (reuse `gather_M`), and `jldsave` to `debug/ho3d_nr$(nranks).jld2`.

- [ ] **Step 2:** Run at 1 and 4 ranks:
```bash
for nr in 1 4; do REPRO_NR=$nr UCX_TLS=sm,self HYQMOM_SKIP_PLOTTING=true CI=true \
  mpiexec -n $nr julia --project=. test/repro/run_ho_3d.jl; done
```
- [ ] **Step 3:** Compare (`debug/ho3d_nr1.jld2` vs `debug/ho3d_nr4.jld2`):
Run a small Julia diff; Expected: `max|Δ| == 0` (bit-identical). If not, the halo path is inconsistent — fix `residual_ho_3d!`/halo handling so the rank-boundary face fluxes match (the Phase-1 lesson). Do not proceed until bit-identical.
- [ ] **Step 4: Commit**
```bash
git add test/repro/run_ho_3d.jl
git commit -m "test: 3D high-order MPI losslessness (1-vs-4 rank bit-identical)"
```

---

### Task 5: Wire into simulation_runner behind `spatial_order`

**Files:**
- Modify: `src/simulation_runner.jl` (read `spatial_order = get(params, :spatial_order, 1)`; in the time loop, when `==2`, replace the first-order flux/HLL/realizability block with a `step_highorder_3d!` call + the same dt computation; keep `==1` path byte-identical).
- Test: extend `test/test_highorder_3d.jl` with a tiny `simulation_runner` run at `spatial_order=2` (Np=16, 2 steps) asserting finite/realizable/mass-conserved output and that it returns the same tuple shape as the first-order path.

- [ ] **Step 1:** Failing test (call `simulation_runner` with `spatial_order=2`).
- [ ] **Step 2:** Run (mpiexec -n 1), expect FAIL (option not handled).
- [ ] **Step 3:** Implement the branch. Keep dt from the existing wave-speed reduction (still needs per-cell eigenvalues for dt — compute as today, or from the stage-1 residual's wave speeds; simplest: keep the existing dt block, then call `step_highorder_3d!` instead of the Euler flux/update/realizability sequence). Default `spatial_order=1` leaves the loop unchanged.
- [ ] **Step 4:** Run, expect PASS; also re-run the core suite to confirm `spatial_order=1` path unaffected.
- [ ] **Step 5: Commit**
```bash
git add src/simulation_runner.jl test/test_highorder_3d.jl
git commit -m "feat: select 3D high-order step via params.spatial_order=2"
```

---

### Task 6: Validation — Ma=2 regression + Ma=100 crossing demo

**Files:**
- Create: `examples/run_3d_highorder_crossing.jl`
- Modify: `test/runtests.jl` (register the serial parts of `test_highorder_3d.jl`)

- [ ] **Step 1:** Driver runs the crossing at `spatial_order=1` and `=2` for Ma=2 (Np=64) and Ma=100 (Np=128), printing for each: steps, mass drift, min density, a sharpness metric (max |∇ρ| or the jet-core peak density retained), and the order-2/order-1 sharpness ratio. Use `scripts/pace_mpi.sh` for multi-rank.
- [ ] **Step 2:** Run Ma=2 (regression): high-order must stay finite, realizable, mass-conserving, and be at least as sharp as first-order. Run Ma=100: high-order should retain a sharper crossing (higher peak / steeper gradients) than first-order. Report ACTUAL numbers; if high-order is not sharper, investigate (limiter, CFL, projection diffusion) and report honestly.
- [ ] **Step 3:** Register the serial high-order testsets in `runtests.jl`.
- [ ] **Step 4: Commit**
```bash
git add examples/run_3d_highorder_crossing.jl test/runtests.jl
git commit -m "feat: 3D high-order crossing validation (Ma=2 regression, Ma=100 demo)"
```

---

## Self-review notes

- **Spec coverage:** Phase-2 spec items — 3D x/y/z high-order (Tasks 1-2), SSP-RK3 + per-stage realizability (Task 3), MPI losslessness (Task 4), `spatial_order` integration keeping first-order intact (Task 5), Ma=2→Ma=100 validation (Task 6). WENO5 / order-degradation-near-vacuum remain Phase 3.
- **Reuse/DRY:** all reconstruction/flux/projection logic comes from Phase-1 functions via the `residual_line` primitive; no duplication of the MUSCL/HLL/projection math.
- **MPI invariant** is an explicit gated task (Task 4) — bit-identical before proceeding, mirroring the Phase-1 fix.
- **Risk:** per-stage `realizable_3D_M4` over all interior cells × 3 RK stages is ~3× the first-order projection cost; if too slow at Np=128, profile and consider projecting once per step instead of per stage (measure realizability impact first). The `face_flux_1d` per interface also re-projects faces — acceptable for correctness; optimize later if needed.
- **dt:** Task 5 keeps the existing (validated) wave-speed dt computation; high-order stability with SSP-RK3 is comfortable at the current CFL (≤1/3); reduce if instability appears.
