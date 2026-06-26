# Staged Riemann-Solver Implementation Plan (3D 35-moment HyQMOM)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> **For humans (Jacob/Rodney):** this is a staged plan to replace the diffusive HLL flux with a clever, realizability-preserving Riemann solver. Spec: `docs/riemann-solver-scope.md`. The opt-in `riemann_solver` selector scaffolding is already in place (commit 20e6ac2).

**Goal:** Add contact/shear-resolving, realizability-preserving interface fluxes (HLLC → HLLEM →
kinetic/relaxation) to the high-order HyQMOM scheme, each opt-in via `riemann_solver`, default `:hll`.

**Architecture:** Each flux is a branch inside `face_flux_1d` (`src/numerics/highorder_flux.jl`),
selected by the `RIEMANN_SOLVER[]` Ref (set from the `riemann_solver` param). The realizable HLL
star state and the realizability oracle (`realizability_margin`/`is_realizable`) are the safety net:
any clever flux whose star/intermediate state leaves the realizable cone R falls back to HLL.

**Tech Stack:** Julia 1.11.3; SSP-RK3; the realizability oracle + `realizable_3D_M4` projection;
golden harness (`debug/golden_kernels.jl`); MATLAB parity (`test/matlab_parity/`).

## Global Constraints

- Branch `projection35-port` only. Never `master`.
- **Every solver is OPT-IN; `riemann_solver=:hll` is the default and BYTE-IDENTICAL** to the current
  scheme. With `:hll`, golden must show 0 entries failing at 1e-10. The `:hll` and `:rusanov` paths and
  all existing code are never removed.
- **Realizability is the binding constraint:** every star/intermediate state a new flux produces must be
  checked with `is_realizable` (smallest eigenvalue of `delta2star3D` ≥ 0) and **fall back to the HLL
  flux for that face when it would leave R.** No new flux may make the Ma=100 case crash at `vacfloor=0`.
- Conservation: total mass must stay conserved to ~1e-12 on the closed-box crossing test (as the current
  scheme does).
- Documented: each solver gets a `face_flux_1d` docstring branch comment, a `HIGHORDER.md` knob-table
  row, and a `docs/` note. (See [[optin-and-documented-features]].)
- **The controller runs the golden gate** — subagents misreport it in this MPI environment. Julia+MPI
  invocation (login node has no `module`):
  ```
  JULIA=/usr/local/pace-apps/manual/packages/julia/1.11.3/bin/julia
  OMPI=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/openmpi-4.1.5-ahgvv7r3aju6cty4nlmcd5hihsckie7j
  export PATH="$OMPI/bin:$PATH" LD_LIBRARY_PATH="$OMPI/lib:$LD_LIBRARY_PATH" UCX_TLS=sm,self HYQMOM_SKIP_PLOTTING=true CI=true
  $JULIA --project=. debug/golden_kernels.jl compare
  ```
- Heavy 3D validation runs on Granite Rapids: `sbatch -p cpu-gnr -q embers --constraint=graniterapids`
  (modest jobs auto-route to slow `cpu-small` without the constraint).

## Already in place (foundation — do not redo)

- `RIEMANN_SOLVER = Ref{Symbol}(:hll)` and the `:hll`/`:rusanov` branch in `face_flux_1d`
  (`src/numerics/highorder_flux.jl`); `riemann_solver` param in `simulation_runner.jl`;
  `test/test_riemann_solver.jl` (5/5). New solvers add a branch here.
- Realizability oracle: `realizability_margin(M)`, `is_realizable(M; lam_min)`
  (`src/realizability/realizability_oracle.jl`).
- Wave speeds + flux helpers in `face_flux_1d`: `realize_and_speed(M, axis, Ma)` → `(Mr, sL, sR)`;
  `_phys_flux(M, axis)` → length-35 physical flux; `realizable_3D_M4(M, Ma)` projection.
- Moment indexing (normal direction per axis): density `M[1]`; normal momentum `M[2]`(x)/`M[6]`(y)/`M[16]`(z);
  normal 2nd moment `M[3]`(x)/`M[10]`(y)/`M[20]`(z). So `u_n = M[mom]/M[1]`, normal stress
  `P_nn = M[mom2] − M[mom]^2/M[1]`.

---

## STAGE A — HLLC (restore the contact)

### Task A1: contact-speed estimate `hllc_contact_speed`

**Files:**
- Modify: `src/numerics/highorder_flux.jl` (add helper)
- Test: `test/test_riemann_solver.jl`

**Interfaces:**
- Consumes: realized states `MLr, MRr`, speeds `sL, sR`, `axis`.
- Produces: `hllc_contact_speed(MLr, MRr, sL, sR, axis)::Float64` — the contact (entropy/material) wave
  speed `S_M`, computed from the HLL star state's normal velocity `u_n = (HLL momentum)/(HLL density)`,
  clamped to `[sL, sR]`.

- [ ] **Step 1: Failing test** — append to `test/test_riemann_solver.jl`:
```julia
@testset "hllc contact speed" begin
    using HyQMOM: hllc_contact_speed, realize_and_speed, realizable_3D_M4
    Mu = InitializeM4_35(1.0, 0.37, 0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    Mr,sL,sR = realize_and_speed(Mu, 1, 0.0)
    # uniform state: contact speed == the bulk normal velocity
    @test isapprox(hllc_contact_speed(Mr, Mr, sL, sR, 1), 0.37; atol=1e-10)
    # bracketed by the HLL wave speeds
    ML = realizable_3D_M4(InitializeM4_35(1.0, 0.5,0,0,1.0,0,0,1,0,1), 2.0)
    MR = realizable_3D_M4(InitializeM4_35(0.3,-0.4,0,0,1.2,0,0,1,0,1), 2.0)
    MLr,lL,_ = realize_and_speed(ML,1,2.0); MRr,_,lR = realize_and_speed(MR,1,2.0)
    s = hllc_contact_speed(MLr, MRr, min(lL,lR), max(lL,lR), 1)
    @test min(lL,lR) <= s <= max(lL,lR)
end
```

- [ ] **Step 2: Run, expect FAIL** (`UndefVarError: hllc_contact_speed`). Command: the Julia+MPI block
  above with `$JULIA --project=. test/test_riemann_solver.jl`.

- [ ] **Step 3: Implement** in `highorder_flux.jl`:
```julia
const _NMOM = (1=>2, 2=>6, 3=>16)   # axis -> normal-momentum index; density is M[1]
"Contact (material) wave speed S_M = normal velocity of the HLL star state, clamped to [sL,sR]."
function hllc_contact_speed(MLr::AbstractVector, MRr::AbstractVector, sL::Real, sR::Real, axis::Int)
    m = _NMOM[axis]
    FL = _phys_flux(MLr, axis); FR = _phys_flux(MRr, axis)
    Uden = (sR*MRr[1] - sL*MLr[1] - (FR[1] - FL[1])) / (sR - sL)        # HLL density
    Umom = (sR*MRr[m] - sL*MLr[m] - (FR[m] - FL[m])) / (sR - sL)        # HLL normal momentum
    return clamp(Umom / Uden, sL, sR)
end
```
Note: `F[1]=` mass flux `= M[mom]`, `F[m]=` normal-momentum flux. Verify these components of
`Flux_closure35_3D` (they should be the momentum and the normal 2nd moment) before relying on them.

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** `feat(flux): HLLC contact-speed estimate`.

### Task A2: realizable HLLC star states `hllc_star` (THE derivation — research)

> **This is the substantive research task.** The 35-moment contact is linearly degenerate at `S_M`;
> the star states must (i) satisfy Rankine–Hugoniot across each acoustic wave, (ii) be consistent with
> the HLL average, and (iii) **be realizable**. For Euler/10-moment this closes via "normal velocity =
> S_M and the normal stress is continuous across the contact" (Toro 1994; Sangam 2008, *IJCSM* 2). The
> derivation here is the 35-moment generalization. Do it with Sangam's ten-moment HLLC as the template
> (tensor pressure), and **gate every star state through the realizability oracle**.

**Files:** Modify `src/numerics/highorder_flux.jl`; Test `test/test_riemann_solver.jl`.

**Interfaces:**
- Produces: `hllc_star(MKr, sK, S_M, axis)::Vector{Float64}` — the star state `U*_K` for side K
  (K∈{L,R}), and `hllc_flux(MLr,MRr,sL,sR,S_M,axis)` assembling the four-region HLLC flux.

- [ ] **Step 1: Derivation note** (write to `docs/riemann-solver-scope.md` §A or a scratch note):
  state the chosen contact closure for the 35-moment system (which moments are continuous across the
  contact, how `U*_K` is built), citing Toro 1994 + Sangam 2008. Required properties to prove/justify:
  Rankine–Hugoniot `F*_K − F_K = sK(U*_K − U_K)`; HLL-consistency
  `((S_M−sL)U*_L + (sR−S_M)U*_R)/(sR−sL) = U_HLL`.

- [ ] **Step 2: Failing tests** (properties, not a hand-computed number):
```julia
@testset "hllc star states" begin
    using HyQMOM: hllc_star, hllc_contact_speed, realize_and_speed, realizable_3D_M4, _phys_flux, is_realizable
    ML = realizable_3D_M4(InitializeM4_35(1.0, 0.5,0,0,1.0,0,0,1,0,1), 2.0)
    MR = realizable_3D_M4(InitializeM4_35(0.3,-0.4,0,0,1.2,0,0,1,0,1), 2.0)
    MLr,lL,_ = realize_and_speed(ML,1,2.0); MRr,_,lR = realize_and_speed(MR,1,2.0)
    sL=min(lL,lR); sR=max(lL,lR); SM=hllc_contact_speed(MLr,MRr,sL,sR,1)
    UsL=hllc_star(MLr,sL,SM,1); UsR=hllc_star(MRr,sR,SM,1)
    # HLL-consistency (the integral constraint)
    Uhll = (sR.*MRr .- sL.*MLr .- (_phys_flux(MRr,1).-_phys_flux(MLr,1)))./(sR-sL)
    @test isapprox(((SM-sL).*UsL .+ (sR-SM).*UsR)./(sR-sL), Uhll; rtol=1e-8)
    # star states realizable (else the scheme must fall back to HLL — see A3)
    @test is_realizable(UsL) && is_realizable(UsR)
end
```

- [ ] **Step 3: Implement** `hllc_star` + `hllc_flux` per the Step-1 derivation. The four-region flux:
```julia
function hllc_flux(MLr,MRr,sL,sR,S_M,axis)
    if sL >= 0;      return _phys_flux(MLr,axis)
    elseif sR <= 0;  return _phys_flux(MRr,axis)
    elseif S_M >= 0; UsL=hllc_star(MLr,sL,S_M,axis); return _phys_flux(MLr,axis) .+ sL.*(UsL .- MLr)
    else             UsR=hllc_star(MRr,sR,S_M,axis); return _phys_flux(MRr,axis) .+ sR.*(UsR .- MRr)
    end
end
```

- [ ] **Step 4: Run, expect PASS** (consistency + realizability properties). If star states are NOT
  realizable for physical inputs, that is the expected hard case → A3's fallback handles it; relax the
  test to "realizable OR flagged for fallback."  **Step 5: Commit** `feat(flux): realizable HLLC star states`.

### Task A3: wire `:hllc` into `face_flux_1d` with HLL fallback

**Files:** Modify `src/numerics/highorder_flux.jl`; Test `test/test_riemann_solver.jl`.

- [ ] **Step 1: Failing test** — `:hllc` is finite, consistent on a uniform state, and (key) keeps a
  near-vacuum line finite:
```julia
@testset "hllc flux branch" begin
    HyQMOM.RIEMANN_SOLVER[]=:hllc
    Mu=InitializeM4_35(1.0,0.25,0,0,1.0,0,0,1,0,1)
    @test isapprox(face_flux_1d(Mu,Mu,1,0.0), HyQMOM._phys_flux(realizable_3D_M4(Mu,0.0),1); atol=1e-10)
    ML=InitializeM4_35(1.0,60.0,0,0,1.0,0,0,1,0,1); MR=InitializeM4_35(1e-5,-60.0,0,0,1.0,0,0,1,0,1)
    @test all(isfinite, face_flux_1d(ML,MR,1,100.0))
    HyQMOM.RIEMANN_SOLVER[]=:hll
end
```

- [ ] **Step 2: Run, expect FAIL** (`:hllc` not in the branch → ArgumentError).

- [ ] **Step 3: Implement** — add to `face_flux_1d`'s selector (after `:rusanov`):
```julia
elseif rs === :hllc
    S_M = hllc_contact_speed(MLr, MRr, sL, sR, axis)
    if sL >= 0; return FL elseif sR <= 0; return FR end
    F = hllc_flux(MLr, MRr, sL, sR, S_M, axis)
    # realizability guard: the implied star update must stay in R; else fall back to HLL
    Ustar = (S_M >= 0) ? hllc_star(MLr, sL, S_M, axis) : hllc_star(MRr, sR, S_M, axis)
    return is_realizable(Ustar) ? F : (sR.*FL .- sL.*FR .+ (sL*sR).*(MRr.-MLr))./(sR-sL)
```

- [ ] **Step 4: Run, expect PASS. Step 5: CONTROLLER runs golden** (`:hll` default must be byte-identical,
  0 failing). **Step 6: Commit** `feat(flux): wire :hllc with HLL realizability fallback`.

### Task A4: validate HLLC vs HLL (sharpness, robustness) + document

- [ ] **Step 1:** run the 1D crossing (`debug/repro_1d_crash.jl`) and a small 3D crossing
  (`debug/run_ma100_demo.jl`, add a `REPRO_RS` env that sets `riemann_solver`) for `:hll` vs `:hllc` at
  Ma=10; report peak density / `max|grad rho|` (expect `:hllc` sharper at the contact) + mass conserved.
- [ ] **Step 2:** Mach-ladder `:hll` vs `:hllc` (reuse `scratchpad/ladder_one_ma.sh` with `REPRO_RS=hllc`)
  on Granite Rapids; confirm no Ma=100 crash, mass conserved, centro-symmetry not worsened.
- [ ] **Step 3:** document — `face_flux_1d` `:hllc` branch comment, `HIGHORDER.md` `riemann_solver` row
  (add `:hllc`), and a results paragraph in `docs/riemann-solver-scope.md`. **Step 4: Commit.**

---

## STAGE B — HLLEM (anti-diffuse the linearly-degenerate sub-block)

> The best-ROI low-diffusion upgrade (Dumbser & Balsara 2016, JCP 304). Anti-diffuses **only** the LD
> (contact/shear) fields using the **inner eigenvectors**, inheriting HLL positivity; auto-reverts to HLL
> at coalescing eigenvalues (= the vacuum boundary). Precedent on a moment system: Ben Nasr et al. 2014.

### Task B1: inner (LD) eigenvectors of the HyQMOM flux Jacobian
- [ ] Derive/extract the linearly-degenerate eigenvectors from the HyQMOM eigenstructure (the
  characteristic polynomial factors into orthogonal polynomials, Fox–Laurent 2022; the Jacobian is
  `jacobian15.jl`). Produce `ld_eigvecs(Mr, axis, Ma)::(R_inner, L_inner, λ_inner)` for the contact +
  shear fields. **Test:** `L_inner * R_inner ≈ I` on the LD subspace; `λ_inner ≈ u_n` for the contact.

### Task B2: HLLEM anti-diffusion term + `:hllem` branch
- [ ] Implement `f_HLLEM = f_HLL − φ·(sL sR)/(sR−sL)·R_inner·δ*·L_inner·(MRr−MLr)`, with
  `δ* = I − Λ⁻*/sL − Λ⁺*/sR` (diagonal, bounded (0,1]) and `φ∈[0,1]` (default 1). Drop the term
  (revert to HLL) when any inner eigenvalues coalesce (gap below a tolerance) — i.e. near vacuum.
  **Test:** `:hllem` finite; equals HLL on a uniform state; sharper than HLL on a contact; near-vacuum
  line stays finite.

### Task B3: realizability guard + validation + docs
- [ ] Guard: confirm the anti-diffused update stays in R via the oracle; fall back to HLL per-face if not.
  **CONTROLLER golden** (`:hll` byte-identical). Mach-ladder `:hll`/`:hllc`/`:hllem` comparison; doc.

---

## STAGE C — the paper (choose one; both novel at 35 moments)

### Path C-kinetic: native abscissa-upwind (KFVS) flux
- [ ] Expose the interface quadrature nodes (weights `n_α`, abscissas `U_α`) from the CHyQMOM inversion;
  implement `F_k = Σ_α n_α U_α^{k+1}` split by `sign(U_α)` (`:kinetic`). Realizable-by-construction at
  first order (Vikas–Fox 2011). **Test:** realizable + finite near vacuum; sharper than HLL; golden
  `:hll` byte-identical. Validate on the Mach ladder.

### Path C-relaxation: Suliciu/relaxation solver
- [ ] Design a relaxation system for the 35-moment closure (relaxation variables, subcharacteristic
  condition); implement the exact-contact relaxation flux (`:relax`) following Chalons–Coulombel–Serre
  2012 and the Bouchut–Klingenberg–Waagan many-wave template. Provable positivity/entropy under the
  subcharacteristic condition. Higher rigor, higher effort.

---

## ORTHOGONAL — symmetry (independent of the flux)

### Task S1: optional symmetric (Strang) operator splitting
- [ ] In `step_highorder_3d!`, add an opt-in `symmetric_split::Bool=false` (param `symmetric_split`,
  default false → current x→y→z, byte-identical). When true, use Strang x→y→z→z→y→x (half steps), 2nd
  order and reflection-symmetric. **Test:** default byte-identical (golden); with `symmetric_split=true`
  the centro-symmetry error on the crossing test drops well below the ~3–9% measured for the split
  scheme. Document. (Addresses the centro-symmetry finding; not a flux change.)

---

## Notes: engineering vs research
- **Concrete engineering (well-specified):** A1, A3, A4, B3, S1, and all the golden/byte-identical gates.
- **Research derivation (formulas + gates given; the implementer derives & validates):** A2 (HLLC star
  states for the 35-moment tensor-pressure contact), B1/B2 (LD eigenvectors + HLLEM term), C (kinetic or
  relaxation). These are the genuinely novel pieces (no published HLLC/HLLEM/relaxation solver exists for
  a 35-moment HyQMOM) and are Jacob's territory; each is gated by realizability + golden + ladder
  validation so a wrong derivation cannot silently corrupt the default scheme.
- **Recommended order:** A (cheap contact win) → B (best-ROI anti-diffusion) → C (the paper); S1 anytime.
