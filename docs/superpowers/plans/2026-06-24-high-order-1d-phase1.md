# High-Order 1D Flux Reconstruction (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 1D, 2nd-order (MUSCL) realizability-preserving spatial reconstruction with SSP-RK3 time integration to HyQMOM.jl, validated for order-of-accuracy, realizability, conservation, and reduced numerical diffusion vs the first-order scheme.

**Architecture:** Reconstruct each cell's 35-moment vector in a *bounded* variable set (density, mean velocity, variances, standardized moments), slope-limit to left/right face states, re-assemble moments, **project each face state** (`realizable_3D_M4` + hyperbolicity correction), compute an HLL interface flux from the projected L/R states, and advance the resulting method-of-lines residual with SSP-RK3. This reuses the validated kernels (`M2CS4_35`, `C4toM4_3D`, `Flux_closure35_3D`, `realizable_3D_M4`, `eigenvalues6_hyperbolic_3D`, `closure_and_eigenvalues`) and adds only spatial assembly + the time stepper. Spatially 1D (x); velocity space stays full 3D / 35 moments.

**Tech Stack:** Julia 1.11, HyQMOM.jl package, `Test` stdlib. No new dependencies.

## Global Constraints

- Julia 1.11; run anything that calls `MPI.Init` under `mpiexec -n 1` — but Phase 1 is serial and does NOT use MPI (no `MPI.Init`), so plain `julia --project=.` is fine.
- Headless env for any run/test: `HYQMOM_SKIP_PLOTTING=true CI=true`.
- Modules on PACE: `module load julia/1.11.3` (OpenMPI not needed for Phase 1).
- Reuse existing validated kernels; do not modify them. Match existing file/style conventions in `src/numerics/`.
- The 28 standardized-moment ordering used everywhere is:
  `S300,S400,S110,S210,S310,S120,S220,S030,S130,S040,S101,S201,S301,S102,S202,S003,S103,S004,S011,S111,S211,S021,S121,S031,S012,S112,S013,S022`
  at `S4` indices `[4,5,7,8,9,11,12,13,14,15,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]`.
- TDD: every task is failing-test → run → implement → run → commit.

---

## File structure

- Create `src/numerics/reconstruction.jl` — variable bijection (`to_recon_vars`/`from_recon_vars`), limiters, MUSCL face states.
- Create `src/numerics/highorder_flux.jl` — single-interface HLL face flux from L/R states; 1D residual operator with order degradation.
- Create `src/numerics/ssp_rk.jl` — SSP-RK3 stepper.
- Create `test/test_highorder_1d.jl` — all Phase-1 tests.
- Modify `src/HyQMOM.jl` — `include` the three new files (after the existing `numerics/` includes) and export the public functions.
- Modify `test/runtests.jl` — add `include("test_highorder_1d.jl")` in the unit-test set.
- Create `examples/run_1d_highorder.jl` — 1D shock-tube driver for the diffusion comparison.

---

### Task 1: Reconstruction-variable bijection

**Files:**
- Create: `src/numerics/reconstruction.jl`
- Modify: `src/HyQMOM.jl` (include + export)
- Test: `test/test_highorder_1d.jl`

**Interfaces:**
- Consumes: `M2CS4_35(M)->(C4,S4)`, `C4toM4_3D(...)->M5(6x6x6)` (existing).
- Produces:
  - `to_recon_vars(M::AbstractVector)::Vector{Float64}` — length-35 bounded variable vector `V = [M000, u, v, w, C200, C020, C002, <28 standardized moments in the canonical order>]`.
  - `from_recon_vars(V::AbstractVector)::Vector{Float64}` — inverse, returns the 35-moment vector `M`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_highorder_1d.jl`:
```julia
using Test
using HyQMOM
using LinearAlgebra

@testset "recon-vars bijection" begin
    # realizable moment vectors from particle samples
    function sample_M(seed)
        # deterministic pseudo-particles (no RNG: fixed lattice + shift)
        rho = 0.7 + 0.1*seed
        u0, v0, w0 = 0.1*seed, -0.05*seed, 0.02*seed
        T = 1.0 + 0.1*seed
        return InitializeM4_35(rho, u0, v0, w0, T, 0.0, 0.0, T, 0.0, T)
    end
    for s in 1:5
        M = sample_M(s)
        V = to_recon_vars(M)
        @test length(V) == 35
        M2 = from_recon_vars(V)
        @test M2 ≈ M atol=1e-10 rtol=1e-10
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using HyQMOM; @assert isdefined(HyQMOM,:to_recon_vars)'`
Expected: FAIL — `to_recon_vars` not defined.

- [ ] **Step 3: Write minimal implementation**

Create `src/numerics/reconstruction.jl`:
```julia
"""
    to_recon_vars(M) / from_recon_vars(V)

Bijection between the 35-moment vector `M` and a bounded reconstruction-variable
vector `V = [M000, u, v, w, C200, C020, C002, <28 standardized moments>]`.
Reconstructing the bounded variables (not raw moments) limits realizability
corruption (cf. Posey/Fox/Houim arXiv:2603.13697). Reuses the same S->C->M
reconstruction as realize_3D_M4.
"""
const _SIDX = [4,5,7,8,9,11,12,13,14,15,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

function to_recon_vars(M::AbstractVector)
    C4, S4 = M2CS4_35(M)
    M000 = M[1]
    u = M[2]/M000; v = M[6]/M000; w = M[16]/M000
    C200 = max(0.0, C4[3]); C020 = max(0.0, C4[10]); C002 = max(0.0, C4[20])
    return vcat([M000, u, v, w, C200, C020, C002], S4[_SIDX])
end

function from_recon_vars(V::AbstractVector)
    M000=V[1]; u=V[2]; v=V[3]; w=V[4]; C200=V[5]; C020=V[6]; C002=V[7]
    S300,S400,S110,S210,S310,S120,S220,S030,S130,S040,
    S101,S201,S301,S102,S202,S003,S103,S004,S011,S111,
    S211,S021,S121,S031,S012,S112,S013,S022 = V[8:35]
    sC200=sqrt(C200); sC020=sqrt(C020); sC002=sqrt(C002)
    C110=S110*sC200*sC020; C101=S101*sC200*sC002; C011=S011*sC020*sC002
    C300=S300*sC200*C200; C210=S210*C200*sC020; C201=S201*C200*sC002
    C120=S120*sC200*C020; C111=S111*sC200*sC020*sC002; C102=S102*sC200*C002
    C030=S030*sC020*C020; C021=S021*C020*sC002; C012=S012*sC020*C002; C003=S003*sC002*C002
    C400=S400*C200^2; C310=S310*sC200*C200*sC020; C301=S301*sC200*C200*sC002
    C220=S220*C200*C020; C211=S211*C200*sC020*sC002; C202=S202*C200*C002
    C130=S130*sC200*sC020*C020; C121=S121*sC200*C020*sC002; C112=S112*sC200*sC020*C002; C103=S103*sC200*sC002*C002
    C040=S040*C020^2; C031=S031*sC020*C020*sC002; C022=S022*C020*C002; C013=S013*sC020*sC002*C002; C004=S004*C002^2
    M5 = C4toM4_3D(M000,u,v,w,C200,C110,C101,C020,C011,C002,
                   C300,C210,C201,C120,C111,C102,C030,C021,C012,C003,
                   C400,C310,C301,C220,C211,C202,C130,C121,C112,C103,C040,C031,C022,C013,C004)
    return [M5[1,1,1],M5[2,1,1],M5[3,1,1],M5[4,1,1],M5[5,1,1],
            M5[1,2,1],M5[2,2,1],M5[3,2,1],M5[4,2,1],
            M5[1,3,1],M5[2,3,1],M5[3,3,1],
            M5[1,4,1],M5[2,4,1],
            M5[1,5,1],
            M5[1,1,2],M5[2,1,2],M5[3,1,2],M5[4,1,2],
            M5[1,1,3],M5[2,1,3],M5[3,1,3],
            M5[1,1,4],M5[2,1,4],
            M5[1,1,5],
            M5[1,2,2],M5[2,2,2],M5[3,2,2],
            M5[1,3,2],M5[2,3,2],
            M5[1,4,2],
            M5[1,2,3],M5[2,2,3],
            M5[1,2,4],
            M5[1,3,3]]
end
```

Add to `src/HyQMOM.jl` after `include("numerics/Flux_closure35_3D.jl")`:
```julia
include("numerics/reconstruction.jl")
```
Add to the export block (near the other numerics exports):
```julia
export to_recon_vars, from_recon_vars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using Pkg; include("test/test_highorder_1d.jl")'`
Expected: PASS — "recon-vars bijection" testset, 5 round-trips within 1e-10.

- [ ] **Step 5: Commit**

```bash
git add src/numerics/reconstruction.jl src/HyQMOM.jl test/test_highorder_1d.jl
git commit -m "feat: reconstruction-variable bijection (to/from_recon_vars)"
```

---

### Task 2: Slope limiter + MUSCL face states

**Files:**
- Modify: `src/numerics/reconstruction.jl`
- Modify: `src/HyQMOM.jl` (export)
- Test: `test/test_highorder_1d.jl`

**Interfaces:**
- Produces:
  - `minmod(a::Real,b::Real)::Float64`
  - `muscl_slopes(Vm1, V0, Vp1; limiter=minmod)::Vector{Float64}` — limited slope per component for one cell given its left/right neighbors (all length-35 recon-var vectors).
  - `muscl_faces(Vm1, V0, Vp1; limiter=minmod)::Tuple{Vector{Float64},Vector{Float64}}` — returns `(Vminus, Vplus)`: the cell's left-face (`V0 - 0.5*slope`) and right-face (`V0 + 0.5*slope`) recon-var states.

- [ ] **Step 1: Write the failing test**

Append to `test/test_highorder_1d.jl`:
```julia
@testset "MUSCL limiter + faces" begin
    @test minmod(2.0, 3.0) == 2.0
    @test minmod(-2.0, 3.0) == 0.0
    @test minmod(-2.0, -5.0) == -2.0
    # On a LINEAR field, minmod returns the exact slope (2nd-order, no clamping)
    Vm1 = fill(1.0, 35); V0 = fill(2.0, 35); Vp1 = fill(3.0, 35)
    s = muscl_slopes(Vm1, V0, Vp1)
    @test all(s .≈ 1.0)
    Vminus, Vplus = muscl_faces(Vm1, V0, Vp1)
    @test all(Vminus .≈ 1.5) && all(Vplus .≈ 2.5)
    # At a local MAX, limiter clamps slope to 0 (1st-order, TVD)
    s2 = muscl_slopes(fill(1.0,35), fill(3.0,35), fill(1.0,35))
    @test all(s2 .== 0.0)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using HyQMOM; @assert isdefined(HyQMOM,:muscl_faces)'`
Expected: FAIL — `muscl_faces` not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `src/numerics/reconstruction.jl`:
```julia
"minmod slope limiter"
@inline function minmod(a::Real, b::Real)
    (a*b <= 0) ? 0.0 : (abs(a) < abs(b) ? Float64(a) : Float64(b))
end

"Per-component limited slope for cell V0 given neighbors Vm1, Vp1."
function muscl_slopes(Vm1::AbstractVector, V0::AbstractVector, Vp1::AbstractVector; limiter=minmod)
    n = length(V0)
    s = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        s[k] = limiter(V0[k]-Vm1[k], Vp1[k]-V0[k])
    end
    return s
end

"Left/right face recon-var states for cell V0 (V0 ∓ 0.5*slope)."
function muscl_faces(Vm1::AbstractVector, V0::AbstractVector, Vp1::AbstractVector; limiter=minmod)
    s = muscl_slopes(Vm1, V0, Vp1; limiter=limiter)
    return (V0 .- 0.5 .* s, V0 .+ 0.5 .* s)
end
```
Add to `src/HyQMOM.jl` export block:
```julia
export minmod, muscl_slopes, muscl_faces
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'include("test/test_highorder_1d.jl")'`
Expected: PASS — "MUSCL limiter + faces".

- [ ] **Step 5: Commit**

```bash
git add src/numerics/reconstruction.jl src/HyQMOM.jl test/test_highorder_1d.jl
git commit -m "feat: minmod limiter and MUSCL face states"
```

---

### Task 3: HLL face flux from left/right states

**Files:**
- Create: `src/numerics/highorder_flux.jl`
- Modify: `src/HyQMOM.jl` (include + export)
- Test: `test/test_highorder_1d.jl`

**Interfaces:**
- Consumes: `eigenvalues6_hyperbolic_3D(M,axis,flag2D,Ma)->(vmin,vmax,Mr)`, `closure_and_eigenvalues(M5slice)->(Mp,vmin,vmax)`, `Flux_closure35_3D(M)->(Fx,Fy,Fz)`, `realizable_3D_M4(M,Ma)` (existing).
- Produces:
  - `realize_and_speed(M, axis, Ma)::Tuple{Vector{Float64},Float64,Float64}` — returns `(Mr, vpmin, vpmax)`: hyperbolicity-corrected `Mr` and combined wave speeds for `axis` (1=x), matching the interior solver path (hyperbolicity + 1D closure abscissa via `closure_and_eigenvalues`).
  - `face_flux_1d(M_L, M_R, axis, Ma)::Vector{Float64}` — HLL interface flux (length 35) from left/right face moment states. Applies `realizable_3D_M4` then `realize_and_speed` to each side before the HLL formula.

- [ ] **Step 1: Write the failing test**

Append to `test/test_highorder_1d.jl`:
```julia
@testset "HLL face flux consistency" begin
    M = InitializeM4_35(1.0, 0.3, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    # uniform L==R: HLL flux must equal the physical x-flux of M
    Fhat = face_flux_1d(copy(M), copy(M), 1, 0.0)
    Fx, _, _ = Flux_closure35_3D(M)
    @test Fhat ≈ Fx atol=1e-10 rtol=1e-10
    @test length(Fhat) == 35
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using HyQMOM; @assert isdefined(HyQMOM,:face_flux_1d)'`
Expected: FAIL — `face_flux_1d` not defined.

- [ ] **Step 3: Write minimal implementation**

Create `src/numerics/highorder_flux.jl`:
```julia
"""
    realize_and_speed(M, axis, Ma)

Hyperbolicity-correct M for the given axis and return (Mr, vpmin, vpmax) with the
combined 6x6 + 1D-closure wave speeds, matching the interior flux path.
"""
function realize_and_speed(M::AbstractVector, axis::Int, Ma::Real)
    if axis == 1
        v6min, v6max, Mr = eigenvalues6_hyperbolic_3D(M, 1, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[[1,2,3,4,5]])
    elseif axis == 2
        v6min, v6max, Mr = eigenvalues6_hyperbolic_3D(M, 2, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[[1,6,10,13,15]])
    else
        v6min, v6max, Mr = eigenvalues6z_hyperbolic_3D(M, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[[1,16,20,23,25]])
    end
    return Mr, min(v5min, v6min), max(v5max, v6max)
end

"Physical flux (length 35) of moment vector M in the given axis direction."
function _phys_flux(M::AbstractVector, axis::Int)
    Fx, Fy, Fz = Flux_closure35_3D(M)
    return axis == 1 ? Fx : (axis == 2 ? Fy : Fz)
end

"""
    face_flux_1d(M_L, M_R, axis, Ma)

HLL interface flux from left/right face moment states. Each side is projected
(realizable_3D_M4) and hyperbolicity-corrected before fluxing.
"""
function face_flux_1d(M_L::AbstractVector, M_R::AbstractVector, axis::Int, Ma::Real)
    ML = realizable_3D_M4(M_L, Ma)
    MR = realizable_3D_M4(M_R, Ma)
    MLr, lminL, lmaxL = realize_and_speed(ML, axis, Ma)
    MRr, lminR, lmaxR = realize_and_speed(MR, axis, Ma)
    FL = _phys_flux(MLr, axis)
    FR = _phys_flux(MRr, axis)
    sL = min(lminL, lminR)
    sR = max(lmaxL, lmaxR)
    if sL >= 0
        return FL
    elseif sR <= 0
        return FR
    else
        return (sR .* FL .- sL .* FR .+ (sL*sR) .* (MRr .- MLr)) ./ (sR - sL)
    end
end
```
Add to `src/HyQMOM.jl` after `include("numerics/reconstruction.jl")`:
```julia
include("numerics/highorder_flux.jl")
```
Export:
```julia
export realize_and_speed, face_flux_1d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'include("test/test_highorder_1d.jl")'`
Expected: PASS — uniform L==R reproduces the physical flux.

- [ ] **Step 5: Commit**

```bash
git add src/numerics/highorder_flux.jl src/HyQMOM.jl test/test_highorder_1d.jl
git commit -m "feat: HLL face flux from reconstructed L/R states"
```

---

### Task 4: 1D spatial residual operator (MUSCL + order degradation)

**Files:**
- Modify: `src/numerics/highorder_flux.jl`
- Modify: `src/HyQMOM.jl` (export)
- Test: `test/test_highorder_1d.jl`

**Interfaces:**
- Consumes: `to_recon_vars`, `from_recon_vars`, `muscl_faces`, `face_flux_1d`.
- Produces:
  - `residual_1d(Mline::AbstractMatrix, dx::Float64, Ma::Real; order::Int=2)::Matrix{Float64}` — `Mline` is `(Ncell, 35)`. Returns `dMdt` `(Ncell, 35)` = `-(Fhat_{i+1/2} - Fhat_{i-1/2})/dx`. `order=1` uses cell-centered states (no reconstruction); `order=2` uses MUSCL. Zero-gradient BC at both ends. Order degrades to 1 locally at a cell whose reconstructed face fails `M000>0`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_highorder_1d.jl`:
```julia
@testset "1D residual" begin
    Ncell = 16
    M0 = InitializeM4_35(1.0, 0.2, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    # uniform field -> zero residual (interior)
    Mline = repeat(reshape(M0,1,35), Ncell, 1)
    R = residual_1d(Mline, 0.1, 0.0; order=2)
    @test maximum(abs.(R[3:Ncell-2, :])) < 1e-9
    @test size(R) == (Ncell, 35)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using HyQMOM; @assert isdefined(HyQMOM,:residual_1d)'`
Expected: FAIL — `residual_1d` not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `src/numerics/highorder_flux.jl`:
```julia
"""
    residual_1d(Mline, dx, Ma; order=2)

Method-of-lines spatial residual for a 1D row of 35-moment cells (Ncell x 35) in
the x-direction. order=1: first-order (cell-centered). order=2: MUSCL on the
bounded reconstruction variables, with local fallback to first order if a
reconstructed face has nonpositive density.
"""
function residual_1d(Mline::AbstractMatrix, dx::Float64, Ma::Real; order::Int=2)
    Nc = size(Mline, 1)
    axis = 1
    # Right-face L/R moment states at each interface i+1/2, i=1..Nc-1
    ML = [zeros(35) for _ in 1:Nc-1]   # left state at interface i+1/2 (from cell i)
    MR = [zeros(35) for _ in 1:Nc-1]   # right state at interface i+1/2 (from cell i+1)
    if order == 1
        for i in 1:Nc-1
            ML[i] = Mline[i, :]; MR[i] = Mline[i+1, :]
        end
    else
        V = [to_recon_vars(Mline[i, :]) for i in 1:Nc]
        # per-cell left/right face recon-vars with zero-gradient BC
        Vminus = [zeros(35) for _ in 1:Nc]; Vplus = [zeros(35) for _ in 1:Nc]
        for i in 1:Nc
            vm = V[max(i-1,1)]; v0 = V[i]; vp = V[min(i+1,Nc)]
            Vminus[i], Vplus[i] = muscl_faces(vm, v0, vp)
        end
        for i in 1:Nc-1
            Li = from_recon_vars(Vplus[i])     # right face of cell i
            Ri = from_recon_vars(Vminus[i+1])  # left face of cell i+1
            # local order degradation: fall back to 1st order on bad reconstruction
            ML[i] = (Li[1] > 0) ? Li : Mline[i, :]
            MR[i] = (Ri[1] > 0) ? Ri : Mline[i+1, :]
        end
    end
    Fhat = [face_flux_1d(ML[i], MR[i], axis, Ma) for i in 1:Nc-1]
    R = zeros(Nc, 35)
    for i in 2:Nc-1
        R[i, :] = -(Fhat[i] .- Fhat[i-1]) ./ dx
    end
    # zero-gradient BC: no net flux at the physical boundary cells
    return R
end
```
Export in `src/HyQMOM.jl`:
```julia
export residual_1d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'include("test/test_highorder_1d.jl")'`
Expected: PASS — uniform residual ≈ 0 in the interior.

- [ ] **Step 5: Commit**

```bash
git add src/numerics/highorder_flux.jl src/HyQMOM.jl test/test_highorder_1d.jl
git commit -m "feat: 1D MUSCL spatial residual with order degradation"
```

---

### Task 5: SSP-RK3 time stepper

**Files:**
- Create: `src/numerics/ssp_rk.jl`
- Modify: `src/HyQMOM.jl` (include + export)
- Test: `test/test_highorder_1d.jl`

**Interfaces:**
- Produces:
  - `ssp_rk3_step(M, dt, L)::typeof(M)` — one SSP-RK3 update for state `M` (any array) given residual function `L(M)->dM/dt`. Stages: `M1=M+dt*L(M)`, `M2=3/4 M+1/4(M1+dt*L(M1))`, `Mnew=1/3 M+2/3(M2+dt*L(M2))`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_highorder_1d.jl`:
```julia
@testset "SSP-RK3 order" begin
    # scalar ODE dy/dt = -y, y(0)=1, exact y(T)=exp(-T)
    L(y) = -y
    T = 1.0
    err(n) = (dt = T/n; y = 1.0; for _ in 1:n; y = ssp_rk3_step(y, dt, L); end; abs(y - exp(-T)))
    e1 = err(10); e2 = err(20)
    @test e2 < e1
    @test log2(e1/e2) > 2.7   # ~3rd-order convergence
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using HyQMOM; @assert isdefined(HyQMOM,:ssp_rk3_step)'`
Expected: FAIL — `ssp_rk3_step` not defined.

- [ ] **Step 3: Write minimal implementation**

Create `src/numerics/ssp_rk.jl`:
```julia
"""
    ssp_rk3_step(M, dt, L)

One 3-stage strong-stability-preserving RK3 update of state `M` with residual
operator `L(M) -> dM/dt`. Works for scalars or arrays.
"""
function ssp_rk3_step(M, dt, L)
    k0 = L(M)
    M1 = M .+ dt .* k0
    k1 = L(M1)
    M2 = (3/4) .* M .+ (1/4) .* (M1 .+ dt .* k1)
    k2 = L(M2)
    return (1/3) .* M .+ (2/3) .* (M2 .+ dt .* k2)
end
```
Add to `src/HyQMOM.jl` after `include("numerics/highorder_flux.jl")`:
```julia
include("numerics/ssp_rk.jl")
```
Export:
```julia
export ssp_rk3_step
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'include("test/test_highorder_1d.jl")'`
Expected: PASS — observed convergence rate > 2.7.

- [ ] **Step 5: Commit**

```bash
git add src/numerics/ssp_rk.jl src/HyQMOM.jl test/test_highorder_1d.jl
git commit -m "feat: SSP-RK3 time stepper"
```

---

### Task 6: 1D order-of-accuracy, realizability, conservation tests

**Files:**
- Modify: `test/test_highorder_1d.jl`
- Modify: `test/runtests.jl` (register the new test file)

**Interfaces:**
- Consumes: `residual_1d`, `ssp_rk3_step`, `to_recon_vars`/`from_recon_vars` (all above).

- [ ] **Step 1: Write the failing test**

Append to `test/test_highorder_1d.jl`:
```julia
# advance a 1D periodic moment field; helper used by the tests below
function _advance_1d(Mline, dx, dt, nsteps, Ma)
    L(M) = residual_1d(M, dx, Ma; order=2)
    for _ in 1:nsteps
        Mline = ssp_rk3_step(Mline, dt, L)
    end
    return Mline
end

@testset "1D smooth order-of-accuracy" begin
    # smooth density bump advecting at u=1; measure self-convergence under refinement
    Ma = 0.0; u = 1.0; tfinal = 0.05
    function setup(N)
        dx = 1.0/N
        Mline = zeros(N, 35)
        for i in 1:N
            x = (i-0.5)*dx
            rho = 1.0 + 0.2*sin(2pi*x)
            Mline[i, :] = InitializeM4_35(rho, u, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
        end
        return Mline, dx
    end
    function run(N)
        Mline, dx = setup(N)
        dt = 0.2*dx/(u + 3.0)          # CFL-safe for the moment wave speeds
        nsteps = ceil(Int, tfinal/dt); dt = tfinal/nsteps
        _advance_1d(Mline, dx, dt, nsteps, Ma)
    end
    # Richardson self-convergence on density (M000): rate between N, 2N, 4N
    d(N) = run(N)[:, 1]
    coarsen(a) = (a[1:2:end] .+ a[2:2:end]) ./ 2
    eC = maximum(abs.(coarsen(d(64)) .- d(32)))
    eF = maximum(abs.(coarsen(d(128)) .- d(64)))
    @test eF < eC
    @test log2(eC/eF) > 1.6        # ~2nd order (limiter may shave it slightly)
end

@testset "1D realizability + conservation (shock tube)" begin
    Ma = 0.0; N = 100; dx = 1.0/N
    Ml = InitializeM4_35(1.0,   0.0,0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    Mr = InitializeM4_35(0.125, 0.0,0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    Mline = zeros(N, 35)
    for i in 1:N
        Mline[i, :] = (i <= N÷2) ? Ml : Mr
    end
    mass0 = sum(Mline[:, 1])
    dt = 0.2*dx/3.0; nsteps = 40
    Mline = _advance_1d(Mline, dx, dt, nsteps, Ma)
    @test all(isfinite, Mline)
    @test minimum(Mline[:, 1]) > 0                     # density positive
    # realizable: variances positive everywhere
    for i in 1:N
        _, S4 = M2CS4_35(Mline[i, :])
        @test (S4[5]-S4[4]^2-1) > -1e-8                # H200 >= 0 (x)
    end
    @test abs(sum(Mline[:, 1]) - mass0) / mass0 < 1e-12  # mass conserved (no through-flow @ walls)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'include("test/test_highorder_1d.jl")'`
Expected: FAIL initially only if a prior task is incomplete; otherwise these run. If the order test fails (rate ≤ 1.6) or realizability fails, debug the reconstruction/limiter before proceeding. Expected once correct: PASS.

- [ ] **Step 3: Register the test file**

In `test/runtests.jl`, add inside the `@testset "Unit Tests"` block, after `include("test_hyqmom_closure_golden.jl")`:
```julia
        include("test_highorder_1d.jl")
```

- [ ] **Step 4: Run the full unit suite to verify no regressions**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true mpiexec -n 1 julia --project=. -e 'using Pkg; Pkg.test()'`
(If `Pkg.test()` is blocked by the known MAT/HDF5 precompile issue, instead run the no-MAT core set plus the new file:
`HYQMOM_SKIP_PLOTTING=true CI=true mpiexec -n 1 julia --project=. -e 'using Test, HyQMOM; include("test/test_highorder_1d.jl")'`)
Expected: PASS — all high-order 1D testsets green; existing tests unaffected.

- [ ] **Step 5: Commit**

```bash
git add test/test_highorder_1d.jl test/runtests.jl
git commit -m "test: 1D high-order order-of-accuracy, realizability, conservation"
```

---

### Task 7: 1D shock-tube driver — diffusion vs first order

**Files:**
- Create: `examples/run_1d_highorder.jl`

**Interfaces:**
- Consumes: `residual_1d`, `ssp_rk3_step`, `InitializeM4_35`, `M2CS4_35`.

- [ ] **Step 1: Write the driver**

Create `examples/run_1d_highorder.jl`:
```julia
# 1D moment shock tube: compare first-order vs MUSCL-2 numerical diffusion.
# Usage: HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. examples/run_1d_highorder.jl
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, Printf

Ma = 0.0; N = 200; dx = 1.0/N; tfinal = 0.1
function ic()
    M = zeros(N, 35)
    for i in 1:N
        rho = (i <= N÷2) ? 1.0 : 0.125
        M[i, :] = InitializeM4_35(rho, 0.0,0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    end
    return M
end
function advance(order)
    M = ic(); dt = 0.2*dx/3.0; nsteps = ceil(Int, tfinal/dt); dt = tfinal/nsteps
    L(x) = residual_1d(x, dx, Ma; order=order)
    for _ in 1:nsteps; M = ssp_rk3_step(M, dt, L); end
    return M
end
M1 = advance(1); M2 = advance(2)
# sharpness metric: max density gradient (higher = less diffused)
g1 = maximum(abs.(diff(M1[:,1]))); g2 = maximum(abs.(diff(M2[:,1])))
@printf("max |drho/dx|: first-order=%.4f  MUSCL-2=%.4f  (ratio %.2fx sharper)\n", g1, g2, g2/g1)
@printf("density range first-order=[%.4f,%.4f]  MUSCL-2=[%.4f,%.4f]\n",
        minimum(M1[:,1]),maximum(M1[:,1]), minimum(M2[:,1]),maximum(M2[:,1]))
@printf("mass conserved: first=%.3e  muscl=%.3e (rel drift)\n",
        abs(sum(M1[:,1])-sum(ic()[:,1]))/sum(ic()[:,1]),
        abs(sum(M2[:,1])-sum(ic()[:,1]))/sum(ic()[:,1]))
```

- [ ] **Step 2: Run the driver**

Run: `HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. examples/run_1d_highorder.jl`
Expected: prints a `max |drho/dx|` ratio > 1 (MUSCL-2 noticeably sharper than first-order), both mass drifts ~1e-13. If the ratio is ≤ 1, the reconstruction is not reducing diffusion — debug before committing.

- [ ] **Step 3: Commit**

```bash
git add examples/run_1d_highorder.jl
git commit -m "feat: 1D shock-tube driver comparing first-order vs MUSCL-2 diffusion"
```

---

## Self-review notes

- **Spec coverage:** reconstruction variable set (Task 1), MUSCL + limiter (Task 2), face projection + HLL (Task 3), residual + order degradation (Task 4), SSP-RK3 (Task 5), order/realizability/conservation validation + diffusion-reduction demo (Tasks 6–7). WENO5, 3D, and MPI are explicitly Phase 2/3 (separate plans), per the spec.
- **Types:** `to_recon_vars`/`from_recon_vars` exchange length-35 `Vector{Float64}`; `muscl_faces` returns a tuple of two length-35 vectors; `face_flux_1d` returns length-35; `residual_1d` takes/returns `(Ncell,35)`; `ssp_rk3_step` is generic over the state type. Consistent across tasks.
- **Open item carried from spec:** if the order test (Task 6) shows projection over-smearing (rate well below 2 or diffusion ratio ≈ 1), revisit the reconstruction variable set / limiter and consider keeping the highest standardized moments first-order (Jacob's abscissa-first-order analogue) before moving to Phase 2.
