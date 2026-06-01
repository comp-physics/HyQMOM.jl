# Bubble convergence study — results (PRIVATE; unpublished)

Job 9371274, 64 MPI ranks (graniterapids), `srun --mpi=pmix`, system OpenMPI 4.1.5.
Bubble: domain [-0.5,0.5]^2, disk r=0.25, rho_in/rho_out=2 (isothermal, T=1), zero
velocity, Nz=1, tmax=0.05, CFL=0.5. Waves stay off the wall (edge<rho>=1.0) and
rho in [1,2] at all resolutions.

## SMOOTH bubble — clean grid convergence (Reviewer #1 HEADLINE)

Gaussian density bump (rho = 1 + 0.5 exp(-r^2/2w^2), w=0.1, isothermal, zero
velocity) -> C-infinity solution, no discontinuity. Job 9374704, 64 ranks.
Density L1 self-difference, observed order p:

| pair      | Kn=0.01  p     | Kn=0.1   p     | Kn=1.0   p     |
|-----------|----------------|----------------|----------------|
| 128-256   | 7.84e-4   –    | 7.94e-4   –    | 7.97e-4   –    |
| 256-512   | 4.03e-4  0.961 | 4.08e-4  0.961 | 4.09e-4  0.962 |
| 512-1024  | 2.04e-4  0.980 | 2.06e-4  0.985 | 2.07e-4  0.985 |

CLEAN first order (p -> 0.98), uniform across all regimes including rarefied
Kn=1. Contrast with the discontinuous bubble below, where Kn=1 collapses to
p~0.13: that collapse is the CONTACT DISCONTINUITY (L1 order ~1/2, plus rarefied
fine structure), NOT a solver/closure defect. The smooth bubble isolates the
scheme's true design order. -> Use this for the R1 convergence demonstration.

Rotational invariance (smooth-region azimuthal density deviation), smooth bubble:

| N    | Kn=0.01  | Kn=0.1   | Kn=1.0   |
|------|----------|----------|----------|
| 128  | 5.01e-4  | 5.71e-4  | 5.71e-4  |
| 256  | 4.84e-4  | 4.80e-4  | 4.79e-4  |
| 512  | 4.74e-4  | 4.70e-4  | 4.69e-4  |
| 1024 | 4.67e-4  | 4.63e-4  | 4.62e-4  |

Residual rotational asymmetry settles to ~0.046% (uniform across Kn), decreasing
only weakly -> this is the closure's INTRINSIC anisotropy (a property of the
closure, not a grid artifact, since the IC/solution are smooth). A clean
quantitative invariance bound for Reviewer #1, Q1 (~0.05%), consistent with Rice
et al.: these closures are not perfectly rotationally invariant but the deviation
is small.

---

## Discontinuous bubble (Rice et al. validation case) — used for R3 diagnostics below

## Grid convergence — density L1 self-difference, observed order p = log2(e_h/e_{h/2})

| pair      | Kn=0.01  p     | Kn=0.1   p     | Kn=1.0   p     |
|-----------|----------------|----------------|----------------|
| 128-256   | 4.58e-3   –    | 4.96e-3   –    | 5.27e-3   –    |
| 256-512   | 2.56e-3  0.84  | 3.06e-3  0.70  | 4.03e-3  0.39  |
| 512-1024  | 1.56e-3  0.72  | 2.34e-3  0.39  | 3.67e-3  0.13  |

Interpretation (HONEST): converges cleanly only near the continuum limit. The
bubble's contact discontinuity caps first-order L1 convergence at ~1/2; in the
rarefied regime (Kn=1) the free-transport HyQMOM closure develops fine structure
and the differences plateau (p~0.13 at finest). The discontinuous bubble is a
poor vehicle for a clean convergence claim. RECOMMENDATION: use a SMOOTH bubble
(Gaussian density bump) for R1's convergence demonstration; keep the discontinuous
case for the correction/conservation/Mach diagnostics below.

## Rotational invariance — azimuthal density deviation (Reviewer #1, Q1)

rms over radial shells (rms_all = interface-dominated; rms_smooth excludes
steep-gradient front shells -> closure intrinsic asymmetry):

| N    | Kn=0.01 rms_smooth | Kn=0.1 | Kn=1.0 |
|------|--------------------|--------|--------|
| 128  | 8.87e-4            | 1.29e-3| 1.30e-3|
| 256  | 5.57e-4            | 1.11e-3| 1.25e-3|
| 512  | 7.21e-4            | 1.54e-3| 1.85e-3|
| 1024 | 6.43e-4            | 1.61e-3| 2.31e-3|

Closure preserves rotational symmetry to <0.1% (continuum) to <0.25% (rarefied)
in smooth regions; residual is a small bound that does not vanish under refinement
(consistent with Rice et al.: these closures are not *perfectly* rotationally
invariant; collisions improve symmetry). A quantitative invariance bound, not a
recovery-to-zero.

## Correction activity & conservation (Reviewer #3) — bubble, Kn x mesh

| Kn   | N    | frac_real | frac_hyp | mean|dM| | max|dM|  | max dConserved/corr |
|------|------|-----------|----------|----------|----------|---------------------|
| 0.01 | 128  | 0.331     | 4.2e-5   | 1.4e-6   | 7.0e-3   | 1.3e-15             |
| 0.01 | 1024 | 0.268     | 1.8e-5   | 2.5e-6   | 4.8e-2   | 1.8e-15             |
| 0.1  | 1024 | (see corrections_Kn0.1.txt)                                       |
| 1.0  | 1024 | (see corrections_Kn1.txt)                                         |

Key R3 findings:
- Hyperbolicity correction almost never fires at Ma=0 (frac_hyp ~ 1e-5).
- Corrections are tiny in magnitude (mean|dM| ~ 1e-6) -> local safety net, not
  solution-shaping (R3-1).
- Correction preserves conserved moments to machine precision, all Kn/mesh
  (max dConserved/correction ~ 1e-15) (R3-5).

## Correction activity vs Mach number (Reviewer #3-4) — crossing jets, N=256, Kn=1

| Ma | frac_real | frac_hyp | mean|dM|  | max|dM|  | dConserved/corr | mass drift |
|----|-----------|----------|-----------|----------|-----------------|------------|
| 0  | 0.171     | 7.9e-5   | 7.6e-6    | 0.053    | 6.7e-16         | 4.3e-15    |
| 2  | 0.181     | 9.2e-4   | 3.9e-4    | 5.16     | 3.8e-15         | 6.6e-9     |
| 4  | 0.259     | 1.4e-3   | 2.0e-3    | 33.9     | 1.6e-14         | 1.5e-7     |

Correction activity grows sharply and monotonically with Ma (frac_hyp x18,
mean|dM| x270 from Ma 0->4) -> confirms R3-4: low-speed needs ~no correction,
high-Ma needs it. The correction preserves conserved moments to ~1e-14 even at
Ma=4. Global mass drift grows with Ma (4e-15 -> 1.5e-7) but stays at 7+ digits.
Job 9371274 COMPLETED (exit 0, 1:54 walltime, 64 ranks).

## Note
Raw moment fields (studies/bubble/out/*.bin) are NOT in git (1024^2 = 287 MB each
> GitHub 100 MB limit). They live on the cluster and are regenerable via
studies/bubble/sweep.sbatch. Only scripts + summaries + correction logs are backed up.
