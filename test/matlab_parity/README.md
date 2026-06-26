# MATLAB ↔ Julia parity harness

Numerical equivalence checks between the Julia kernels and the reference MATLAB
implementation in `Code_Riemann_3D_35mom_july2026` (the revised moment-projection
method, Appendix B). Used to validate the `hyqmom_3D` closure update and the
`projection35` / `realizable_3D_M4` port.

## Environment (PACE / compute node)

```bash
module load julia/1.11.3 openmpi/4.1.5     # OpenMPI matches the pinned ABI in LocalPreferences.toml
module load matlab/r2024b
```

The MATLAB source path is hard-coded near the top of the `.m` files
(`/storage/.../Code_Riemann_3D_35mom_july2026_GT/src`); edit if it moves.

## Scripts

| Script | Purpose |
|---|---|
| `gen_julia.jl` / `gen_matlab.m` | Kernel parity: `M2CS4_35`, `delta2star3D`, `C4toM4_3D`, `hyqmom_3D`, `realizability_S2`, `realizability_S220` (MATLAB spells it `realizablity_S220`) on identical particle-sampled moment vectors. |
| `gen_proj_inputs.jl` | Generates realizable + deliberately unrealizable `M4` test vectors. |
| `eval_proj_julia.jl` / `eval_proj_matlab.m` | Projection parity: full `realizable_3D_M4(M4,Ma)` wrapper (Ma=2,5) and `projection35` in isolation. |

## Running

```bash
cd <repo>
mkdir -p /tmp/kdiff
julia --project=. test/matlab_parity/gen_julia.jl
matlab -nodisplay -batch "run('test/matlab_parity/gen_matlab.m')"
julia --project=. test/matlab_parity/gen_proj_inputs.jl
julia --project=. test/matlab_parity/eval_proj_julia.jl
matlab -nodisplay -batch "run('test/matlab_parity/eval_proj_matlab.m')"
# then diff the jl_*.txt vs ml_*.txt files in /tmp/kdiff
```

## Last results (julia 1.11.3 vs matlab r2024b)

| Kernel | max abs Δ |
|---|---|
| `M2CS4_35`, `delta2star3D`, `C4toM4_3D`, `realizability_S220`, `realizability_S2` | ≤ 1e-12 |
| `hyqmom_3D` (after closure update) | 8.7e-14 |
| `projection35` (isolated, 289/300 triggered) | 3.9e-14 |
| `realizable_3D_M4` wrapper (Ma=2 and Ma=5) | 1.8e-12 |

The frozen `hyqmom_3D` reference vectors live in `test/goldenfiles/` and are
checked by `test/test_hyqmom_closure_golden.jl` (no MATLAB needed at test time).
