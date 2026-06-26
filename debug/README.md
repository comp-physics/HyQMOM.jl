# debug/ — reproducers and validation tooling

Standalone scripts (run with `julia --project=. debug/<script>.jl`). They are not
part of the package and are excluded from CI.

- **`run_ma100_demo.jl`** — runs the 3D crossing-jets demo and saves the moment
  field. Configurable via env vars `REPRO_NP`, `REPRO_MA`, `REPRO_TMAX`,
  `REPRO_ORDER` (1=first-order HLL, 2=high-order HLL+MUSCL), `REPRO_VACFLOOR`
  (near-vacuum first-order fallback density; see `ho_vacuum_floor`).

- **`repro_1d_crash.jl`** — cheap serial 1D analog of the high-order crossing
  (two dense Mach-`Ma` slabs colliding through near-vacuum). Reproduces the
  high-order near-vacuum behaviour in seconds. Env: `R1D_MA`, `R1D_N`,
  `R1D_ORDER`, `R1D_VACFLOOR`. See `docs/ma100-highorder-crash-analysis.md`.

- **`golden_kernels.jl`** — bit-level regression harness for the realizability /
  eigenvalue / flux kernels. `... golden_kernels.jl capture` writes a reference;
  `... golden_kernels.jl compare` checks the current code against it. Used to gate
  numerics-preserving refactors.
