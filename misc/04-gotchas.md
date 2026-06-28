# Gotchas (read before debugging anything weird)

## Environment / ABI

- **HOME is over quota.** Any write to `$HOME` (depot artifacts, logs, data) fails or
  silently truncates. Set `JULIA_DEPOT_PATH`/`TMPDIR` to scratch (see `01-environment.md`).
- **System OpenMPI is `--without-cuda`.** CUDA-aware MPI does NOT work here
  (`ompi_info | grep cuda` shows the API stub but the transport is not CUDA-aware).
  Halos are **host-staged** (GPU→host→MPI→host→GPU). This is the right portable design
  anyway; the halo data is tiny (`35·n·n·halo`).
- **UCX init is flaky.** For multi-rank runs disable it:
  `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader`. (Fine — host-staged needs no UCX.)
- **Reading `.mat` (MAT.jl) clashes with the system-MPI binding.**
  `MAT → HDF5_jll → OpenMPI_jll` fails to load when the system OpenMPI is on
  `LD_LIBRARY_PATH`: `libmpi_mpifh.so: undefined symbol: ompi_instance_count`.
  Fix: read `.mat` in a **separate env without the system-OpenMPI `LD_LIBRARY_PATH`**
  (the JLL libs then load consistently; reading a `.mat` needs no MPI). A throwaway
  `matenv` with just `MAT` is used to dump goldens to fp64, which gpuenv2 then reads.
- **CI: transient `pkg.julialang.org` registry 404.** `Pkg.instantiate()` sometimes
  fails fast (~30–40 s) with `HTTP/2 404 while requesting .../registry/...` — a CDN
  snapshot flake, NOT a code error; the next push usually passes. The `CI` workflow
  retries 4× then falls back to the git registry (`JULIA_PKG_SERVER=""`). Check the log
  for the 404 before debugging code.

## Floating-point parity

- **Shared `@fastmath` central-moment helpers MUST be `@noinline`.** `@fastmath` lets
  LLVM reassociate the catastrophic-cancellation central-moment formulas depending on
  the *surrounding* context. Inlined into `to_recon_vars_dev`/`from_recon_vars_dev`,
  the centrals drifted ~1 ULP from the standalone autogen `M4toC4_3D`/`C4toM4_3D`,
  amplified to ~2e-7 by the `1/sC200^4` standardization on deep-vacuum (ρ~1e-5) cells —
  which broke the golden battery (1250 entries). `@noinline` pins each helper to the
  autogen's compilation. **Do not change `_recon_centrals` / `_c4tom4_35` to `@inline`.**
- **`@fastmath` stays OFF in the wave-speed path.** GPU `rsqrt` under `@fastmath` can
  flip the hyperbolicity discriminant's sign → wrong wave speeds.
- **fp64 everywhere.** fp32 is catastrophic on the companion-like 4×4 wave-speed blocks
  (percent-level error); the `schur4` eps constants are fp64-specific.
- **`schur4` (GPU) vs LAPACK (CPU) differ by ~4.6e-8** on ill-conditioned high-Ma
  companion blocks (verified vs 300-bit BigFloat — genuine `schur4` error, ~1000× less
  accurate than LAPACK there, but far below the CFL/wave-speed tolerance and below fp32
  eps). This is why the GPU residual matches the CPU only to ~1e-6, and why the CPU
  keeps LAPACK rather than adopting `schur4`.
- **Multi-step high-order cross moments are conditioning-limited.** At shocks
  `dt·R ≫ M`, so the high-order moments are not bit-reproducible CPU↔GPU (a 1e-15 CPU
  self-perturbation diverges O(1)). Validate the dt sequence (EXACT) + density/low-order
  moments, not the high-order moments.

## CUDA kernel authoring

- **No tuple-splat in device code.** `f(x...)` lowers to `_apply_iterate`, unsupported
  in GPU kernels. The device functions take explicit scalar args / index NTuples
  explicitly (that's why there are `_tup` wrappers).
- **Top-level `for`-loop scope.** Variables reassigned in a top-level loop become new
  locals (Julia soft-scope) — wrap timing/accumulation loops in a function, or use
  `global`, or you'll get `UndefVarError`.
- **`MPI.Sendrecv!` needs the `comm` positional arg:** `Sendrecv!(send, recv, comm; dest=…, source=…)`.
  Omitting `comm` gives a confusing `MethodError`.

## Decomposition / dimensionality

- **2D = `nz=1`, single-GPU only.** The 35-moment velocity space is always 3D; a 2D run
  is a `nz=1` spatial grid. `flag2D` is a legacy no-op. The z-slab multi-GPU march
  asserts `nz_loc >= halo`, so it cannot decompose `nz=1` — 2D runs on one GPU by design.
- **z-slab ghosts: outflow at the global boundary must replicate the edge plane** to
  match the cubic index-clamp exactly; internal seams use neighbor real planes. Get this
  wrong and the boundary-slab residual silently diverges from single-GPU.
