ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, Printf

Ma = 10.0
# a realizable, slightly off-equilibrium state
M = collect(InitializeM4_35(1.0, 3.0, 1.0, 0.5, 1.2, 0.1, 0.05, 0.9, 0.02, 1.1))
Mr = realizable_3D_M4(M, Ma)           # ensure realizable
V  = to_recon_vars(Mr)

bench(f, n) = (f(); t=@timed (for _ in 1:n; f(); end); (us=t.time/n*1e6, bytes=t.bytes/n, gcpct=100*t.gctime/t.time))

n = 200_000
r1 = bench(()->realizable_3D_M4(Mr, Ma), n)
r2 = bench(()->face_flux_1d(Mr, Mr, 1, Ma), n)
r3 = bench(()->from_recon_vars(to_recon_vars(Mr)), n)
r4 = bench(()->realize_and_speed(Mr, 1, Ma), n)

@printf("%-26s %10s %12s %8s\n", "kernel (per call)", "time[us]", "alloc[B]", "gc%")
@printf("%-26s %10.3f %12.0f %8.1f\n", "realizable_3D_M4", r1.us, r1.bytes, r1.gcpct)
@printf("%-26s %10.3f %12.0f %8.1f\n", "face_flux_1d (2 sides)", r2.us, r2.bytes, r2.gcpct)
@printf("%-26s %10.3f %12.0f %8.1f\n", "recon roundtrip", r3.us, r3.bytes, r3.gcpct)
@printf("%-26s %10.3f %12.0f %8.1f\n", "realize_and_speed", r4.us, r4.bytes, r4.gcpct)

# rough per-cell-per-step model for unsplit RK3 order=2:
#   3 RK stages x [ 3 dirs x 2 faces x face_flux_1d  +  1 realizable_3D_M4 projection ]
per_cell = 3*(3*2*r2.us + r1.us)
@printf("\nmodeled order-2 cost/cell/step ~ %.1f us  (3 RK x [6 face_flux + 1 proj])\n", per_cell)
@printf("=> Np=128 (2.1e6 cells) on 64 ranks ~ %.1f s/step\n", per_cell*1e-6*2.1e6/64)
