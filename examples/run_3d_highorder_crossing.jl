"""
3D High-Order vs First-Order Crossing Validation Driver

Compares spatial_order=1 (first-order HLL/Euler) vs spatial_order=2 (high-order SSP-RK3)
on the crossing-jets IC (ic_type=:crossing_matlab) to demonstrate reduced numerical diffusion.

Sharpness metric: maximum density gradient magnitude |∇ρ| over the domain
(finite differences of M_final[:,:,:,1]).  High-order should yield larger
max|∇ρ| (sharper interface) and higher peak density in the collision core.

Usage:
  UCX_TLS=sm,self HYQMOM_SKIP_PLOTTING=true CI=true \\
    mpiexec -n 8 julia --project=. examples/run_3d_highorder_crossing.jl

  # or serial
  HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. examples/run_3d_highorder_crossing.jl
"""

ENV["HYQMOM_SKIP_PLOTTING"] = "true"
ENV["CI"] = "true"

using MPI
MPI.Initialized() || MPI.Init()
using HyQMOM
using Printf
using LinearAlgebra

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# ---------------------------------------------------------------------------
# Helper: compute max |∇ρ| over the global domain using finite differences.
# Uses proper one-sided differences at domain boundaries so the metric is
# well-defined everywhere (no silent zero-gradient at boundary cells):
#   interior:     (rho[i+1] - rho[i-1]) / (2*dx)   (centered)
#   low boundary: (rho[2]   - rho[1])   / dx         (forward)
#   high boundary:(rho[N]   - rho[N-1]) / dx         (backward)
# and analogously for y, z.
# ---------------------------------------------------------------------------
function max_density_gradient(rho::Array{Float64,3}, dx, dy, dz)
    Nx, Ny, Nz = size(rho)
    max_g = 0.0
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # x-direction
        if i == 1
            drx = (rho[2,j,k] - rho[1,j,k]) / dx
        elseif i == Nx
            drx = (rho[Nx,j,k] - rho[Nx-1,j,k]) / dx
        else
            drx = (rho[i+1,j,k] - rho[i-1,j,k]) / (2*dx)
        end
        # y-direction
        if j == 1
            dry = (rho[i,2,k] - rho[i,1,k]) / dy
        elseif j == Ny
            dry = (rho[i,Ny,k] - rho[i,Ny-1,k]) / dy
        else
            dry = (rho[i,j+1,k] - rho[i,j-1,k]) / (2*dy)
        end
        # z-direction
        if k == 1
            drz = (rho[i,j,2] - rho[i,j,1]) / dz
        elseif k == Nz
            drz = (rho[i,j,Nz] - rho[i,j,Nz-1]) / dz
        else
            drz = (rho[i,j,k+1] - rho[i,j,k-1]) / (2*dz)
        end
        g = sqrt(drx^2 + dry^2 + drz^2)
        if g > max_g; max_g = g; end
    end
    return max_g
end

# ---------------------------------------------------------------------------
# Run helper: run simulation_runner for a given spatial_order
# ---------------------------------------------------------------------------
function run_order(order::Int, Np::Int, Ma::Float64, tmax::Float64; label::String="")
    params = (
        Nx = Np, Ny = Np, Nz = Np,
        Nmom = 35,
        tmax = tmax,
        Kn = 1000.0,
        Ma = Ma,
        flag2D = 0,
        CFL = 1/3,
        nnmax = 100000,
        dtmax = 1000.0,
        rhol = 1.0,
        rhor = 0.001,
        T = 1.0,
        r110 = 0.0,
        r101 = 0.0,
        r011 = 0.0,
        symmetry_check_interval = 1000,
        homogeneous_z = false,
        debug_output = false,
        snapshot_interval = 0,
        ic_type = :crossing_matlab,
        spatial_order = order,
    )

    if rank == 0
        println("\n" * "="^70)
        @printf("  %s  spatial_order=%d  Np=%d  Ma=%.1f  tmax=%.4f\n",
                label, order, Np, Ma, tmax)
        println("="^70)
        flush(stdout)
    end

    result = simulation_runner(params)
    return result
end

# ---------------------------------------------------------------------------
# Analysis helper: print stats + return sharpness
# ---------------------------------------------------------------------------
function analyze_result(result, Np::Int, order::Int, Ma::Float64, tmax::Float64;
                        label::String="")
    M_final, t_final, steps, grid = result
    if rank != 0
        return nothing
    end

    dx = 1.0/Np; dy = 1.0/Np; dz = 1.0/Np
    rho = M_final[:,:,:,1]

    mass_final = sum(rho) * dx * dy * dz

    # Compute initial mass analytically (crossing_matlab IC):
    # two cubes of side floor(0.1*Np)+1 cells at rhol, rest at rhor
    Csize = floor(Int, 0.1 * Np)
    ncube = (Csize + 1)^3
    ntotal = Np^3
    mass0 = (2*ncube * 1.0 + (ntotal - 2*ncube) * 0.001) * dx * dy * dz
    rel_drift = abs(mass_final - mass0) / mass0

    rho_min = minimum(rho)
    rho_max = maximum(rho)
    rho_peak = rho_max  # peak retained density in collision core

    sharpness = max_density_gradient(rho, dx, dy, dz)

    println()
    @printf("  %-30s order=%d\n", label, order)
    @printf("    steps            = %d\n", steps)
    @printf("    t_final          = %.6f  (target %.4f)\n", t_final, tmax)
    @printf("    mass0            = %.6e\n", mass0)
    @printf("    mass_final       = %.6e\n", mass_final)
    @printf("    rel_mass_drift   = %.3e\n", rel_drift)
    @printf("    rho_min          = %.6e\n", rho_min)
    @printf("    rho_max (peak)   = %.6e\n", rho_max)
    @printf("    max|grad(rho)|   = %.6e   <-- sharpness metric\n", sharpness)
    @printf("    all_finite       = %s\n", string(all(isfinite, M_final)))
    @printf("    rho_positive     = %s\n", string(rho_min > 0.0))
    println()
    flush(stdout)

    return sharpness
end

# ===========================================================================
# MAIN COMPARISON  — Np=48, Ma=4, tmax=0.01
# ===========================================================================
Np_main = 48
Ma_main = 4.0
tmax_main = 0.01

if rank == 0
    println("\n" * "#"^70)
    println("# HIGH-ORDER VS FIRST-ORDER 3D CROSSING  —  MAIN COMPARISON")
    println("#   Np=$(Np_main)  Ma=$(Ma_main)  tmax=$(tmax_main)")
    println("#"^70)
    flush(stdout)
end

result1 = run_order(1, Np_main, Ma_main, tmax_main; label="MAIN Ma=$(Ma_main)")
result2 = run_order(2, Np_main, Ma_main, tmax_main; label="MAIN Ma=$(Ma_main)")

if rank == 0
    println("\n--- Analysis: MAIN comparison ---")
end
sharp1 = analyze_result(result1, Np_main, 1, Ma_main, tmax_main; label="MAIN Ma=$(Ma_main)")
sharp2 = analyze_result(result2, Np_main, 2, Ma_main, tmax_main; label="MAIN Ma=$(Ma_main)")

if rank == 0 && sharp1 !== nothing && sharp2 !== nothing
    ratio = sharp2 / sharp1
    @printf("\n  === SHARPNESS RATIO order2/order1 (Ma=%.1f, Np=%d) ===\n", Ma_main, Np_main)
    @printf("      max|grad(rho)| order=1  : %.6e\n", sharp1)
    @printf("      max|grad(rho)| order=2  : %.6e\n", sharp2)
    @printf("      ratio order2/order1     : %.4f\n", ratio)
    if ratio > 1.0
        @printf("  ==> high-order produces sharper gradients (ratio = %.4f) at the same\n", ratio)
        @printf("      final time t=%.4f — consistent with reduced numerical diffusion.\n", tmax_main)
        println("      (A sharper max|∇ρ| is consistent with, but not a rigorous proof of,")
        println("       reduced diffusion; MUSCL can locally over-sharpen at a compression.)")
    else
        println("  ==> HIGH-ORDER is NOT sharper (ratio <= 1).")
        println("      Possible causes: face projection diffusion, limiter, CFL, short tmax.")
    end
    println()
    flush(stdout)
end

# ===========================================================================
# SANITY / REGRESSION CHECK  — Np=32, Ma=2, tmax=0.01
# ===========================================================================
Np_reg = 32
Ma_reg = 2.0
tmax_reg = 0.01

if rank == 0
    println("\n" * "#"^70)
    println("# REGRESSION SANITY CHECK  —  Np=$(Np_reg)  Ma=$(Ma_reg)  tmax=$(tmax_reg)")
    println("#"^70)
    flush(stdout)
end

result_r1 = run_order(1, Np_reg, Ma_reg, tmax_reg; label="REGRESSION Ma=$(Ma_reg)")
result_r2 = run_order(2, Np_reg, Ma_reg, tmax_reg; label="REGRESSION Ma=$(Ma_reg)")

if rank == 0
    println("\n--- Analysis: REGRESSION check ---")
end
sharp_r1 = analyze_result(result_r1, Np_reg, 1, Ma_reg, tmax_reg; label="REGRESSION Ma=$(Ma_reg)")
sharp_r2 = analyze_result(result_r2, Np_reg, 2, Ma_reg, tmax_reg; label="REGRESSION Ma=$(Ma_reg)")

if rank == 0 && sharp_r1 !== nothing && sharp_r2 !== nothing
    ratio_r = sharp_r2 / sharp_r1
    @printf("\n  === SHARPNESS RATIO order2/order1 (Ma=%.1f, Np=%d) ===\n", Ma_reg, Np_reg)
    @printf("      max|grad(rho)| order=1  : %.6e\n", sharp_r1)
    @printf("      max|grad(rho)| order=2  : %.6e\n", sharp_r2)
    @printf("      ratio order2/order1     : %.4f\n", ratio_r)
    if ratio_r > 1.0
        @printf("  ==> high-order produces sharper gradients (ratio = %.4f) at the same\n", ratio_r)
        @printf("      final time t=%.4f — consistent with reduced numerical diffusion.\n", tmax_reg)
        println("      Regression check passes.")
    else
        println("  ==> HIGH-ORDER is NOT sharper (ratio <= 1) at Ma=$(Ma_reg).")
    end
    println()
    flush(stdout)
end

# ===========================================================================
# SUMMARY
# ===========================================================================
if rank == 0
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)

    if sharp1 !== nothing && sharp2 !== nothing
        _, t1, steps1, _ = result1
        _, t2, steps2, _ = result2
        ratio = sharp2 / sharp1
        # Both orders are compared at the SAME final time (tmax_main).  The time
        # loop clips the last dt to land exactly on tmax, so t_final is identical
        # for both runs even though step counts may differ (adaptive dt differs
        # between orders).  We assert this to catch any regression in the solver.
        @assert isapprox(t1, t2; atol=1e-12) "t_final mismatch: order1=$(t1) order2=$(t2)"
        @printf("  NOTE: both orders reach the same t_final=%.6f (matched-time comparison).\n", t1)
        @printf("        Step counts may differ due to adaptive dt (order1: %d, order2: %d).\n",
                steps1, steps2)
        @printf("  MAIN  Np=%d Ma=%.1f tmax=%.4f:\n", Np_main, Ma_main, tmax_main)
        @printf("    order=1  steps=%d  t=%.6f  sharp=%.4e\n", steps1, t1, sharp1)
        @printf("    order=2  steps=%d  t=%.6f  sharp=%.4e\n", steps2, t2, sharp2)
        @printf("    sharpness ratio order2/order1 = %.4f\n", ratio)
    end

    if sharp_r1 !== nothing && sharp_r2 !== nothing
        _, tr1, stepsr1, _ = result_r1
        _, tr2, stepsr2, _ = result_r2
        ratio_r = sharp_r2 / sharp_r1
        # Same matched-time guarantee as MAIN — assert identical t_final.
        @assert isapprox(tr1, tr2; atol=1e-12) "t_final mismatch: order1=$(tr1) order2=$(tr2)"
        @printf("  NOTE: both orders reach the same t_final=%.6f (matched-time comparison).\n", tr1)
        @printf("        Step counts may differ due to adaptive dt (order1: %d, order2: %d).\n",
                stepsr1, stepsr2)
        @printf("  REG   Np=%d Ma=%.1f tmax=%.4f:\n", Np_reg, Ma_reg, tmax_reg)
        @printf("    order=1  steps=%d  t=%.6f  sharp=%.4e\n", stepsr1, tr1, sharp_r1)
        @printf("    order=2  steps=%d  t=%.6f  sharp=%.4e\n", stepsr2, tr2, sharp_r2)
        @printf("    sharpness ratio order2/order1 = %.4f\n", ratio_r)
    end
    println("="^70)
    flush(stdout)
end

MPI.Finalize()
