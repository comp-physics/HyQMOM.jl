"""
MPI losslessness gate for 3D high-order scheme.

Run 3 steps of step_highorder_3d! on a crossing_matlab-style IC and save the
gathered global field to debug/ho3d_nr<tag>.jld2.  Compare the 1-rank and
4-rank outputs to verify bit-identical results (MPI losslessness).

Usage:
    # precompile once first:
    HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. -e 'using HyQMOM'

    # 1-rank run:
    REPRO_NP=24 REPRO_NR=1 UCX_TLS=sm,self HYQMOM_SKIP_PLOTTING=true CI=true \\
        mpiexec -n 1 julia --project=. test/repro/run_ho_3d.jl

    # 4-rank run:
    REPRO_NP=24 REPRO_NR=4 UCX_TLS=sm,self HYQMOM_SKIP_PLOTTING=true CI=true \\
        mpiexec -n 4 julia --project=. test/repro/run_ho_3d.jl

    # Compare:
    julia --project=. -e '
        using JLD2
        M1 = load("debug/ho3d_nr1.jld2","M")
        M4 = load("debug/ho3d_nr4.jld2","M")
        println("max|Δ| = ", maximum(abs.(M1 .- M4)))
    '
"""

ENV["HYQMOM_SKIP_PLOTTING"] = "true"
ENV["CI"] = "true"

using HyQMOM, MPI, JLD2, Printf

MPI.Init()

comm  = MPI.COMM_WORLD
rank  = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

Np   = parse(Int, get(ENV, "REPRO_NP", "24"))
tag  = get(ENV, "REPRO_NR", string(nranks))

halo = 2
Nmom = 35

# ------------------------------------------------------------------
# 1. Decompose
# ------------------------------------------------------------------
decomp = setup_mpi_cartesian_3d(Np, Np, Np, halo, comm)
nx, ny, nz = decomp.local_size
i0, i1 = decomp.istart_iend
j0, j1 = decomp.jstart_jend
k0, k1 = decomp.kstart_kend   # always (1, Np) for 3-D case

# ------------------------------------------------------------------
# 2. Build global IC via global indices (crossing_matlab style)
#    Background: rho=0.001, u=v=w=0, C200=C020=C002=1, off-diag=0
#    Bottom cube: rho=1, u=v=w=0  (global indices Minb:Maxb in each axis)
#    Top    cube: rho=1, u=v=w=0  (global indices Mint:Maxt in each axis)
# ------------------------------------------------------------------
T_val  = 1.0
C200   = T_val; C020 = T_val; C002 = T_val
C110   = 0.0;   C101 = 0.0;   C011 = 0.0

Mr_bg = InitializeM4_35(0.001, 0.0, 0.0, 0.0, C200, C110, C101, C020, C011, C002)
Mr_cube = InitializeM4_35(1.0,  0.0, 0.0, 0.0, C200, C110, C101, C020, C011, C002)

Csize = floor(Int, 0.1 * Np)
Minb  = div(Np, 2) - Csize;  Maxb = div(Np, 2)
Mint  = div(Np, 2) + 1;      Maxt = div(Np, 2) + 1 + Csize

M = zeros(Float64, nx + 2halo, ny + 2halo, nz, Nmom)

# Fill interior (local indices i in 1:nx, j in 1:ny, k in 1:nz)
for k in 1:nz
    gk = k0 + k - 1
    for i in 1:nx
        gi = i0 + i - 1
        for j in 1:ny
            gj = j0 + j - 1
            Mr = Mr_bg
            # bottom cube
            if Minb <= gi <= Maxb && Minb <= gj <= Maxb && Minb <= gk <= Maxb
                Mr = Mr_cube
            end
            # top cube
            if Mint <= gi <= Maxt && Mint <= gj <= Maxt && Mint <= gk <= Maxt
                Mr = Mr_cube
            end
            M[i + halo, j + halo, k, :] = Mr
        end
    end
end

# ------------------------------------------------------------------
# 3. Initial halo exchange
# ------------------------------------------------------------------
halo_exchange_3d!(M, decomp, :copy)

# ------------------------------------------------------------------
# 4. Time-step parameters (stable for Np=24, Ma=0)
# ------------------------------------------------------------------
dx = 1.0 / Np
dy = 1.0 / Np
dz = 1.0 / Np
Ma = 0.0
dt = 0.15 * dx / 4.5   # CFL-safe for low-speed IC

nsteps = 3

# ------------------------------------------------------------------
# 5. Advance 3 steps
# ------------------------------------------------------------------
for s in 1:nsteps
    step_highorder_3d!(M, dt, decomp, :copy, nx, ny, nz, halo, dx, dy, dz, Ma; order=2)
end

# ------------------------------------------------------------------
# 6. Gather interior to rank 0 and save
# ------------------------------------------------------------------
M_interior = M[halo+1:halo+nx, halo+1:halo+ny, 1:nz, :]

Mglobal = HyQMOM.gather_M(M_interior, decomp.istart_iend, decomp.jstart_jend,
                           decomp.kstart_kend, Np, Np, Np, Nmom, comm)

if rank == 0
    outdir = joinpath(@__DIR__, "..", "..", "debug")
    mkpath(outdir)
    outfile = joinpath(outdir, "ho3d_nr$(tag).jld2")
    jldsave(outfile; M=Mglobal)
    @printf("DONE nranks=%d tag=%s Np=%d dt=%.6e nsteps=%d  ->  %s\n",
            nranks, tag, Np, dt, nsteps, abspath(outfile))
end

MPI.Finalize()
