using MPI
MPI.Initialized() || MPI.Init()
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

@testset "residual_ho_3d uniform -> 0" begin
    halo=2; nx=6; ny=6; nz=6
    M0 = InitializeM4_35(1.0, 0.1, -0.1, 0.05, 1.0,0.0,0.0,1.0,0.0,1.0)
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for i in 1:nx+2halo, j in 1:ny+2halo, k in 1:nz; M[i,j,k,:]=M0; end
    R = zeros(size(M))
    residual_ho_3d!(R, M, nx,ny,nz,halo, 0.1,0.1,0.1, 0.0; order=2)
    @test maximum(abs.(R[halo+1:halo+nx, halo+1:halo+ny, :, :])) < 1e-9
end

@testset "residual_ho_3d directed per-axis" begin
    halo = 2; nx = 6; ny = 6; nz = 6
    dx = 0.1; dy = 0.2; dz = 0.05
    Ma = 0.0

    # -----------------------------------------------------------------------
    # X-only gradient: density varies only along i (the x-index in extended
    # array), uniform in j and k.  The extended array has nx+2halo rows in i.
    # -----------------------------------------------------------------------
    Mx = zeros(nx+2halo, ny+2halo, nz, 35)
    for ih in 1:(nx+2halo), jh in 1:(ny+2halo), k in 1:nz
        rho = 1.0 + 0.1 * ih          # gentle ramp along i (global index)
        Mx[ih, jh, k, :] = InitializeM4_35(rho, 0.0, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    end
    Rx = zeros(size(Mx))
    residual_ho_3d!(Rx, Mx, nx, ny, nz, halo, dx, dy, dz, Ma; order=2)
    R_int_x = Rx[halo+1:halo+nx, halo+1:halo+ny, :, :]
    # interior residual must be non-zero (gradient is present)
    @test maximum(abs.(R_int_x)) > 1e-6
    # pick representative interior (jh, k) and compare against pure x-line residual
    jh_rep = halo + 3; k_rep = 3
    Mext_x = Mx[:, jh_rep, k_rep, :]                # (nx+2halo, 35)
    Rline_x = residual_line(Mext_x, dx, 1, Ma; order=2, g=halo)  # (nx, 35)
    # Since field is uniform in y/z, y- and z-sweeps contribute 0; 3D == x-line
    @test maximum(abs.(Rx[halo+1:halo+nx, jh_rep, k_rep, :] .- Rline_x)) < 1e-12

    # -----------------------------------------------------------------------
    # Y-only gradient: density varies only along j (extended j index jh).
    # -----------------------------------------------------------------------
    My = zeros(nx+2halo, ny+2halo, nz, 35)
    for ih in 1:(nx+2halo), jh in 1:(ny+2halo), k in 1:nz
        rho = 1.0 + 0.1 * jh
        My[ih, jh, k, :] = InitializeM4_35(rho, 0.0, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    end
    Ry = zeros(size(My))
    residual_ho_3d!(Ry, My, nx, ny, nz, halo, dx, dy, dz, Ma; order=2)
    R_int_y = Ry[halo+1:halo+nx, halo+1:halo+ny, :, :]
    @test maximum(abs.(R_int_y)) > 1e-6
    # pick representative interior (ih, k)
    ih_rep = halo + 3; k_rep2 = 3
    Mext_y = My[ih_rep, :, k_rep2, :]               # (ny+2halo, 35)
    Rline_y = residual_line(Mext_y, dy, 2, Ma; order=2, g=halo)  # (ny, 35)
    @test maximum(abs.(Ry[ih_rep, halo+1:halo+ny, k_rep2, :] .- Rline_y)) < 1e-12

    # -----------------------------------------------------------------------
    # Z-only gradient: density varies only along k.  No halo in z, so
    # residual_ho_3d! pads with outflow ghosts — we must replicate that here.
    # -----------------------------------------------------------------------
    Mz = zeros(nx+2halo, ny+2halo, nz, 35)
    for ih in 1:(nx+2halo), jh in 1:(ny+2halo), k in 1:nz
        rho = 1.0 + 0.1 * k
        Mz[ih, jh, k, :] = InitializeM4_35(rho, 0.0, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    end
    Rz = zeros(size(Mz))
    residual_ho_3d!(Rz, Mz, nx, ny, nz, halo, dx, dy, dz, Ma; order=2)
    R_int_z = Rz[halo+1:halo+nx, halo+1:halo+ny, :, :]
    @test maximum(abs.(R_int_z)) > 1e-6
    # representative (ih, jh); replicate outflow padding used inside residual_ho_3d!
    ih_rep2 = halo + 3; jh_rep2 = halo + 3
    col_z = Mz[ih_rep2, jh_rep2, :, :]              # (nz, 35)
    Mext_z = vcat(repeat(col_z[1:1,:], halo, 1), col_z, repeat(col_z[nz:nz,:], halo, 1))
    Rline_z = residual_line(Mext_z, dz, 3, Ma; order=2, g=halo)  # (nz, 35)
    @test maximum(abs.(Rz[ih_rep2, jh_rep2, :, :] .- Rline_z)) < 1e-12
end

@testset "step_highorder_3d serial conservation+realizability" begin
    halo=2; nx=8; ny=8; nz=8
    mid = 4.0
    decomp = setup_mpi_cartesian_3d(nx,ny,nz,halo,MPI.COMM_WORLD)  # serial (1 rank)
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    # density blob at rest (u=v=w=0): with copy BC and zero bulk velocity,
    # boundary flux is ~0, so mass must conserve to machine precision
    for k in 1:nz, j in 1:ny, i in 1:nx
        rho = 1.0 + 0.3*exp(-(((i-mid))^2+((j-mid))^2+((k-mid))^2)/8)
        M[i+halo,j+halo,k,:] = InitializeM4_35(rho, 0.0,0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    end
    mass0 = sum(M[halo+1:halo+nx, halo+1:halo+ny, :, 1])
    dt = 0.15*(1.0/nx)/4.5
    for _ in 1:5
        step_highorder_3d!(M, dt, decomp, :copy, nx,ny,nz,halo, 1.0/nx,1.0/ny,1.0/nz, 0.0; order=2)
    end
    Min = M[halo+1:halo+nx, halo+1:halo+ny, :, :]
    @test all(isfinite, Min)
    @test minimum(Min[:,:,:,1]) > 0
    rel_mass_err = abs(sum(Min[:,:,:,1]) - mass0)/mass0
    @info "mass conservation error (u=0, copy BC)" rel_mass_err
    @test rel_mass_err < 1e-11   # u=0 => zero boundary flux => machine-precision conservation
end
