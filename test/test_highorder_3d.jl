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
