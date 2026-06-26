# test/test_scaling_limiter.jl
using Test
using HyQMOM

@testset "scaling limiter" begin
    cellM(s) = InitializeM4_35(0.8+0.1s, 0.1s, 0.0, 0.0, 1.0+0.05s, 0.0,0.0, 1.0, 0.0, 1.0)
    Vm1 = to_recon_vars(cellM(0)); V0 = to_recon_vars(cellM(1)); Vp1 = to_recon_vars(cellM(2))

    # Smooth, well-resolved field -> limiter is inactive (theta == 1), faces realizable.
    Vminus, Vplus, θ = scaling_limited_faces(Vm1, V0, Vp1)
    @test θ == 1.0
    @test is_realizable(from_recon_vars(Vminus))
    @test is_realizable(from_recon_vars(Vplus))

    # Constructed kurtosis (S400) jump that pushes a face out of R: theta < 1, faces still realizable.
    # V0[9] = S400 = 3 (Gaussian). Vhi[9] = -2 (non-realizable, below oracle boundary ~1),
    # Vlo[9] = 8 (high kurtosis). Both differences are negative so minmod is -5; Vplus[9] = 0.5 < 1
    # → non-realizable at theta=1. Bisection finds theta ≈ 0.8 where Vplus[9] ≈ 1 (boundary).
    Vhi = copy(V0); Vhi[9] = -2.0   # S400 far below realizable cone on right
    Vlo = copy(V0); Vlo[9] = 8.0    # large S400 on left
    Vminus2, Vplus2, θ2 = scaling_limited_faces(Vlo, V0, Vhi)
    @test 0.0 <= θ2 < 1.0
    @test is_realizable(from_recon_vars(Vminus2))
    @test is_realizable(from_recon_vars(Vplus2))

    # theta == 0 reproduces the cell mean exactly on both faces (first-order fallback).
    Vminus0, Vplus0, _ = scaling_limited_faces(V0, V0, V0)   # zero slope
    @test from_recon_vars(Vminus0) ≈ from_recon_vars(V0) atol=1e-12
end
