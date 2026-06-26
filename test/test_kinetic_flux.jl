using Test
using HyQMOM
using LinearAlgebra
using HyQMOM: kinetic_flux, realize_and_speed, realizable_3D_M4, _phys_flux,
              _NMOM, chyqmom_nodes_3d

# 35-moment exponent ordering (n = 1..35 <-> (i,j,k), M_n = <vx^i vy^j vz^k f>).
const KIN_TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),
 (0,2,0),(1,2,0),(2,2,0),
 (0,3,0),(1,3,0),
 (0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),
 (0,0,2),(1,0,2),(2,0,2),
 (0,0,3),(1,0,3),
 (0,0,4),
 (0,1,1),(1,1,1),(2,1,1),
 (0,2,1),(1,2,1),
 (0,3,1),
 (0,1,2),(1,1,2),
 (0,1,3),
 (0,2,2)]
const KIN_IDX = Dict(t => n for (n, t) in enumerate(KIN_TRIPLES))

# Build the realized left/right interior states exactly as face_flux_1d does.
function realized_state(M, axis, Ma)
    Mr, lmin, lmax = realize_and_speed(realizable_3D_M4(M, Ma), axis, Ma)
    return Mr, lmin, lmax
end

# Independent re-implementation of the byte-identical :hll branch (golden gate).
function manual_hll(M_L, M_R, axis, Ma)
    ML = realizable_3D_M4(M_L, Ma); MR = realizable_3D_M4(M_R, Ma)
    MLr, lminL, lmaxL = realize_and_speed(ML, axis, Ma)
    MRr, lminR, lmaxR = realize_and_speed(MR, axis, Ma)
    FL = _phys_flux(MLr, axis); FR = _phys_flux(MRr, axis)
    sL = min(lminL, lminR); sR = max(lmaxL, lmaxR)
    return sL >= 0 ? FL :
           (sR <= 0 ? FR :
            (sR .* FL .- sL .* FR .+ (sL*sR) .* (MRr .- MLr)) ./ (sR - sL))
end

@testset "kinetic (abscissa-upwind) Riemann flux :kinetic" begin

    # ------------------------------------------------------------------ (1)
    @testset "consistency direction (ML==MR low order vs :hll)" begin
        Ma = 2.0
        Mu = InitializeM4_35(1.0, 0.3, 0.1, -0.05, 1.0, 0.0, 0.0, 1.1, 0.0, 0.9)
        try
            HyQMOM.RIEMANN_SOLVER[] = :kinetic
            Fk = face_flux_1d(Mu, Mu, 1, Ma)
            HyQMOM.RIEMANN_SOLVER[] = :hll
            Fh = face_flux_1d(Mu, Mu, 1, Ma)

            @test all(isfinite, Fk)
            # On ML==MR, :hll returns the analytic flux. The kinetic flux must
            # reproduce it on the well-recovered LOW order (mass, momentum, the
            # x-marginal moments 1..5 for axis 1). High-order cross moments use a
            # DIFFERENT realizable closure and are NOT asserted to match.
            @test isapprox(Fk[1:5], Fh[1:5]; rtol=1e-6, atol=1e-6)

            # Mass flux (n=1) equals the quadrature normal momentum EXACTLY (the
            # ML==MR kinetic mass flux sums n*Ux over all nodes), and matches the
            # state's normal momentum component to the CHyQMOM recovery tolerance.
            MLr, _, _ = realized_state(Mu, 1, Ma)
            nL, UL = chyqmom_nodes_3d(MLr)
            @test isapprox(Fk[1], sum(nL .* UL[:, 1]); atol=1e-10)
            @test isapprox(Fk[1], MLr[_NMOM[1]]; atol=1e-6)
        finally
            HyQMOM.RIEMANN_SOLVER[] = :hll
        end
    end

    # ------------------------------------------------------------------ (2)
    @testset "realizable split (non-negative half-sums, complete partition)" begin
        Ma = 2.0
        ML = InitializeM4_35(1.0,  0.4, 0.2, -0.1, 1.0, 0.0, 0.0, 1.1, 0.0, 0.9)
        MR = InitializeM4_35(0.6, -0.3, -0.1, 0.05, 1.2, 0.0, 0.0, 0.9, 0.0, 1.1)
        for axis in (1, 2, 3)
            MLr, _, _ = realized_state(ML, axis, Ma)
            MRr, _, _ = realized_state(MR, axis, Ma)
            nL, UL = chyqmom_nodes_3d(MLr)
            nR, UR = chyqmom_nodes_3d(MRr)
            # half weight-sums that actually feed the kinetic flux are non-negative
            wL = sum(nL[k] for k in eachindex(nL) if UL[k, axis] > 0; init=0.0)
            wR = sum(nR[k] for k in eachindex(nR) if UR[k, axis] < 0; init=0.0)
            @test wL >= -1e-12
            @test wR >= -1e-12
            # partition is complete: kept (>0) + dropped (<=0) weights == total,
            # i.e. every node is counted exactly once.
            wLpos = sum(nL[k] for k in eachindex(nL) if UL[k, axis] > 0; init=0.0)
            wLnon = sum(nL[k] for k in eachindex(nL) if UL[k, axis] <= 0; init=0.0)
            @test isapprox(wLpos + wLnon, sum(nL); atol=1e-12 * max(1.0, sum(nL)))
        end
    end

    # ------------------------------------------------------------------ (3)
    @testset "upwind limits (all-positive normal velocity -> only L nodes)" begin
        Ma = 2.0
        # strongly positive normal mean velocity, small variance -> all nodes ux>0
        Mp = InitializeM4_35(1.0, 5.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02, 0.0, 0.02)
        MLr, _, _ = realized_state(Mp, 1, Ma)
        nL, UL = chyqmom_nodes_3d(MLr)
        @test all(UL[:, 1] .> 0)                       # premise: every node ux>0
        try
            HyQMOM.RIEMANN_SOLVER[] = :kinetic
            Fk = face_flux_1d(Mp, Mp, 1, Ma)           # ML==MR (same positive state)
            # R contributes nothing (no node has ux<0) so the flux is the pure-L
            # moment flux; mass flux == full normal momentum of the quadrature.
            @test isapprox(Fk[1], sum(nL .* UL[:, 1]); atol=1e-10)
        finally
            HyQMOM.RIEMANN_SOLVER[] = :hll
        end

        # mirror: strongly NEGATIVE normal velocity -> only R nodes contribute
        Mn = InitializeM4_35(1.0, -5.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02, 0.0, 0.02)
        MRr, _, _ = realized_state(Mn, 1, Ma)
        nR, UR = chyqmom_nodes_3d(MRr)
        @test all(UR[:, 1] .< 0)
        try
            HyQMOM.RIEMANN_SOLVER[] = :kinetic
            Fk = face_flux_1d(Mn, Mn, 1, Ma)
            @test isapprox(Fk[1], sum(nR .* UR[:, 1]); atol=1e-10)
        finally
            HyQMOM.RIEMANN_SOLVER[] = :hll
        end
    end

    # ------------------------------------------------------------------ (4)
    @testset "degenerate node inversion -> HLL fallback" begin
        Ma = 2.0
        axis = 1
        ML = InitializeM4_35(1.0,  0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        MR = InitializeM4_35(0.3, -0.4, 0.0, 0.0, 1.2, 0.0, 0.0, 1.0, 0.0, 1.0)
        MLr, lminL, lmaxL = realized_state(ML, axis, Ma)
        MRr, lminR, lmaxR = realized_state(MR, axis, Ma)
        FL = _phys_flux(MLr, axis); FR = _phys_flux(MRr, axis)
        sL = min(lminL, lminR); sR = max(lmaxL, lmaxR)
        @test sL < 0 < sR                              # genuine two-sided fan
        # Force a degenerate inversion: a non-positive density makes
        # chyqmom_nodes_3d throw, so kinetic_flux must fall back to the HLL flux
        # computed from the passed-in FL/FR/sL/sR/MLr/MRr.
        MLdeg = collect(float.(MLr)); MLdeg[1] = 0.0
        expected = (sR .* FL .- sL .* FR .+ (sL*sR) .* (MRr .- MLdeg)) ./ (sR - sL)
        F = kinetic_flux(MLdeg, MRr, FL, FR, sL, sR, axis, Ma)
        @test all(isfinite, F)
        @test isapprox(F, expected; atol=1e-12, rtol=1e-12)

        # near-vacuum Ma=100 collision: fallback keeps the flux finite (all axes).
        MLv = InitializeM4_35(1.0,   60.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 1e-4, 0.0, 1e-4)
        MRv = InitializeM4_35(1e-5, -60.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 1e-4, 0.0, 1e-4)
        try
            HyQMOM.RIEMANN_SOLVER[] = :kinetic
            for ax in 1:3
                @test all(isfinite, face_flux_1d(MLv, MRv, ax, 100.0))
            end
        finally
            HyQMOM.RIEMANN_SOLVER[] = :hll
        end
    end

    # ------------------------------------------------------------------ (5)
    @testset "axis symmetry (x vs y-permuted)" begin
        Ma = 2.0
        # x/y-asymmetric realizable state.
        M = InitializeM4_35(1.0, 0.35, -0.15, 0.05, 1.25, 0.0, 0.0, 0.85, 0.0, 0.95)
        # x<->y permutation of the 35-vector: moment (i,j,k) maps to (j,i,k).
        pidx = [KIN_IDX[(t[2], t[1], t[3])] for t in KIN_TRIPLES]
        Mp = M[pidx]
        try
            HyQMOM.RIEMANN_SOLVER[] = :kinetic
            F1 = face_flux_1d(M,  M,  1, Ma)   # axis-1 flux of x-state
            F2 = face_flux_1d(Mp, Mp, 2, Ma)   # axis-2 flux of y-permuted state
            # covariance: F2 == permutation of F1 (pidx is an involution).
            @test isapprox(F1, F2[pidx]; atol=1e-12, rtol=1e-10)
        finally
            HyQMOM.RIEMANN_SOLVER[] = :hll
        end
    end

    # ------------------------------------------------------------------ (6)
    @testset "default :hll unchanged (byte-identical gate)" begin
        @test HyQMOM.RIEMANN_SOLVER[] === :hll
        ML = InitializeM4_35(1.0,  0.3, 0.1, -0.05, 1.0, 0.0, 0.0, 1.1, 0.0, 0.9)
        MR = InitializeM4_35(0.6, -0.2, -0.1, 0.05, 1.2, 0.0, 0.0, 0.9, 0.0, 1.1)
        Mu = InitializeM4_35(1.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        for (A, B, axis, Ma) in ((ML, MR, 1, 2.0), (ML, MR, 2, 2.0),
                                 (ML, MR, 3, 2.0), (Mu, Mu, 1, 0.0))
            HyQMOM.RIEMANN_SOLVER[] = :hll
            @test face_flux_1d(A, B, axis, Ma) == manual_hll(A, B, axis, Ma)
        end
        HyQMOM.RIEMANN_SOLVER[] = :hll          # reset; don't leak global state
    end
end
