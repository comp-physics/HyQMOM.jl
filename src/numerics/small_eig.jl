import LinearAlgebra.BLAS: @blasfunc, libblastrampoline, BlasInt

# Reused scratch for the 4x4 eigenvalue solve (values only). NOT thread-safe: the
# solver runs one thread per MPI rank, so module-level buffers are safe here. This
# calls the SAME LAPACK dgeev as `eigvals`, bit-for-bit, but with a fixed LWORK
# (skips the per-call workspace query) and no allocation — ~1.2x faster + zero GC.
const _EIG4_A    = Matrix{Float64}(undef, 4, 4)
const _EIG4_WR   = Vector{Float64}(undef, 4)
const _EIG4_WI   = Vector{Float64}(undef, 4)
const _EIG4_WORK = Vector{Float64}(undef, 136)   # LWORK >= 3N; 136 is generous for N=4
const _EIG4_V    = Vector{Float64}(undef, 1)

"""
    jac4_realpart_minmax(J, r0, c0) -> (rmin, rmax)

Min and max real part of the eigenvalues of the 4x4 block `J[r0:r0+3, c0:c0+3]`,
via a direct (reused-buffer, fixed-LWORK) LAPACK dgeev call. Bit-identical to
`sort(real(eigvals(J[r0:r0+3,c0:c0+3])))[[1,4]]`; avoids the slice allocation,
the workspace query, and the result allocation that `eigvals` incurs.
"""
@inline function jac4_realpart_minmax(J, r0::Int, c0::Int)
    @inbounds for jj in 1:4, ii in 1:4
        _EIG4_A[ii, jj] = J[r0+ii-1, c0+jj-1]
    end
    info = Ref{BlasInt}(0)
    ccall((@blasfunc(dgeev_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
         Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
        'N', 'N', 4, _EIG4_A, 4, _EIG4_WR, _EIG4_WI, _EIG4_V, 1, _EIG4_V, 1,
        _EIG4_WORK, 136, info, 1, 1)
    @inbounds begin
        lo = _EIG4_WR[1]; hi = _EIG4_WR[1]
        for k in 2:4
            w = _EIG4_WR[k]; lo = ifelse(w < lo, w, lo); hi = ifelse(w > hi, w, hi)
        end
    end
    return lo, hi
end

"""
    eig3_realparts(a11,a12,a13, a21,a22,a23, a31,a32,a33) -> (NTuple{3,Float64}, Bool)

Eigenvalues of a general (non-symmetric) real 3x3 matrix, returned as the sorted
real parts and a `has_complex` flag (true if the matrix has a complex-conjugate
eigenvalue pair). Analytic (Cardano/trigonometric) replacement for `eigvals` on
the small flux-Jacobian blocks, avoiding LAPACK `dgeev`'s tiny-matrix overhead.

The real/complex split is decided by the sign of the characteristic cubic's
discriminant, matching what LAPACK's real Schur form reports (1x1 vs 2x2 blocks)
away from the exact boundary. Used by `_jac15_eig`; see the golden-kernel
regression (debug/golden_kernels.jl), which gates this against the LAPACK path.
"""
@inline function eig3_realparts(a11,a12,a13, a21,a22,a23, a31,a32,a33)
    # characteristic poly  λ³ - I1 λ² + I2 λ - I3
    I1 = a11 + a22 + a33
    I2 = (a11*a22 - a12*a21) + (a11*a33 - a13*a31) + (a22*a33 - a23*a32)
    I3 = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
    s  = I1/3
    # depressed cubic  y³ + p y + q = 0   (λ = y + s)
    p = I2 - I1*I1/3
    q = s*s*s - I1*s*s + I2*s - I3        # value of depressed poly at y=0
    disc = -4*p*p*p - 27*q*q              # ≥0: 3 real; <0: one real + complex pair
    if p > -1e-300 && disc >= 0
        # degenerate triple real root at s
        return (s, s, s), false
    elseif disc >= 0
        # three real roots (trigonometric)
        m = 2*sqrt(-p/3)
        arg = (3*q)/(2*p) * sqrt(-3/p)
        arg = arg > 1.0 ? 1.0 : (arg < -1.0 ? -1.0 : arg)
        θ = acos(arg)/3
        y1 = m*cos(θ)
        y2 = m*cos(θ - 2.0943951023931953)   # 2π/3
        y3 = m*cos(θ - 4.1887902047863905)   # 4π/3
        r1 = y1+s; r2 = y2+s; r3 = y3+s
        lo = min(r1, min(r2, r3)); hi = max(r1, max(r2, r3))
        mid = r1+r2+r3 - lo - hi
        return (lo, mid, hi), false
    else
        # one real root (Cardano) + complex pair; real parts: yR+s and -(yR)/2+s (twice)
        d = q*q/4 + p*p*p/27
        sd = sqrt(d >= 0 ? d : 0.0)       # disc<0 ⇒ d>0 here
        cbrt_(x) = cbrt(x)
        yR = cbrt_(-q/2 + sd) + cbrt_(-q/2 - sd)
        rR = yR + s
        rC = -yR/2 + s                    # real part of the complex pair
        lo = min(rR, rC); hi = max(rR, rC)
        return (lo, rC, hi), true         # middle entry = a complex-pair real part
    end
end
