"""
    schur4.jl — CPU prototype of a GPU eigensolver kernel.

Fixed-size, allocation-free, eigenvalues-only real-Schur eigensolver for a general
(non-symmetric) real 4×4 matrix. Returns the min/max of the eigenvalue REAL PARTS
plus a status flag. This is the CPU prototype of a batched CUDA kernel: it is written
in register/scalar style (a single stack-allocated `MMatrix` workspace, no heap, no
LAPACK), so the body ports verbatim to a per-thread CUDA kernel.

Algorithm: Householder reduction to upper Hessenberg + Francis implicit double-shift
QR with deflation (handles complex-conjugate eigenvalue pairs without complex
arithmetic). NOT a closed-form quartic.

This module is a pure addition under `gpu/` — it is NOT wired into production.
"""
module Schur4

using StaticArrays

export schur4_realpart_minmax

# EPS² floor: when the reflector "tail" (the part to be zeroed) is below machine
# precision relative to the head, the reflection is a no-op to working precision but
# its construction is numerically unstable (v1² underflows → β≈0, v2≈∞ → the
# I-βvvᵀ identity breaks and eigenvalues are corrupted). Skip it in that regime.
const _TAILTOL = 4.930380657631324e-32   # eps(Float64)^2  -- fp64-SPECIFIC.
# NOTE for the GPU port: this and `EPS` in schur4_realpart_minmax are fp64 constants.
# A single-precision kernel MUST use eps(Float32)/eps(Float32)^2 (parameterize on the
# input eltype). And per the validator's BigFloat check, fp32 is NOT safe for the
# ill-conditioned high-Ma companion blocks (percent-level error) -- keep this kernel
# fp64, or detect companion structure and fall back. See gpu/validate_schur4.jl.

# Householder reflector that maps (x,y,z) -> (α,0,0). Returns (v2, v3, β) with the
# implicit convention v = (1, v2, v3), so that (I - β v vᵀ)(x,y,z)ᵀ = (α,0,0)ᵀ.
@inline function _house3(x::Float64, y::Float64, z::Float64)
    σ = y*y + z*z
    if σ <= _TAILTOL * (x * x)
        return 0.0, 0.0, 0.0
    end
    μ = sqrt(x*x + σ)
    v1 = x <= 0.0 ? (x - μ) : (-σ / (x + μ))
    β = 2.0 * v1 * v1 / (σ + v1 * v1)
    inv = 1.0 / v1
    return y * inv, z * inv, β
end

# Householder reflector that maps (x,y) -> (α,0). Returns (v2, β), v = (1, v2).
@inline function _house2(x::Float64, y::Float64)
    σ = y * y
    if σ <= _TAILTOL * (x * x)
        return 0.0, 0.0
    end
    μ = sqrt(x * x + σ)
    v1 = x <= 0.0 ? (x - μ) : (-σ / (x + μ))
    β = 2.0 * v1 * v1 / (σ + v1 * v1)
    return y / v1, β
end

# One Francis implicit double-shift sweep on the unreduced Hessenberg block
# H[lo:hi, lo:hi] (window size nw = hi-lo+1 ≥ 3). Eigenvalues-only: the similarity
# updates are confined to the window (the off-block couplings do not affect the
# block's eigenvalues), so we never need to touch entries outside [lo:hi].
@inline function _francis!(H, lo::Int, hi::Int)
    @inbounds begin
        # shift from trailing 2×2 of the window
        s = H[hi-1, hi-1] + H[hi, hi]
        t = H[hi-1, hi-1] * H[hi, hi] - H[hi-1, hi] * H[hi, hi-1]
        # first column of (H - λ1 I)(H - λ2 I)
        x = H[lo, lo] * H[lo, lo] + H[lo, lo+1] * H[lo+1, lo] - s * H[lo, lo] + t
        y = H[lo+1, lo] * (H[lo, lo] + H[lo+1, lo+1] - s)
        z = H[lo+1, lo] * H[lo+2, lo+1]

        nw = hi - lo + 1
        # bulge chase
        for kk in 0:(nw - 3)
            base = lo + kk
            v2, v3, β = _house3(x, y, z)
            if β != 0.0
                # left apply P over columns lo:hi
                for j in lo:hi
                    a1 = H[base, j]; a2 = H[base+1, j]; a3 = H[base+2, j]
                    w = β * (a1 + v2 * a2 + v3 * a3)
                    H[base, j]   = a1 - w
                    H[base+1, j] = a2 - v2 * w
                    H[base+2, j] = a3 - v3 * w
                end
                # right apply P over rows lo:hi
                for i in lo:hi
                    a1 = H[i, base]; a2 = H[i, base+1]; a3 = H[i, base+2]
                    w = β * (a1 + v2 * a2 + v3 * a3)
                    H[i, base]   = a1 - w
                    H[i, base+1] = a2 - v2 * w
                    H[i, base+2] = a3 - v3 * w
                end
            end
            x = H[base+1, base]
            y = H[base+2, base]
            if kk < nw - 3
                z = H[base+3, base]
            end
        end

        # final 2×2 Householder on the last two rows of the window
        base = hi - 1
        v2, β = _house2(x, y)
        if β != 0.0
            for j in lo:hi
                a1 = H[base, j]; a2 = H[base+1, j]
                w = β * (a1 + v2 * a2)
                H[base, j]   = a1 - w
                H[base+1, j] = a2 - v2 * w
            end
            for i in lo:hi
                a1 = H[i, base]; a2 = H[i, base+1]
                w = β * (a1 + v2 * a2)
                H[i, base]   = a1 - w
                H[i, base+1] = a2 - v2 * w
            end
        end
    end
    return nothing
end

"""
    schur4_realpart_minmax(a11,a12,a13,a14, a21,..., a44) -> (rmin, rmax, status)

Min/max of the REAL PARTS of the 4 eigenvalues of the general real 4×4 matrix given
by its 16 scalar entries (row-major arg order). `status`: 0 = converged cleanly,
1 = suspicious (no deflation within the sweep cap, NaN/Inf, degenerate) → the caller
should fall back to LAPACK. Allocation-free; no LAPACK / `eigvals` inside.
"""
@inline function schur4_realpart_minmax(
        a11::Float64, a12::Float64, a13::Float64, a14::Float64,
        a21::Float64, a22::Float64, a23::Float64, a24::Float64,
        a31::Float64, a32::Float64, a33::Float64, a34::Float64,
        a41::Float64, a42::Float64, a43::Float64, a44::Float64)

    # --- 1. scale by max |a_ij| ---
    s = abs(a11)
    s = max(s, abs(a12)); s = max(s, abs(a13)); s = max(s, abs(a14))
    s = max(s, abs(a21)); s = max(s, abs(a22)); s = max(s, abs(a23)); s = max(s, abs(a24))
    s = max(s, abs(a31)); s = max(s, abs(a32)); s = max(s, abs(a33)); s = max(s, abs(a34))
    s = max(s, abs(a41)); s = max(s, abs(a42)); s = max(s, abs(a43)); s = max(s, abs(a44))
    if s == 0.0
        return 0.0, 0.0, 0
    end
    if !isfinite(s)
        return 0.0, 0.0, 1
    end
    si = 1.0 / s

    H = MMatrix{4,4,Float64}(undef)
    @inbounds begin
        H[1,1]=a11*si; H[1,2]=a12*si; H[1,3]=a13*si; H[1,4]=a14*si
        H[2,1]=a21*si; H[2,2]=a22*si; H[2,3]=a23*si; H[2,4]=a24*si
        H[3,1]=a31*si; H[3,2]=a32*si; H[3,3]=a33*si; H[3,4]=a34*si
        H[4,1]=a41*si; H[4,2]=a42*si; H[4,3]=a43*si; H[4,4]=a44*si
    end

    # --- 2. reduce to upper Hessenberg ---
    @inbounds begin
        # column 1: zero H[3,1], H[4,1] (reflector on rows 2,3,4)
        v2, v3, β = _house3(H[2,1], H[3,1], H[4,1])
        if β != 0.0
            for j in 1:4
                a2 = H[2,j]; a3 = H[3,j]; a4 = H[4,j]
                w = β * (a2 + v2 * a3 + v3 * a4)
                H[2,j] = a2 - w; H[3,j] = a3 - v2 * w; H[4,j] = a4 - v3 * w
            end
            for i in 1:4
                a2 = H[i,2]; a3 = H[i,3]; a4 = H[i,4]
                w = β * (a2 + v2 * a3 + v3 * a4)
                H[i,2] = a2 - w; H[i,3] = a3 - v2 * w; H[i,4] = a4 - v3 * w
            end
        end
        # column 2: zero H[4,2] (reflector on rows 3,4)
        v2b, βb = _house2(H[3,2], H[4,2])
        if βb != 0.0
            for j in 1:4
                a3 = H[3,j]; a4 = H[4,j]
                w = βb * (a3 + v2b * a4)
                H[3,j] = a3 - w; H[4,j] = a4 - v2b * w
            end
            for i in 1:4
                a3 = H[i,3]; a4 = H[i,4]
                w = βb * (a3 + v2b * a4)
                H[i,3] = a3 - w; H[i,4] = a4 - v2b * w
            end
        end
    end

    # --- 3. Francis double-shift QR with deflation ---
    EPS = 2.220446049250313e-16
    maxsweep = 40
    nsweep = 0
    status = 0
    rmin = Inf
    rmax = -Inf

    # global scale (matrix ∞-style norm) used as a floor for the deflation test:
    # the purely relative criterion EPS*(|h[i-1,i-1]|+|h[i,i]|) collapses to ~0 when
    # both diagonals vanish (common for these companion-like blocks), so a converged
    # subdiagonal is never accepted. Flooring by EPS*anorm gives a stable noise floor.
    anorm = 0.0
    @inbounds for j in 1:4, i in 1:4
        anorm = max(anorm, abs(H[i, j]))
    end

    hi = 4
    @inbounds while hi >= 1
        # zero negligible subdiagonals in the active region
        for i in 2:hi
            thresh = EPS * max(abs(H[i-1, i-1]) + abs(H[i, i]), anorm)
            if abs(H[i, i-1]) <= thresh
                H[i, i-1] = 0.0
            end
        end
        # bottom unreduced block [lo, hi]
        lo = hi
        while lo > 1 && H[lo, lo-1] != 0.0
            lo -= 1
        end

        if lo == hi
            # 1×1 block
            r = H[hi, hi]
            rmin = min(rmin, r); rmax = max(rmax, r)
            hi -= 1
        elseif lo == hi - 1
            # 2×2 block -> real parts
            a = H[lo, lo]; b = H[lo, hi]; c = H[hi, lo]; d = H[hi, hi]
            tr = a + d
            disc = (a - d) * (a - d) + 4.0 * b * c
            if disc >= 0.0
                rd = sqrt(disc)
                r1 = 0.5 * (tr + rd); r2 = 0.5 * (tr - rd)
                rmin = min(rmin, min(r1, r2)); rmax = max(rmax, max(r1, r2))
            else
                rp = 0.5 * tr
                rmin = min(rmin, rp); rmax = max(rmax, rp)
            end
            hi -= 2
        else
            # block size >= 3: one Francis sweep
            if nsweep >= maxsweep
                status = 1
                break
            end
            _francis!(H, lo, hi)
            nsweep += 1
        end
    end

    rmin *= s; rmax *= s
    if !(isfinite(rmin) && isfinite(rmax))
        status = 1
    end
    return rmin, rmax, status
end

end # module
