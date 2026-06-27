"""
    schur4_gpu.jl — batched CUDA port of the validated CPU prototype `gpu/schur4.jl`.

One GPU thread computes the min/max of the eigenvalue REAL PARTS of one general
(non-symmetric) real 4×4 matrix, using the SAME fp64 algorithm as
`Schur4.schur4_realpart_minmax`:

    scale by max|a|  →  Householder upper-Hessenberg  →  Francis implicit
    double-shift QR with deflation (sweep cap)  →  1×1 / 2×2 block real parts
    →  (rmin, rmax, status)

including the two robustness fixes ported verbatim:
  * eps²-tail Householder skip (`_house3_gpu` / `_house2_gpu`), and
  * norm-floored deflation test in the QR loop.

Everything is register/local-scalar resident inside the kernel: the 4×4 workspace
is a per-thread `MArray` (lowered by GPUCompiler to a thread-local `alloca`, NOT
heap), the helper reflectors are `@inline` scalar device functions, and there is no
dynamic allocation, no LAPACK, no host call from the kernel.

fp64 throughout — fp32 is catastrophic on the ill-conditioned companion blocks.

LAYOUT: the batch `A` is a `CuMatrix{Float64}` of size `(16, B)`; column `k` holds
the 16 entries of block `k` in COLUMN-MAJOR 4×4 order (a11,a21,a31,a41,a12,...,a44).
Eigenvalues are transpose-invariant, so row- vs column-major of each block is
irrelevant to correctness; this layout matches the on-disk `real_blocks.f64`
(`reshape(raw, 16, nb)`) so no host-side transpose is needed.

Pure addition under `gpu/` — NOT wired into production, CUDA NOT added to the main
HyQMOM Project.
"""
module Schur4GPU

using CUDA
using StaticArrays

export schur4_batched!, schur4_batched

const _TAILTOL = 4.930380657631324e-32   # eps(Float64)^2
const _EPS     = 2.220446049250313e-16   # eps(Float64)

# Householder reflector mapping (x,y,z) -> (α,0,0); returns (v2,v3,β), v=(1,v2,v3).
@inline function _house3_gpu(x::Float64, y::Float64, z::Float64)
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

# Householder reflector mapping (x,y) -> (α,0); returns (v2,β), v=(1,v2).
@inline function _house2_gpu(x::Float64, y::Float64)
    σ = y * y
    if σ <= _TAILTOL * (x * x)
        return 0.0, 0.0
    end
    μ = sqrt(x * x + σ)
    v1 = x <= 0.0 ? (x - μ) : (-σ / (x + μ))
    β = 2.0 * v1 * v1 / (σ + v1 * v1)
    return y / v1, β
end

# One Francis implicit double-shift sweep on the unreduced Hessenberg window
# H[lo:hi, lo:hi] (window size ≥ 3). Eigenvalues-only: updates confined to window.
@inline function _francis_gpu!(H, lo::Int, hi::Int)
    @inbounds begin
        s = H[hi-1, hi-1] + H[hi, hi]
        t = H[hi-1, hi-1] * H[hi, hi] - H[hi-1, hi] * H[hi, hi-1]
        x = H[lo, lo] * H[lo, lo] + H[lo, lo+1] * H[lo+1, lo] - s * H[lo, lo] + t
        y = H[lo+1, lo] * (H[lo, lo] + H[lo+1, lo+1] - s)
        z = H[lo+1, lo] * H[lo+2, lo+1]

        nw = hi - lo + 1
        for kk in 0:(nw - 3)
            base = lo + kk
            v2, v3, β = _house3_gpu(x, y, z)
            if β != 0.0
                for j in lo:hi
                    a1 = H[base, j]; a2 = H[base+1, j]; a3 = H[base+2, j]
                    w = β * (a1 + v2 * a2 + v3 * a3)
                    H[base, j]   = a1 - w
                    H[base+1, j] = a2 - v2 * w
                    H[base+2, j] = a3 - v3 * w
                end
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

        base = hi - 1
        v2, β = _house2_gpu(x, y)
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

# Per-thread solve: 16 column-major scalars -> (rmin, rmax, status). Faithful port
# of Schur4.schur4_realpart_minmax (arg order here is COLUMN-major: a_ij = a[col][row]).
@inline function _schur4_device(
        a11::Float64, a21::Float64, a31::Float64, a41::Float64,
        a12::Float64, a22::Float64, a32::Float64, a42::Float64,
        a13::Float64, a23::Float64, a33::Float64, a43::Float64,
        a14::Float64, a24::Float64, a34::Float64, a44::Float64)

    # --- 1. scale by max |a_ij| ---
    s = abs(a11)
    s = max(s, abs(a12)); s = max(s, abs(a13)); s = max(s, abs(a14))
    s = max(s, abs(a21)); s = max(s, abs(a22)); s = max(s, abs(a23)); s = max(s, abs(a24))
    s = max(s, abs(a31)); s = max(s, abs(a32)); s = max(s, abs(a33)); s = max(s, abs(a34))
    s = max(s, abs(a41)); s = max(s, abs(a42)); s = max(s, abs(a43)); s = max(s, abs(a44))
    if s == 0.0
        return 0.0, 0.0, Int32(0)
    end
    if !isfinite(s)
        return 0.0, 0.0, Int32(1)
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
        v2, v3, β = _house3_gpu(H[2,1], H[3,1], H[4,1])
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
        v2b, βb = _house2_gpu(H[3,2], H[4,2])
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
    maxsweep = 40
    nsweep = 0
    status = Int32(0)
    rmin = Inf
    rmax = -Inf

    anorm = 0.0
    @inbounds for j in 1:4, i in 1:4
        anorm = max(anorm, abs(H[i, j]))
    end

    hi = 4
    @inbounds while hi >= 1
        for i in 2:hi
            thresh = _EPS * max(abs(H[i-1, i-1]) + abs(H[i, i]), anorm)
            if abs(H[i, i-1]) <= thresh
                H[i, i-1] = 0.0
            end
        end
        lo = hi
        while lo > 1 && H[lo, lo-1] != 0.0
            lo -= 1
        end

        if lo == hi
            r = H[hi, hi]
            rmin = min(rmin, r); rmax = max(rmax, r)
            hi -= 1
        elseif lo == hi - 1
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
            if nsweep >= maxsweep
                status = Int32(1)
                break
            end
            _francis_gpu!(H, lo, hi)
            nsweep += 1
        end
    end

    rmin *= s; rmax *= s
    if !(isfinite(rmin) && isfinite(rmax))
        status = Int32(1)
    end
    return rmin, rmax, status
end

# ---------------------------------------------------------------------------
# CUDA kernel: one thread per matrix. A is (16, B) column-major-per-block.
function _schur4_kernel!(rmin, rmax, status, A, B::Int)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= B
        @inbounds begin
            a11 = A[1,k];  a21 = A[2,k];  a31 = A[3,k];  a41 = A[4,k]
            a12 = A[5,k];  a22 = A[6,k];  a32 = A[7,k];  a42 = A[8,k]
            a13 = A[9,k];  a23 = A[10,k]; a33 = A[11,k]; a43 = A[12,k]
            a14 = A[13,k]; a24 = A[14,k]; a34 = A[15,k]; a44 = A[16,k]
            lo, hi, st = _schur4_device(
                a11,a21,a31,a41, a12,a22,a32,a42,
                a13,a23,a33,a43, a14,a24,a34,a44)
            rmin[k] = lo
            rmax[k] = hi
            status[k] = st
        end
    end
    return nothing
end

"""
    schur4_batched!(rmin, rmax, status, A; threads=256)

In-place batched solve. `A::CuMatrix{Float64}` is `(16, B)` (column k = 16
column-major entries of block k). `rmin`, `rmax` are `CuVector{Float64}` length B,
`status` is `CuVector{Int32}` length B. Launches one thread per block.
"""
function schur4_batched!(rmin::CuVector{Float64}, rmax::CuVector{Float64},
                         status::CuVector{Int32}, A::CuMatrix{Float64};
                         threads::Int=256)
    B = size(A, 2)
    @assert size(A, 1) == 16
    @assert length(rmin) == B && length(rmax) == B && length(status) == B
    nblocks = cld(B, threads)
    @cuda threads=threads blocks=nblocks _schur4_kernel!(rmin, rmax, status, A, B)
    return nothing
end

"""
    schur4_batched(A_host::AbstractMatrix{Float64}; threads=256) -> (rmin, rmax, status)

End-to-end host convenience: upload `(16, B)` host matrix, solve, return host Arrays.
"""
function schur4_batched(A_host::AbstractMatrix{Float64}; threads::Int=256)
    @assert size(A_host, 1) == 16
    B = size(A_host, 2)
    Ad = CuArray(A_host)
    rmin = CUDA.zeros(Float64, B)
    rmax = CUDA.zeros(Float64, B)
    status = CUDA.zeros(Int32, B)
    schur4_batched!(rmin, rmax, status, Ad; threads=threads)
    CUDA.synchronize()
    return Array(rmin), Array(rmax), Array(status)
end

end # module
