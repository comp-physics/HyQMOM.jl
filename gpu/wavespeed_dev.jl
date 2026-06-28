"""
    wavespeed_dev.jl — alloc-free, device-compatible per-cell wave-speed path.

Faithful scalar/register port of the CPU chain `realize_and_speed(M, axis, Ma)`
(`src/numerics/highorder_flux.jl`) and everything it calls:

    realize_and_speed
      ├─ eigenvalues6{,z}_hyperbolic_3D  (2 transverse planes -> jacobian15
      │     3x3 block [13:15] via eig3_realparts, 4x4 block [6:9] via Schur4)
      │     + correct_moments_hyperbolic_3D on the complex-eigenvalue branch
      └─ closure_and_eigenvalues(Mr[marginal])  (Chebyshev recurrence -> 3x3 Jacobi eig)

Everything here is plain fp64 scalar arithmetic, no allocation, no LAPACK, no
dynamic dispatch -> it GPU-compiles directly (one cell per thread).

IMPORTANT — NO `@fastmath`: the production `jacobian15`/`M4toC4`/`C4toM4` use
`@fastmath`, but it is deliberately NOT used here. The `has_complex` discriminant in
`eig3_realparts_dev` (and thus whether the `correct_moments` branch fires) is sign-
sensitive at the hyperbolicity boundary; GPU `@fastmath` (reciprocal/rsqrt
approximations) shifts the jacobian entries enough to FLIP that sign on ~0.5% of real
states, swinging the wave speed by ~1e-4. With correctly-rounded div/sqrt (no
`@fastmath`) the GPU matches the CPU reference to ~1e-13 with zero boundary flips.

REGISTER-PRESSURE NOTE: `jac15_blocks_dev` reproduces the full autogen `jacobian15`
intermediate (`t`) block verbatim but materializes ONLY the 10 nontrivial entries of
the [13:15,13:15] (3x3) and [6:9,6:9] (4x4) blocks; LLVM dead-code-elimination prunes
the `t`-chains that feed none of them, so the full 15x15 is never held. The 3x3 block
is [[0,1,0],[e194,e209,e224],[e195,e210,e225]]; the 4x4 block is the companion matrix
[[0,1,0,0],[0,0,1,0],[0,0,0,1],[e84,e99,e114,e129]].

`Ma` is carried through for caller-signature parity but is UNUSED in the wave-speed
path (it is unused in the CPU `realize_and_speed` chain too).

Pure addition under `gpu/`; not wired into production; fp64.
"""
module WavespeedDev

include(joinpath(@__DIR__, "schur4.jl"))
using .Schur4: schur4_realpart_minmax

export realize_and_speed_dev, realize_and_speed_Mr_dev, jac15_eig_dev, closure5_dev,
       correct_moments_dev, eig3_realparts_dev

# ---------------------------------------------------------------------------
# eig3_realparts_dev: analytic eigenvalues (real parts) of a general real 3x3,
# returning (lo, mid, hi, has_complex). Verbatim port of `eig3_realparts`
# (src/numerics/small_eig.jl), which the golden kernel gates against LAPACK.
# ---------------------------------------------------------------------------
@inline function eig3_realparts_dev(a11, a12, a13, a21, a22, a23, a31, a32, a33)
    begin
        I1 = a11 + a22 + a33
        I2 = (a11*a22 - a12*a21) + (a11*a33 - a13*a31) + (a22*a33 - a23*a32)
        I3 = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
        s  = I1/3
        p  = I2 - I1*I1/3
        q  = s*s*s - I1*s*s + I2*s - I3
        disc = -4*p*p*p - 27*q*q
        if p > -1e-300 && disc >= 0
            return s, s, s, false
        elseif disc >= 0
            m = 2*sqrt(-p/3)
            arg = (3*q)/(2*p) * sqrt(-3/p)
            arg = arg > 1.0 ? 1.0 : (arg < -1.0 ? -1.0 : arg)
            θ = acos(arg)/3
            y1 = m*cos(θ)
            y2 = m*cos(θ - 2.0943951023931953)
            y3 = m*cos(θ - 4.1887902047863905)
            r1 = y1+s; r2 = y2+s; r3 = y3+s
            lo = min(r1, min(r2, r3)); hi = max(r1, max(r2, r3))
            mid = r1+r2+r3 - lo - hi
            return lo, mid, hi, false
        else
            d = q*q/4 + p*p*p/27
            sd = sqrt(d >= 0 ? d : 0.0)
            yR = cbrt(-q/2 + sd) + cbrt(-q/2 - sd)
            rR = yR + s
            rC = -yR/2 + s
            lo = min(rR, rC); hi = max(rR, rC)
            return lo, rC, hi, true
        end
    end
end

# ---------------------------------------------------------------------------
# closure5_dev: closure_and_eigenvalues for the marginal length-5 moment vector.
# N=2 Chebyshev recurrence (hand-specialized from src/numerics/closure_and_eigenvalues.jl)
# -> 3x3 symmetric-tridiagonal Jacobi matrix. Its eigenvalues depend only on the
# diagonal (a1,a2,a3) and the off-diagonal PRODUCTS (b2,b3) (z[i,i+1]*z[i+1,i] = b),
# so feeding the non-symmetric companion [[a1,b2,0],[1,a2,b3],[0,1,a3]] to eig3 gives
# the identical characteristic polynomial -> identical eigenvalue real parts as the
# CPU `eigvals(z)` (handles b<0 / sqrt(Complex) without complex arithmetic).
# Returns (v5min, v5max).
# ---------------------------------------------------------------------------
@inline function closure5_dev(w1, w2, w3, w4, w5)
    begin
        a1  = w2 / w1
        s33 = w3 - a1*w2
        s34 = w4 - a1*w3
        s35 = w5 - a1*w4
        a2  = s34/s33 - w2/w1
        b2  = s33 / w1
        s44 = s35 - a2*s34 - b2*w3
        b3  = s44 / s33
        a3  = (a1 + a2) / 2
        b3  = b3 * 5.0 / 2.0          # b[N+1] *= (2N+1)/N = 5/2  (N=2)
        lo, _, hi, _ = eig3_realparts_dev(a1, b2, 0.0,  1.0, a2, b3,  0.0, 1.0, a3)
        return lo, hi
    end
end

# ---------------------------------------------------------------------------
# correct_moments_dev: hyperbolicity correction (port of
# correct_moments_hyperbolic_3D, src/numerics/eigenvalues6_hyperbolic_3D.jl).
# Computes the needed central moments (M4toC4_3D), standardizes (M2CS4_35 rules,
# eps-floored variances), zeros the cross 3rd-order standardized moments, floors
# S220/S202/S022 at 1/3, rebuilds central moments (0-floored variances), and
# converts back to raw moments (C4toM4_3D). Returns the 35 corrected raw moments
# in canonical order. @noinline to cap kernel register pressure.
# ---------------------------------------------------------------------------
@noinline function correct_moments_dev(
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
        m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
    begin
        M000 = m1
        u = m2/m1; v = m6/m1; w = m16/m1
        t8 = 1.0/m1; t9 = t8*t8; t10 = t9*t8; t11 = t9*t9
        # central moments (M4toC4_3D expressions, only those used downstream)
        C200 = m3*t8 - m2^2*t9
        C300 = m4*t8 + m2^3*t10*2.0 - m2*m3*t9*3.0
        C400 = m5*t8 - m2^4*t11*3.0 - m2*m4*t9*4.0 + m3*m2^2*t10*6.0
        C110 = m7*t8 - m6*m2*t9
        C310 = m9*t8 - m6*m4*t9 - m2*m8*t9*3.0 - m6*m2^3*t11*3.0 + m7*m2^2*t10*3.0 + m6*m2*m3*t10*3.0
        C020 = m10*t8 - m6^2*t9
        C220 = m12*t8 - m6*m8*t9*2.0 - m2*m11*t9*2.0 + m10*m2^2*t10 + m3*m6^2*t10 - m6^2*m2^2*t11*3.0 + m6*m2*m7*t10*4.0
        C030 = m13*t8 + m6^3*t10*2.0 - m6*m10*t9*3.0
        C130 = m14*t8 - m6*m11*t9*3.0 - m13*m2*t9 - m2*m6^3*t11*3.0 + m7*m6^2*t10*3.0 + m6*m10*m2*t10*3.0
        C040 = m15*t8 - m6^4*t11*3.0 - m6*m13*t9*4.0 + m10*m6^2*t10*6.0
        C101 = m17*t8 - m16*m2*t9
        C301 = m19*t8 - m16*m4*t9 - m2*m18*t9*3.0 - m16*m2^3*t11*3.0 + m17*m2^2*t10*3.0 + m16*m2*m3*t10*3.0
        C002 = m20*t8 - m16^2*t9
        C202 = m22*t8 - m16*m18*t9*2.0 - m2*m21*t9*2.0 + m20*m2^2*t10 + m3*m16^2*t10 - m16^2*m2^2*t11*3.0 + m16*m2*m17*t10*4.0
        C003 = m23*t8 + m16^3*t10*2.0 - m16*m20*t9*3.0
        C103 = m24*t8 - m16*m21*t9*3.0 - m23*m2*t9 + m17*m16^2*t10*3.0 - m2*m16^3*t11*3.0 + m16*m20*m2*t10*3.0
        C004 = m25*t8 - m16^4*t11*3.0 - m16*m23*t9*4.0 + m20*m16^2*t10*6.0
        C011 = m26*t8 - m16*m6*t9
        C111 = m27*t8 - m16*m7*t9 - m6*m17*t9 - m26*m2*t9 + m16*m6*m2*t10*2.0
        C031 = m31*t8 - m16*m13*t9 - m6*m29*t9*3.0 - m16*m6^3*t11*3.0 + m26*m6^2*t10*3.0 + m16*m6*m10*t10*3.0
        C013 = m34*t8 - m16*m32*t9*3.0 - m23*m6*t9 + m26*m16^2*t10*3.0 - m6*m16^3*t11*3.0 + m16*m20*m6*t10*3.0
        C022 = m35*t8 - m16*m29*t9*2.0 - m6*m32*t9*2.0 + m20*m6^2*t10 + m10*m16^2*t10 - m16^2*m6^2*t11*3.0 + m16*m6*m26*t10*4.0

        # standardized moments (M2CS4_35: variances eps-floored)
        EPS = 2.220446049250313e-16
        sC200 = sqrt(max(C200, EPS)); sC020 = sqrt(max(C020, EPS)); sC002 = sqrt(max(C002, EPS))
        S110 = C110/(sC200*sC020); S101 = C101/(sC200*sC002); S011 = C011/(sC020*sC002)
        S220 = C220/(sC200^2*sC020^2); S202 = C202/(sC200^2*sC002^2); S022 = C022/(sC020^2*sC002^2)

        # force hyperbolic set
        s22min = 1.0/3.0
        S112 = S110; S121 = S101; S211 = S011
        S220 = max(S220, s22min); S202 = max(S202, s22min); S022 = max(S022, s22min)

        # rebuild central moments (variances 0-floored, cross 3rd-order zeroed)
        sC200c = sqrt(max(0.0, C200)); sC020c = sqrt(max(0.0, C020)); sC002c = sqrt(max(0.0, C002))
        C112 = S112*sC200c*sC020c*C002
        C121 = S121*sC200c*C020*sC002c
        C211 = S211*C200*sC020c*sC002c
        C220 = S220*C200*C020
        C202 = S202*C200*C002
        C022 = S022*C020*C002

        # raw moments (C4toM4_3D; C210=C120=C201=C102=C021=C012=0)
        tu2 = u*u; tu3 = tu2*u; tv2 = v*v; tv3 = tv2*v; tw2 = w*w; tw3 = tw2*w
        o1  = M000
        o2  = M000*u
        o3  = M000*tu2 + C200*M000
        o4  = M000*tu3 + C300*M000 + C200*M000*u*3.0
        o5  = M000*tu2^2 + C400*M000 + C200*M000*tu2*6.0 + C300*M000*u*4.0
        o6  = M000*v
        o7  = C110*M000 + M000*u*v
        o8  = C110*M000*u*2.0 + C200*M000*v + M000*tu2*v
        o9  = C310*M000 + C110*M000*tu2*3.0 + C300*M000*v + M000*tu3*v + C200*M000*u*v*3.0
        o10 = M000*tv2 + C020*M000
        o11 = C020*M000*u + C110*M000*v*2.0 + M000*tv2*u
        o12 = C220*M000 + C020*M000*tu2 + C200*M000*tv2 + M000*tu2*tv2 + C110*M000*u*v*4.0
        o13 = M000*tv3 + C030*M000 + C020*M000*v*3.0
        o14 = C130*M000 + C110*M000*tv2*3.0 + C030*M000*u + M000*tv3*u + C020*M000*u*v*3.0
        o15 = M000*tv2^2 + C040*M000 + C020*M000*tv2*6.0 + C030*M000*v*4.0
        o16 = M000*w
        o17 = C101*M000 + M000*u*w
        o18 = C101*M000*u*2.0 + C200*M000*w + M000*tu2*w
        o19 = C301*M000 + C101*M000*tu2*3.0 + C300*M000*w + M000*tu3*w + C200*M000*u*w*3.0
        o20 = M000*tw2 + C002*M000
        o21 = C002*M000*u + C101*M000*w*2.0 + M000*tw2*u
        o22 = C202*M000 + C002*M000*tu2 + C200*M000*tw2 + M000*tu2*tw2 + C101*M000*u*w*4.0
        o23 = M000*tw3 + C003*M000 + C002*M000*w*3.0
        o24 = C103*M000 + C101*M000*tw2*3.0 + C003*M000*u + M000*tw3*u + C002*M000*u*w*3.0
        o25 = M000*tw2^2 + C004*M000 + C002*M000*tw2*6.0 + C003*M000*w*4.0
        o26 = C011*M000 + M000*v*w
        o27 = C111*M000 + C011*M000*u + C101*M000*v + C110*M000*w + M000*u*v*w
        o28 = C211*M000 + C011*M000*tu2 + C111*M000*u*2.0 + M000*tu2*v*w + C101*M000*u*v*2.0 + C110*M000*u*w*2.0 + C200*M000*v*w
        o29 = C011*M000*v*2.0 + C020*M000*w + M000*tv2*w
        o30 = C121*M000 + C101*M000*tv2 + C111*M000*v*2.0 + M000*tv2*u*w + C011*M000*u*v*2.0 + C020*M000*u*w + C110*M000*v*w*2.0
        o31 = C031*M000 + C011*M000*tv2*3.0 + C030*M000*w + M000*tv3*w + C020*M000*v*w*3.0
        o32 = C002*M000*v + C011*M000*w*2.0 + M000*tw2*v
        o33 = C112*M000 + C110*M000*tw2 + C111*M000*w*2.0 + M000*tw2*u*v + C002*M000*u*v + C011*M000*u*w*2.0 + C101*M000*v*w*2.0
        o34 = C013*M000 + C011*M000*tw2*3.0 + C003*M000*v + M000*tw3*v + C002*M000*v*w*3.0
        o35 = C022*M000 + C002*M000*tv2 + C020*M000*tw2 + M000*tv2*tw2 + C011*M000*v*w*4.0
        return (o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,o16,o17,o18,
                o19,o20,o21,o22,o23,o24,o25,o26,o27,o28,o29,o30,o31,o32,o33,o34,o35)
    end
end

# ---------------------------------------------------------------------------
# jac15_blocks_dev: the 10 nontrivial entries of the 3x3 [13:15] and 4x4 [6:9]
# blocks of jacobian15 (args in MATLAB jacobian15 order). Full autogen `t`-block
# embedded verbatim; only the 10 outputs are materialized (LLVM DCE prunes the rest).
# @noinline to cap register pressure. Returns
#   (e84, e99, e114, e129,  e194, e195, e209, e210, e224, e225).
# ---------------------------------------------------------------------------
@noinline function jac15_blocks_dev(m00,m01,m02,m03,m04,m10,m11,m12,m13,m20,m21,m22,m30,m31,m40)
    begin
    t2 = m01^2
    t3 = m01^3
    t5 = m10^2
    t6 = m10^3
    t8 = 1.0/m00
    t4 = t2^2
    t7 = t5^2
    t9 = t8^2
    t10 = t8^3
    t12 = t8^5
    t13 = m02*t8
    t14 = m03*t8
    t16 = m04*t8
    t19 = m11*t8
    t20 = m12*t8
    t22 = m13*t8
    t25 = m20*t8
    t26 = m21*t8
    t28 = m22*t8
    t31 = m30*t8
    t32 = m31*t8
    t35 = m40*t8
    t11 = t9^2
    t15 = m02*t9
    t17 = m03*t9
    t18 = m04*t9
    t21 = m11*t9
    t23 = m12*t9
    t24 = m13*t9
    t27 = m20*t9
    t29 = m21*t9
    t30 = m22*t9
    t33 = m30*t9
    t34 = m31*t9
    t36 = m40*t9
    t37 = m01*m10*t9
    t49 = t28*3.0
    t53 = m01*m02*t10*3.0
    t55 = m01*m02*t10*6.0
    t57 = m01*m10*t10*2.0
    t59 = m02*m10*t10*2.0
    t61 = m01*m10*t10*4.0
    t62 = m02*m10*t10*3.0
    t63 = m03*m10*t10*2.0
    t65 = m01*m11*t10*4.0
    t66 = m01*m11*t10*6.0
    t67 = m01*m12*t10*6.0
    t71 = m01*m20*t10*2.0
    t72 = m01*m20*t10*3.0
    t73 = m10*m11*t10*4.0
    t74 = m01*m21*t10*4.0
    t75 = m10*m12*t10*4.0
    t78 = m10*m11*t10*6.0
    t81 = m01*m30*t10*2.0
    t82 = m10*m20*t10*3.0
    t83 = m10*m20*t10*6.0
    t84 = m10*m21*t10*6.0
    t86 = t2*t9
    t87 = t5*t9
    t94 = m01*m03*t10*8.0
    t101 = m01*m02*t10*1.2e+1
    t128 = m10*m20*t10*1.2e+1
    t130 = m10*m30*t10*8.0
    t131 = m02*t5*t10
    t132 = m20*t2*t10
    t136 = t2*t10*2.0
    t137 = t3*t10*2.0
    t139 = t2*t10*6.0
    t142 = t5*t10*2.0
    t143 = t6*t10*2.0
    t145 = t5*t10*6.0
    t156 = m11*t2*t10*3.0
    t165 = m11*t5*t10*3.0
    t169 = m01*m10*m11*t10*1.2e+1
    t181 = t4*t12*1.2e+1
    t183 = t7*t12*1.2e+1
    t190 = m01*t6*t12*1.2e+1
    t191 = m10*t3*t12*1.2e+1
    t200 = t2*t5*t12*1.2e+1
    t38 = m10*t15
    t39 = m10*t17
    t40 = m01*t27
    t41 = m01*t33
    t42 = t15*3.0
    t43 = t17*4.0
    t44 = t21*2.0
    t45 = t23*2.0
    t46 = t23*3.0
    t47 = t27*3.0
    t48 = t29*2.0
    t50 = t29*3.0
    t51 = t33*4.0
    t58 = t37*4.0
    t64 = t37*6.0
    t76 = m01*t29*6.0
    t77 = m10*t23*6.0
    t88 = m01*t15*-3.0
    t89 = -t53
    t90 = m01*t17*-4.0
    t91 = -t55
    t92 = -t37
    t95 = m01*t21*-2.0
    t97 = -t57
    t99 = -t59
    t100 = m01*t23*-3.0
    t102 = -t61
    t103 = -t62
    t104 = -t63
    t106 = -t65
    t107 = -t66
    t108 = -t67
    t110 = m10*t21*-2.0
    t111 = m01*t29*-2.0
    t112 = m10*t23*-2.0
    t113 = -t71
    t114 = -t72
    t115 = -t73
    t116 = -t74
    t117 = -t75
    t120 = -t78
    t122 = m10*t27*-3.0
    t123 = m10*t29*-3.0
    t124 = -t81
    t125 = -t82
    t126 = -t83
    t127 = -t84
    t129 = m10*t33*-4.0
    t133 = m10*t53
    t134 = m11*t61
    t135 = m10*t72
    t138 = t3*t11*3.0
    t140 = t4*t11*3.0
    t141 = t3*t11*6.0
    t144 = t6*t11*3.0
    t146 = t7*t11*3.0
    t147 = t6*t11*6.0
    t148 = -t94
    t149 = -t101
    t150 = -t128
    t151 = -t130
    t152 = m02*t139
    t153 = m01*t142
    t154 = m10*t136
    t155 = t131*3.0
    t158 = m02*t5*t11*3.0
    t160 = m01*t145
    t161 = m10*t139
    t162 = m01*t5*t11*6.0
    t163 = m10*t2*t11*6.0
    t164 = t132*3.0
    t166 = m20*t2*t11*3.0
    t167 = m20*t145
    t168 = m01*m02*m10*t11*9.0
    t170 = m01*m10*m11*t11*1.2e+1
    t171 = m01*m10*m20*t11*9.0
    t172 = -t86
    t173 = -t136
    t174 = -t139
    t176 = -t87
    t177 = -t142
    t178 = t3*t11*1.2e+1
    t179 = -t145
    t182 = t6*t11*1.2e+1
    t184 = m01*t6*t11*-3.0
    t185 = m10*t3*t11*-3.0
    t186 = m01*t5*t11*9.0
    t187 = m10*t2*t11*9.0
    t188 = m02*t2*t11*1.8e+1
    t189 = m11*t2*t11*9.0
    t192 = m11*t5*t11*9.0
    t193 = m20*t5*t11*1.8e+1
    t194 = -t181
    t195 = -t183
    t196 = -t190
    t197 = -t191
    t198 = t2*t5*t11*3.0
    t201 = t2*t5*t11*1.8e+1
    t202 = -t200
    t93 = -t38
    t96 = -t39
    t98 = -t58
    t105 = -t64
    t109 = -t40
    t118 = -t76
    t119 = -t77
    t121 = -t41
    t175 = -t140
    t180 = -t146
    t199 = -t198
    t203 = -t201
    t204 = t19+t92
    t205 = t21+t97
    t206 = t13+t172
    t207 = t15+t173
    t208 = t25+t176
    t209 = t27+t177
    t210 = t44+t102
    t211 = t42+t174
    t212 = t47+t179
    t235 = t14+t88+t137
    t236 = t17+t89+t138
    t237 = t17+t91+t141
    t238 = t31+t122+t143
    t239 = t33+t125+t144
    t240 = t33+t126+t147
    t246 = t43+t149+t178
    t247 = t51+t150+t182
    t252 = t23+t99+t106+t163
    t254 = t29+t113+t115+t162
    t257 = t45+t99+t106+t163
    t258 = t48+t113+t115+t162
    t259 = t46+t103+t107+t187
    t260 = t50+t114+t120+t186
    t261 = t18+t148+t188+t194
    t262 = t36+t151+t193+t195
    t284 = t24+t104+t108+t168+t189+t197
    t285 = t34+t124+t127+t171+t192+t196
    t301 = t30+t116+t117+t158+t166+t170+t202
    t213 = t206^2
    t214 = t208^2
    t215 = 1.0/t206
    t219 = 1.0/t208
    t223 = sqrt(t206)
    t225 = sqrt(t208)
    t233 = m01*m10*t8*t210*6.0
    t234 = t37*t204*1.2e+1
    t242 = t87*t206*3.0
    t243 = t86*t208*3.0
    t244 = t235^2
    t245 = t238^2
    t248 = t206*t208
    t249 = t206*t209
    t250 = t207*t208
    t251 = t20+t93+t95+t154
    t253 = t26+t109+t110+t153
    t255 = t16+t90+t152+t175
    t256 = t35+t129+t167+t180
    t279 = t22+t96+t100+t133+t156+t185
    t280 = t32+t121+t123+t135+t165+t184
    t297 = t28+t111+t112+t131+t132+t134+t199
    t216 = 1.0/t213
    t217 = t215^3
    t220 = 1.0/t214
    t221 = t219^3
    t224 = t223^3
    t226 = t225^3
    t227 = 1.0/t223
    t230 = 1.0/t225
    t241 = -t233
    t263 = 1.0/sqrt(t248)
    t265 = m10*t8*t251*6.0
    t266 = m01*t8*t253*6.0
    t269 = t2*t10*t215*t219*3.0
    t270 = t5*t10*t215*t219*3.0
    t302 = t249+t250
    t305 = t215*t219*t257*3.0
    t306 = t215*t219*t258*3.0
    t307 = t215*t219*t297*3.0
    t312 = t215*t219*t301*3.0
    t218 = t216^2
    t222 = t220^2
    t228 = 1.0/t224
    t229 = t227^5
    t231 = 1.0/t226
    t232 = t230^5
    t264 = t263^3
    t267 = t217*t244*3.0
    t268 = t221*t245*3.0
    t271 = t217*t244*9.0
    t272 = t217*t244*1.5e+1
    t273 = t221*t245*9.0
    t274 = t221*t245*1.5e+1
    t275 = t8*t221*t238*6.0
    t276 = m10*t9*t221*t238*1.8e+1
    t281 = t216*t255*2.0
    t282 = t220*t256*2.0
    t283 = t220*t256*5.0
    t287 = t216*t255*8.0
    t288 = t216*t255*1.0e+1
    t291 = t220*t256*8.0
    t292 = t220*t256*1.0e+1
    t298 = t212*t221*t238*6.0
    t303 = t221*t238*t240*6.0
    t310 = t307-1.0
    t313 = t8*t215*t220*t297*3.0
    t314 = t8*t216*t219*t297*3.0
    t315 = -t312
    t316 = m01*t9*t216*t219*t297*6.0
    t317 = m10*t9*t215*t220*t297*6.0
    t326 = t217*t230*t235*t279*2.0
    t327 = t221*t227*t238*t280*2.0
    t328 = t207*t216*t219*t297*3.0
    t329 = t209*t215*t220*t297*3.0
    t277 = t8*t222*t245*9.0
    t278 = m10*t9*t222*t245*1.8e+1
    t286 = -t281
    t289 = -t282
    t290 = -t283
    t293 = -t287
    t294 = -t288
    t295 = -t291
    t296 = -t292
    t299 = -t298
    t300 = t209*t222*t245*9.0
    t304 = -t303
    t322 = -t313
    t323 = -t314
    t324 = -t316
    t325 = -t317
    t330 = (t228*t235*t310)/2.0
    t331 = (t231*t238*t310)/2.0
    t348 = t315+t328+t329
    t308 = t267+t286
    t309 = t268+t289
    t311 = t268+t290+1.0
    t318 = t271+t293+4.0
    t319 = t272+t294+6.0
    t320 = t273+t295+4.0
    t321 = t274+t296+6.0
    t332 = t269+t322
    t333 = t270+t323
    t338 = t305+t325
    t339 = t306+t324
    t334 = (t219*t227*t253*t308)/2.0
    t335 = (t215*t230*t251*t309)/2.0
    t340 = (t215*t230*t251*t319)/4.0
    t341 = (t219*t227*t253*t321)/4.0
    t344 = (t204*t228*t235*t263*t318)/4.0
    t345 = (t204*t231*t238*t263*t320)/4.0
    t336 = -t334
    t337 = -t335
    t342 = -t340
    t343 = -t341
    t346 = t330+t336
    t347 = t331+t337
    t349 = t326+t342+t344
    t350 = t327+t343+t345
        # --- 4x4 block [6:9,6:9] companion row (linear idx 84,99,114,129) ---
        e84 = t35+t129+t167-m10*t239*4.0-t7*t11*6.0+t87*t208*6.0+m10*t8*t238*4.0-t5*t8*t209*6.0-m00*t214*t223*(t221*t227*t238*t239*2.0-(t209*t219*t227*t321)/4.0-m01*t9*t221*t228*t238*t280*2.0+(m01*t9*t219*t228*t253*t321)/4.0+(m10*t9*t231*t238*t263*t320)/4.0-(m01*t9*t204*t230*t238*t264*t320)/4.0)-m01*t8*t214*t227*t350
        e99 = t6*t10*4.0+m00*t214*t223*(t145*t221*t227*t238+(m10*t9*t219*t227*t321)/2.0+(t8*t231*t238*t263*t320)/4.0)
        e114 = t87*-6.0-m00*t214*t223*((t8*t219*t227*t321)/4.0+m10*t9*t221*t227*t238*6.0)
        e129 = m10*t8*4.0+t219*t238*2.0
        # --- 3x3 block [13:15,13:15] (linear idx 194,195,209,210,224,225) ---
        e194 = t176+m00*t208*t224*((t8*t228*t310)/2.0-(t219*t227*t253*(m01*t9*t216*8.0+t8*t217*t235*6.0))/2.0)
        e195 = t98+m00*t213*t225*(t8*t217*t230*t279*2.0-(t215*t230*t251*(m01*t9*t216*4.0e+1+t8*t217*t235*3.0e+1))/4.0-m10*t9*t217*t230*t235*2.0+(t8*t204*t228*t263*t318)/4.0+(t204*t228*t235*t263*(m01*t9*t216*3.2e+1+t8*t217*t235*1.8e+1))/4.0)
        e209 = m10*t8*2.0
        e210 = m01*t8*4.0+t215*t235*2.0
        e224 = t215*t253
        e225 = m10*t8+m00*t213*t225*(t8*t217*t230*t251*(5.0/2.0)-t8*t204*t227^7*t235*t263*2.0)
        return (e84, e99, e114, e129, e194, e195, e209, e210, e224, e225)
    end
end

# ---------------------------------------------------------------------------
# jac15_eig_dev: (vmin, vmax, has_complex) for one transverse plane, matching
# `_jac15_eig` (src/numerics/eigenvalues6_hyperbolic_3D.jl). 3x3 block real parts
# via eig3 (+ complex flag); 4x4 companion block real-part min/max via Schur4.
# If any block entry is non-finite -> (NaN, NaN, false) (matches `any(!isfinite,J)`).
# ---------------------------------------------------------------------------
@inline function jac15_eig_dev(m00,m01,m02,m03,m04,m10,m11,m12,m13,m20,m21,m22,m30,m31,m40)
    e84,e99,e114,e129,e194,e195,e209,e210,e224,e225 =
        jac15_blocks_dev(m00,m01,m02,m03,m04,m10,m11,m12,m13,m20,m21,m22,m30,m31,m40)
    ok = isfinite(e84) & isfinite(e99) & isfinite(e114) & isfinite(e129) &
         isfinite(e194) & isfinite(e195) & isfinite(e209) & isfinite(e210) &
         isfinite(e224) & isfinite(e225)
    if !ok
        return NaN, NaN, false
    end
    # 3x3 block J[13:15,13:15] = [[0,1,0],[e194,e209,e224],[e195,e210,e225]]
    r3lo, _, r3hi, hc = eig3_realparts_dev(0.0, 1.0, 0.0, e194, e209, e224, e195, e210, e225)
    # 4x4 companion block J[6:9,6:9]
    e4lo, e4hi, _ = schur4_realpart_minmax(0.0, 1.0, 0.0, 0.0,
                                           0.0, 0.0, 1.0, 0.0,
                                           0.0, 0.0, 0.0, 1.0,
                                           e84, e99, e114, e129)
    vmin = min(r3lo, e4lo)
    vmax = max(r3hi, e4hi)
    return vmin, vmax, hc
end

# 15-moment transverse-plane slices (MATLAB jacobian15 arg order); see
# src/numerics/eigenvalues6_hyperbolic_3D.jl `_plane_*`.
@inline _eig6_dev(axis, M) = _eig6_dev(axis,
    M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8],M[9],M[10],M[11],M[12],M[13],M[14],M[15],M[16],M[17],
    M[18],M[19],M[20],M[21],M[22],M[23],M[24],M[25],M[26],M[27],M[28],M[29],M[30],M[31],M[32],M[33],M[34],M[35])

@inline function _eig6_dev(axis,
        M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,M16,M17,M18,M19,M20,
        M21,M22,M23,M24,M25,M26,M27,M28,M29,M30,M31,M32,M33,M34,M35)
    if axis == 1
        # plane UV then UW
        a_min,a_max,hca = jac15_eig_dev(M1,M6,M10,M13,M15, M2,M7,M11,M14, M3,M8,M12, M4,M9, M5)
        b_min,b_max,hcb = jac15_eig_dev(M1,M16,M20,M23,M25, M2,M17,M21,M24, M3,M18,M22, M4,M19, M5)
    elseif axis == 2
        # plane VU then VW
        a_min,a_max,hca = jac15_eig_dev(M1,M2,M3,M4,M5, M6,M7,M8,M9, M10,M11,M12, M13,M14, M15)
        b_min,b_max,hcb = jac15_eig_dev(M1,M16,M20,M23,M25, M6,M26,M32,M34, M10,M29,M35, M13,M31, M15)
    else
        # plane WU then WV
        a_min,a_max,hca = jac15_eig_dev(M1,M2,M3,M4,M5, M16,M17,M18,M19, M20,M21,M22, M23,M24, M25)
        b_min,b_max,hcb = jac15_eig_dev(M1,M6,M10,M13,M15, M16,M26,M29,M31, M20,M32,M35, M23,M34, M25)
    end
    return min(a_min,b_min), max(a_max,b_max), (hca | hcb)
end

"""
    realize_and_speed_dev(m1..m35, axis, Ma) -> (vmin, vmax)

Device port of `realize_and_speed`: the combined 6x6-plane + 1D-closure wave speeds
for the given axis. `Ma` is accepted but unused (as in the CPU path).
"""
@inline function realize_and_speed_dev(
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
        m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35, axis, Ma)
    v6min, v6max, hc = _eig6_dev(axis,
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
        m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
    if hc
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
        c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35 =
            correct_moments_dev(
                m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
                m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
        v6min, v6max, _ = _eig6_dev(axis,
            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
            c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35)
        if axis == 1
            v5min, v5max = closure5_dev(c1, c2, c3, c4, c5)
        elseif axis == 2
            v5min, v5max = closure5_dev(c1, c6, c10, c13, c15)
        else
            v5min, v5max = closure5_dev(c1, c16, c20, c23, c25)
        end
        return min(v5min, v6min), max(v5max, v6max)
    else
        if axis == 1
            v5min, v5max = closure5_dev(m1, m2, m3, m4, m5)
        elseif axis == 2
            v5min, v5max = closure5_dev(m1, m6, m10, m13, m15)
        else
            v5min, v5max = closure5_dev(m1, m16, m20, m23, m25)
        end
        return min(v5min, v6min), max(v5max, v6max)
    end
end

"""
    realize_and_speed_Mr_dev(m1..m35, axis, Ma) -> (Mr::NTuple{35}, vmin, vmax)

Extended device port of the FULL CPU `realize_and_speed` (`src/numerics/highorder_flux.jl`):
identical wave-speed logic to `realize_and_speed_dev`, but ALSO returns the
hyperbolicity-corrected 35-moment state `Mr` (the second output of
`eigenvalues6{,z}_hyperbolic_3D`). On the real-eigenvalue branch `Mr` is the input
moments verbatim; on the complex branch it is `correct_moments_hyperbolic_3D(M)`
(== `correct_moments_dev` here) — exactly mirroring the CPU `Mr = M` / `Mr = correct_…`
split. Needed by the HLL face flux, which fluxes (and diffuses) the CORRECTED states.
`@fastmath` deliberately OFF (see module docstring). `Ma` accepted but unused.
"""
@inline function realize_and_speed_Mr_dev(
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
        m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35, axis, Ma)
    v6min, v6max, hc = _eig6_dev(axis,
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
        m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
    if hc
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
        c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35 =
            correct_moments_dev(
                m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
                m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
        v6min, v6max, _ = _eig6_dev(axis,
            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
            c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35)
        if axis == 1
            v5min, v5max = closure5_dev(c1, c2, c3, c4, c5)
        elseif axis == 2
            v5min, v5max = closure5_dev(c1, c6, c10, c13, c15)
        else
            v5min, v5max = closure5_dev(c1, c16, c20, c23, c25)
        end
        Mr = (c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
              c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35)
        return Mr, min(v5min, v6min), max(v5max, v6max)
    else
        if axis == 1
            v5min, v5max = closure5_dev(m1, m2, m3, m4, m5)
        elseif axis == 2
            v5min, v5max = closure5_dev(m1, m6, m10, m13, m15)
        else
            v5min, v5max = closure5_dev(m1, m16, m20, m23, m25)
        end
        Mr = (m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
              m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
        return Mr, min(v5min, v6min), max(v5max, v6max)
    end
end

end # module
