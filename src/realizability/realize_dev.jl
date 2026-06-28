"""
    realize_dev.jl — device-compatible, ALLOC-FREE port of the realizability
    projection `realizable_3D_M4(M4, Ma) -> M4r` (`src/realizability/`).

Exposes (all scalar / NTuple, no heap allocation, fp64):

  * `realizable_3D_M4_dev(m1,...,m35, Ma) -> NTuple{35}` : raw 35-moment vector
    (M4 canonical order) + Mach number -> corrected raw 35-moment vector. Direct
    port of `realizable_3D_M4` (= M2CS4_35 -> univariate floors / skewness cap ->
    `realizability_S2` -> `realizability_S220` -> `projection35` -> standardized_to_M4).

  * helpers: `realizability_S2_dev`, `realizability_S220_dev`,
    `delta2star_mineig_dev` (builds the 6x6 symmetric `delta2star3D` and returns its
    smallest eigenvalue via an in-kernel cyclic Jacobi `sym6_mineig`), `projection35_dev`.

The M2CS4_35 (central + standardize) and standardized_to_M4 arithmetic is reused
verbatim from the earlier validated `src/numerics/recon_dev.jl` (`to_recon_vars_dev` /
`from_recon_vars_dev`), which are byte-for-byte with the CPU chain.

FP parity notes (so the per-cell output matches the CPU `realizable_3D_M4`):
  * `delta2star3D` is the verbatim `@fastmath` autogen build — the matrix handed to
    the eigensolver is identical to the CPU's, so the Jacobi min-eig agrees with the
    CPU's `_geigvals` to ~1e-13 and the (sign-only) projection BRANCH decisions match.
  * everything else (univariate floors, skewness cap, `realizability_S2` bisection,
    `realizability_S220`, `projection35` target formulas) is NOT `@fastmath`, matching
    the CPU sources, so the returned moments are byte-identical.

The smallest eigenvalue of the 6x6 SYMMETRIC matrix is computed by a register/local
cyclic Jacobi sweep (eigenvalues-only). A 6x6 `MMatrix` lives in local memory (the
same pattern `schur4` uses for its 4x4 workspace); no heap, no cuSOLVER — callable
per-thread inside the projection.

Single-sourced under `src/realizability/`: the CPU `realizable_3D_M4` delegates here
and the GPU kernel modules `include` this same file. No CUDA dependency here — plain
Julia + StaticArrays.
"""
module RealizeDev

using StaticArrays

# Single-source: reuse the already-loaded ReconDev (M2CS4_35 / standardized_to_M4
# arithmetic) from the PARENT module rather than re-`include`ing recon_dev.jl, which
# would define a second, stale copy of module `ReconDev`. Every context that includes
# this file (HyQMOM in src/, plus the GPU modules RealizeGPU / Residual2GPU /
# Residual3DGPU) `include`s `recon_dev.jl` as a sibling module FIRST, so `..ReconDev`
# always resolves.
using ..ReconDev: to_recon_vars_dev, from_recon_vars_dev

export realizable_3D_M4_dev, realizable_3D_M4_corr_dev, projection35_dev,
       delta2star_mineig_dev, sym6_mineig, realizability_S2_dev, realizability_S220_dev

# ---------------------------------------------------------------------------
# realizability_S2  (port of src/realizability/realizability_S2.jl, NOT @fastmath)
# Returns (S110r, S101r, S011r, S2r). The bisection (find_zero_bisection) is
# inlined; the polynomial Y(x) is written verbatim so the FP op order matches.
# ---------------------------------------------------------------------------
@inline function realizability_S2_dev(S110, S101, S011)
    S2 = 1 + 2*S110*S101*S011 - (S110^2 + S101^2 + S011^2)
    xr = 1.0
    if S2 < 0
        # find_zero_bisection(Y, 0.0, 1.0); tol=1e-12, maxiter=100
        a = 0.0; b = 1.0
        fa = 1 + 2*S110*S101*S011*a^3 - (S110^2 + S101^2 + S011^2)*a^2   # Y(0)
        xr = (a + b) / 2   # value returned if loop completes without early return
        for _ in 1:100
            c = (a + b) / 2
            fc = 1 + 2*S110*S101*S011*c^3 - (S110^2 + S101^2 + S011^2)*c^2
            if abs(fc) < 1e-12 || (b - a) / 2 < 1e-12
                xr = c
                break
            end
            if sign(fc) == sign(fa)
                a = c; fa = fc
            else
                b = c
            end
            xr = (a + b) / 2   # keep in sync with the "loop finished" return value
        end
    end

    xr = 0.9999 * xr
    S110r = xr * S110
    S101r = xr * S101
    S011r = xr * S011
    S2r = 1 + 2*S110r*S101r*S011r - (S110r^2 + S101r^2 + S011r^2)
    return S110r, S101r, S011r, S2r
end

# ---------------------------------------------------------------------------
# realizability_S220 (port of src/realizability/realizability_S220.jl, @fastmath)
# ---------------------------------------------------------------------------
@inline @fastmath function realizability_S220_dev(S110, S220, A220)
    S220r = S220
    s220min = max(S110^2, 1 - A220)
    s220max = 1 + A220
    if S220 < s220min
        S220r = s220min
    elseif S220 > s220max
        S220r = s220max
    end
    return S220r
end

# ---------------------------------------------------------------------------
# 6x6 SYMMETRIC smallest eigenvalue: in-kernel cyclic Jacobi (eigenvalues only).
# Inputs are the 21 upper-triangular entries. Register/local MMatrix workspace.
# ---------------------------------------------------------------------------
@inline function sym6_mineig(e11, e12, e13, e14, e15, e16,
                             e22, e23, e24, e25, e26,
                             e33, e34, e35, e36,
                             e44, e45, e46,
                             e55, e56,
                             e66)
    A = MMatrix{6,6,Float64}(undef)
    @inbounds begin
        A[1,1]=e11; A[1,2]=e12; A[1,3]=e13; A[1,4]=e14; A[1,5]=e15; A[1,6]=e16
        A[2,1]=e12; A[2,2]=e22; A[2,3]=e23; A[2,4]=e24; A[2,5]=e25; A[2,6]=e26
        A[3,1]=e13; A[3,2]=e23; A[3,3]=e33; A[3,4]=e34; A[3,5]=e35; A[3,6]=e36
        A[4,1]=e14; A[4,2]=e24; A[4,3]=e34; A[4,4]=e44; A[4,5]=e45; A[4,6]=e46
        A[5,1]=e15; A[5,2]=e25; A[5,3]=e35; A[5,4]=e45; A[5,5]=e55; A[5,6]=e56
        A[6,1]=e16; A[6,2]=e26; A[6,3]=e36; A[6,4]=e46; A[6,5]=e56; A[6,6]=e66

        # Frobenius² for a relative off-diagonal convergence threshold.
        fnorm = 0.0
        for i in 1:6, j in 1:6
            fnorm += A[i,j] * A[i,j]
        end
        thresh = 1.0e-30 * fnorm

        for _ in 1:100
            off = 0.0
            for p in 1:5, q in (p+1):6
                off += A[p,q] * A[p,q]
            end
            (off <= thresh) && break

            for p in 1:5
                for q in (p+1):6
                    apq = A[p,q]
                    if apq != 0.0
                        app = A[p,p]; aqq = A[q,q]
                        tau = (aqq - app) / (2.0 * apq)
                        t = (tau >= 0.0 ? 1.0 : -1.0) / (abs(tau) + sqrt(tau*tau + 1.0))
                        c = 1.0 / sqrt(t*t + 1.0)
                        s = t * c
                        A[p,p] = app - t * apq
                        A[q,q] = aqq + t * apq
                        A[p,q] = 0.0; A[q,p] = 0.0
                        for k in 1:6
                            if k != p && k != q
                                akp = A[k,p]; akq = A[k,q]
                                A[k,p] = c*akp - s*akq
                                A[p,k] = A[k,p]
                                A[k,q] = s*akp + c*akq
                                A[q,k] = A[k,q]
                            end
                        end
                    end
                end
            end
        end

        m = A[1,1]
        for i in 2:6
            ai = A[i,i]
            m = ai < m ? ai : m
        end
        return m
    end
end

# ---------------------------------------------------------------------------
# delta2star3D smallest eigenvalue: build the 6x6 symmetric matrix (verbatim
# @fastmath port of src/autogen/delta2star3D.jl, alloc-free) and return its
# smallest eigenvalue via sym6_mineig.
# Argument order matches the autogen signature.
# ---------------------------------------------------------------------------
@inline function delta2star_mineig_dev(
        s300, s400, s110, s210, s310, s120, s220, s030, s130, s040,
        s101, s201, s301, s102, s202, s003, s103, s004, s011, s111,
        s211, s021, s121, s031, s012, s112, s013, s022)
    @fastmath begin
        t41 = s003*s003; t42 = s011*s011; t43 = t42*s011
        t44 = s012*s012; t45 = s021*s021; t46 = s030*s030
        t47 = s101*s101; t48 = t47*s101; t49 = s102*s102
        t50 = s110*s110; t51 = t50*s110; t52 = s111*s111
        t53 = s120*s120; t54 = s201*s201; t55 = s210*s210
        t56 = s300*s300

        t2 = s003*s012; t3 = s003*s021; t4 = s012*s021
        t5 = s012*s030; t6 = s021*s030; t7 = s003*s102
        t8 = s011*s101; t9 = s003*s111; t10 = s012*s102
        t11 = s011*s110; t12 = s012*s111; t13 = s021*s102
        t14 = s012*s120; t15 = s021*s111; t16 = s021*s120
        t17 = s030*s111; t18 = s030*s120; t19 = s003*s201
        t20 = s101*s110; t21 = s012*s201; t22 = s102*s111
        t23 = s012*s210; t24 = s021*s201; t25 = s102*s120
        t26 = s021*s210; t27 = s111*s120; t28 = s030*s210
        t29 = s102*s201; t30 = s102*s210; t31 = s111*s201
        t32 = s111*s210; t33 = s120*s201; t34 = s120*s210
        t35 = s102*s300; t36 = s111*s300; t37 = s201*s210
        t38 = s120*s300; t39 = s201*s300; t40 = s210*s300

        t58 = s003*s011*s030; t63 = s003*s011*s120
        t68 = s011*s030*s102; t73 = s003*s011*s210
        t74 = s003*s101*s120; t85 = s011*s030*s201
        t87 = s030*s102*s110; t91 = s003*s101*s210
        t109 = s030*s110*s201; t112 = s003*s101*s300
        t114 = s012*s101*s300; t117 = s012*s110*s300
        t118 = s021*s101*s300; t121 = s021*s110*s300
        t124 = s030*s110*s300

        t132 = -s013; t133 = -s031; t134 = -s103
        t135 = -s112; t136 = -s121; t137 = -s130
        t138 = -s211; t139 = -s301; t140 = -s310

        t57 = s011*t3; t59 = s011*t4; t60 = s011*t5
        t61 = s011*t9; t62 = s011*t10; t64 = s011*t12
        t65 = s011*t13; t66 = s011*t14; t67 = s011*t15
        t69 = s011*t16; t70 = s011*t17; t71 = s101*t9
        t72 = s101*t10; t75 = s011*t21; t76 = s101*t13
        t77 = s011*t23; t78 = s011*t24; t79 = s011*t25
        t80 = s101*t14; t81 = s110*t12; t82 = s101*t15
        t83 = s110*t13; t84 = s011*t26; t86 = s110*t14
        t88 = s110*t16; t89 = s110*t17; t90 = s101*t19
        t92 = s101*t21; t93 = s101*t22; t94 = s011*t30
        t95 = s011*t31; t96 = s101*t23; t97 = s110*t21
        t98 = s101*t24; t99 = s101*t25; t100 = s110*t22
        t101 = s011*t32; t102 = s011*t33; t103 = s110*t23
        t104 = s101*t26; t105 = s110*t24; t106 = s101*t27
        t107 = s110*t25; t108 = s110*t26; t110 = s110*t27
        t111 = s110*t28; t113 = s101*t29; t115 = s101*t30
        t116 = s101*t31; t119 = s101*t33; t120 = s110*t30
        t122 = s110*t32; t123 = s110*t33; t125 = s110*t34
        t126 = s101*t35; t127 = s101*t36; t128 = s101*t37
        t129 = s110*t36; t130 = s110*t37; t131 = s110*t38

        t141 = t9*t11; t142 = t3*t20; t143 = t8*t12
        t144 = t10*t11; t145 = t8*t13; t146 = s003*s120*t11
        t147 = s003*s030*t20; t148 = t8*t14; t149 = t11*t13
        t150 = s030*s102*t8; t151 = t4*t20; t152 = t11*t14
        t153 = t8*t16; t154 = t11*t15; t155 = t8*t17
        t156 = t5*t20; t157 = t11*t19; t158 = t9*t20
        t159 = t8*t21; t160 = t8*t22; t161 = t10*t20
        t162 = s003*s210*t11; t163 = s003*s120*t20; t164 = t8*t23
        t165 = t11*t21; t166 = t8*t24; t167 = t8*t25
        t168 = t11*t22; t169 = t12*t20; t170 = t13*t20
        t171 = t11*t23; t172 = t8*t26; t173 = t11*t24
        t174 = s030*s201*t8; t175 = t8*t27; t176 = t11*t25
        t177 = t14*t20; t178 = t15*t20; t179 = s030*s102*t20
        t180 = t11*t26; t181 = t8*t28; t182 = t11*t27
        t183 = t16*t20; t184 = t17*t20; t185 = s003*s300*t11
        t186 = s003*s210*t20; t187 = s012*s300*t8; t188 = t8*t30
        t189 = t11*t29; t190 = t20*t21; t191 = s012*s300*t11
        t192 = s021*s300*t8; t193 = t8*t32; t194 = t8*t33
        t195 = t11*t30; t196 = t11*t31; t197 = t20*t23
        t198 = t20*t24; t199 = t20*t25; t200 = s021*s300*t11
        t201 = s030*s300*t8; t202 = t8*t34; t203 = t11*t33
        t204 = t20*t26; t205 = s030*s201*t20; t206 = t8*t36
        t207 = t8*t37; t208 = t11*t35; t209 = t20*t30
        t210 = t20*t31; t211 = t8*t38; t212 = t11*t36
        t213 = t11*t37; t214 = t20*t32; t215 = t20*t33

        t216=-t3; t217=-t5; t218=-t8; t219=-t10; t220=-t11
        t221=-t12; t222=-t15; t223=-t16; t224=-t19; t225=-t20
        t226=-t22; t227=-t23; t228=-t24; t229=-t25; t230=-t27
        t231=-t28; t232=-t31; t233=-t32; t234=-t35; t235=-t37
        t236=-t38

        t237 = s011*t44; t238 = s013*t42; t239 = s011*t45
        t240 = s022*t42; t241 = s031*t42; t242 = s101*t8
        t243 = s011*t8; t244 = t8*t47; t245 = t8*t42
        t246 = s013*t47; t247 = s103*t42; t248 = s110*t11
        t250 = t11*t50; t251 = s011*t52; t252 = t11*t42
        t253 = s013*t50; t254 = s022*t47; t255 = s112*t42
        t256 = s022*t50; t257 = s031*t47; t258 = s121*t42
        t259 = s031*t50; t260 = s130*t42; t261 = s101*t49
        t262 = s103*t47; t263 = s110*t20; t265 = t20*t50
        t266 = s101*t52; t267 = t20*t47; t268 = s103*t50
        t269 = s202*t42; t270 = s112*t47; t271 = s110*t52
        t272 = s211*t42; t273 = s121*t47; t274 = s112*t50
        t275 = s110*t53; t276 = s220*t42; t277 = s130*t47
        t278 = s121*t50; t279 = s130*t50; t280 = s101*t54
        t281 = s202*t47; t282 = s301*t42; t283 = s211*t47
        t284 = s202*t50; t285 = s110*t55; t286 = s310*t42
        t287 = s220*t47; t288 = s211*t50; t289 = s220*t50
        t290 = s301*t47; t291 = s310*t47; t292 = s301*t50
        t293 = s310*t50

        t294 = s110*t8*2.0
        t295 = t2*t50; t296 = t3*t50; t297 = t4*t47
        t298 = t4*t50; t299 = t5*t47; t300 = t6*t47
        t301 = t7*t50; t302 = t8*t50; t303 = t8*t20
        t304 = t8*t11; t306 = t11*t49; t307 = t20*t44
        t308 = t9*t50; t309 = t10*t50; t312 = t8*t53
        t314 = t20*t45; t315 = t14*t47; t316 = t12*t50
        t317 = t13*t50; t318 = t15*t47; t319 = t25*t42
        t320 = t16*t47; t322 = t17*t47; t324 = t18*t47
        t325 = t19*t50; t326 = t29*t42; t327 = t8*t55
        t328 = t11*t54; t330 = t23*t47; t331 = t21*t50
        t332 = t22*t50; t333 = t30*t42; t334 = t31*t42
        t335 = t26*t47; t336 = t24*t50; t337 = t32*t42
        t338 = t33*t42; t339 = t27*t47; t340 = t28*t47
        t341 = t34*t42; t342 = t29*t50; t343 = t35*t42
        t344 = t36*t42; t345 = t37*t42; t348 = t38*t42
        t349 = t34*t47; t350 = t39*t42; t351 = t40*t42

        t364 = -t43; t365 = -t44; t366 = -t45
        t367 = -t48; t368 = -t49; t369 = -t51
        t370 = -t52; t371 = -t53; t372 = -t54
        t373 = -t55; t378 = -t63; t381 = -t68
        t396 = -t91; t410 = -t109; t413 = -t114
        t415 = -t121; t423 = t8*t8; t424 = t11*t11
        t425 = t20*t20

        t429 = s013*s110*t8*-2.0
        t436 = s031*s110*t8*-2.0
        t437 = s103*s110*t8*-2.0
        t443 = s110*s112*t8*-2.0
        t449 = s110*s121*t8*-2.0
        t450 = s110*s130*t8*-2.0
        t462 = s110*s211*t8*-2.0
        t469 = s110*s301*t8*-2.0
        t470 = s110*s310*t8*-2.0

        t479 = t42*t135; t481 = t42*t136
        t486 = t47*t135; t487 = t42*t138
        t488 = t47*t136; t489 = t50*t135
        t492 = t50*t136; t495 = t47*t138
        t499 = t50*t138

        t353 = s022*t294; t356 = s112*t294; t357 = s121*t294
        t359 = s202*t294; t360 = s211*t294; t361 = s220*t294

        t374=-t57; t375=-t60; t376=-t61; t377=-t62; t379=-t64
        t380=-t67; t382=-t69; t383=-t70; t384=-t71; t385=-t72
        t386=-t294; t387=-t77; t388=-t78; t389=-t80; t390=-t81
        t391=-t82; t392=-t83; t393=-t88; t394=-t89; t395=-t90
        t397=-t93; t398=-t94; t399=-t95; t400=-t97; t401=-t98
        t402=-t99; t403=-t100; t404=-t101; t405=-t102; t406=-t103
        t407=-t104; t408=-t106; t409=-t107; t411=-t110; t412=-t111
        t414=-t116; t416=-t122; t417=-t126; t418=-t127; t419=-t128
        t420=-t129; t421=-t130; t422=-t131

        t426 = t302*2.0; t427 = t303*2.0; t428 = t304*2.0

        t430=-t146; t431=-t147; t432=-t148; t433=-t149; t434=-t150
        t435=-t151; t438=-t165; t439=-t166; t440=-t168; t441=-t169
        t442=-t170; t444=-t171; t445=-t172; t446=-t175; t447=-t177
        t448=-t178; t451=-t185; t452=-t186; t453=-t187; t454=-t188
        t455=-t189; t456=-t190; t457=-t193; t458=-t194; t459=-t195
        t460=-t196; t461=-t199; t463=-t200; t464=-t201; t465=-t202
        t466=-t203; t467=-t204; t468=-t205

        t471=-t237; t472=-t239; t473=-t240; t474=s101*t218
        t475=s011*t218; t476=s110*t220; t477=s011*t220
        t478=-t254; t480=-t256; t482=-t261; t483=s110*t225
        t484=s101*t225; t485=-t269; t490=-t275; t491=-t276
        t493=-t280; t494=-t281; t496=-t284; t497=-t285; t498=-t287
        t500=-t289; t501=-t295; t502=-t297; t503=-t298; t504=-t300
        t505=-t301; t506=t52*t218; t507=-t308; t508=t47*t221
        t509=t42*t226; t510=t52*t220; t511=-t315; t512=-t317
        t513=t50*t222; t514=-t322; t515=t42*t230; t516=-t324
        t517=-t326; t518=t52*t225

        t519 = -t331; t520 = -t333; t521 = -t335
        t522 = -t338; t523 = -t341; t524 = -t342
        t525 = -t344; t526 = t47*t233; t527 = t50*t232
        t528 = -t349; t529 = -t350; t530 = -t351
        t531 = t8*t263*2.0; t532 = t8*t248*2.0
        t536 = s110*t423*-2.0; t534 = -t531; t535 = -t532

        t537 = t42+t47+t50+t386-1.0
        t542 = s110+t9+t14+t30+t135+t162+t163+t164+t167+t168+t169+t255+t270+t274+t369+t378+t379+t396+t397+t406+t409+t426+t443+t477+t484+t507+t511+t520
        t543 = s101+t13+t17+t33+t136+t173+t174+t175+t176+t178+t179+t258+t273+t278+t367+t380+t381+t401+t402+t410+t411+t427+t449+t475+t483+t512+t514+t522
        t544 = s011+t21+t26+t36+t138+t191+t192+t193+t196+t197+t198+t272+t283+t288+t364+t387+t388+t413+t414+t415+t416+t428+t462+t474+t476+t519+t521+t525
        t545 = s011+t2+t4+t22+t132+t141+t142+t143+t144+t145+t238+t246+t253+t307+t364+t374+t384+t385+t390+t392+t428+t429+t471+t474+t476+t501+t502+t509
        t546 = s011+t4+t6+t27+t133+t152+t153+t154+t155+t156+t241+t257+t259+t314+t364+t375+t389+t391+t393+t394+t428+t436+t472+t474+t476+t503+t504+t515
        t547 = s101+t7+t12+t29+t134+t157+t158+t159+t160+t161+t247+t262+t268+t306+t367+t376+t377+t395+t400+t403+t427+t437+t475+t482+t483+t505+t508+t517
        t548 = s110+t15+t18+t34+t137+t180+t181+t182+t183+t184+t260+t277+t279+t312+t369+t382+t383+t407+t408+t412+t426+t450+t477+t484+t490+t513+t516+t523
        t549 = s101+t29+t32+t39+t139+t206+t207+t208+t209+t210+t282+t290+t292+t328+t367+t398+t399+t417+t420+t421+t427+t469+t475+t483+t493+t524+t526+t529
        t550 = s110+t31+t34+t40+t140+t211+t212+t213+t214+t215+t286+t291+t293+t327+t369+t404+t405+t418+t419+t422+t426+t470+t477+t484+t497+t527+t528+t530
        t551 = s112+t64+t65+t92+t93+t105+t218+t219+t222+t232+t244+t245+t271+t302+t309+t318+t334+t356+t438+t439+t440+t441+t442+t479+t486+t489+t506+t536

        t538 = 1.0/t537
        t539 = s022+t58+t59+t74+t76+t86+t87+t216+t217+t229+t296+t299+t319+t353+t430+t431+t432+t433+t434+t435+t473+t478+t480+t537
        t540 = s202+t73+t75+t112+t113+t117+t120+t224+t227+t234+t325+t330+t343+t359+t451+t452+t453+t454+t455+t456+t485+t494+t496+t537
        t541 = s220+t84+t85+t118+t119+t124+t125+t228+t231+t236+t336+t340+t348+t361+t463+t464+t465+t466+t467+t468+t491+t498+t500+t537
        t552 = s121+t66+t67+t96+t108+t110+t220+t221+t223+t233+t250+t252+t266+t303+t316+t320+t337+t357+t444+t445+t446+t447+t448+t481+t488+t492+t510+t535
        t553 = s211+t79+t115+t116+t122+t123+t225+t226+t230+t235+t251+t265+t267+t304+t332+t339+t345+t360+t457+t458+t459+t460+t461+t487+t495+t499+t518+t534

        t554 = t538*t539; t555 = t538*t540; t556 = t538*t541
        t560 = t538*t542; t561 = t538*t543; t562 = t538*t544
        t563 = t538*t545; t564 = t538*t546; t565 = t538*t547
        t566 = t538*t548; t567 = t538*t549; t568 = t538*t550
        t569 = t538*t551; t570 = t538*t552; t571 = t538*t553

        t557 = -t554; t558 = -t555; t559 = -t556
        t572 = -t569; t573 = -t570; t574 = -t571

        E11 = -t538*(s400-t56+t372+t373+t537+s011*t37*2.0+s101*t39*2.0+s110*t40*2.0-s400*t42-s400*t47-s400*t50+s400*t294-t8*t40*2.0-t11*t39*2.0-t20*t37*2.0+t42*t56+t47*t55+t50*t54)
        E12 = t568
        E13 = t567
        E14 = t559
        E15 = t562
        E16 = t558

        E22 = -t538*(s220-t50+t125*2.0-t202*2.0+t361+t370+t371+t373+t424+t425+t491+t498+t500+s011*t27*2.0+s101*t32*2.0-t11*t32*2.0-t20*t27*2.0-t8*t51*2.0+t42*t55+t47*t53+t50*t52+t50^2)
        E23 = t574
        E24 = t566
        E25 = t573
        E26 = t560

        E33 = -t538*(s202-t47+t113*2.0-t189*2.0+t359+t368+t370+t372+t423+t425+t485+t494+t496+s011*t22*2.0+s110*t31*2.0-t8*t31*2.0-t20*t22*2.0+t42*t54+t47*t52+t49*t50-t20*t242*2.0+t47^2)
        E34 = t561
        E35 = t572
        E36 = t565

        E44 = -t538*(s040-t46+t366+t371+t537+s011*t6*2.0-s040*t42-s040*t47-s040*t50+s101*t16*2.0+s110*t18*2.0+s040*t294-t6*t20*2.0-t8*t18*2.0-t11*t16*2.0+t46*t47+t42*t53+t45*t50)
        E45 = t564
        E46 = t557

        E55 = -t538*(s022-t42+t59*2.0-t151*2.0+t353+t365+t366+t370+t423+t424+t473+t478+t480+s101*t12*2.0+s110*t15*2.0-t8*t15*2.0-t11*t12*2.0+t45*t47+t42*t52+t44*t50-t11*t243*2.0+t42^2)
        E56 = t563

        E66 = -t538*(s004-t41+t365+t368+t537+s011*t2*2.0-s004*t42-s004*t47-s004*t50+s101*t7*2.0+s110*t10*2.0+s004*t294-t7*t11*2.0-t8*t10*2.0-t2*t20*2.0+t41*t50+t42*t49+t44*t47)

        return sym6_mineig(E11, E12, E13, E14, E15, E16,
                           E22, E23, E24, E25, E26,
                           E33, E34, E35, E36,
                           E44, E45, E46,
                           E55, E56,
                           E66)
    end
end

# ---------------------------------------------------------------------------
# projection35 (port of src/realizability/projection35.jl, NOT @fastmath).
# Returns the 28 standardized moments (same order as the args), corrected if
# necessary. At most 2 eigenvalue evaluations.
# ---------------------------------------------------------------------------
@inline function projection35_dev(
        S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
        S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
        S211, S021, S121, S031, S012, S112, S013, S022)

    lam = delta2star_mineig_dev(
        S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
        S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
        S211, S021, S121, S031, S012, S112, S013, S022)

    if lam >= 0
        return (S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                S211, S021, S121, S031, S012, S112, S013, S022)
    end

    H200 = S400 - S300^2 - 1
    H020 = S040 - S030^2 - 1
    H002 = S004 - S003^2 - 1

    S2 = 1 + 2*S110*S101*S011 - S110^2 - S101^2 - S011^2
    R101 = sign(S101)
    R110 = sign(S110)
    S3 = (S300 + R110*S030 + R101*S003) / 3
    S4 = 1 + S3^2 + S2*(H200 + H020 + H002) / 3

    # 3D target moments (S110,S101,S011 unchanged on this pass)
    nS300 = S3
    nS030 = S110*S3
    nS003 = S101*S3
    nS111 = S011*S3
    nS120 = nS300
    nS102 = nS300
    nS210 = nS030
    nS012 = nS030
    nS201 = nS003
    nS021 = nS003
    nS310 = S110*S4
    nS130 = S110*S4
    nS112 = S110*S4
    nS301 = S101*S4
    nS103 = S101*S4
    nS121 = S101*S4
    nS031 = S011*S4
    nS013 = S011*S4
    nS211 = S011*S4
    nS400 = S4
    nS040 = S4
    nS004 = S4
    nS220 = S4
    nS202 = S4
    nS022 = S4

    lam2 = delta2star_mineig_dev(
        nS300, nS400, S110, nS210, nS310, nS120, nS220, nS030, nS130, nS040,
        S101, nS201, nS301, nS102, nS202, nS003, nS103, nS004, S011, nS111,
        nS211, nS021, nS121, nS031, nS012, nS112, nS013, nS022)

    if lam2 > -1.0e-6
        return (nS300, nS400, S110, nS210, nS310, nS120, nS220, nS030, nS130, nS040,
                S101, nS201, nS301, nS102, nS202, nS003, nS103, nS004, S011, nS111,
                nS211, nS021, nS121, nS031, nS012, nS112, nS013, nS022)
    end

    # force realizability
    fS110 = S110; fS101 = S101; fS011 = S011
    if abs(S110) <= abs(S101) && abs(S110) <= abs(S011)
        fS110 = S101*S011
    elseif abs(S101) <= abs(S110) && abs(S101) <= abs(S011)
        fS101 = S110*S011
    else
        fS011 = S110*S101
    end

    # rebuild target (S3, S4, S2 retained; S110/S101/S011 -> fS*)
    nS300 = S3
    nS030 = fS110*S3
    nS003 = fS101*S3
    nS111 = fS011*S3
    nS120 = nS300
    nS102 = nS300
    nS210 = nS030
    nS012 = nS030
    nS201 = nS003
    nS021 = nS003
    nS310 = fS110*S4
    nS130 = fS110*S4
    nS112 = fS110*S4
    nS301 = fS101*S4
    nS103 = fS101*S4
    nS121 = fS101*S4
    nS031 = fS011*S4
    nS013 = fS011*S4
    nS211 = fS011*S4
    nS400 = S4
    nS040 = S4
    nS004 = S4
    nS220 = S4
    nS202 = S4
    nS022 = S4

    return (nS300, nS400, fS110, nS210, nS310, nS120, nS220, nS030, nS130, nS040,
            fS101, nS201, nS301, nS102, nS202, nS003, nS103, nS004, fS011, nS111,
            nS211, nS021, nS121, nS031, nS012, nS112, nS013, nS022)
end

# ---------------------------------------------------------------------------
# Realizability CORRECTION (alloc-free). 35 raw moments + Ma in; the corrected
# RECON-VARS out as NTuple{35}: (M000, umean, vmean, wmean, C200, C020, C002,
# <28 corrected standardized moments>) — exactly the argument layout shared by
# `from_recon_vars_dev` (GPU reconstruction) and the autogen `standardized_to_M4`
# (CPU reference reconstruction). Returning the pre-reconstruction state lets each
# caller pick its reconstruction:
#   * `realizable_3D_M4_dev` (GPU / device) reconstructs with `from_recon_vars_dev`;
#   * CPU `realizable_3D_M4` reconstructs with the autogen `standardized_to_M4`,
#     which is byte-identical to the legacy inline path (and thus the golden battery).
# All correction stages here are byte-identical to the CPU sources (verified 0.0 over
# the 1200-state battery); only the final S->C->M reconstruction has a ~1 ULP
# @fastmath reassociation difference between `_c4tom4_35` and the autogen `C4toM4_3D`,
# which this split keeps out of the CPU path.
# ---------------------------------------------------------------------------
@inline function realizable_3D_M4_corr_dev(
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,
        m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,
        m31,m32,m33,m34,m35, Ma)

    s3max = 4.0 + abs(Ma)/2.0
    h2min = 1.0e-6
    S2min = 1.0e-6

    # central + standardized (= M2CS4_35; variances floored eps(), C2** floored 1e-12)
    V = to_recon_vars_dev(
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,
        m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,
        m31,m32,m33,m34,m35)

    M000 = V[1]; umean = V[2]; vmean = V[3]; wmean = V[4]
    C200 = V[5]; C020 = V[6]; C002 = V[7]
    S300 = V[8];  S400 = V[9];  S110 = V[10]; S210 = V[11]; S310 = V[12]
    S120 = V[13]; S220 = V[14]; S030 = V[15]; S130 = V[16]; S040 = V[17]
    S101 = V[18]; S201 = V[19]; S301 = V[20]; S102 = V[21]; S202 = V[22]
    S003 = V[23]; S103 = V[24]; S004 = V[25]; S011 = V[26]; S111 = V[27]
    S211 = V[28]; S021 = V[29]; S121 = V[30]; S031 = V[31]; S012 = V[32]
    S112 = V[33]; S013 = V[34]; S022 = V[35]

    # --- univariate moments ---
    H200 = S400 - S300^2 - 1
    H020 = S040 - S030^2 - 1
    H002 = S004 - S003^2 - 1
    if H200 <= h2min; H200 = h2min; S400 = H200 + S300^2 + 1; end
    if H020 <= h2min; H020 = h2min; S040 = H020 + S030^2 + 1; end
    if H002 <= h2min; H002 = h2min; S004 = H002 + S003^2 + 1; end
    if S300 < -s3max; S300 = -s3max; S400 = H200 + S300^2 + 1
    elseif S300 > s3max; S300 = s3max; S400 = H200 + S300^2 + 1; end
    if S030 < -s3max; S030 = -s3max; S040 = H020 + S030^2 + 1
    elseif S030 > s3max; S030 = s3max; S040 = H020 + S030^2 + 1; end
    if S003 < -s3max; S003 = -s3max; S004 = H002 + S003^2 + 1
    elseif S003 > s3max; S003 = s3max; S004 = H002 + S003^2 + 1; end
    S400 = max(S400, S300^2 + 1 + h2min)
    S040 = max(S040, S030^2 + 1 + h2min)
    S004 = max(S004, S003^2 + 1 + h2min)

    # --- 2nd-order cross moments ---
    S110 = min(1.0, max(S110, -1.0))
    S101 = min(1.0, max(S101, -1.0))
    S011 = min(1.0, max(S011, -1.0))
    S110, S101, S011, S2 = realizability_S2_dev(S110, S101, S011)
    if S2 < S2min
        R = 1 - h2min
        S110 = R*S110
        S101 = R*S101
        S011 = R*S011
    end

    # --- 4th-order: max bounds on S220, S202, S022 ---
    A220 = sqrt((H200 + S300^2)*(H020 + S030^2))
    S220max = realizability_S220_dev(S110, S220, A220)
    A202 = sqrt((H200 + S300^2)*(H002 + S003^2))
    S202max = realizability_S220_dev(S101, S202, A202)
    A022 = sqrt((H020 + S030^2)*(H002 + S003^2))
    S022max = realizability_S220_dev(S011, S022, A022)
    S220 = min(S220, S220max)
    S202 = min(S202, S202max)
    S022 = min(S022, S022max)

    # --- 3rd/4th-order: projection ---
    P = projection35_dev(
        S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
        S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
        S211, S021, S121, S031, S012, S112, S013, S022)

    S300 = P[1];  S400 = P[2];  S110 = P[3];  S210 = P[4];  S310 = P[5]
    S120 = P[6];  S220 = P[7];  S030 = P[8];  S130 = P[9];  S040 = P[10]
    S101 = P[11]; S201 = P[12]; S301 = P[13]; S102 = P[14]; S202 = P[15]
    S003 = P[16]; S103 = P[17]; S004 = P[18]; S011 = P[19]; S111 = P[20]
    S211 = P[21]; S021 = P[22]; S121 = P[23]; S031 = P[24]; S012 = P[25]
    S112 = P[26]; S013 = P[27]; S022 = P[28]

    # --- corrected recon-vars (pre-reconstruction); see header note ---
    return (M000, umean, vmean, wmean, C200, C020, C002,
            S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
            S101, S201, S301, S102, S202, S003, S103, S004,
            S011, S111, S211, S021, S121, S031, S012, S112, S013, S022)
end

# ---------------------------------------------------------------------------
# Top-level device realizable_3D_M4 (alloc-free): correction + device-side
# reconstruction. 35 raw moments + Ma in, 35 corrected raw moments out
# (NTuple{35}, M4 canonical order). This is the GPU per-cell entry point; its
# output is byte-identical to the prior monolithic kernel. (Explicit indexing,
# not splat, into from_recon_vars_dev — splat lowers to _apply_iterate, which is
# unsupported on device.)
# ---------------------------------------------------------------------------
@inline function realizable_3D_M4_dev(
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,
        m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,
        m31,m32,m33,m34,m35, Ma)
    c = realizable_3D_M4_corr_dev(
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,
        m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,
        m31,m32,m33,m34,m35, Ma)
    return from_recon_vars_dev(
        c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12],
        c[13], c[14], c[15], c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23],
        c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31], c[32], c[33], c[34], c[35])
end

end # module
