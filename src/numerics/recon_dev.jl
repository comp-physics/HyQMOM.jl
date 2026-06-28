"""
    recon_dev.jl — device-compatible, ALLOC-FREE port of the high-order (order-2)
    MUSCL reconstruction helpers from `src/numerics/reconstruction.jl` +
    `src/moments/M2CS4_35.jl`.

Exposes (all scalar / NTuple, no heap allocation, fp64):

  * `to_recon_vars_dev(m1,...,m35) -> NTuple{35}` : raw 35-moment vector (M4 canonical
    order) -> bounded reconstruction-variable vector
    `V = [M000, u, v, w, C200, C020, C002, <28 standardized moments>]`.
    Reproduces `to_recon_vars(M)` = `M2CS4_35(M)` (central via `@fastmath` M4toC4_3D
    subset + plain standardization with `eps()` floor) + the `_SIDX` extraction +
    the `max(1e-12, ·)` variance floor on C200/C020/C002.

  * `from_recon_vars_dev(v1,...,v35) -> NTuple{35}` : recon-var vector -> raw 35-moment
    vector. Reproduces `from_recon_vars` = `standardized_to_M4` (plain S->C assignment
    block + `@fastmath` C4toM4_3D), extracting the 35 raw moments in canonical layout.

  * `minmod(a,b)` slope limiter.

  * NTuple wrappers `to_recon_vars_tup`, `from_recon_vars_tup`, and `recon_vars_ok_tup`
    so callers (GPU kernels) avoid tuple-splatting (`f(t...)` lowers to `_apply_iterate`,
    unsupported on device) — they index the tuple explicitly.

FP parity notes (byte-for-byte with the CPU chain):
  * the central-moment block (`_recon_centrals`) is the verbatim `@fastmath` subset of
    `M4toC4_3D` (identical to `FluxClosureDev._m4toc4_subset`);
  * the standardization in `to_recon_vars_dev` is plain (NOT `@fastmath`), matching
    `M2CS4_35`, variances floored by `eps(Float64)`;
  * the S->C block in `from_recon_vars_dev` is plain (NOT `@fastmath`), transcribed
    verbatim from `standardized_to_M4` (note `S300*sC200*C200`, NOT `S300*sC200^3`);
  * the C4toM4_3D block (`_c4tom4_35`) is `@fastmath` (matching the autogen) and the 35
    output expressions are transcribed verbatim from the `C4toM4_3D` reshape literal at
    the linear indices `_M2CS4_IDX`.

  * BOTH `@fastmath` helpers are `@noinline` (NOT `@inline`). This is load-bearing for
    byte parity: `@fastmath` lets LLVM reassociate the catastrophic-cancellation central-
    moment formulas, and the chosen reassociation depends on the *surrounding* expression
    context. Inlined into `to_recon_vars_dev`/`from_recon_vars_dev`, the centrals get
    reassociated differently than the standalone autogen `M4toC4_3D`/`C4toM4_3D`, drifting
    ~1 ULP — invisible normally but amplified to ~2e-7 by the `1/sC200^4` divisor on deep-
    vacuum states (rho~1e-5). `@noinline` pins each helper to the same standalone
    compilation the autogen uses, restoring exact agreement. Do NOT change to `@inline`.
    (GPU-safe: CUDA.jl emits these as device function calls.)

Pure addition under `gpu/`. No CUDA dependency here — plain Julia, `include`d by both a
CPU validator and the GPU kernel module.
"""
module ReconDev

export to_recon_vars_dev, from_recon_vars_dev, minmod,
       to_recon_vars_tup, from_recon_vars_tup, recon_vars_ok_tup

const _EPSF = 2.220446049250313e-16   # eps(Float64)

"minmod slope limiter (== reconstruction.jl `minmod`)."
@inline function minmod(a, b)
    (a*b <= 0.0) ? 0.0 : (abs(a) < abs(b) ? Float64(a) : Float64(b))
end

# ---------------------------------------------------------------------------
# Central moments from raw (verbatim @fastmath subset of autogen M4toC4_3D,
# identical to FluxClosureDev._m4toc4_subset). Returns the 31 central moments
# consumed by the standardization, in a flat NTuple.
# Argument order is the M4 canonical raw-moment order.
# ---------------------------------------------------------------------------
@noinline @fastmath function _recon_centrals(
        M000,M100,M200,M300,M400,M010,M110,M210,M310,M020,M120,M220,M030,M130,M040,
        M001,M101,M201,M301,M002,M102,M202,M003,M103,M004,M011,M111,M211,M021,M121,
        M031,M012,M112,M013,M022)
    t2 = M001^2; t3 = M001^3; t4 = M010^2; t5 = M010^3; t6 = M100^2; t7 = M100^3
    t8 = 1.0/M000; t9 = t8^2; t10 = t8^3; t11 = t9^2

    cC200 = M200*t8-t6*t9
    cC020 = M020*t8-t4*t9
    cC002 = M002*t8-t2*t9
    cC300 = M300*t8+t7*t10*2.0-M100*M200*t9*3.0
    cC400 = M400*t8-t6^2*t11*3.0-M100*M300*t9*4.0+M200*t6*t10*6.0
    cC110 = M110*t8-M010*M100*t9
    cC210 = M210*t8-M010*M200*t9-M100*M110*t9*2.0+M010*t6*t10*2.0
    cC310 = M310*t8-M010*M300*t9-M100*M210*t9*3.0-M010*t7*t11*3.0+M110*t6*t10*3.0+M010*M100*M200*t10*3.0
    cC120 = M120*t8-M010*M110*t9*2.0-M020*M100*t9+M100*t4*t10*2.0
    cC220 = M220*t8-M010*M210*t9*2.0-M100*M120*t9*2.0+M020*t6*t10+M200*t4*t10-t4*t6*t11*3.0+M010*M100*M110*t10*4.0
    cC030 = M030*t8+t5*t10*2.0-M010*M020*t9*3.0
    cC130 = M130*t8-M010*M120*t9*3.0-M030*M100*t9-M100*t5*t11*3.0+M110*t4*t10*3.0+M010*M020*M100*t10*3.0
    cC040 = M040*t8-t4^2*t11*3.0-M010*M030*t9*4.0+M020*t4*t10*6.0
    cC101 = M101*t8-M001*M100*t9
    cC201 = M201*t8-M001*M200*t9-M100*M101*t9*2.0+M001*t6*t10*2.0
    cC301 = M301*t8-M001*M300*t9-M100*M201*t9*3.0-M001*t7*t11*3.0+M101*t6*t10*3.0+M001*M100*M200*t10*3.0
    cC011 = M011*t8-M001*M010*t9
    cC111 = M111*t8-M001*M110*t9-M010*M101*t9-M011*M100*t9+M001*M010*M100*t10*2.0
    cC211 = M211*t8-M001*M210*t9-M010*M201*t9-M100*M111*t9*2.0+M011*t6*t10+M001*M010*M200*t10+M001*M100*M110*t10*2.0+M010*M100*M101*t10*2.0-M001*M010*t6*t11*3.0
    cC021 = M021*t8-M001*M020*t9-M010*M011*t9*2.0+M001*t4*t10*2.0
    cC121 = M121*t8-M001*M120*t9-M010*M111*t9*2.0-M021*M100*t9+M101*t4*t10+M001*M010*M110*t10*2.0+M001*M020*M100*t10+M010*M011*M100*t10*2.0-M001*M100*t4*t11*3.0
    cC031 = M031*t8-M001*M030*t9-M010*M021*t9*3.0-M001*t5*t11*3.0+M011*t4*t10*3.0+M001*M010*M020*t10*3.0
    cC102 = M102*t8-M001*M101*t9*2.0-M002*M100*t9+M100*t2*t10*2.0
    cC202 = M202*t8-M001*M201*t9*2.0-M100*M102*t9*2.0+M002*t6*t10+M200*t2*t10-t2*t6*t11*3.0+M001*M100*M101*t10*4.0
    cC012 = M012*t8-M001*M011*t9*2.0-M002*M010*t9+M010*t2*t10*2.0
    cC112 = M112*t8-M001*M111*t9*2.0-M010*M102*t9-M012*M100*t9+M110*t2*t10+M001*M010*M101*t10*2.0+M001*M011*M100*t10*2.0+M002*M010*M100*t10-M010*M100*t2*t11*3.0
    cC022 = M022*t8-M001*M021*t9*2.0-M010*M012*t9*2.0+M002*t4*t10+M020*t2*t10-t2*t4*t11*3.0+M001*M010*M011*t10*4.0
    cC003 = M003*t8+t3*t10*2.0-M001*M002*t9*3.0
    cC103 = M103*t8-M001*M102*t9*3.0-M003*M100*t9+M101*t2*t10*3.0-M100*t3*t11*3.0+M001*M002*M100*t10*3.0
    cC013 = M013*t8-M001*M012*t9*3.0-M003*M010*t9+M011*t2*t10*3.0-M010*t3*t11*3.0+M001*M002*M010*t10*3.0
    cC004 = M004*t8-t2^2*t11*3.0-M001*M003*t9*4.0+M002*t2*t10*6.0

    return (cC200,cC020,cC002,cC300,cC400,cC110,cC210,cC310,cC120,cC220,cC030,cC130,cC040,
            cC101,cC201,cC301,cC011,cC111,cC211,cC021,cC121,cC031,cC102,cC202,cC012,cC112,
            cC022,cC003,cC103,cC013,cC004)
end

# ---------------------------------------------------------------------------
# to_recon_vars: raw 35-moment vector (M4 canonical order) -> recon-var NTuple{35}.
# V = [M000, u, v, w, C200, C020, C002, S300,S400,S110,S210,S310,S120,S220,S030,
#      S130,S040,S101,S201,S301,S102,S202,S003,S103,S004,S011,S111,S211,S021,S121,
#      S031,S012,S112,S013,S022]
# ---------------------------------------------------------------------------
@inline function to_recon_vars_dev(
        M000,M100,M200,M300,M400,M010,M110,M210,M310,M020,M120,M220,M030,M130,M040,
        M001,M101,M201,M301,M002,M102,M202,M003,M103,M004,M011,M111,M211,M021,M121,
        M031,M012,M112,M013,M022)

    u = M100/M000; v = M010/M000; w = M001/M000

    (cC200,cC020,cC002,cC300,cC400,cC110,cC210,cC310,cC120,cC220,cC030,cC130,cC040,
     cC101,cC201,cC301,cC011,cC111,cC211,cC021,cC121,cC031,cC102,cC202,cC012,cC112,
     cC022,cC003,cC103,cC013,cC004) =
        _recon_centrals(
            M000,M100,M200,M300,M400,M010,M110,M210,M310,M020,M120,M220,M030,M130,M040,
            M001,M101,M201,M301,M002,M102,M202,M003,M103,M004,M011,M111,M211,M021,M121,
            M031,M012,M112,M013,M022)

    # --- standardization (M2CS4_35), variances floored by eps(); NOT @fastmath ---
    sC200 = sqrt(max(cC200, _EPSF))
    sC020 = sqrt(max(cC020, _EPSF))
    sC002 = sqrt(max(cC002, _EPSF))

    S110 = cC110/(sC200*sC020)
    S101 = cC101/(sC200*sC002)
    S011 = cC011/(sC020*sC002)
    S300 = cC300/sC200^3
    S210 = cC210/(sC200^2*sC020)
    S201 = cC201/(sC200^2*sC002)
    S120 = cC120/(sC200*sC020^2)
    S111 = cC111/(sC200*sC020*sC002)
    S102 = cC102/(sC200*sC002^2)
    S030 = cC030/sC020^3
    S021 = cC021/(sC020^2*sC002)
    S012 = cC012/(sC020*sC002^2)
    S003 = cC003/sC002^3
    S400 = cC400/sC200^4
    S310 = cC310/(sC200^3*sC020)
    S301 = cC301/(sC200^3*sC002)
    S220 = cC220/(sC200^2*sC020^2)
    S211 = cC211/(sC200^2*sC020*sC002)
    S202 = cC202/(sC200^2*sC002^2)
    S130 = cC130/(sC200*sC020^3)
    S121 = cC121/(sC200*sC020^2*sC002)
    S112 = cC112/(sC200*sC020*sC002^2)
    S103 = cC103/(sC200*sC002^3)
    S040 = cC040/sC020^4
    S031 = cC031/(sC020^3*sC002)
    S022 = cC022/(sC020^2*sC002^2)
    S013 = cC013/(sC020*sC002^3)
    S004 = cC004/sC002^4

    # variance floor (to_recon_vars c2min), matches realizable_3D_M4
    C200 = max(1.0e-12, cC200)
    C020 = max(1.0e-12, cC020)
    C002 = max(1.0e-12, cC002)

    return (M000, u, v, w, C200, C020, C002,
            S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
            S101, S201, S301, S102, S202, S003, S103, S004,
            S011, S111, S211, S021, S121, S031, S012, S112, S013, S022)
end

# ---------------------------------------------------------------------------
# C4toM4_3D (verbatim @fastmath transcription of the 35 output raw moments, in
# M4 canonical layout, from the autogen reshape literal at indices _M2CS4_IDX).
# Inputs: M000, means, and central moments (S->C output).
# ---------------------------------------------------------------------------
@noinline @fastmath function _c4tom4_35(
        M000, umean, vmean, wmean,
        C200, C110, C101, C020, C011, C002,
        C300, C210, C201, C120, C111, C102, C030, C021, C012, C003,
        C400, C310, C301, C220, C211, C202, C130, C121, C112, C103,
        C040, C031, C022, C013, C004)
    t2 = umean^2
    t3 = umean^3
    t4 = vmean^2
    t5 = vmean^3
    t6 = wmean^2
    t7 = wmean^3

    rM000 = M000
    rM100 = M000*umean
    rM200 = M000*t2+C200*M000
    rM300 = M000*t3+C300*M000+C200*M000*umean*3.0
    rM400 = M000*t2^2+C400*M000+C200*M000*t2*6.0+C300*M000*umean*4.0
    rM010 = M000*vmean
    rM110 = C110*M000+M000*umean*vmean
    rM210 = C210*M000+C110*M000*umean*2.0+C200*M000*vmean+M000*t2*vmean
    rM310 = C310*M000+C110*M000*t2*3.0+C210*M000*umean*3.0+C300*M000*vmean+M000*t3*vmean+C200*M000*umean*vmean*3.0
    rM020 = M000*t4+C020*M000
    rM120 = C120*M000+C020*M000*umean+C110*M000*vmean*2.0+M000*t4*umean
    rM220 = C220*M000+C020*M000*t2+C200*M000*t4+C120*M000*umean*2.0+C210*M000*vmean*2.0+M000*t2*t4+C110*M000*umean*vmean*4.0
    rM030 = M000*t5+C030*M000+C020*M000*vmean*3.0
    rM130 = C130*M000+C110*M000*t4*3.0+C030*M000*umean+C120*M000*vmean*3.0+M000*t5*umean+C020*M000*umean*vmean*3.0
    rM040 = M000*t4^2+C040*M000+C020*M000*t4*6.0+C030*M000*vmean*4.0
    rM001 = M000*wmean
    rM101 = C101*M000+M000*umean*wmean
    rM201 = C201*M000+C101*M000*umean*2.0+C200*M000*wmean+M000*t2*wmean
    rM301 = C301*M000+C101*M000*t2*3.0+C201*M000*umean*3.0+C300*M000*wmean+M000*t3*wmean+C200*M000*umean*wmean*3.0
    rM002 = M000*t6+C002*M000
    rM102 = C102*M000+C002*M000*umean+C101*M000*wmean*2.0+M000*t6*umean
    rM202 = C202*M000+C002*M000*t2+C200*M000*t6+C102*M000*umean*2.0+C201*M000*wmean*2.0+M000*t2*t6+C101*M000*umean*wmean*4.0
    rM003 = M000*t7+C003*M000+C002*M000*wmean*3.0
    rM103 = C103*M000+C101*M000*t6*3.0+C003*M000*umean+C102*M000*wmean*3.0+M000*t7*umean+C002*M000*umean*wmean*3.0
    rM004 = M000*t6^2+C004*M000+C002*M000*t6*6.0+C003*M000*wmean*4.0
    rM011 = C011*M000+M000*vmean*wmean
    rM111 = C111*M000+C011*M000*umean+C101*M000*vmean+C110*M000*wmean+M000*umean*vmean*wmean
    rM211 = C211*M000+C011*M000*t2+C111*M000*umean*2.0+C201*M000*vmean+C210*M000*wmean+M000*t2*vmean*wmean+C101*M000*umean*vmean*2.0+C110*M000*umean*wmean*2.0+C200*M000*vmean*wmean
    rM021 = C021*M000+C011*M000*vmean*2.0+C020*M000*wmean+M000*t4*wmean
    rM121 = C121*M000+C101*M000*t4+C021*M000*umean+C111*M000*vmean*2.0+C120*M000*wmean+M000*t4*umean*wmean+C011*M000*umean*vmean*2.0+C020*M000*umean*wmean+C110*M000*vmean*wmean*2.0
    rM031 = C031*M000+C011*M000*t4*3.0+C021*M000*vmean*3.0+C030*M000*wmean+M000*t5*wmean+C020*M000*vmean*wmean*3.0
    rM012 = C012*M000+C002*M000*vmean+C011*M000*wmean*2.0+M000*t6*vmean
    rM112 = C112*M000+C110*M000*t6+C012*M000*umean+C102*M000*vmean+C111*M000*wmean*2.0+M000*t6*umean*vmean+C002*M000*umean*vmean+C011*M000*umean*wmean*2.0+C101*M000*vmean*wmean*2.0
    rM013 = C013*M000+C011*M000*t6*3.0+C003*M000*vmean+C012*M000*wmean*3.0+M000*t7*vmean+C002*M000*vmean*wmean*3.0
    rM022 = C022*M000+C002*M000*t4+C020*M000*t6+C012*M000*vmean*2.0+C021*M000*wmean*2.0+M000*t4*t6+C011*M000*vmean*wmean*4.0

    return (rM000, rM100, rM200, rM300, rM400,
            rM010, rM110, rM210, rM310,
            rM020, rM120, rM220,
            rM030, rM130,
            rM040,
            rM001, rM101, rM201, rM301,
            rM002, rM102, rM202,
            rM003, rM103,
            rM004,
            rM011, rM111, rM211,
            rM021, rM121,
            rM031,
            rM012, rM112,
            rM013,
            rM022)
end

# ---------------------------------------------------------------------------
# from_recon_vars: recon-var vector -> raw 35-moment NTuple{35}.
# Reproduces standardized_to_M4: plain S->C block (NOT @fastmath) then C4toM4_3D.
# Argument order matches the recon-var layout V[1..35].
# ---------------------------------------------------------------------------
@inline function from_recon_vars_dev(
        M000, umean, vmean, wmean, C200, C020, C002,
        S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
        S101, S201, S301, S102, S202, S003, S103, S004,
        S011, S111, S211, S021, S121, S031, S012, S112, S013, S022)

    sC200 = sqrt(C200); sC020 = sqrt(C020); sC002 = sqrt(C002)
    C110 = S110*sC200*sC020
    C101 = S101*sC200*sC002
    C011 = S011*sC020*sC002
    C300 = S300*sC200*C200
    C210 = S210*C200*sC020
    C201 = S201*C200*sC002
    C120 = S120*sC200*C020
    C111 = S111*sC200*sC020*sC002
    C102 = S102*sC200*C002
    C030 = S030*sC020*C020
    C021 = S021*C020*sC002
    C012 = S012*sC020*C002
    C003 = S003*sC002*C002
    C400 = S400*C200^2
    C310 = S310*sC200*C200*sC020
    C301 = S301*sC200*C200*sC002
    C220 = S220*C200*C020
    C211 = S211*C200*sC020*sC002
    C202 = S202*C200*C002
    C130 = S130*sC200*sC020*C020
    C121 = S121*sC200*C020*sC002
    C112 = S112*sC200*sC020*C002
    C103 = S103*sC200*sC002*C002
    C040 = S040*C020^2
    C031 = S031*sC020*C020*sC002
    C022 = S022*C020*C002
    C013 = S013*sC020*sC002*C002
    C004 = S004*C002^2

    return _c4tom4_35(M000, umean, vmean, wmean,
                      C200, C110, C101, C020, C011, C002,
                      C300, C210, C201, C120, C111, C102, C030, C021, C012, C003,
                      C400, C310, C301, C220, C211, C202, C130, C121, C112, C103,
                      C040, C031, C022, C013, C004)
end

# ---------------------------------------------------------------------------
# NTuple wrappers (explicit indexing — no splat, which lowers to _apply_iterate).
# ---------------------------------------------------------------------------
@inline to_recon_vars_tup(M::NTuple{35}) = to_recon_vars_dev(
    M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8],M[9],M[10],M[11],M[12],M[13],M[14],M[15],
    M[16],M[17],M[18],M[19],M[20],M[21],M[22],M[23],M[24],M[25],M[26],M[27],M[28],M[29],M[30],
    M[31],M[32],M[33],M[34],M[35])

@inline from_recon_vars_tup(V::NTuple{35}) = from_recon_vars_dev(
    V[1],V[2],V[3],V[4],V[5],V[6],V[7],V[8],V[9],V[10],V[11],V[12],V[13],V[14],V[15],
    V[16],V[17],V[18],V[19],V[20],V[21],V[22],V[23],V[24],V[25],V[26],V[27],V[28],V[29],V[30],
    V[31],V[32],V[33],V[34],V[35])

"recon_vars_ok: all finite, V[1]>0, V[5]>0, V[6]>0, V[7]>0 (== reconstruction.jl)."
@inline function recon_vars_ok_tup(V::NTuple{35,Float64})
    fin = true
    @inbounds for k in 1:35
        fin &= isfinite(V[k])
    end
    @inbounds return fin && V[1] > 0.0 && V[5] > 0.0 && V[6] > 0.0 && V[7] > 0.0
end

end # module
