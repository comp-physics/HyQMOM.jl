"""
    flux_closure_dev.jl — device-compatible, ALLOC-FREE port of `Flux_closure35_3D`.

`flux_closure35_dev(M000,...,M022) -> NTuple{105,Float64}` returns `(Fx[1:35],
Fy[1:35], Fz[1:35])` flattened: indices 1..35 = Fx, 36..70 = Fy, 71..105 = Fz.

The 35 scalar inputs are the raw moments in the canonical `M4` vector order
(M4[1]=M000, M4[2]=M100, ..., M4[35]=M022) — i.e. the same argument order as
`M4toC4_3D` / `M2CS4_35`.

This reproduces the EXACT floating-point arithmetic of the CPU chain
    Flux_closure35_3D = M2CS4_35 (M4toC4_3D + standardize) -> hyqmom_3D ->
                        S_to_C_batch -> C5toM5_3D -> assemble Fx/Fy/Fz
but with NO heap allocation: only scalars and a returned immutable NTuple.
Only the central / raw moments actually consumed downstream are computed (the
autogen `reshape([...],5,5,5)` / `(6,6,6)` arrays are sparse — most entries are
0.0 and unused). The kept entries are transcribed verbatim from the autogen
literals, so the FP operation order is identical.

`@fastmath` is applied to match the autogen `M4toC4_3D` / `C5toM5_3D` (which are
`@fastmath`); `hyqmom_3D`, `M2CS4_35` standardization and `S_to_C_batch` are NOT
`@fastmath` on the CPU, so those blocks are left outside the `@fastmath` macro by
splitting into helper functions. This keeps byte-for-byte FP parity.

Single source of the per-cell flux closure: this file lives in `src/numerics/` and
is `include`d by both the CPU `Flux_closure35_3D` (which delegates here) and the GPU
kernel module (`gpu/flux_closure_gpu.jl`). No CUDA dependency here — plain Julia.
"""
module FluxClosureDev

export flux_closure35_dev

const _EPSF = 2.220446049250313e-16   # eps(Float64)

# ---------------------------------------------------------------------------
# Block A: central moments from raw (subset of autogen M4toC4_3D, @fastmath).
# Returns the central moments consumed downstream, in a flat NTuple.
# ---------------------------------------------------------------------------
@inline @fastmath function _m4toc4_subset(
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
# Block B: raw moments from central (subset of autogen C5toM5_3D, @fastmath).
# Returns the 55 raw moments consumed by Fx/Fy/Fz (M000 omitted — trivial).
# ---------------------------------------------------------------------------
@inline @fastmath function _c5tom5_subset(
        M000,umean,vmean,wmean,C200,C110,C101,C020,C011,C002,
        C300,C210,C201,C120,C111,C102,C030,C021,C012,C003,
        C400,C310,C301,C220,C211,C202,C130,C121,C112,C103,C040,C031,C022,C013,C004,
        C500,C410,C320,C230,C140,C401,C302,C203,C104,C311,C221,C131,C212,C113,C122,
        C050,C041,C032,C023,C014,C005)
    t2 = umean^2; t3 = umean^3; t5 = vmean^2; t6 = vmean^3; t8 = wmean^2; t9 = wmean^3
    t4 = t2^2; t7 = t5^2; t10 = t8^2

    M100 = M000*umean
    M200 = M000*t2+C200*M000
    M300 = M000*t3+C300*M000+C200*M000*umean*3.0
    M400 = M000*t4+C400*M000+C200*M000*t2*6.0+C300*M000*umean*4.0
    M500 = M000*umean^5+C500*M000+C200*M000*t3*1.0e+1+C300*M000*t2*1.0e+1+C400*M000*umean*5.0
    M010 = M000*vmean
    M110 = C110*M000+M000*umean*vmean
    M210 = C210*M000+C110*M000*umean*2.0+C200*M000*vmean+M000*t2*vmean
    M310 = C310*M000+C110*M000*t2*3.0+C210*M000*umean*3.0+C300*M000*vmean+M000*t3*vmean+C200*M000*umean*vmean*3.0
    M410 = C410*M000+C110*M000*t3*4.0+C210*M000*t2*6.0+C310*M000*umean*4.0+C400*M000*vmean+M000*t4*vmean+C200*M000*t2*vmean*6.0+C300*M000*umean*vmean*4.0
    M020 = M000*t5+C020*M000
    M120 = C120*M000+C020*M000*umean+C110*M000*vmean*2.0+M000*t5*umean
    M220 = C220*M000+C020*M000*t2+C200*M000*t5+C120*M000*umean*2.0+C210*M000*vmean*2.0+M000*t2*t5+C110*M000*umean*vmean*4.0
    M320 = C320*M000+C020*M000*t3+C120*M000*t2*3.0+C300*M000*t5+C220*M000*umean*3.0+C310*M000*vmean*2.0+M000*t3*t5+C200*M000*t5*umean*3.0+C110*M000*t2*vmean*6.0+C210*M000*umean*vmean*6.0
    M030 = M000*t6+C030*M000+C020*M000*vmean*3.0
    M130 = C130*M000+C110*M000*t5*3.0+C030*M000*umean+C120*M000*vmean*3.0+M000*t6*umean+C020*M000*umean*vmean*3.0
    M230 = C230*M000+C030*M000*t2+C200*M000*t6+C210*M000*t5*3.0+C130*M000*umean*2.0+C220*M000*vmean*3.0+M000*t2*t6+C110*M000*t5*umean*6.0+C020*M000*t2*vmean*3.0+C120*M000*umean*vmean*6.0
    M040 = M000*t7+C040*M000+C020*M000*t5*6.0+C030*M000*vmean*4.0
    M140 = C140*M000+C110*M000*t6*4.0+C120*M000*t5*6.0+C040*M000*umean+C130*M000*vmean*4.0+M000*t7*umean+C020*M000*t5*umean*6.0+C030*M000*umean*vmean*4.0
    M050 = M000*vmean^5+C050*M000+C020*M000*t6*1.0e+1+C030*M000*t5*1.0e+1+C040*M000*vmean*5.0
    M001 = M000*wmean
    M101 = C101*M000+M000*umean*wmean
    M201 = C201*M000+C101*M000*umean*2.0+C200*M000*wmean+M000*t2*wmean
    M301 = C301*M000+C101*M000*t2*3.0+C201*M000*umean*3.0+C300*M000*wmean+M000*t3*wmean+C200*M000*umean*wmean*3.0
    M401 = C401*M000+C101*M000*t3*4.0+C201*M000*t2*6.0+C301*M000*umean*4.0+C400*M000*wmean+M000*t4*wmean+C200*M000*t2*wmean*6.0+C300*M000*umean*wmean*4.0
    M011 = C011*M000+M000*vmean*wmean
    M111 = C111*M000+C011*M000*umean+C101*M000*vmean+C110*M000*wmean+M000*umean*vmean*wmean
    M211 = C211*M000+C011*M000*t2+C111*M000*umean*2.0+C201*M000*vmean+C210*M000*wmean+M000*t2*vmean*wmean+C101*M000*umean*vmean*2.0+C110*M000*umean*wmean*2.0+C200*M000*vmean*wmean
    M311 = C311*M000+C011*M000*t3+C111*M000*t2*3.0+C211*M000*umean*3.0+C301*M000*vmean+C310*M000*wmean+M000*t3*vmean*wmean+C101*M000*t2*vmean*3.0+C110*M000*t2*wmean*3.0+C201*M000*umean*vmean*3.0+C210*M000*umean*wmean*3.0+C300*M000*vmean*wmean+C200*M000*umean*vmean*wmean*3.0
    M021 = C021*M000+C011*M000*vmean*2.0+C020*M000*wmean+M000*t5*wmean
    M121 = C121*M000+C101*M000*t5+C021*M000*umean+C111*M000*vmean*2.0+C120*M000*wmean+M000*t5*umean*wmean+C011*M000*umean*vmean*2.0+C020*M000*umean*wmean+C110*M000*vmean*wmean*2.0
    M221 = C221*M000+C021*M000*t2+C201*M000*t5+C121*M000*umean*2.0+C211*M000*vmean*2.0+C220*M000*wmean+M000*t2*t5*wmean+C101*M000*t5*umean*2.0+C011*M000*t2*vmean*2.0+C020*M000*t2*wmean+C200*M000*t5*wmean+C111*M000*umean*vmean*4.0+C120*M000*umean*wmean*2.0+C210*M000*vmean*wmean*2.0+C110*M000*umean*vmean*wmean*4.0
    M031 = C031*M000+C011*M000*t5*3.0+C021*M000*vmean*3.0+C030*M000*wmean+M000*t6*wmean+C020*M000*vmean*wmean*3.0
    M131 = C131*M000+C101*M000*t6+C111*M000*t5*3.0+C031*M000*umean+C121*M000*vmean*3.0+C130*M000*wmean+M000*t6*umean*wmean+C011*M000*t5*umean*3.0+C110*M000*t5*wmean*3.0+C021*M000*umean*vmean*3.0+C030*M000*umean*wmean+C120*M000*vmean*wmean*3.0+C020*M000*umean*vmean*wmean*3.0
    M041 = C041*M000+C011*M000*t6*4.0+C021*M000*t5*6.0+C031*M000*vmean*4.0+C040*M000*wmean+M000*t7*wmean+C020*M000*t5*wmean*6.0+C030*M000*vmean*wmean*4.0
    M002 = M000*t8+C002*M000
    M102 = C102*M000+C002*M000*umean+C101*M000*wmean*2.0+M000*t8*umean
    M202 = C202*M000+C002*M000*t2+C200*M000*t8+C102*M000*umean*2.0+C201*M000*wmean*2.0+M000*t2*t8+C101*M000*umean*wmean*4.0
    M302 = C302*M000+C002*M000*t3+C102*M000*t2*3.0+C300*M000*t8+C202*M000*umean*3.0+C301*M000*wmean*2.0+M000*t3*t8+C200*M000*t8*umean*3.0+C101*M000*t2*wmean*6.0+C201*M000*umean*wmean*6.0
    M012 = C012*M000+C002*M000*vmean+C011*M000*wmean*2.0+M000*t8*vmean
    M112 = C112*M000+C110*M000*t8+C012*M000*umean+C102*M000*vmean+C111*M000*wmean*2.0+M000*t8*umean*vmean+C002*M000*umean*vmean+C011*M000*umean*wmean*2.0+C101*M000*vmean*wmean*2.0
    M212 = C212*M000+C012*M000*t2+C210*M000*t8+C112*M000*umean*2.0+C202*M000*vmean+C211*M000*wmean*2.0+M000*t2*t8*vmean+C110*M000*t8*umean*2.0+C002*M000*t2*vmean+C200*M000*t8*vmean+C011*M000*t2*wmean*2.0+C102*M000*umean*vmean*2.0+C111*M000*umean*wmean*4.0+C201*M000*vmean*wmean*2.0+C101*M000*umean*vmean*wmean*4.0
    M022 = C022*M000+C002*M000*t5+C020*M000*t8+C012*M000*vmean*2.0+C021*M000*wmean*2.0+M000*t5*t8+C011*M000*vmean*wmean*4.0
    M122 = C122*M000+C102*M000*t5+C120*M000*t8+C022*M000*umean+C112*M000*vmean*2.0+C121*M000*wmean*2.0+M000*t5*t8*umean+C002*M000*t5*umean+C020*M000*t8*umean+C110*M000*t8*vmean*2.0+C101*M000*t5*wmean*2.0+C012*M000*umean*vmean*2.0+C021*M000*umean*wmean*2.0+C111*M000*vmean*wmean*4.0+C011*M000*umean*vmean*wmean*4.0
    M032 = C032*M000+C002*M000*t6+C012*M000*t5*3.0+C030*M000*t8+C022*M000*vmean*3.0+C031*M000*wmean*2.0+M000*t6*t8+C020*M000*t8*vmean*3.0+C011*M000*t5*wmean*6.0+C021*M000*vmean*wmean*6.0
    M003 = M000*t9+C003*M000+C002*M000*wmean*3.0
    M103 = C103*M000+C101*M000*t8*3.0+C003*M000*umean+C102*M000*wmean*3.0+M000*t9*umean+C002*M000*umean*wmean*3.0
    M203 = C203*M000+C003*M000*t2+C200*M000*t9+C201*M000*t8*3.0+C103*M000*umean*2.0+C202*M000*wmean*3.0+M000*t2*t9+C101*M000*t8*umean*6.0+C002*M000*t2*wmean*3.0+C102*M000*umean*wmean*6.0
    M013 = C013*M000+C011*M000*t8*3.0+C003*M000*vmean+C012*M000*wmean*3.0+M000*t9*vmean+C002*M000*vmean*wmean*3.0
    M113 = C113*M000+C110*M000*t9+C111*M000*t8*3.0+C013*M000*umean+C103*M000*vmean+C112*M000*wmean*3.0+M000*t9*umean*vmean+C011*M000*t8*umean*3.0+C101*M000*t8*vmean*3.0+C003*M000*umean*vmean+C012*M000*umean*wmean*3.0+C102*M000*vmean*wmean*3.0+C002*M000*umean*vmean*wmean*3.0
    M023 = C023*M000+C003*M000*t5+C020*M000*t9+C021*M000*t8*3.0+C013*M000*vmean*2.0+C022*M000*wmean*3.0+M000*t5*t9+C011*M000*t8*vmean*6.0+C002*M000*t5*wmean*3.0+C012*M000*vmean*wmean*6.0
    M004 = M000*t10+C004*M000+C002*M000*t8*6.0+C003*M000*wmean*4.0
    M104 = C104*M000+C101*M000*t9*4.0+C102*M000*t8*6.0+C004*M000*umean+C103*M000*wmean*4.0+M000*t10*umean+C002*M000*t8*umean*6.0+C003*M000*umean*wmean*4.0
    M014 = C014*M000+C011*M000*t9*4.0+C012*M000*t8*6.0+C004*M000*vmean+C013*M000*wmean*4.0+M000*t10*vmean+C002*M000*t8*vmean*6.0+C003*M000*vmean*wmean*4.0
    M005 = M000*wmean^5+C005*M000+C002*M000*t9*1.0e+1+C003*M000*t8*1.0e+1+C004*M000*wmean*5.0

    return (M100,M200,M300,M400,M500,M010,M110,M210,M310,M410,M020,M120,M220,M320,
            M030,M130,M230,M040,M140,M050,M001,M101,M201,M301,M401,M011,M111,M211,M311,
            M021,M121,M221,M031,M131,M041,M002,M102,M202,M302,M012,M112,M212,M022,M122,
            M032,M003,M103,M203,M013,M113,M023,M004,M104,M014,M005)
end

# ---------------------------------------------------------------------------
# Top-level device flux closure (alloc-free). 35 raw moment scalars in, 105 out.
# ---------------------------------------------------------------------------
@inline function flux_closure35_dev(
        M000,M100,M200,M300,M400,M010,M110,M210,M310,M020,M120,M220,M030,M130,M040,
        M001,M101,M201,M301,M002,M102,M202,M003,M103,M004,M011,M111,M211,M021,M121,
        M031,M012,M112,M013,M022)

    umean = M100/M000
    vmean = M010/M000
    wmean = M001/M000

    # --- central moments (subset of M2CS4_35's M4toC4_3D) ---
    (cC200,cC020,cC002,cC300,cC400,cC110,cC210,cC310,cC120,cC220,cC030,cC130,cC040,
     cC101,cC201,cC301,cC011,cC111,cC211,cC021,cC121,cC031,cC102,cC202,cC012,cC112,
     cC022,cC003,cC103,cC013,cC004) =
        _m4toc4_subset(M000,M100,M200,M300,M400,M010,M110,M210,M310,M020,M120,M220,M030,M130,M040,
                       M001,M101,M201,M301,M002,M102,M202,M003,M103,M004,M011,M111,M211,M021,M121,
                       M031,M012,M112,M013,M022)

    # --- standardize (M2CS4_35): variances floored by eps() ---
    si200 = sqrt(max(cC200, _EPSF)); si020 = sqrt(max(cC020, _EPSF)); si002 = sqrt(max(cC002, _EPSF))

    S110 = cC110/(si200*si020)
    S101 = cC101/(si200*si002)
    S011 = cC011/(si020*si002)
    S300 = cC300/si200^3
    S210 = cC210/(si200^2*si020)
    S201 = cC201/(si200^2*si002)
    S120 = cC120/(si200*si020^2)
    S111 = cC111/(si200*si020*si002)
    S102 = cC102/(si200*si002^2)
    S030 = cC030/si020^3
    S021 = cC021/(si020^2*si002)
    S012 = cC012/(si020*si002^2)
    S003 = cC003/si002^3
    S400 = cC400/si200^4
    S310 = cC310/(si200^3*si020)
    S301 = cC301/(si200^3*si002)
    S220 = cC220/(si200^2*si020^2)
    S211 = cC211/(si200^2*si020*si002)
    S202 = cC202/(si200^2*si002^2)
    S130 = cC130/(si200*si020^3)
    S121 = cC121/(si200*si020^2*si002)
    S112 = cC112/(si200*si020*si002^2)
    S103 = cC103/(si200*si002^3)
    S040 = cC040/si020^4
    S031 = cC031/(si020^3*si002)
    S022 = cC022/(si020^2*si002^2)
    S013 = cC013/(si020*si002^3)
    S004 = cC004/si002^4

    # --- 5th-order HyQMOM closures (verbatim hyqmom_3D, NOT @fastmath) ---
    S500 = 0.5*S300*(5*S400 - 3*S300^2 - 1)
    S050 = 0.5*S030*(5*S040 - 3*S030^2 - 1)
    S005 = 0.5*S003*(5*S004 - 3*S003^2 - 1)
    S320 = (S400 - (3*S300^2)/2)*S120 + ((3*S300)/2)*S220 - S300/2
    S230 = (S040 - (3*S030^2)/2)*S210 + ((3*S030)/2)*S220 - S030/2
    S302 = (S400 - (3*S300^2)/2)*S102 + ((3*S300)/2)*S202 - S300/2
    S203 = (S004 - (3*S003^2)/2)*S201 + ((3*S003)/2)*S202 - S003/2
    S032 = (S040 - (3*S030^2)/2)*S012 + ((3*S030)/2)*S022 - S030/2
    S023 = (S004 - (3*S003^2)/2)*S021 + ((3*S003)/2)*S022 - S003/2
    S410 = (S300 - 2*S300*S400 + (9*S300^3)/4)*S110 + ((5*S400)/2 - (15*S300^2)/4 - 3/2)*S210 + (2*S300)*S310
    S140 = (S030 - 2*S030*S040 + (9*S030^3)/4)*S110 + ((5*S040)/2 - (15*S030^2)/4 - 3/2)*S120 + (2*S030)*S130
    S401 = (S300 - 2*S300*S400 + (9*S300^3)/4)*S101 + ((5*S400)/2 - (15*S300^2)/4 - 3/2)*S201 + (2*S300)*S301
    S104 = (S003 - 2*S003*S004 + (9*S003^3)/4)*S101 + ((5*S004)/2 - (15*S003^2)/4 - 3/2)*S102 + (2*S003)*S103
    S041 = (S030 - 2*S030*S040 + (9*S030^3)/4)*S011 + ((5*S040)/2 - (15*S030^2)/4 - 3/2)*S021 + (2*S030)*S031
    S014 = (S003 - 2*S003*S004 + (9*S003^3)/4)*S011 + ((5*S004)/2 - (15*S003^2)/4 - 3/2)*S012 + (2*S003)*S013
    S311 = 3*(S300*S211)/2 + (2*S400 - 3*S300^2)*S111/2 - (S300*S011)/2
    S131 = 3*(S030*S121)/2 + (2*S040 - 3*S030^2)*S111/2 - (S030*S101)/2
    S113 = 3*(S003*S112)/2 + (2*S004 - 3*S003^2)*S111/2 - (S003*S110)/2
    S221 = S300*(S121 - S101) + S030*(S211 - S011) + S201 + S021 - S300*S030*S111
    S212 = S300*(S112 - S110) + S003*(S211 - S011) + S210 + S012 - S300*S003*S111
    S122 = S003*(S121 - S101) + S030*(S112 - S110) + S102 + S120 - S003*S030*S111

    # --- standardized -> central (S_to_C_batch), outer variances floored by 0 ---
    C200 = max(0.0, cC200); C020 = max(0.0, cC020); C002 = max(0.0, cC002)
    so200 = sqrt(C200); so020 = sqrt(C020); so002 = sqrt(C002)

    C110 = S110 * so200 * so020
    C101 = S101 * so200 * so002
    C011 = S011 * so020 * so002
    C300 = S300 * so200^3
    C210 = S210 * so200^2 * so020
    C201 = S201 * so200^2 * so002
    C120 = S120 * so200 * so020^2
    C111 = S111 * so200 * so020 * so002
    C102 = S102 * so200 * so002^2
    C030 = S030 * so020^3
    C021 = S021 * so020^2 * so002
    C012 = S012 * so020 * so002^2
    C003 = S003 * so002^3
    C400 = S400 * so200^4
    C310 = S310 * so200^3 * so020
    C301 = S301 * so200^3 * so002
    C220 = S220 * so200^2 * so020^2
    C211 = S211 * so200^2 * so020 * so002
    C202 = S202 * so200^2 * so002^2
    C130 = S130 * so200 * so020^3
    C121 = S121 * so200 * so020^2 * so002
    C112 = S112 * so200 * so020 * so002^2
    C103 = S103 * so200 * so002^3
    C040 = S040 * so020^4
    C031 = S031 * so020^3 * so002
    C022 = S022 * so020^2 * so002^2
    C013 = S013 * so020 * so002^3
    C004 = S004 * so002^4
    C500 = S500 * so200^5
    C410 = S410 * so200^4 * so020
    C401 = S401 * so200^4 * so002
    C320 = S320 * so200^3 * so020^2
    C311 = S311 * so200^3 * so020 * so002
    C302 = S302 * so200^3 * so002^2
    C230 = S230 * so200^2 * so020^3
    C221 = S221 * so200^2 * so020^2 * so002
    C212 = S212 * so200^2 * so020 * so002^2
    C203 = S203 * so200^2 * so002^3
    C140 = S140 * so200 * so020^4
    C131 = S131 * so200 * so020^3 * so002
    C122 = S122 * so200 * so020^2 * so002^2
    C113 = S113 * so200 * so020 * so002^3
    C104 = S104 * so200 * so002^4
    C050 = S050 * so020^5
    C041 = S041 * so020^4 * so002
    C032 = S032 * so020^3 * so002^2
    C023 = S023 * so020^2 * so002^3
    C014 = S014 * so020 * so002^4
    C005 = S005 * so002^5

    # --- central -> raw (subset of C5toM5_3D) ---
    (rM100,rM200,rM300,rM400,rM500,rM010,rM110,rM210,rM310,rM410,rM020,rM120,rM220,rM320,
     rM030,rM130,rM230,rM040,rM140,rM050,rM001,rM101,rM201,rM301,rM401,rM011,rM111,rM211,rM311,
     rM021,rM121,rM221,rM031,rM131,rM041,rM002,rM102,rM202,rM302,rM012,rM112,rM212,rM022,rM122,
     rM032,rM003,rM103,rM203,rM013,rM113,rM023,rM004,rM104,rM014,rM005) =
        _c5tom5_subset(M000,umean,vmean,wmean,C200,C110,C101,C020,C011,C002,
                       C300,C210,C201,C120,C111,C102,C030,C021,C012,C003,
                       C400,C310,C301,C220,C211,C202,C130,C121,C112,C103,C040,C031,C022,C013,C004,
                       C500,C410,C320,C230,C140,C401,C302,C203,C104,C311,C221,C131,C212,C113,C122,
                       C050,C041,C032,C023,C014,C005)

    # --- assemble Fx | Fy | Fz (flattened 105-tuple) ---
    return (
        # Fx (1..35)
        rM100,rM200,rM300,rM400,rM500,rM110,rM210,rM310,rM410,rM120,rM220,rM320,rM130,rM230,rM140,
        rM101,rM201,rM301,rM401,rM102,rM202,rM302,rM103,rM203,rM104,rM111,rM211,rM311,rM121,rM221,
        rM131,rM112,rM212,rM113,rM122,
        # Fy (36..70)
        rM010,rM110,rM210,rM310,rM410,rM020,rM120,rM220,rM320,rM030,rM130,rM230,rM040,rM140,rM050,
        rM011,rM111,rM211,rM311,rM012,rM112,rM212,rM013,rM113,rM014,rM021,rM121,rM221,rM031,rM131,
        rM041,rM022,rM122,rM023,rM032,
        # Fz (71..105)
        rM001,rM101,rM201,rM301,rM401,rM011,rM111,rM211,rM311,rM021,rM121,rM221,rM031,rM131,rM041,
        rM002,rM102,rM202,rM302,rM003,rM103,rM203,rM004,rM104,rM005,rM012,rM112,rM212,rM022,rM122,
        rM032,rM013,rM113,rM014,rM023)
end

end # module
