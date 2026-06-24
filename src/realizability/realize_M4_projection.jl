"""
    realizable_3D_M4(M4, Ma)

Check and correct 3D unrealizable moments using the revised moment-projection
method (Appendix B). Direct port of `realizable_3D.m` from
Code_Riemann_3D_35mom_july2026.

Takes the 35-moment vector `M4` (orders 0-4, standard layout) and the Mach number
`Ma`, and returns the corrected 35-moment vector `M4r`. Internally: convert to
central/standardized moments, enforce univariate realizability (with an
Ma-dependent skewness cap), correct the 2nd-order cross moments, bound
S220/S202/S022, apply `projection35`, then reconstruct the raw moments.

This is the projection-based replacement for the legacy minor-cascade
`realizable_3D` (28-argument standardized-moment corrector). It is provided
alongside the legacy path; wiring it into the solver is a separate step.
"""
function realizable_3D_M4(M4::AbstractVector, Ma::Real)
    s3max = 4.0 + abs(Ma)/2.0   # maximum skewness
    c2min = 1.0e-12             # distance from boundary of 2nd-order moment space
    h2min = 1.0e-6              # distance from boundary of 4th-order moment space
    S2min = 1.0e-6

    # mean velocities
    M000 = M4[1]
    umean = M4[2]/M000
    vmean = M4[6]/M000
    wmean = M4[16]/M000

    # central and standardized moments
    C4, S4 = M2CS4_35(M4)
    C200 = max(c2min, C4[3])
    C020 = max(c2min, C4[10])
    C002 = max(c2min, C4[20])

    S300=S4[4];  S400=S4[5];  S110=S4[7];  S210=S4[8];  S310=S4[9]
    S120=S4[11]; S220=S4[12]; S030=S4[13]; S130=S4[14]; S040=S4[15]
    S101=S4[17]; S201=S4[18]; S301=S4[19]; S102=S4[21]; S202=S4[22]
    S003=S4[23]; S103=S4[24]; S004=S4[25]; S011=S4[26]; S111=S4[27]
    S211=S4[28]; S021=S4[29]; S121=S4[30]; S031=S4[31]; S012=S4[32]
    S112=S4[33]; S013=S4[34]; S022=S4[35]

    # --- univariate moments ---
    H200 = S400 - S300^2 - 1
    H020 = S040 - S030^2 - 1
    H002 = S004 - S003^2 - 1
    if H200 <= h2min; H200 = h2min; S400 = H200 + S300^2 + 1; end
    if H020 <= h2min; H020 = h2min; S040 = H020 + S030^2 + 1; end
    if H002 <= h2min; H002 = h2min; S004 = H002 + S003^2 + 1; end
    # cap skewness at +/- s3max (MATLAB applies this block twice; it is idempotent)
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
    S110, S101, S011, S2 = realizability_S2(S110, S101, S011)
    if S2 < S2min
        R = 1 - h2min
        S110 = R*S110
        S101 = R*S101
        S011 = R*S011
    end

    # --- 4th-order: max bounds on S220, S202, S022 ---
    A220 = sqrt((H200 + S300^2)*(H020 + S030^2))
    S220max = realizablity_S220(S110, S220, A220)
    A202 = sqrt((H200 + S300^2)*(H002 + S003^2))
    S202max = realizablity_S220(S101, S202, A202)
    A022 = sqrt((H020 + S030^2)*(H002 + S003^2))
    S022max = realizablity_S220(S011, S022, A022)
    S220 = min(S220, S220max)
    S202 = min(S202, S202max)
    S022 = min(S022, S022max)

    # --- 3rd/4th-order: projection ---
    (S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
     S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
     S211, S021, S121, S031, S012, S112, S013, S022) =
        projection35(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                     S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                     S211, S021, S121, S031, S012, S112, S013, S022)

    # --- central moments from corrected standardized moments ---
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

    # --- raw moments from central moments ---
    M5 = C4toM4_3D(M000, umean, vmean, wmean, C200, C110, C101, C020, C011, C002,
                   C300, C210, C201, C120, C111, C102, C030, C021, C012, C003,
                   C400, C310, C301, C220, C211, C202, C130, C121, C112, C103,
                   C040, C031, C022, C013, C004)

    M4r = [M5[1,1,1], M5[2,1,1], M5[3,1,1], M5[4,1,1], M5[5,1,1],
           M5[1,2,1], M5[2,2,1], M5[3,2,1], M5[4,2,1],
           M5[1,3,1], M5[2,3,1], M5[3,3,1],
           M5[1,4,1], M5[2,4,1],
           M5[1,5,1],
           M5[1,1,2], M5[2,1,2], M5[3,1,2], M5[4,1,2],
           M5[1,1,3], M5[2,1,3], M5[3,1,3],
           M5[1,1,4], M5[2,1,4],
           M5[1,1,5],
           M5[1,2,2], M5[2,2,2], M5[3,2,2],
           M5[1,3,2], M5[2,3,2],
           M5[1,4,2],
           M5[1,2,3], M5[2,2,3],
           M5[1,2,4],
           M5[1,3,3]]
    return M4r
end
