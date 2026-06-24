"""
    Flux_closure35_3D(M4)

Compute the 3D HLL flux moments for all 35 moments via the HyQMOM closure, with
NO realizability correction. Direct port of `Flux_closure35_3D.m` from
Code_Riemann_3D_35mom_july2026 (the revised solver, where realizability is a
separate step applied after the spatial update).

The input `M4` is assumed already hyperbolicity-corrected (e.g. by
`eigenvalues6_hyperbolic_3D`). Returns `(Fx, Fy, Fz)`, each a 35-vector of flux
moments in the standard ordering.

This is the pure-flux companion to `realizable_3D_M4`; together they replace the
combined `Flux_closure35_and_realizable_3D` in the projection-based solver.
"""
function Flux_closure35_3D(M4::AbstractVector)
    M000 = M4[1]
    umean = M4[2]/M000
    vmean = M4[6]/M000
    wmean = M4[16]/M000

    # central and standardized moments
    C4, S4 = M2CS4_35(M4)
    C200 = max(0.0, C4[3])
    C020 = max(0.0, C4[10])
    C002 = max(0.0, C4[20])

    S300=S4[4];  S400=S4[5];  S110=S4[7];  S210=S4[8];  S310=S4[9]
    S120=S4[11]; S220=S4[12]; S030=S4[13]; S130=S4[14]; S040=S4[15]
    S101=S4[17]; S201=S4[18]; S301=S4[19]; S102=S4[21]; S202=S4[22]
    S003=S4[23]; S103=S4[24]; S004=S4[25]; S011=S4[26]; S111=S4[27]
    S211=S4[28]; S021=S4[29]; S121=S4[30]; S031=S4[31]; S012=S4[32]
    S112=S4[33]; S013=S4[34]; S022=S4[35]

    # 3D HyQMOM closures for 5th-order standardized moments
    (S500, S410, S320, S230, S140, S401, S302, S203, S104, S311,
     S221, S131, S212, S113, S122, S050, S041, S032, S023, S014, S005) =
        hyqmom_3D(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                  S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                  S211, S021, S121, S031, S012, S112, S013, S022)

    # standardized -> central
    sC200 = sqrt(C200); sC020 = sqrt(C020); sC002 = sqrt(C002)
    (C110, C101, C011, C300, C210, C201, C120, C111, C102, C030, C021, C012, C003,
     C400, C310, C301, C220, C211, C202, C130, C121, C112, C103, C040, C031, C022, C013, C004,
     C500, C410, C401, C320, C311, C302, C230, C221, C212, C203, C140, C131, C122, C113, C104,
     C050, C041, C032, C023, C014, C005) =
        S_to_C_batch(S110, S101, S011, S300, S210, S201, S120, S111, S102, S030, S021, S012, S003,
                     S400, S310, S301, S220, S211, S202, S130, S121, S112, S103, S040, S031, S022, S013, S004,
                     S500, S410, S401, S320, S311, S302, S230, S221, S212, S203, S140, S131, S122, S113, S104,
                     S050, S041, S032, S023, S014, S005,
                     sC200, sC020, sC002)

    # central -> raw (5th order)
    M5 = C5toM5_3D(M000, umean, vmean, wmean, C200, C110, C101, C020, C011, C002,
                   C300, C210, C201, C120, C111, C102, C030, C021, C012, C003,
                   C400, C310, C301, C220, C211, C202, C130, C121, C112, C103, C040, C031, C022, C013, C004,
                   C500, C410, C320, C230, C140, C401, C302, C203, C104, C311, C221, C131, C212, C113, C122,
                   C050, C041, C032, C023, C014, C005)

    (M000, M100, M010, M001, M200, M110, M101, M020, M011, M002,
     M300, M210, M201, M120, M111, M102, M030, M021, M012, M003,
     M400, M310, M301, M220, M211, M202, M130, M121, M112, M103, M040, M031, M022, M013, M004,
     M500, M410, M320, M230, M140, M401, M302, M203, M104, M311, M221, M131, M212, M113, M122,
     M050, M041, M032, M023, M014, M005) = M5_to_vars(M5)

    Fx = [M100,M200,M300,M400,M500,M110,M210,M310,M410,M120,M220,M320,M130,M230,M140,
          M101,M201,M301,M401,M102,M202,M302,M103,M203,M104,M111,M211,M311,M121,M221,
          M131,M112,M212,M113,M122]
    Fy = [M010,M110,M210,M310,M410,M020,M120,M220,M320,M030,M130,M230,M040,M140,M050,
          M011,M111,M211,M311,M012,M112,M212,M013,M113,M014,M021,M121,M221,M031,M131,
          M041,M022,M122,M023,M032]
    Fz = [M001,M101,M201,M301,M401,M011,M111,M211,M311,M021,M121,M221,M031,M131,M041,
          M002,M102,M202,M302,M003,M103,M203,M004,M104,M005,M012,M112,M212,M022,M122,
          M032,M013,M113,M014,M023]
    return Fx, Fy, Fz
end
