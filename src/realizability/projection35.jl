"""
    projection35(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                 S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                 S211, S021, S121, S031, S012, S112, S013, S022)

Check and correct realizability of the 3rd/4th-order cross moments by projection
onto a realizable target (revised moment-projection method, Appendix B of the
HyQMOM paper). Direct port of `projection35.m`.

Returns the 28 standardized moments (corrected if necessary), in the same order
as the arguments.

# Algorithm
1. Build the 6x6 realizability matrix `E1 = delta2star3D(...)`. If its smallest
   eigenvalue is non-negative the moments are realizable -> return unchanged.
2. Otherwise project onto a one-parameter realizable target defined by the
   invariants S2, S3 (mean skewness) and S4, setting every 3rd/4th-order cross
   moment to a simple product of (S110,S101,S011) with S3 or S4.
3. Recheck; if the target is still (barely) unrealizable, force realizability by
   collapsing the smallest-magnitude correlation to the product of the other two
   and rebuild the target.
"""
function projection35(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                      S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                      S211, S021, S121, S031, S012, S112, S013, S022)

    # <p2p2> interior of 2nd-order moment space
    E1 = delta2star3D(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                      S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                      S211, S021, S121, S031, S012, S112, S013, S022)
    lambda = sort(real(_geigvals(E1)))
    if lambda[1] >= 0
        # no projection required
        return (S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                S211, S021, S121, S031, S012, S112, S013, S022)
    end

    H200 = S400 - S300^2 - 1
    H020 = S040 - S030^2 - 1
    H002 = S004 - S003^2 - 1

    S2 = 1 + 2*S110*S101*S011 - S110^2 - S101^2 - S011^2
    # 3D target moments
    R101 = sign(S101)
    R110 = sign(S110)
    S3 = (S300 + R110*S030 + R101*S003) / 3
    S4 = 1 + S3^2 + S2*(H200 + H020 + H002) / 3

    S300 = S3
    S030 = S110*S3
    S003 = S101*S3
    S111 = S011*S3
    S120 = S300
    S102 = S300
    S210 = S030
    S012 = S030
    S201 = S003
    S021 = S003
    S310 = S110*S4
    S130 = S110*S4
    S112 = S110*S4
    S301 = S101*S4
    S103 = S101*S4
    S121 = S101*S4
    S031 = S011*S4
    S013 = S011*S4
    S211 = S011*S4
    S400 = S4
    S040 = S4
    S004 = S4
    S220 = S4
    S202 = S4
    S022 = S4

    # check realizability of target moments
    E1 = delta2star3D(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                      S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                      S211, S021, S121, S031, S012, S112, S013, S022)
    lambda = sort(real(_geigvals(E1)))
    if lambda[1] > -1.0e-6
        return (S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                S211, S021, S121, S031, S012, S112, S013, S022)
    end

    # force realizability
    if abs(S110) <= abs(S101) && abs(S110) <= abs(S011)
        S110 = S101*S011
    elseif abs(S101) <= abs(S110) && abs(S101) <= abs(S011)
        S101 = S110*S011
    else
        S011 = S110*S101
    end

    # rebuild target (S3, S4, S2 retained from above)
    S300 = S3
    S030 = S110*S3
    S003 = S101*S3
    S111 = S011*S3
    S120 = S300
    S102 = S300
    S210 = S030
    S012 = S030
    S201 = S003
    S021 = S003
    S310 = S110*S4
    S130 = S110*S4
    S112 = S110*S4
    S301 = S101*S4
    S103 = S101*S4
    S121 = S101*S4
    S031 = S011*S4
    S013 = S011*S4
    S211 = S011*S4
    S400 = S4
    S040 = S4
    S004 = S4
    S220 = S4
    S202 = S4
    S022 = S4

    return (S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
            S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
            S211, S021, S121, S031, S012, S112, S013, S022)
end
