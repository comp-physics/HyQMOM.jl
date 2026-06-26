"""
    realizable_3D(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                  S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                  S211, S021, S121, S031, S012, S112, S013, S022)

Check and correct realizability of cross moments in 3D.

This is the most complex realizability function, handling 3D cross-moment
constraints. It enforces positive-definiteness of the 6x6 realizability matrix.

# Arguments
- 28 standardized moments from 3rd and 4th order

# Returns
- 28 corrected moments + flag220 indicator

# Algorithm
1. Compute H200, H020, H002 (variance-related quantities)
2. Check maximum bounds on S220, S202, S022
3. Check and correct S110, S101, S011 (2nd-order)
4. Handle degenerate cases (faces) when S2 ~= 0
5. Handle interior cases:
   - Check diagonal elements of delta2star3D matrix
   - Correct 3rd-order moments (S210, S201, S120, S021, S102, S012)
   - Correct S111
   - Apply S310_220 constraints
   - Compute lower bounds for S220, S202, S022
   - Recheck diagonal elements
   - Compute bound values for off-diagonal moments using bound_minor1
   - Apply corrections based on which diagonal elements are problematic
6. Iteratively refine until all constraints are satisfied

# Notes
- This is a 830-line function in MATLAB - the full port requires careful
  validation against test cases
- The current implementation provides the structure; full details need to be
  filled in by carefully translating each section of the MATLAB code
"""
function realizable_3D(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                       S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                       S211, S021, S121, S031, S012, S112, S013, S022)
    
    flag220 = 0
    diagmin = 1.0e-10
    h2min = 1.0e-10
    S2min = 1.0e-12
    
    # Compute H quantities
    H200 = max(eps(), S400 - S300^2 - 1)
    H020 = max(eps(), S040 - S030^2 - 1)
    H002 = max(eps(), S004 - S003^2 - 1)
    
    # Check maximum bounds on 4th-order moments
    A220 = sqrt((H200 + S300^2) * (H020 + S030^2))
    S220max = realizability_S220(S110, S220, A220)
    A202 = sqrt((H200 + S300^2) * (H002 + S003^2))
    S202max = realizability_S220(S101, S202, A202)
    A022 = sqrt((H020 + S030^2) * (H002 + S003^2))
    S022max = realizability_S220(S011, S022, A022)
    S220 = min(S220, S220max)
    S202 = min(S202, S202max)
    S022 = min(S022, S022max)
    
    # Check and correct realizability of S110, S101, S011
    S110, S101, S011, S2 = realizability_S2(S110, S101, S011)
    
    # Compute R quantities
    R110 = max(0.0, 1 - S110^2)
    R110 = sqrt(R110)
    R101 = max(0.0, 1 - S101^2)
    R101 = sqrt(R101)
    R011 = max(0.0, 1 - S011^2)
    R011 = sqrt(R011)
    
    if R110 * R101 * R011 == 0
        @warn "R110*R101*R011 == 0"
    end
    
    if S2 <= S2min  # Treat faces (degenerate cases)
        @info "Treating faces in realizable_3D"
        Rmax = max(R110, R101, R011)
        
        if R110 == Rmax  # Sij0 is known
            gam1 = (S101 - S110*S011) / (1 - S110^2)
            gam2 = (S011 - S110*S101) / (1 - S110^2)
            S021 = gam2*S030 + gam1*S120
            S201 = gam2*S210 + gam1*S300
            S031 = gam2*S040 + gam1*S130
            S301 = gam2*S310 + gam1*S400
            S111 = gam2*S120 + gam1*S210
            S211 = gam2*S220 + gam1*S310
            S121 = gam2*S130 + gam1*S220
            S012 = gam2^2*S030 + 2*gam1*gam2*S120 + gam1^2*S210
            S102 = gam2^2*S120 + 2*gam1*gam2*S210 + gam1^2*S300
            S022 = gam2^2*S040 + 2*gam1*gam2*S130 + gam1^2*S220
            S202 = gam2^2*S220 + 2*gam1*gam2*S310 + gam1^2*S400
            S112 = gam2^2*S130 + 2*gam1*gam2*S220 + gam1^2*S310
            S003 = gam2^3*S030 + 3*gam1*gam2^2*S120 + 3*gam1^2*gam2*S210 + gam1^3*S300
            S013 = gam2^3*S040 + 3*gam1*gam2^2*S130 + 3*gam1^2*gam2*S220 + gam1^3*S310
            S103 = gam2^3*S130 + 3*gam1*gam2^2*S220 + 3*gam1^2*gam2*S310 + gam1^3*S400
            S004 = gam2^4*S040 + 4*gam1*gam2^3*S130 + 6*gam1^2*gam2^2*S220 + 4*gam1^3*gam2*S310 + gam1^4*S400
            
        elseif R101 == Rmax  # Si0k is known
            gam3 = (S110 - S101*S011) / (1 - S101^2)
            gam4 = (S011 - S110*S101) / (1 - S101^2)
            S012 = gam4*S003 + gam3*S102
            S210 = gam4*S201 + gam3*S300
            S013 = gam4*S004 + gam3*S103
            S310 = gam4*S301 + gam3*S400
            S111 = gam4*S102 + gam3*S201
            S211 = gam4*S202 + gam3*S301
            S112 = gam4*S103 + gam3*S202
            S021 = gam4^2*S003 + 2*gam3*gam4*S102 + gam3^2*S201
            S120 = gam4^2*S102 + 2*gam3*gam4*S201 + gam3^2*S300
            S022 = gam4^2*S004 + 2*gam3*gam4*S103 + gam3^2*S202
            S220 = gam4^2*S202 + 2*gam3*gam4*S301 + gam3^2*S400
            S121 = gam4^2*S103 + 2*gam3*gam4*S202 + gam3^2*S301
            S030 = gam4^3*S003 + 3*gam3*gam4^2*S102 + 3*gam3^2*gam4*S201 + gam3^3*S300
            S031 = gam4^3*S004 + 3*gam3*gam4^2*S103 + 3*gam3^2*gam4*S202 + gam3^3*S301
            S130 = gam4^3*S103 + 3*gam3*gam4^2*S202 + 3*gam3^2*gam4*S301 + gam3^3*S400
            S040 = gam4^4*S004 + 4*gam3*gam4^3*S103 + 6*gam3^2*gam4^2*S202 + 4*gam3^3*gam4*S301 + gam3^4*S400
            
        else  # S0jk is known
            gam5 = (S101 - S110*S011) / (1 - S011^2)
            gam6 = (S110 - S101*S011) / (1 - S011^2)
            S102 = gam6*S003 + gam5*S012
            S120 = gam6*S021 + gam5*S030
            S103 = gam6*S004 + gam5*S013
            S130 = gam6*S031 + gam5*S040
            S111 = gam6*S012 + gam5*S021
            S121 = gam6*S022 + gam5*S031
            S112 = gam6*S013 + gam5*S022
            S201 = gam6^2*S003 + 2*gam5*gam6*S012 + gam5^2*S021
            S210 = gam6^2*S012 + 2*gam5*gam6*S021 + gam5^2*S030
            S202 = gam6^2*S004 + 2*gam5*gam6*S013 + gam5^2*S022
            S220 = gam6^2*S022 + 2*gam5*gam6*S031 + gam5^2*S040
            S211 = gam6^2*S013 + 2*gam5*gam6*S022 + gam5^2*S031
            S300 = gam6^3*S003 + 3*gam5*gam6^2*S012 + 3*gam5^2*gam6*S021 + gam5^3*S030
            S301 = gam6^3*S004 + 3*gam5*gam6^2*S013 + 3*gam5^2*gam6*S022 + gam5^3*S031
            S310 = gam6^3*S013 + 3*gam5*gam6^2*S022 + 3*gam5^2*gam6*S031 + gam5^3*S040
            S400 = gam6^4*S004 + 4*gam5*gam6^3*S013 + 6*gam5^2*gam6^2*S022 + 4*gam5^3*gam6*S031 + gam5^4*S040
        end
        
    else  # Treat interior of 2nd-order moment space
        beta = 1.0
        
        # Check diagonal elements of E1 matrix
        E1 = delta2star3D(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                          S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                          S211, S021, S121, S031, S012, S112, S013, S022)
        
        # Check and correct 3rd-order moments
        flagE11 = 0
        if E1[1,1] < 0
            S210, S201 = realizability_S210(S110, S101, S011, S300, S210, S201, H200, beta)
            flagE11 = 1
        end
        
        flagE44 = 0
        if E1[4,4] < 0
            S120, S021 = realizability_S210(S110, S011, S101, S030, S120, S021, H020, beta)
            flagE44 = 1
        end
        
        flagE66 = 0
        if E1[6,6] < 0
            S012, S102 = realizability_S210(S011, S101, S110, S003, S012, S102, H002, beta)
            flagE66 = 1
        end
        
        # Check and correct S111
        S111r = realizability_S111(S110, S101, S011, S210, S201, S120, S021, S102, S012, S111)
        if S111r != S111
            S111 = S111r
        end
        
        # Set minimum values based on S310
        S220_310 = realizability_S310_220(S110, S101, S011, S210, S120, S111, S220)
        S202_310 = realizability_S310_220(S101, S110, S011, S201, S102, S111, S202)
        S022_310 = realizability_S310_220(S011, S110, S101, S021, S012, S111, S022)
        
        # Lower bounds for positive diagonal elements
        S220_diag = S220
        S202_diag = S202
        S022_diag = S022
        
        S22min = lower_bound_S220(S011, S012, S021, S101, S102, S110, S111, S120, S201, S210)
        
        flagE22 = 0
        if S220 < S22min[1]
            S220_diag = S22min[1]
            flagE22 = 1
            flag220 = 1
        end
        
        flagE33 = 0
        if S202 < S22min[2]
            S202_diag = S22min[2]
            flagE33 = 1
            flag220 = 1
        end
        
        flagE55 = 0
        if S022 < S22min[3]
            S022_diag = S22min[3]
            flagE55 = 1
            flag220 = 1
        end
        
        # Recheck diagonal elements
        E1 = delta2star3D(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                          S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                          S211, S021, S121, S031, S012, S112, S013, S022)
        
        # Check for too small diagonal elements
        if E1[1,1] < diagmin; flagE11 = 1; end
        if E1[2,2] < diagmin; flagE22 = 1; end
        if E1[3,3] < diagmin; flagE33 = 1; end
        if E1[4,4] < diagmin; flagE44 = 1; end
        if E1[5,5] < diagmin; flagE55 = 1; end
        if E1[6,6] < diagmin; flagE66 = 1; end
        
        # Compute bound values for off-diagonal moments if needed
        if flagE11 == 1 || flagE22 == 1 || flagE33 == 1 || flagE44 == 1 || flagE55 == 1 || flagE66 == 1
            Mbound = bound_minor1(S003, S011, S012, S021, S030, S101, S102, S110, S111, S120, S201, S210, S300)
            
            # Extract bound values
            S310b = Mbound[1]
            S301b = Mbound[2]
            S130b = Mbound[3]
            S103b = Mbound[4]
            S031b = Mbound[5]
            S013b = Mbound[6]
            S211b = Mbound[10]
            S121b = Mbound[11]
            S112b = Mbound[12]
            
            # Apply corrections based on flags
            if flagE11 == 1
                S310 = S310b
                S301 = S301b
                S211 = S211b
            end
            if flagE22 == 1
                S310 = S310b
                S130 = S130b
                S211 = S211b
                S121 = S121b
                S112 = S112b
            end
            if flagE33 == 1
                S301 = S301b
                S103 = S103b
                S211 = S211b
                S121 = S121b
                S112 = S112b
            end
            if flagE44 == 1
                S130 = S130b
                S031 = S031b
                S121 = S121b
            end
            if flagE55 == 1
                S031 = S031b
                S013 = S013b
                S211 = S211b
                S121 = S121b
                S112 = S112b
            end
            if flagE66 == 1
                S103 = S103b
                S013 = S013b
                S112 = S112b
            end
        end
        
        # Return if all off-diagonal terms have been fixed
        if flagE11 == 1 && flagE22 == 1 && flagE33 == 1 && flagE44 == 1 && flagE55 == 1 && flagE66 == 1
            return S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                   S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                   S211, S021, S121, S031, S012, S112, S013, S022, flag220
        end
        
        ## ITERATIVE REFINEMENT: Check 2x2 minors and apply realizability_S310
        S220a = S220
        S220b = S220
        S202a = S202
        S202b = S202
        S022a = S022
        S022b = S022
        
        # Recompute E matrices with permutations
        E1, E2, E3, E4, E5, E6 = delta2star3D_permutation(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                                                           S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                                                           S211, S021, S121, S031, S012, S112, S013, S022)
        
        flag310 = 0
        
        # Check E1 and E4 (2x2 minor)
        if flagE11 == 0 && flagE44 == 0
            if det(E1[1:2, 1:2]) < h2min
                S310, S220a = realizability(:S310, S110, S101, S011, S300, S210, S120, S201, S111, S310, S220, H200, beta)
                flag310 = 1
            end
        end
        
        # Check E2 and E2 (2x2 minor)
        if flagE22 == 0 && flagE44 == 0
            if det(E3[1:2, 1:2]) < h2min
                S130, S220b = realizability(:S310, S110, S011, S101, S030, S120, S021, S210, S111, S130, S220, H020, beta)
                flag310 = 1
            end
        end
        
        # Check E1 and E3 (2x2 minor)
        if flagE11 == 0 && flagE33 == 0
            if det(E2[1:2, 1:2]) < h2min
                S301, S202a = realizability(:S310, S101, S110, S011, S300, S201, S210, S102, S111, S301, S202, H200, beta)
                flag310 = 1
            end
        end
        
        # Check E3 and E6 (2x2 minor)
        if flagE33 == 0 && flagE66 == 0
            if det(E4[1:2, 1:2]) < h2min
                S103, S202b = realizability(:S310, S101, S011, S110, S003, S102, S012, S201, S111, S103, S202, H002, beta)
                flag310 = 1
            end
        end
        
        # Check E4 and E5 (2x2 minor)
        if flagE44 == 0 && flagE55 == 0
            if det(E5[1:2, 1:2]) < h2min
                S031, S022a = realizability(:S310, S011, S110, S101, S030, S021, S120, S012, S111, S031, S022, H020, beta)
                flag310 = 1
            end
        end
        
        # Check E5 and E6 (2x2 minor)
        if flagE55 == 0 && flagE66 == 0
            if det(E6[1:2, 1:2]) < h2min
                S013, S022b = realizability(:S310, S011, S101, S110, S003, S012, S102, S021, S111, S013, S022, H002, beta)
                flag310 = 1
            end
        end
        
        # Take maximum of S220 corrections
        S220r = max(S220a, S220b)
        S220_310r = S220_310
        if S220 < S220r
            S220_310r = S220r
            flag220 = 1
        end
        
        S022r = max(S022a, S022b)
        S022_310r = S022_310
        if S022 < S022r
            S022_310r = S022r
            flag220 = 1
        end
        
        S202r = max(S202a, S202b)
        S202_310r = S202_310
        if S202 < S202r
            S202_310r = S202r
            flag220 = 1
        end
        
        # If moments have been corrected, recompute matrices
        if flag310 == 1
            E1, E2, E3, E4, E5, E6 = delta2star3D_permutation(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                                                               S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                                                               S211, S021, S121, S031, S012, S112, S013, S022)
            if minimum([det(E1[1:2, 1:2]), det(E2[1:2, 1:2]), det(E3[1:2, 1:2]), 
                       det(E4[1:2, 1:2]), det(E5[1:2, 1:2]), det(E6[1:2, 1:2])]) < 0
                flag220 = 1
            end
        end
        
        ## Check 3x3 minors and apply realizability_S211
        S220a = S220
        S220b = S220
        S202a = S202
        S202c = S202
        S022b = S022
        S022c = S022
        
        # Check E1 and E2 (3x3 minor) - S211
        if flagE11 == 0 && flagE22 == 0 && flagE33 == 0 && flagE55 == 0
            if det(E1[1:3, 1:3]) <= h2min || det(E2[1:3, 1:3]) <= h2min
                e11a = E1[1,1]
                e12a = E1[1,2]
                e13a = E1[1,3]
                e22a = E1[2,2]
                e23a = E1[2,3]
                e33a = E1[3,3]
                d23a = S211 - e23a
                d22a = S220 - e22a
                d33a = S202 - e33a
                if e11a > diagmin
                    S220a = d22a + e13a^2 / e11a
                    S202a = d33a + e12a^2 / e11a
                end
                S211 = realizability(:S211, e11a, e22a, e33a, e12a, e13a, d23a, S211, beta)
                flag220 = 1
            end
        end
        
        # Check E3 and E5 (3x3 minor) - S121
        if flagE22 == 0 && flagE33 == 0 && flagE44 == 0 && flagE55 == 0
            if det(E3[1:3, 1:3]) < h2min || det(E5[1:3, 1:3]) < h2min
                e11b = E3[1,1]
                e12b = E3[1,2]
                e13b = E3[1,3]
                e22b = E3[2,2]
                e23b = E3[2,3]
                e33b = E3[3,3]
                d23b = S121 - e23b
                d22b = S220 - e22b
                d33b = S022 - e33b
                if e11b > diagmin
                    S220b = d22b + e13b^2 / e11b
                    S022b = d33b + e12b^2 / e11b
                end
                S121 = realizability(:S211, e11b, e22b, e33b, e12b, e13b, d23b, S121, beta)
                flag220 = 1
            end
        end
        
        # Check E4 and E6 (3x3 minor) - S112
        if flagE22 == 0 && flagE33 == 0 && flagE55 == 0 && flagE66 == 0
            if det(E4[1:3, 1:3]) < h2min || det(E6[1:3, 1:3]) < h2min
                e11c = E6[1,1]
                e12c = E6[1,2]
                e13c = E6[1,3]
                e22c = E6[2,2]
                e23c = E6[2,3]
                e33c = E6[3,3]
                d23c = S112 - e23c
                d22c = S022 - e22c
                d33c = S202 - e33c
                if e11c > diagmin
                    S202c = d22c + e13c^2 / e11c
                    S022c = d33c + e12c^2 / e11c
                end
                S112 = realizability(:S211, e11c, e22c, e33c, e12c, e13c, d23c, S112, beta)
                flag220 = 1
            end
        end
        
        S220_211 = max(S220a, S220b)
        S202_211 = max(S202a, S202c)
        S022_211 = max(S022b, S022c)
        
        ## Check 4x4 minors using rootsR_X_Y to find bounds
        
        # Recompute E1 and E3 for 4x4 checks
        E1 = delta2star3D(S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
                          S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
                          S211, S021, S121, S031, S012, S112, S013, S022)
        
        E3 = delta2star3D(S030, S040, S110, S120, S130, S210, S220, S300, S310, S400,
                          S011, S021, S031, S012, S022, S003, S013, S004, S101, S111,
                          S121, S201, S211, S301, S102, S112, S103, S202)
        
        s220mina = S220
        s220maxa = S220
        if det(E1[1:4, 1:4]) < 0
            flag220 = 1
            E = E1
            e11, e12, e13, e14 = E[1,1], E[1,2], E[1,3], E[1,4]
            e22, e23, e24 = E[2,2], E[2,3], E[2,4]
            e33, e34 = E[3,3], E[3,4]
            e44 = E[4,4]
            
            ex = -e22 + e14
            d22 = S220 - e22
            Y = e33
            Ra = rootsR_X_Y(Y, e11, e12, e13, e23, e24, e34, e44, ex)
            Rr = sort(real(Ra))
            if maximum(abs.(imag(Ra))) / maximum(abs.(Ra)) < 1000*eps()
                s220mina = Rr[2] + d22
                s220maxa = Rr[3] + d22
            end
        end
        
        s220minb = S220
        s220maxb = S220
        if det(E3[1:4, 1:4]) < 0
            flag220 = 1
            E = E3
            e11, e12, e13, e14 = E[1,1], E[1,2], E[1,3], E[1,4]
            e22, e23, e24 = E[2,2], E[2,3], E[2,4]
            e33, e34 = E[3,3], E[3,4]
            e44 = E[4,4]
            
            ex = -e22 + e14
            d22 = S220 - e22
            Y = e33
            Ra = rootsR_X_Y(Y, e11, e12, e13, e23, e24, e34, e44, ex)
            Rr = sort(real(Ra))
            if maximum(abs.(imag(Ra))) / maximum(abs.(Ra)) < 1000*eps()
                s220minb = Rr[2] + d22
                s220maxb = Rr[3] + d22
            end
        end
        
        # Recompute E2 and E4 for S202
        E2 = delta2star3D(S300, S400, S101, S201, S301, S102, S202, S003, S103, S004,
                          S110, S210, S310, S120, S220, S030, S130, S040, S011, S111,
                          S211, S012, S112, S013, S021, S121, S031, S022)
        
        E4 = delta2star3D(S003, S004, S101, S102, S103, S201, S202, S300, S301, S400,
                          S011, S012, S013, S021, S022, S030, S031, S040, S110, S111,
                          S112, S210, S211, S310, S120, S121, S130, S220)
        
        s202mina = S202
        s202maxa = S202
        if det(E2[1:4, 1:4]) < 0
            flag220 = 1
            E = E2
            e11, e12, e13, e14 = E[1,1], E[1,2], E[1,3], E[1,4]
            e22, e23, e24 = E[2,2], E[2,3], E[2,4]
            e33, e34 = E[3,3], E[3,4]
            e44 = E[4,4]
            
            ex = -e22 + e14
            d22 = S202 - e22
            Y = e33
            Ra = rootsR_X_Y(Y, e11, e12, e13, e23, e24, e34, e44, ex)
            Rr = sort(real(Ra))
            if maximum(abs.(imag(Ra))) / maximum(abs.(Ra)) < 1000*eps()
                s202mina = Rr[2] + d22
                s202maxa = Rr[3] + d22
            end
        end
        
        s202minb = S202
        s202maxb = S202
        if det(E4[1:4, 1:4]) < 0
            flag220 = 1
            E = E4
            e11, e12, e13, e14 = E[1,1], E[1,2], E[1,3], E[1,4]
            e22, e23, e24 = E[2,2], E[2,3], E[2,4]
            e33, e34 = E[3,3], E[3,4]
            e44 = E[4,4]
            
            ex = -e22 + e14
            d22 = S202 - e22
            Y = e33
            Ra = rootsR_X_Y(Y, e11, e12, e13, e23, e24, e34, e44, ex)
            Rr = sort(real(Ra))
            if maximum(abs.(imag(Ra))) / maximum(abs.(Ra)) < 1000*eps()
                s202minb = Rr[2] + d22
                s202maxb = Rr[3] + d22
            end
        end
        
        # Recompute E5 and E6 for S022
        E5 = delta2star3D(S030, S040, S011, S021, S031, S012, S022, S003, S013, S004,
                          S110, S120, S130, S210, S220, S300, S310, S400, S101, S111,
                          S121, S102, S112, S103, S201, S211, S301, S202)
        
        E6 = delta2star3D(S003, S004, S011, S012, S013, S021, S022, S030, S031, S040,
                          S101, S102, S103, S201, S202, S300, S301, S400, S110, S111,
                          S112, S120, S121, S130, S210, S211, S310, S220)
        
        s022mina = S022
        s022maxa = S022
        if det(E5[1:4, 1:4]) < 0
            flag220 = 1
            E = E5
            e11, e12, e13, e14 = E[1,1], E[1,2], E[1,3], E[1,4]
            e22, e23, e24 = E[2,2], E[2,3], E[2,4]
            e33, e34 = E[3,3], E[3,4]
            e44 = E[4,4]
            
            ex = -e22 + e14
            d22 = S022 - e22
            Y = e33
            Ra = rootsR_X_Y(Y, e11, e12, e13, e23, e24, e34, e44, ex)
            Rr = sort(real(Ra))
            if maximum(abs.(imag(Ra))) / maximum(abs.(Ra)) < 1000*eps()
                s022mina = Rr[2] + d22
                s022maxa = Rr[3] + d22
            end
        end
        
        s022minb = S022
        s022maxb = S022
        if det(E6[1:4, 1:4]) < 0
            flag220 = 1
            E = E6
            e11, e12, e13, e14 = E[1,1], E[1,2], E[1,3], E[1,4]
            e22, e23, e24 = E[2,2], E[2,3], E[2,4]
            e33, e34 = E[3,3], E[3,4]
            e44 = E[4,4]
            
            ex = -e22 + e14
            d22 = S022 - e22
            Y = e33
            Ra = rootsR_X_Y(Y, e11, e12, e13, e23, e24, e34, e44, ex)
            Rr = sort(real(Ra))
            if maximum(abs.(imag(Ra))) / maximum(abs.(Ra)) < 1000*eps()
                s022minb = Rr[2] + d22
                s022maxb = Rr[3] + d22
            end
        end
        
        # Final bounds: take maximum of all lower bounds, minimum of all upper bounds
        S220 = maximum([s220mina, s220minb, S220_diag, S220_310, S220_310r, S220_211])
        S220 = minimum([S220, S220max, s220maxa, s220maxb])
        
        S202 = maximum([s202mina, s202minb, S202_diag, S202_310, S202_310r, S202_211])
        S202 = minimum([S202, S202max, s202maxa, s202maxb])
        
        S022 = maximum([s022mina, s022minb, S022_diag, S022_310, S022_310r, S022_211])
        S022 = minimum([S022, S022max, s022maxa, s022maxb])
    end
    
    return S300, S400, S110, S210, S310, S120, S220, S030, S130, S040,
           S101, S201, S301, S102, S202, S003, S103, S004, S011, S111,
           S211, S021, S121, S031, S012, S112, S013, S022, flag220
end
