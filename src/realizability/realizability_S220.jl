"""
    realizability_S220(S110, S220, A220)

Check maximum bounds and correct S220.

# Arguments
- S110: Second-order standardized moment
- S220: Fourth-order moment to correct
- A220: Bound parameter (typically sqrt((H200+S300^2)*(H020+S030^2)))

# Returns
- S220r: Corrected S220

# Algorithm
Clamps S220 to the range [s220min, s220max] where:
- s220min = max(S110^2, 1 - A220)
- s220max = 1 + A220
"""
@fastmath function realizability_S220(S110, S220, A220)
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
