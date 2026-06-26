"""
Realizability module - consolidated utility functions for moment realizability checks.

This module acts as a dispatcher for various realizability helpers:
- `realizable_2D`: Find realizable moment set in 2D
- `realizable_3D`: Check and correct realizability of cross moments in 3D
- `realizability_S111`: Check and correct realizability of S111
- `realizability_S2`: Check and correct realizability of 2nd-order moments
- `realizability_S210`: Check and correct realizability of S210, S201
- `realizability_S211`: Check and correct realizability of S211
- `realizability_S310`: Check and correct realizability of S310 and S220
- `realizability_S310_220`: Check and correct realizability of S220 for S310
- `realizability_S220`: Check maximum bounds and correct S220
"""

# Include all realizability sub-functions
include("realizability_S111.jl")
include("realizability_S2.jl")
include("realizability_S210.jl")
include("realizability_S211.jl")
include("realizability_S310.jl")
include("realizability_S310_220.jl")
include("realizability_S220.jl")
include("realizable_2D.jl")
include("realizable_3D.jl")
# Revised moment-projection method (Appendix B): projection35 + M4 wrapper
include("projection35.jl")
include("realize_M4_projection.jl")

"""
    realizability(operation::Symbol, args...)

Dispatcher for realizability operations.

# Operations
- `:2D` - Find realizable moment set in 2D (5 outputs)
- `:3D` - Check and correct realizability of cross moments in 3D (29 outputs)
- `:S111` - Check and correct realizability of S111 (1 output)
- `:S2` - Check and correct realizability of 2nd-order moments (4 outputs)
- `:S210` - Check and correct realizability of S210, S201 (2 outputs)
- `:S211` - Check and correct realizability of S211 (1 output)
- `:S310` - Check and correct realizability of S310 and S220 (2 outputs)
- `:S310_220` - Check and correct realizability of S220 for S310 (1 output)
- `:S220` - Check maximum bounds and correct S220 (1 output)

# Examples
```julia
# 2D realizability
S21, S12, S31, S22, S13 = realizability(:2D, S30, S40, S11, S21, S31, S12, S22, S03, S13, S04)

# 3D realizability
S300, ..., flag220 = realizability(:3D, S300, ..., S022)

# S111 correction
S111r = realizability(:S111, S110, S101, S011, S210, S201, S120, S021, S102, S012, S111)

# S2 correction
S110r, S101r, S011r, S2r = realizability(:S2, S110, S101, S011)
```
"""
function realizability(operation::Symbol, args...)
    if operation == Symbol("2D") || operation == Symbol("2d")
        return realizable_2D(args...)
    elseif operation == Symbol("3D") || operation == Symbol("3d")
        return realizable_3D(args...)
    elseif operation == :S111 || operation == :s111
        return realizability_S111(args...)
    elseif operation == :S2 || operation == :s2
        return realizability_S2(args...)
    elseif operation == :S210 || operation == :s210
        return realizability_S210(args...)
    elseif operation == :S211 || operation == :s211
        return realizability_S211(args...)
    elseif operation == :S310 || operation == :s310
        return realizability_S310(args...)
    elseif operation == :S310_220 || operation == :s310_220
        return realizability_S310_220(args...)
    elseif operation == :S220 || operation == :s220
        return realizability_S220(args...)
    else
        error("realizability: Unknown operation: $operation")
    end
end
