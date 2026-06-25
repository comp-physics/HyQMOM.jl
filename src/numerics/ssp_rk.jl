"""
    ssp_rk3_step(M, dt, L)

One 3-stage strong-stability-preserving RK3 update of state `M` with residual
operator `L(M) -> dM/dt`. Works for scalars or arrays.
"""
function ssp_rk3_step(M, dt, L)
    k0 = L(M)
    M1 = M .+ dt .* k0
    k1 = L(M1)
    M2 = (3/4) .* M .+ (1/4) .* (M1 .+ dt .* k1)
    k2 = L(M2)
    return (1/3) .* M .+ (2/3) .* (M2 .+ dt .* k2)
end
