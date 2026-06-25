# 1D moment shock tube: compare first-order vs MUSCL-2 numerical diffusion.
# Usage: HYQMOM_SKIP_PLOTTING=true CI=true julia --project=. examples/run_1d_highorder.jl
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, Printf

Ma = 0.0; N = 200; dx = 1.0/N; tfinal = 0.1
function ic()
    M = zeros(N, 35)
    for i in 1:N
        rho = (i <= N÷2) ? 1.0 : 0.125
        M[i, :] = InitializeM4_35(rho, 0.0,0.0,0.0, 1.0,0.0,0.0,1.0,0.0,1.0)
    end
    return M
end
function advance(order)
    M = ic(); dt = 0.2*dx/3.0; nsteps = ceil(Int, tfinal/dt); dt = tfinal/nsteps
    L(x) = residual_1d(x, dx, Ma; order=order)
    for _ in 1:nsteps; M = ssp_rk3_step(M, dt, L); end
    return M
end
M1 = advance(1); M2 = advance(2)
# sharpness metric: max density gradient (higher = less diffused)
g1 = maximum(abs.(diff(M1[:,1]))); g2 = maximum(abs.(diff(M2[:,1])))
@printf("max |drho/dx|: first-order=%.4f  MUSCL-2=%.4f  (ratio %.2fx sharper)\n", g1, g2, g2/g1)
@printf("density range first-order=[%.4f,%.4f]  MUSCL-2=[%.4f,%.4f]\n",
        minimum(M1[:,1]),maximum(M1[:,1]), minimum(M2[:,1]),maximum(M2[:,1]))
@printf("mass conserved: first=%.3e  muscl=%.3e (rel drift)\n",
        abs(sum(M1[:,1])-sum(ic()[:,1]))/sum(ic()[:,1]),
        abs(sum(M2[:,1])-sum(ic()[:,1]))/sum(ic()[:,1]))
