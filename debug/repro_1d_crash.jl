ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"; ENV["HO_DEBUG"]="1"
using HyQMOM, Printf

Ma   = parse(Float64, get(ENV,"R1D_MA","100.0"))
N    = parse(Int,     get(ENV,"R1D_N","512"))
nsmax= parse(Int,     get(ENV,"R1D_NS","400"))
order= parse(Int,     get(ENV,"R1D_ORDER","2"))
rhol = 1.0; rhor = 0.001; T = 1.0
Uc   = Ma/sqrt(2.0)

dx = 1.0/N
xc = [(i-0.5)*dx for i in 1:N]

# Build a 35-moment state: density rho, x-velocity U, isotropic variance T
state(rho,U) = InitializeM4_35(rho, U, 0.0, 0.0, T, 0.0, 0.0, T, 0.0, T)

# Colliding-jets-with-vacuum IC (1D analog of the crossing): two dense slabs move
# toward each other through a near-vacuum background. Reproduces both the strong
# central compression AND the sharp jet/vacuum edges of the 3D crossing.
M = zeros(N, 35)
for i in 1:N
    x = xc[i]
    if 0.25 <= x < 0.5
        M[i,:] = state(rhol, +Uc)
    elseif 0.5 <= x < 0.75
        M[i,:] = state(rhol, -Uc)
    else
        M[i,:] = state(rhor, 0.0)
    end
end

# fixed dt from a generous wave-speed bound (Uc + thermal tail)
vmax = Uc + 4.0*2.334*sqrt(T)
dt = (1/3)*dx/vmax
@printf("1D repro: Ma=%.0f N=%d order=%d  Uc=%.3f dx=%.3e dt=%.3e\n", Ma,N,order,Uc,dx,dt)

project_cells!(U) = (for i in 1:size(U,1); U[i,:] = realizable_3D_M4(U[i,:], Ma); end)

L(U) = residual_1d(U, dx, Ma; order=order, bc=:outflow)

# Report the first non-finite cell in R (a residual) and dump the reconstruction
# stencil from the state U that produced it, to locate the BIRTH of the NaN/Inf.
function check_residual(R, U, tag)
    for i in 1:size(R,1)
        if !all(isfinite, @view R[i,:])
            @printf("\n>>> FIRST non-finite residual at %s, cell i=%d (rho_cell=%.4e)\n", tag, i, U[i,1])
            for j in max(i-2,1):min(i+2,size(U,1))
                m = @view U[j,:]
                # central variance of this stencil cell
                C4,_ = M2CS4_35(collect(m))
                @printf("    stencil cell %d: rho=%.4e u=%.3f C200=%.4e C020=%.4e C002=%.4e finite=%s\n",
                        j, m[1], m[2]/m[1], C4[3], C4[10], C4[20], all(isfinite,m))
            end
            # show the reconstructed faces around cell i (right face of i-1, both faces of i)
            for ii in (i-1, i)
                (1 <= ii-1 && ii+1 <= size(U,1)) || continue
                Vl = muscl_faces(to_recon_vars(U[ii-1,:]), to_recon_vars(U[ii,:]), to_recon_vars(U[ii+1,:]))
                Lface = from_recon_vars(Vl[2]); Rface = from_recon_vars(Vl[1])
                @printf("    cell %d recon faces: rho_Lface=%.4e rho_Rface=%.4e finiteL=%s finiteR=%s\n",
                        ii, Lface[1], Rface[1], all(isfinite,Lface), all(isfinite,Rface))
            end
            return true
        end
    end
    return false
end

function step!(M, dt, n)
    R0 = L(M); check_residual(R0, M, "step $n stage1 L(M)") && (get(ENV,"R1D_NOFATAL","")!="1" && error("non-finite residual born"))
    M1 = M .+ dt.*R0;                            project_cells!(M1)
    R1 = L(M1); check_residual(R1, M1, "step $n stage2 L(M1)") && (get(ENV,"R1D_NOFATAL","")!="1" && error("non-finite residual born"))
    M2 = 0.75.*M .+ 0.25.*(M1 .+ dt.*R1);       project_cells!(M2)
    R2 = L(M2); check_residual(R2, M2, "step $n stage3 L(M2)") && (get(ENV,"R1D_NOFATAL","")!="1" && error("non-finite residual born"))
    M3 = (1/3).*M .+ (2/3).*(M2 .+ dt.*R2);     project_cells!(M3)
    M .= M3
end

t = 0.0
for n in 1:nsmax
    try
        step!(M, dt, n)
    catch e
        @printf("\n*** CRASH at step %d (t=%.6e) ***\n", n, t)
        @printf("exception: %s\n", sprint(showerror, e))
        rho = M[:,1]
        @printf("pre-crash field rho[min,max]=[%.4e,%.4e]\n", minimum(rho), maximum(rho))
        rethrow(e)
    end
    global t += dt
    if n % 20 == 0
        rho = M[:,1]
        @printf("step %4d t=%.5e rho[min,max]=[%.4e,%.4e]\n", n,t,minimum(rho),maximum(rho))
    end
end
@printf("COMPLETED %d steps (no crash). final rho[min,max]=[%.4e,%.4e]\n",
        nsmax, minimum(M[:,1]), maximum(M[:,1]))
