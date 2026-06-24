using LinearAlgebra, Random, DelimitedFiles
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM
OUT="/tmp/kdiff"

triples = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),
 (0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),
 (0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),
 (1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

# ---- A) full wrapper realizable_3D_M4 ----
M4mat = readdlm(joinpath(OUT,"proj_M4.txt"))
N = size(M4mat,1)
for Ma in (2.0, 5.0)
    R = zeros(N,35)
    for c in 1:N
        R[c,:] = realizable_3D_M4(M4mat[c,:], Ma)
    end
    writedlm(joinpath(OUT,"jl_M4r_Ma$(Int(Ma)).txt"), R, ' ')
end

# ---- B) projection35 in isolation, on triggering 28-vectors ----
Random.seed!(4242)
sidx=[4,5,7,8,9,11,12,13,14,15,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
function s28_from_M4(M4)
    _, S4 = M2CS4_35(M4)
    return S4[sidx]
end
function rho_mu_A(jit)
    rho=0.5+rand(); mu=randn(3).*0.8; A=randn(3,3).*0.7+I*1.0
    V=A*randn(3,5000) .+ mu
    M4=zeros(35)
    for n in 1:35; i,j,k=triples[n]; M4[n]=rho*sum(@. V[1,:]^i*V[2,:]^j*V[3,:]^k)/5000; end
    if jit>0
        for n in 1:35; i,j,k=triples[n]; if i+j+k==4; M4[n]*=(1-jit*rand()); end; end
    end
    M4
end
M=300
inp=zeros(M,28); out=zeros(M,28); trig=falses(M)
for c in 1:M
    v = s28_from_M4(rho_mu_A(0.3+0.6*rand()))
    inp[c,:]=v
    # was it triggered? (min eig of delta2star3D < 0)
    E=delta2star3D(v...); trig[c]= sort(real(eigvals(E)))[1] < 0
    out[c,:]=collect(projection35(v...))
end
writedlm(joinpath(OUT,"proj28_in.txt"), inp, ' ')
writedlm(joinpath(OUT,"jl_proj28_out.txt"), out, ' ')
writedlm(joinpath(OUT,"proj28_trig.txt"), Int.(trig), ' ')
println("wrapper: $N M4 vectors x {Ma2,Ma5}; projection35 isolation: $M cases, $(sum(trig)) triggered")
