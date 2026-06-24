using LinearAlgebra
using Random
using Printf
using DelimitedFiles

const SRC = "/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/src"
include(joinpath(SRC, "autogen/delta2star3D.jl"))
include(joinpath(SRC, "autogen/M4toC4_3D.jl"))
include(joinpath(SRC, "autogen/C4toM4_3D.jl"))
include(joinpath(SRC, "moments/M2CS4_35.jl"))
include(joinpath(SRC, "moments/hyqmom_3D.jl"))
include(joinpath(SRC, "realizability/realizablity_S220.jl"))
include(joinpath(SRC, "realizability/realizability_S2.jl"))

Random.seed!(12345)
OUT = "/tmp/kdiff"

# Exponent triples for the 35-moment ordering used throughout the code
triples = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),
 (0,2,0),(1,2,0),(2,2,0),
 (0,3,0),(1,3,0),
 (0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),
 (0,0,2),(1,0,2),(2,0,2),
 (0,0,3),(1,0,3),
 (0,0,4),
 (0,1,1),(1,1,1),(2,1,1),
 (0,2,1),(1,2,1),
 (0,3,1),
 (0,1,2),(1,1,2),
 (0,1,3),
 (0,2,2)]

# Build a realizable M4 from particle samples
function make_M4(Npart)
    rho = 0.5 + rand()                       # positive density
    mu  = randn(3) .* 0.7                     # mean velocity (nonzero -> exercises shifts)
    A   = randn(3,3) .* 0.6 + I*1.0           # random covariance factor
    V   = A*randn(3,Npart) .+ mu              # 3 x Npart samples
    M4 = zeros(35)
    for n in 1:35
        i,j,k = triples[n]
        M4[n] = rho * sum(@. V[1,:]^i * V[2,:]^j * V[3,:]^k) / Npart
    end
    return M4
end

# Extract the 28 standardized moments (delta2star/hyqmom argument order) from S4
function s28(S4)
    return (S4[4],S4[5],S4[7],S4[8],S4[9],S4[11],S4[12],S4[13],S4[14],S4[15],
            S4[17],S4[18],S4[19],S4[21],S4[22],S4[23],S4[24],S4[25],S4[26],S4[27],
            S4[28],S4[29],S4[30],S4[31],S4[32],S4[33],S4[34],S4[35])
end

NCASE = 60
M4mat = zeros(NCASE, 35)
C4mat = zeros(NCASE, 35)
S4mat = zeros(NCASE, 35)
Emat  = zeros(NCASE, 36)   # flattened 6x6
hyq   = zeros(NCASE, 21)
roundtrip = zeros(NCASE, 35)  # C4toM4_3D(M2CS4 central) -> should recover M4

for c in 1:NCASE
    M4 = make_M4(4000)
    M4mat[c,:] = M4
    C4, S4 = M2CS4_35(M4)
    C4mat[c,:] = C4
    S4mat[c,:] = S4
    args = s28(S4)
    E = delta2star3D(args...)
    Emat[c,:] = vec(E)
    hyq[c,:] = collect(hyqmom_3D(args...))
    # round-trip: rebuild raw moments from central moments
    M000=M4[1]; um=M4[2]/M000; vm=M4[6]/M000; wm=M4[16]/M000
    C=C4
    Mr = C4toM4_3D(M000,um,vm,wm, C[3],C[7],C[17],C[10],C[26],C[20],
                   C[4],C[8],C[18],C[11],C[27],C[21],C[13],C[29],C[32],C[23],
                   C[5],C[9],C[19],C[12],C[28],C[22],C[14],C[30],C[33],C[24],C[15],C[31],C[35],C[34],C[25])
    # C4toM4_3D returns a 5x5x5 array; pull the 35 in order
    for n in 1:35
        i,j,k = triples[n]
        roundtrip[c,n] = Mr[i+1,j+1,k+1]
    end
end

writedlm(joinpath(OUT,"M4.txt"), M4mat, ' ')
writedlm(joinpath(OUT,"jl_C4.txt"), C4mat, ' ')
writedlm(joinpath(OUT,"jl_S4.txt"), S4mat, ' ')
writedlm(joinpath(OUT,"jl_E.txt"), Emat, ' ')
writedlm(joinpath(OUT,"jl_hyq.txt"), hyq, ' ')
writedlm(joinpath(OUT,"jl_roundtrip.txt"), roundtrip, ' ')

# --- realizability_S2: random (S110,S101,S011), exercise S2<0 branch (fzero vs bisection) ---
Random.seed!(777)
NS2 = 400
s2in = zeros(NS2,3); s2out = zeros(NS2,4)
for c in 1:NS2
    a = (rand()*2.4-1.2); b=(rand()*2.4-1.2); d=(rand()*2.4-1.2)  # may exceed [-1,1]
    a=clamp(a,-1,1); b=clamp(b,-1,1); d=clamp(d,-1,1)            # mimic pre-clamp in realizable_3D.m
    s2in[c,:] = [a,b,d]
    r = realizability_S2(a,b,d)
    s2out[c,:] = collect(r)
end
writedlm(joinpath(OUT,"s2_in.txt"), s2in, ' ')
writedlm(joinpath(OUT,"jl_s2_out.txt"), s2out, ' ')

# --- realizablity_S220: random (S110,S220,A220) ---
Random.seed!(999)
N220=400
s220in=zeros(N220,3); s220out=zeros(N220,1)
for c in 1:N220
    S110=rand()*2-1; A220=rand()*2.0; S220=rand()*3-0.5
    s220in[c,:]=[S110,S220,A220]
    s220out[c,1]=realizablity_S220(S110,S220,A220)
end
writedlm(joinpath(OUT,"s220_in.txt"), s220in, ' ')
writedlm(joinpath(OUT,"jl_s220_out.txt"), s220out, ' ')

println("Julia outputs written. NCASE=$NCASE NS2=$NS2 N220=$N220")
