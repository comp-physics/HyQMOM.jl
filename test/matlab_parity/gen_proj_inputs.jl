# Generate test inputs for projection35 + realizable_3D_M4 validation.
# Mix of realizable (pass-through) and deliberately unrealizable (trigger) moment vectors.
using LinearAlgebra, Random, DelimitedFiles
Random.seed!(2026)
OUT="/tmp/kdiff"

triples = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),
 (0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),
 (0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),
 (1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

function make_M4(Npart, jitter)
    rho = 0.5 + rand()
    mu  = randn(3) .* 0.8
    A   = randn(3,3) .* 0.7 + I*1.0
    V   = A*randn(3,Npart) .+ mu
    M4 = zeros(35)
    for n in 1:35
        i,j,k = triples[n]
        M4[n] = rho * sum(@. V[1,:]^i * V[2,:]^j * V[3,:]^k) / Npart
    end
    # Optionally corrupt 4th-order moments to push outside the realizable set
    if jitter > 0
        for n in 1:35
            i,j,k = triples[n]
            if i+j+k == 4
                M4[n] *= (1.0 - jitter*rand())   # shrink kurtosis-type moments -> unrealizable
            end
        end
    end
    return M4
end

NR = 40   # realizable
NU = 40   # unrealizable (jittered)
M4mat = zeros(NR+NU, 35)
for c in 1:NR;  M4mat[c,:]      = make_M4(5000, 0.0); end
for c in 1:NU;  M4mat[NR+c,:]   = make_M4(5000, 0.5 + 0.4*rand()); end
writedlm(joinpath(OUT,"proj_M4.txt"), M4mat, ' ')
println("wrote $(NR+NU) M4 vectors ($NR realizable, $NU jittered)")
