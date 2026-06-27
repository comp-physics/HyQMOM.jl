import Pkg; Pkg.activate(@__DIR__)
using CUDA, CUDA.CUSOLVER, LinearAlgebra, Printf, Random

const N = 6
const B = parse(Int, get(ENV, "NBATCH", "2097152"))   # 128^3
const CHUNK = 1_000_000
Random.seed!(1)

println("building $B symmetric $N×$N matrices..."); flush(stdout)
Acpu = Array{Float64,3}(undef, N, N, B)
@inbounds for k in 1:B
    M = randn(N, N); M = (M + M') / 2
    Acpu[:, :, k] = M
end

# ---- CPU baseline: LAPACK smallest eigenvalue per matrix ----
function cpu_mineig(A, rng)
    out = Vector{Float64}(undef, length(rng)); buf = Matrix{Float64}(undef, N, N)
    @inbounds for (o,k) in enumerate(rng)
        for j in 1:N, i in 1:N; buf[i,j] = A[i,j,k]; end
        out[o] = eigvals(Symmetric(buf))[1]
    end
    out
end
cpu_mineig(Acpu, 1:1000)                          # warmup
t_cpu = @elapsed wc = cpu_mineig(Acpu, 1:B)
@printf("CPU  LAPACK 6x6 mineig: %d mats in %.3f s  (%.3f Mmat/s)\n", B, t_cpu, B/t_cpu/1e6); flush(stdout)

# ---- GPU: cuSOLVER batched symmetric eigensolver, chunked ----
gpu_mineig(Achunk) = Array(CUSOLVER.syevjBatched!('N','U', CuArray(Achunk))[1, :])  # smallest eig
function gpu_all!(wg, A, B, chunk)
    i = 1
    while i <= B
        j = min(i+chunk-1, B)
        wg[i:j] = gpu_mineig(view(A,:,:,i:j))
        i = j+1
    end
    CUDA.synchronize()
end
gpu_mineig(view(Acpu,:,:,1:1000)); CUDA.synchronize()                              # warmup
wg = Vector{Float64}(undef, B)
t_gpu = CUDA.@elapsed gpu_all!(wg, Acpu, B, CHUNK)
@printf("GPU  cuSOLVER syevjBatched (incl H2D, chunked %d): %d mats in %.3f s  (%.3f Mmat/s)\n",
        CHUNK, B, t_gpu, B/t_gpu/1e6)

err = maximum(abs.(wg .- wc))
@printf("\nmax |min-eig GPU - CPU| over %d mats = %.3e\n", B, err)
@printf("SPEEDUP GPU vs CPU (end-to-end, FP64) = %.1fx\n", t_cpu/t_gpu)
