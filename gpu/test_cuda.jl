import Pkg; Pkg.activate(@__DIR__)
using CUDA
println("CUDA.functional() = ", CUDA.functional())
CUDA.versioninfo()
if CUDA.functional()
    a = CUDA.rand(Float32, 1_000_000); b = CUDA.rand(Float32, 1_000_000)
    CUDA.@sync c = a .* b .+ 1f0
    println("sample = ", Array(c[1:3]))
    println("KERNEL OK on ", CUDA.name(CUDA.device()))
end
