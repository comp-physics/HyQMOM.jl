using MPI, CUDA, Printf
const HERE = "/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl/gpu"
include(joinpath(HERE, "realize_gpu.jl"));  using .RealizeGPU
const DATA = "/storage/scratch1/6/sbryngelson3/gpudata"

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)
CUDA.device!(rank % CUDA.ndevices())

# every rank reads the shared data (small); decompose the nb columns across ranks
nb  = parse(Int, strip(read(joinpath(DATA, "proj.meta"), String)))
M   = reshape(reinterpret(Float64, read(joinpath(DATA, "proj_M.f64"))),   35, nb)
Ref = reshape(reinterpret(Float64, read(joinpath(DATA, "proj_ref.f64"))), 35, nb)
Ma  = collect(reinterpret(Float64, read(joinpath(DATA, "proj_Ma.f64"))))

# column ranges (as even as possible)
base = div(nb, nranks); rem = nb % nranks
counts = [base + (r < rem ? 1 : 0) for r in 0:nranks-1]
offs   = cumsum([0; counts[1:end-1]])
c0 = offs[rank+1] + 1; c1 = offs[rank+1] + counts[rank+1]
myM  = Matrix(M[:, c0:c1]); myMa = Ma[c0:c1]

# --- each GPU projects its own slab with the REAL kernel ---
myOut = RealizeGPU.realizable_batched(myM, myMa)     # (35, mycols) on this rank's GPU -> host
@assert size(myOut) == (35, counts[rank+1])

for r in 0:nranks-1
    r == rank && (@printf("rank %d GPU %d: projected cols %d:%d (%d cells)\n",
                          rank, CUDA.deviceid(CUDA.device()), c0, c1, counts[rank+1]); flush(stdout))
    MPI.Barrier(comm)
end

# --- gather all slabs back to rank 0 (host-staged) ---
sendbuf = vec(myOut)                                  # 35*mycols Float64
recvbuf = rank == 0 ? Vector{Float64}(undef, 35*nb) : Float64[]
recvcounts = 35 .* counts
MPI.Gatherv!(sendbuf, rank == 0 ? MPI.VBuffer(recvbuf, recvcounts) : nothing, comm; root=0)

if rank == 0
    Out = reshape(recvbuf, 35, nb)
    d = maximum(abs.(Out .- Ref))
    rscale = max.(abs.(Ref), 1.0)
    r = maximum(abs.(Out .- Ref) ./ rscale)
    @printf("2-GPU projected field vs CPU reference (%d cells):\n", nb)
    @printf("  max abs diff = %.3e\n  max rel diff = %.3e\n", d, r)
    println(r < 1e-12 ? "GPU+MPI REALIZE TEST PASS (matches CPU ref)" : "GPU+MPI REALIZE TEST: diff above 1e-12")
end
MPI.Barrier(comm); MPI.Finalize()
