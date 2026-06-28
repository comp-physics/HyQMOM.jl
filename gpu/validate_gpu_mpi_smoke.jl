using MPI, CUDA

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# --- bind each rank to a distinct GPU ---
ndev = CUDA.ndevices()
CUDA.device!(rank % ndev)
dev = CUDA.device()
name = CUDA.name(dev)
# serialize prints by rank
for r in 0:nranks-1
    if r == rank
        println("rank $rank/$nranks  ->  GPU $(CUDA.deviceid(dev)) ($name)  ndev=$ndev")
        flush(stdout)
    end
    MPI.Barrier(comm)
end

# --- each rank builds a GPU field tagged with its rank ---
n = 256
d_field = CUDA.fill(Float64(rank), n)        # whole buffer = rank value
# the "halo face" we send to the right neighbor = our buffer's last element region
face_len = 8
d_send = d_field[end-face_len+1:end]          # CuArray slice (on GPU)

# --- HOST-STAGED halo exchange (no CUDA-aware MPI needed) ---
h_send = Array(d_send)                         # GPU -> host
h_recv = similar(h_send)
left  = (rank - 1 + nranks) % nranks
right = (rank + 1) % nranks
# ring: send my face to the right, receive into halo from the left
MPI.Sendrecv!(h_send, h_recv, comm; dest=right, sendtag=0, source=left, recvtag=0)
d_halo = CuArray(h_recv)                        # host -> GPU

# --- verify: received halo must equal the LEFT neighbor's rank value ---
expected = Float64(left)
got = Array(d_halo)
ok = all(got .== expected)
# also confirm the data actually round-tripped through the GPU
gpu_ok = (d_halo isa CuArray) && all(Array(d_halo .+ 0.0) .== expected)

for r in 0:nranks-1
    if r == rank
        println("rank $rank: recv halo from left=$left  expected=$expected  got=$(got[1])  OK=$(ok && gpu_ok)")
        flush(stdout)
    end
    MPI.Barrier(comm)
end

# --- global reduction across GPUs (host-staged): sum of all rank ids ---
local_sum = sum(Array(d_field)) / n            # = rank
total = MPI.Allreduce(local_sum, +, comm)
expected_total = sum(0:nranks-1)
if rank == 0
    println("Allreduce(sum of ranks) = $total  expected=$expected_total  OK=$(total == expected_total)")
    allok = (total == expected_total)
    println(allok ? "SMOKE TEST PASS" : "SMOKE TEST FAIL")
end
MPI.Barrier(comm)
MPI.Finalize()
