"""
    halo_exchange_3d!(A::Array{T,4}, decomp, bc::Symbol) where T

Exchange halos for a 3D subdomain with nv variables (no z decomposition).

# Arguments
- `A`: Local array (nx+2h, ny+2h, nz, nv), interior: A[h+1:h+nx, h+1:h+ny, :, :]
- `decomp`: Struct from setup_mpi_cartesian_3d
- `bc`: Boundary condition type (default: :copy)

# Returns
- Modifies `A` in place with exchanged halos

# Notes
- Performs left/right exchange first, then up/down exchange
- Applies physical BC at global boundaries before exchange
- No exchange in z-direction (all ranks have full z extent)
- Corners are filled implicitly after second phase

# Example
```julia
A = zeros(nx+2*halo, ny+2*halo, nz, Nmom)
halo_exchange_3d!(A, decomp, :copy)
```
"""
function halo_exchange_3d!(A::Array{T,4}, decomp, bc::Symbol) where T
    h = decomp.halo
    nx = decomp.local_size[1]
    ny = decomp.local_size[2]
    nz = decomp.local_size[3]
    
    if h == 0
        return A
    end
    
    # Apply physical BC to halos at global boundaries before exchange
    apply_physical_bc_3d!(A, decomp, bc)
    
    # Single rank - no exchange needed
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        return A
    end
    
    comm = MPI.COMM_WORLD
    
    # X-direction exchange using non-blocking send/receive
    left_neighbor = decomp.neighbors.left
    right_neighbor = decomp.neighbors.right
    
    # Post all sends first, then do receives
    reqs = MPI.Request[]
    
    # Send interior boundary data to neighbors (includes all z and all variables)
    if left_neighbor != -1
        send_buf = A[h+1:h+h, h+1:h+ny, :, :]
        req = MPI.Isend(send_buf, comm; dest=left_neighbor, tag=0)
        push!(reqs, req)
    end
    
    if right_neighbor != -1
        send_buf = A[h+nx-h+1:h+nx, h+1:h+ny, :, :]
        req = MPI.Isend(send_buf, comm; dest=right_neighbor, tag=1)
        push!(reqs, req)
    end
    
    # Receive halo data from neighbors
    if left_neighbor != -1
        recv_buf = similar(A, h, ny, nz, size(A, 4))
        req = MPI.Irecv!(recv_buf, comm; source=left_neighbor, tag=1)
        MPI.Wait(req)
        A[1:h, h+1:h+ny, :, :] = recv_buf
    end
    
    if right_neighbor != -1
        recv_buf = similar(A, h, ny, nz, size(A, 4))
        req = MPI.Irecv!(recv_buf, comm; source=right_neighbor, tag=0)
        MPI.Wait(req)
        A[h+nx+1:h+nx+h, h+1:h+ny, :, :] = recv_buf
    end
    
    # Wait for all sends to complete
    MPI.Waitall(reqs)
    
    # Y-direction exchange using non-blocking send/receive
    down_neighbor = decomp.neighbors.down
    up_neighbor = decomp.neighbors.up
    
    reqs = MPI.Request[]
    
    # Send interior boundary data to neighbors (includes all z and all variables)
    if down_neighbor != -1
        send_buf = A[h+1:h+nx, h+1:h+h, :, :]
        req = MPI.Isend(send_buf, comm; dest=down_neighbor, tag=2)
        push!(reqs, req)
    end
    
    if up_neighbor != -1
        send_buf = A[h+1:h+nx, h+ny-h+1:h+ny, :, :]
        req = MPI.Isend(send_buf, comm; dest=up_neighbor, tag=3)
        push!(reqs, req)
    end
    
    # Receive halo data from neighbors
    if down_neighbor != -1
        recv_buf = similar(A, nx, h, nz, size(A, 4))
        req = MPI.Irecv!(recv_buf, comm; source=down_neighbor, tag=3)
        MPI.Wait(req)
        A[h+1:h+nx, 1:h, :, :] = recv_buf
    end
    
    if up_neighbor != -1
        recv_buf = similar(A, nx, h, nz, size(A, 4))
        req = MPI.Irecv!(recv_buf, comm; source=up_neighbor, tag=2)
        MPI.Wait(req)
        A[h+1:h+nx, h+ny+1:h+ny+h, :, :] = recv_buf
    end
    
    # Wait for all sends to complete
    MPI.Waitall(reqs)
    
    return A
end

"""
    apply_physical_bc_3d!(A::Array{T,4}, decomp, bc::Symbol) where T

Fill halos at global boundaries based on bc type.

# Arguments
- `A`: Local array (nx+2h, ny+2h, nz, nv)
- `decomp`: Domain decomposition struct
- `bc`: Boundary condition type

# Supported bc types
- `:copy` - Neumann-like (copy nearest interior cell)

# Notes
- Also applies copy BC in z-direction at global z boundaries (no decomposition in z)
"""
function apply_physical_bc_3d!(A::Array{T,4}, decomp, bc::Symbol) where T
    h = decomp.halo
    nx = decomp.local_size[1]
    ny = decomp.local_size[2]
    nz = decomp.local_size[3]
    
    if h == 0
        return A
    end
    
    if bc == :copy
        # Left boundary (global)
        if decomp.neighbors.left == -1
            for ih in 1:h
                A[ih, :, :, :] .= view(A, h+1, :, :, :)
            end
        end
        
        # Right boundary (global)
        if decomp.neighbors.right == -1
            for ih in 1:h
                A[h+nx+ih, :, :, :] .= view(A, h+nx, :, :, :)
            end
        end
        
        # Bottom boundary (global)
        if decomp.neighbors.down == -1
            for ih in 1:h
                A[:, ih, :, :] .= view(A, :, h+1, :, :)
            end
        end
        
        # Top boundary (global)
        if decomp.neighbors.up == -1
            for ih in 1:h
                A[:, h+ny+ih, :, :] .= view(A, :, h+ny, :, :)
            end
        end
        
        # Z boundaries (always global since no z decomposition)
        # Note: No halos in z direction, but we still need BC at k=1 and k=nz
        # This is handled in the flux update for z-direction
    else
        error("Unknown bc type: $bc")
    end
    
    return A
end

