"""
    compute_halo_fluxes_and_wavespeeds_3d!(Fx, Fy, vpxmin_ext, vpxmax_ext, vpymin_ext, vpymax_ext,
                                           M, vpxmin, vpxmax, vpymin, vpymax, vpzmin, vpzmax,
                                           nx, ny, nz, halo, flag2D, Ma)

Compute fluxes and wave speeds in halo cells (3D physical space).

# Description
After halo exchange, the moment data M in halo cells is available from neighbors.
This function computes the corresponding fluxes and wave speeds in those halo cells,
which are needed for the pas_HLL stencil at processor boundaries.

# Arguments
- `Fx`: X-flux array with halos (nx+2*halo, ny+2*halo, nz, Nmom) [modified in place]
- `Fy`: Y-flux array with halos (nx+2*halo, ny+2*halo, nz, Nmom) [modified in place]
- `vpxmin_ext`: Extended min x-wave speed (nx+2*halo, ny, nz) [output]
- `vpxmax_ext`: Extended max x-wave speed (nx+2*halo, ny, nz) [output]
- `vpymin_ext`: Extended min y-wave speed (nx, ny+2*halo, nz) [output]
- `vpymax_ext`: Extended max y-wave speed (nx, ny+2*halo, nz) [output]
- `M`: Moment array with halos (nx+2*halo, ny+2*halo, nz, Nmom)
- `vpxmin`: Min x-wave speed, interior only (nx, ny, nz)
- `vpxmax`: Max x-wave speed, interior only (nx, ny, nz)
- `vpymin`: Min y-wave speed, interior only (nx, ny, nz)
- `vpymax`: Max y-wave speed, interior only (nx, ny, nz)
- `vpzmin`: Min z-wave speed, interior only (nx, ny, nz)
- `vpzmax`: Max z-wave speed, interior only (nx, ny, nz)
- `nx`: Interior size in x
- `ny`: Interior size in y
- `nz`: Interior size in z
- `halo`: Halo width
- `flag2D`: 2D flag for flux closure
- `Ma`: Mach number for flux closure
"""
function compute_halo_fluxes_and_wavespeeds_3d!(Fx::Array{Float64,4}, Fy::Array{Float64,4},
                                                vpxmin_ext::Array{Float64,3}, vpxmax_ext::Array{Float64,3},
                                                vpymin_ext::Array{Float64,3}, vpymax_ext::Array{Float64,3},
                                                M::Array{Float64,4},
                                                vpxmin::Array{Float64,3}, vpxmax::Array{Float64,3},
                                                vpymin::Array{Float64,3}, vpymax::Array{Float64,3},
                                                vpzmin::Array{Float64,3}, vpzmax::Array{Float64,3},
                                                nx::Int, ny::Int, nz::Int, halo::Int, flag2D::Int, Ma::Float64)
    
    # Copy interior wave speeds to extended arrays
    vpxmin_ext[halo+1:halo+nx, :, :] = vpxmin
    vpxmax_ext[halo+1:halo+nx, :, :] = vpxmax
    vpymin_ext[:, halo+1:halo+ny, :] = vpymin
    vpymax_ext[:, halo+1:halo+ny, :] = vpymax
    
    # Compute Fx, Fy, and wave speeds in halo cells (they have M data from exchange)
    
    # Left halo (i=1:halo)
    for k in 1:nz
        for i in 1:halo
            for j in 1:ny
                jh = j + halo
                MOM = M[i, jh, k, :]
                # Must match the interior flux path in simulation_runner EXACTLY
                # (hyperbolicity correction from MOM + pure flux, no realizability)
                # so rank-boundary cells are bit-identical to a single-rank run.
                v6x_min, v6x_max, Mr = eigenvalues6_hyperbolic_3D(MOM, 1, flag2D, Ma)
                v6y_min, v6y_max, Mr = eigenvalues6_hyperbolic_3D(Mr, 2, flag2D, Ma)
                v6z_min, v6z_max, Mr = eigenvalues6z_hyperbolic_3D(Mr, flag2D, Ma)
                Mx, My, _ = Flux_closure35_3D(Mr)
                Fx[i, jh, k, :] = Mx
                Fy[i, jh, k, :] = My
                _, v5x_min, v5x_max = closure_and_eigenvalues(Mr[[1,2,3,4,5]])
                vpxmin_ext[i, j, k] = min(v5x_min, v6x_min)
                vpxmax_ext[i, j, k] = max(v5x_max, v6x_max)
            end
        end
    end
    
    # Right halo (i=halo+nx+1:nx+2*halo)
    for k in 1:nz
        for i in halo+nx+1:nx+2*halo
            for j in 1:ny
                jh = j + halo
                MOM = M[i, jh, k, :]
                # Must match the interior flux path in simulation_runner EXACTLY
                # (hyperbolicity correction from MOM + pure flux, no realizability)
                # so rank-boundary cells are bit-identical to a single-rank run.
                v6x_min, v6x_max, Mr = eigenvalues6_hyperbolic_3D(MOM, 1, flag2D, Ma)
                v6y_min, v6y_max, Mr = eigenvalues6_hyperbolic_3D(Mr, 2, flag2D, Ma)
                v6z_min, v6z_max, Mr = eigenvalues6z_hyperbolic_3D(Mr, flag2D, Ma)
                Mx, My, _ = Flux_closure35_3D(Mr)
                Fx[i, jh, k, :] = Mx
                Fy[i, jh, k, :] = My
                _, v5x_min, v5x_max = closure_and_eigenvalues(Mr[[1,2,3,4,5]])
                vpxmin_ext[i, j, k] = min(v5x_min, v6x_min)
                vpxmax_ext[i, j, k] = max(v5x_max, v6x_max)
            end
        end
    end
    
    # Bottom halo (j=1:halo)
    for k in 1:nz
        for i in 1:nx
            ih = i + halo
            for j in 1:halo
                MOM = M[ih, j, k, :]
                # Must match the interior flux path in simulation_runner EXACTLY
                # (hyperbolicity correction from MOM + pure flux, no realizability)
                # so rank-boundary cells are bit-identical to a single-rank run.
                v6x_min, v6x_max, Mr = eigenvalues6_hyperbolic_3D(MOM, 1, flag2D, Ma)
                v6y_min, v6y_max, Mr = eigenvalues6_hyperbolic_3D(Mr, 2, flag2D, Ma)
                v6z_min, v6z_max, Mr = eigenvalues6z_hyperbolic_3D(Mr, flag2D, Ma)
                Mx, My, _ = Flux_closure35_3D(Mr)
                Fx[ih, j, k, :] = Mx
                Fy[ih, j, k, :] = My
                _, v5y_min, v5y_max = closure_and_eigenvalues(Mr[[1,6,10,13,15]])
                vpymin_ext[i, j, k] = min(v5y_min, v6y_min)
                vpymax_ext[i, j, k] = max(v5y_max, v6y_max)
            end
        end
    end
    
    # Top halo (j=halo+ny+1:ny+2*halo)
    for k in 1:nz
        for i in 1:nx
            ih = i + halo
            for j in halo+ny+1:ny+2*halo
                MOM = M[ih, j, k, :]
                # Must match the interior flux path in simulation_runner EXACTLY
                # (hyperbolicity correction from MOM + pure flux, no realizability)
                # so rank-boundary cells are bit-identical to a single-rank run.
                v6x_min, v6x_max, Mr = eigenvalues6_hyperbolic_3D(MOM, 1, flag2D, Ma)
                v6y_min, v6y_max, Mr = eigenvalues6_hyperbolic_3D(Mr, 2, flag2D, Ma)
                v6z_min, v6z_max, Mr = eigenvalues6z_hyperbolic_3D(Mr, flag2D, Ma)
                Mx, My, _ = Flux_closure35_3D(Mr)
                Fx[ih, j, k, :] = Mx
                Fy[ih, j, k, :] = My
                _, v5y_min, v5y_max = closure_and_eigenvalues(Mr[[1,6,10,13,15]])
                vpymin_ext[i, j, k] = min(v5y_min, v6y_min)
                vpymax_ext[i, j, k] = max(v5y_max, v6y_max)
            end
        end
    end
    
    return nothing
end

