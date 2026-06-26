using JLD2, Printf, Statistics

# Density grid-convergence analysis (Rodney: use M000). Loads conv_o{order}_np*.jld2
# density fields at factor-2 resolutions and reports:
#   (1) self-convergence: ||rho_h - restrict(rho_{h/2})||_1 between successive grids,
#       and the observed order from successive ratios;
#   (2) a diffusion proxy: max|grad rho| and the count of "transition" cells
#       (0.1<rho<0.9*peak), which should sharpen toward the resolved limit.
# Pass the order to analyze as ARGS[1] (default 1), Ma as ARGS[2] (default 10).

order = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
Ma    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
D = "/storage/project/r-sbryngelson3-0/sbryngelson3/debug"

# conservative 2x coarsening of a density cube (block-average over 2x2x2)
function restrict2(r)
    n = size(r,1); m = n ÷ 2
    c = Array{Float64,3}(undef, m,m,m)
    @inbounds for k in 1:m, j in 1:m, i in 1:m
        s = 0.0
        for dk in 0:1, dj in 0:1, di in 0:1
            s += r[2i-di, 2j-dj, 2k-dk]
        end
        c[i,j,k] = s/8
    end
    c
end

maxgrad(r) = (n=size(r,1); dx=1.0/n; g=0.0;
    for k in 2:n-1, j in 2:n-1, i in 2:n-1
        gx=(r[i+1,j,k]-r[i-1,j,k]); gy=(r[i,j+1,k]-r[i,j-1,k]); gz=(r[i,j,k+1]-r[i,j,k-1])
        g=max(g, sqrt(gx^2+gy^2+gz^2)/(2dx))
    end; g)

# discover available resolutions
res = Int[]
for np in (64,128,256,512,1024,2048)
    isfile(@sprintf("%s/conv_o%d_np%d_ma%d.jld2", D, order, np, Ma)) && push!(res, np)
end
isempty(res) && (println("no conv_o$(order)_np*_ma$(Ma).jld2 files found"); exit())

fields = Dict{Int,Array{Float64,3}}()
for np in res
    f = jldopen(@sprintf("%s/conv_o%d_np%d_ma%d.jld2", D, order, np, Ma)); fields[np]=f["rho"]; close(f)
end

@printf("Density convergence  order=%d  Ma=%d  resolutions=%s\n", order, Ma, string(res))
@printf("%-6s | %-12s | %-12s | %-22s | %-10s\n", "Np", "peak rho", "max|grad|", "||d-R(d/2)||_1 vs next", "order p")
println("-"^78)
prev_err = NaN
for (idx,np) in enumerate(res)
    r = fields[np]
    err = "-"; p = "-"
    if np*2 in res
        rc = restrict2(fields[np*2])           # coarsen the finer grid to this one
        e = mean(abs.(r .- rc))                 # L1 density difference
        err = @sprintf("%.4e", e)
        if isfinite(prev_err) && prev_err > 0
            p = @sprintf("%.2f", log2(prev_err / e))
        end
        global prev_err = e
    end
    @printf("%-6d | %-12.5f | %-12.3f | %-22s | %-10s\n", np, maximum(r), maxgrad(r), err, p)
end
println("\nNotes: ||.||_1 is mean abs density diff between grid Np and the next-finer")
println("grid block-averaged down to Np. Decreasing with a clear order p => converging.")
