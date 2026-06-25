ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, JLD2, Printf, LinearAlgebra

# Internal functions (not exported) reached via the module
const to_recon   = HyQMOM.to_recon_vars
const from_recon = HyQMOM.from_recon_vars
const muscl      = HyQMOM.muscl_faces
const facflux    = HyQMOM.face_flux_1d
const realize    = HyQMOM.realizable_3D_M4
const realspeed  = HyQMOM.realize_and_speed

Ma = 100.0

# Load the real Np=128 order=1 field (resolution matches the crashing order=2 run)
f = jldopen("/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma100_o1.jld2")
M = f["M"]; close(f)
nx,ny,nz,nm = size(M)
@printf("Loaded field %d x %d x %d x %d  rho[min,max]=[%.3e,%.3e]\n",
        nx,ny,nz,nm, minimum(M[:,:,:,1]), maximum(M[:,:,:,1]))

finite(v) = all(isfinite, v)

# stage tallies
stats = Dict(:recon=>0, :muscl=>0, :fromrecon=>0, :realizable=>0, :facflux_throw=>0, :ok=>0)
first_bad = nothing   # (i,j,k,axis,stage, info)
worst_mag = 0.0; worst_state = nothing

# scan interior triples along x (axis=1). Mirror logic of residual_line face_states (order=2).
function probe_axis!(M, axis, stats)
    nx,ny,nz,_ = size(M)
    rng_i = axis==1 ? (2:nx-2) : (1:nx)
    # to keep it cheap, subsample the two transverse directions but cover the
    # full sweep direction; the crossing structure is symmetric enough.
    for k in 1:max(1,nz÷16):nz, j in 1:max(1,ny÷16):ny
      for ii in 2:(axis==1 ? nx-2 : nx-2)
        # build the 4-point stencil along the chosen axis centered so we
        # reconstruct the interface between cell iL and iL+1
        get(idx) = axis==1 ? M[idx,j,k,:] : (axis==2 ? M[j,idx,k,:] : M[j,k,idx,:])
        N = axis==1 ? nx : (axis==2 ? ny : nz)
        iL = ii
        (iL-1 >= 1 && iL+2 <= N) || continue
        Mm = get(iL-1); M0 = get(iL); Mp = get(iL+1); Mpp = get(iL+2)
        # left face of interface = right face of cell iL ; right = left face of cell iL+1
        local Vl, Vr, Li, Ri
        try
            Vl = muscl(to_recon(Mm), to_recon(M0), to_recon(Mp))[2]
            Vr = muscl(to_recon(M0), to_recon(Mp), to_recon(Mpp))[1]
        catch e
            stats[:recon]+=1; continue
        end
        if !finite(Vl) || !finite(Vr); stats[:muscl]+=1
            if first_bad===nothing; global first_bad=(iL,j,k,axis,:muscl_recvars,(Vl,Vr,M0,Mp)); end
            continue
        end
        Li = from_recon(Vl); Ri = from_recon(Vr)
        if !finite(Li) || !finite(Ri)
            stats[:fromrecon]+=1
            mg = maximum(abs.(filter(isfinite, vcat(Li,Ri)); init=0.0))
            if first_bad===nothing; global first_bad=(iL,j,k,axis,:from_recon,(M0,Mp,Vl,Vr,Li,Ri)); end
            continue
        end
        # density-positivity fallback (matches residual_line): if either nonpositive, 1st order
        ML,MR = (Li[1]>0 && Ri[1]>0) ? (Li,Ri) : (M0,Mp)
        # realizable correction of each face (first thing face_flux_1d does)
        local MLc, MRc
        try
            MLc = realize(ML, Ma); MRc = realize(MR, Ma)
        catch e
            stats[:realizable]+=1
            if first_bad===nothing; global first_bad=(iL,j,k,axis,:realizable_throw,(ML,MR,e)); end
            continue
        end
        if !finite(MLc) || !finite(MRc)
            stats[:realizable]+=1
            if first_bad===nothing; global first_bad=(iL,j,k,axis,:realizable_nonfinite,(ML,MR,MLc,MRc)); end
            continue
        end
        # full face flux (includes eigenvalue/closure solves -> the crash site)
        try
            F = facflux(ML, MR, axis, Ma)
            if finite(F); stats[:ok]+=1 else
                stats[:facflux_throw]+=1
                if first_bad===nothing; global first_bad=(iL,j,k,axis,:facflux_nonfinite,(ML,MR,F)); end
            end
        catch e
            stats[:facflux_throw]+=1
            if first_bad===nothing; global first_bad=(iL,j,k,axis,:facflux_throw,(ML,MR,e)); end
        end
      end
    end
end

for ax in (1,2,3)
    probe_axis!(M, ax, stats)
end

println("\n=== STAGE TALLIES (count of offending interfaces) ===")
for kk in (:muscl,:fromrecon,:realizable,:facflux_throw,:ok)
    @printf("  %-16s %d\n", kk, stats[kk])
end

if first_bad !== nothing
    iL,j,k,axis,stage,info = first_bad
    @printf("\n=== FIRST OFFENDING INTERFACE ===\n")
    @printf("  axis=%d  cell (i=%d,j=%d,k=%d) interface i+1/2  stage=%s\n", axis,iL,j,k,stage)
    if stage == :from_recon
        M0,Mp,Vl,Vr,Li,Ri = info
        @printf("  cell L rho=%.4e  cell R rho=%.4e\n", M0[1], Mp[1])
        @printf("  recon-var L (C200,C020,C002)=(%.3e,%.3e,%.3e)\n", Vl[5],Vl[6],Vl[7])
        @printf("  recon-var R (C200,C020,C002)=(%.3e,%.3e,%.3e)\n", Vr[5],Vr[6],Vr[7])
        bad = findall(x->!isfinite(x), Li)
        @printf("  from_recon(Vl) non-finite indices: %s\n", string(bad))
        @printf("  from_recon(Vr) non-finite indices: %s\n", string(findall(x->!isfinite(x),Ri)))
        @printf("  max|finite Li|=%.3e  max|finite Ri|=%.3e\n",
                maximum(abs.(filter(isfinite,Li));init=0.0), maximum(abs.(filter(isfinite,Ri));init=0.0))
    elseif stage in (:facflux_throw,:realizable_throw)
        ML,MR,e = info
        @printf("  cell L rho=%.4e  cell R rho=%.4e\n", ML[1], MR[1])
        @printf("  exception: %s\n", sprint(showerror, e))
    else
        @printf("  info: %s\n", string(info)[1:min(end,400)])
    end
else
    println("\nNo offending interface found in the sampled set.")
end
