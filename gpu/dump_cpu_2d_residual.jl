using HyQMOM
DATA="/storage/scratch1/6/sbryngelson3/gpudata"
nb=parse(Int,strip(read("$DATA/proj.meta",String)))
src=reshape(reinterpret(Float64,read("$DATA/proj_M.f64")),35,nb)
n=24; dx=1.0/n; Ma=2.0; g=2; nz=1
idx=round.(Int, range(1,nb,length=n*n))
# CPU layout: (nx+2g, ny+2g, nz, 35), moment-last, x/y stored halos, z no halo
M=zeros(n+2g,n+2g,nz,35)
for c in 1:n*n; i=(c-1)%n+1; j=(c-1)÷n+1; @views M[i+g,j+g,1,:].=src[:,idx[c]]; end
for k in 1:nz
  for j in 1:n+2g, hh in 1:g; @views M[hh,j,k,:].=M[g+1,j,k,:]; @views M[n+g+hh,j,k,:].=M[n+g,j,k,:]; end
  for i in 1:n+2g, hh in 1:g; @views M[i,hh,k,:].=M[i,g+1,k,:]; @views M[i,n+g+hh,k,:].=M[i,n+g,k,:]; end
end
R=zeros(n+2g,n+2g,nz,35)
residual_ho_3d!(R,M,n,n,nz,g,dx,dx,dx,Ma; order=2)
Rint=permutedims(R[g+1:g+n,g+1:g+n,:,:],(4,1,2,3))   # (35,n,n,nz)
Mgpu=permutedims(M[g+1:g+n,g+1:g+n,:,:],(4,1,2,3))   # (35,n,n,nz)
write("$DATA/r2d_M.f64", reinterpret(UInt8, vec(Float64.(Mgpu))))
write("$DATA/r2d_R.f64", reinterpret(UInt8, vec(Float64.(Rint))))
write("$DATA/r2d.meta", "$n\n$dx\n$Ma\n$g\n")
println("wrote r2d ref: n=$n dx=$dx Ma=$Ma  Rint range=", extrema(Rint), "  nonzero=", maximum(abs.(Rint))>1.0)
