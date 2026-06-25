ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using HyQMOM, Printf, LinearAlgebra
import HyQMOM: jacobian15, C4toM4_3D, M2CS4_35, _plane_UV, eigenvalues6_hyperbolic_3D,
               eigenvalues6z_hyperbolic_3D, closure_and_eigenvalues, realize_and_speed,
               realizable_3D_M4, Flux_closure35_3D

Ma=10.0
M = realizable_3D_M4(collect(InitializeM4_35(1.0,3.0,1.0,0.5,1.2,0.1,0.05,0.9,0.02,1.1)), Ma)
m15 = _plane_UV(M)
J = jacobian15(m15...)
b3 = J[13:15,13:15]; b4 = J[6:9,6:9]
m5 = M[[1,2,3,4,5]]

bench(f,n)=(f(); t=@timed(for _ in 1:n; f(); end); (us=t.time/n*1e6, B=t.bytes/n))
n=300_000
for (name,f) in (
    ("jacobian15 build",        ()->jacobian15(m15...)),
    ("eigvals 3x3 (LAPACK)",    ()->eigvals(b3)),
    ("eigvals 4x4 (LAPACK)",    ()->eigvals(b4)),
    ("slice J[13:15,13:15]",    ()->J[13:15,13:15]),
    ("M2CS4_35",                ()->M2CS4_35(M)),
    ("C4toM4_3D",               ()->C4toM4_3D(1.0,3.0,1.0,0.5,1.2,0.1,0.05,0.9,0.02,1.1,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,3.,0.,0.,1.,0.,1.,0.,0.,0.,0.,3.,0.,1.,0.,3.)),
    ("closure_and_eigenvalues", ()->closure_and_eigenvalues(m5)),
    ("eigenvalues6 (x)",        ()->eigenvalues6_hyperbolic_3D(M,1,0,Ma)),
    ("eigenvalues6z",           ()->eigenvalues6z_hyperbolic_3D(M,0,Ma)),
    ("realize_and_speed",       ()->realize_and_speed(M,1,Ma)),
    ("realizable_3D_M4",        ()->realizable_3D_M4(M,Ma)),
    ("Flux_closure35_3D",       ()->Flux_closure35_3D(M)),
   )
    r=bench(f,n)
    @printf("%-26s %9.3f us  %10.0f B\n", name, r.us, r.B)
end
