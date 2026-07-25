using SLCEDynamics
using Aqua

@testset "Aqua" begin
    # SLCE is a genuine src dependency since the S(q,ω) layer (sqw.jl uses
    # Crystal / n_atoms / lattice.reciprocal for the q-space geometry).
    Aqua.test_all(SLCEDynamics)
end
