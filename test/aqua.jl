using SCESpinDynamics
using Aqua

@testset "Aqua" begin
    # SCEFitting sits in [deps] only because it is an unregistered transitive
    # dependency (of the path-dev'd SCEMonteCarlo) that must itself be path-dev'd;
    # src/ deliberately never loads it (the boundary is SCEMonteCarlo's public
    # surface), so exempt it from the stale-deps check.
    Aqua.test_all(SCESpinDynamics; stale_deps = (ignore = [:SCEFitting],))
end
