using SCESpinDynamics
using JET

@testset "JET" begin
    JET.test_package(SCESpinDynamics; target_modules = (SCESpinDynamics,))
end
