using SLCEDynamics
using JET

@testset "JET" begin
    JET.test_package(SLCEDynamics; target_modules = (SLCEDynamics,))
end
