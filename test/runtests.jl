using Test

const TEST_MODE = get(ENV, "TEST_MODE", "default")

@testset "SCESpinDynamics.jl" begin
    if TEST_MODE in ("default", "all", "unit")
        include("unit/fixtures.jl")
        include("unit/test_units.jl")
        include("unit/test_problem.jl")
        include("unit/test_single_spin.jl")
        include("unit/test_two_spin.jl")
        include("unit/test_conservation.jl")
        include("unit/test_observables.jl")
        include("unit/test_thermostat.jl")
    end
    if TEST_MODE in ("default", "all", "aqua")
        include("aqua.jl")
    end
    if TEST_MODE in ("all", "jet")
        include("jet.jl")
    end
end
