# Validation gate (b) and the reproducibility contract on a genuinely anisotropic
# multi-l model: α = 0 energy conservation at O(dt²), norm preservation, strict
# dissipation at α > 0, integrator cross-agreement, bitwise determinism, and
# ntasks independence.

@testset "conservation and determinism (gate b)" begin
    H = MC.TiledHamiltonian(_biquadratic_model(7); dims = (2, 2, 1))
    rng = MersenneTwister(21)
    config0 = _rand_config(rng, H)
    prob0 = LLGProblem(H; magmom = 2.0)                  # α = 0

    _drift(dt, nsteps; kw...) = begin
        r = run_llg(prob0, config0; dt = dt, nsteps = nsteps,
                    measure_interval = 10, renorm_interval = 0, kw...)
        maximum(abs.(r.energies .- r.energies[1]))
    end

    # This model is stiff and chaotic (max|G| ≈ 2.4 eV ⇒ ω_max ≈ 3.7 rad/fs,
    # Lyapunov growth on the fs scale), so the order checks run over short
    # horizons where the O(dt²) asymptotics are clean (measured ratios ≈ 4.0-4.2).

    @testset "α = 0: energy drift is small and O(dt²)" begin
        d1 = _drift(0.02, 250)                           # t = 5 fs
        d2 = _drift(0.01, 500)                           # same physical time
        @test d1 < 1e-2
        @test 3.0 < d1 / d2 < 5.5                        # ≈ 4 (order 2)
    end

    @testset "Depondt preserves norms to rounding (no renormalization)" begin
        r = run_llg(prob0, config0; dt = 0.01, nsteps = 1000, renorm_interval = 0)
        @test maximum(abs(norm(e) - 1) for e in r.config) < 1e-12
    end

    @testset "α > 0: dissipation" begin
        prob = LLGProblem(H; magmom = 2.0, alpha = 0.5)
        r = run_llg(prob, config0; dt = 0.01, nsteps = 2000)
        # tolerance = the O(dt²) drift floor at this dt (α = 0 drift ≈ 1e-3 over
        # 5 fs at dt = 0.02, §above); dissipation dominates it by orders
        @test all(diff(r.energies) .< 1e-6)
        @test r.energies[end] < r.energies[1] - 0.1      # genuinely relaxed
    end

    @testset "integrators agree at O(dt²)" begin
        final(dt, integ) = run_llg(prob0, config0; dt = dt,
                                   nsteps = round(Int, 1.0 / dt),
                                   integrator = integ).config
        gap(dt) = maximum(norm.(final(dt, DepondtMertens()) .-
                                final(dt, HeunProjected())))
        r = gap(0.02) / gap(0.01)
        @test gap(0.02) < 1e-3
        @test 3.0 < r < 5.5                              # measured 4.00
    end

    @testset "bitwise determinism, ntasks independence" begin
        r1 = run_llg(prob0, config0; dt = 0.01, nsteps = 300)
        r2 = run_llg(prob0, config0; dt = 0.01, nsteps = 300)
        @test r1.config == r2.config
        @test r1.energies == r2.energies
        r4 = run_llg(prob0, config0; dt = 0.01, nsteps = 300, ntasks = 4)
        @test r4.config == r1.config
        @test r4.energies == r1.energies
        # config0 is not mutated
        @test config0 == _rand_config(MersenneTwister(21), H)
    end

    @testset "driver validation and measurement layout" begin
        @test_throws DimensionMismatch run_llg(prob0, MC.SpinConfig([config0[1]]);
                                               dt = 0.1, nsteps = 1)
        @test_throws ArgumentError run_llg(prob0, config0; dt = 0.0, nsteps = 1)
        @test_throws ArgumentError run_llg(prob0, config0; dt = Inf, nsteps = 1)
        @test_throws ArgumentError run_llg(prob0, config0; dt = 0.1, nsteps = -1)
        @test_throws ArgumentError run_llg(prob0, config0; dt = 0.1, nsteps = 1,
                                           measure_interval = 0)
        @test_throws ArgumentError run_llg(prob0, config0; dt = 0.1, nsteps = 1,
                                           ntasks = 0)
        bad = copy(config0)
        bad[1] = SVector(2.0, 0.0, 0.0)
        @test_throws ArgumentError run_llg(prob0, bad; dt = 0.1, nsteps = 1)
        # the final step is ALWAYS measured, on-grid or not
        r = run_llg(prob0, config0; dt = 0.5, nsteps = 25, measure_interval = 10)
        @test r.times == [0.0, 5.0, 10.0, 12.5]          # steps 0, 10, 20, 25
        @test length(r.energies) == 4 && length(r.mean_spins) == 4
        @test r.energies[end] == total_energy(prob0, r.config)
        rg = run_llg(prob0, config0; dt = 0.5, nsteps = 20, measure_interval = 10)
        @test rg.times == [0.0, 5.0, 10.0]               # on-grid: no duplicate
        @test rg.energies[end] == total_energy(prob0, rg.config)
        # nsteps = 0: a single measurement of the initial state
        r0 = run_llg(prob0, config0; dt = 0.5, nsteps = 0)
        @test r0.times == [0.0] && r0.config == config0
        @test r0.energies[1] == total_energy(prob0, config0)
        @test occursin("LLGResult", sprint(show, r))
    end
end
