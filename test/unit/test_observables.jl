# User-defined observables: `run_llg` accepts the SAME `Observable(name, ncomp, f)`
# definitions as the SLCEMonteCarlo drivers (`f(config, energy, H)` with the SCE
# energy) and records one `ncomp × n_measurements` time-series matrix per name.

@testset "observables" begin
    H = MC.TiledHamiltonian(_biquadratic_model(7); dims = (2, 2, 1))
    config0 = _rand_config(MersenneTwister(33), H)
    prob = LLGProblem(H; magmom = 2.0)

    @testset "standard_observables plug in unchanged" begin
        obs = standard_observables(H)
        r = run_llg(prob, config0; dt = 0.01, nsteps = 100, measure_interval = 20,
                    observables = obs)
        @test Set(keys(r.series)) == Set(o.name for o in obs)
        nm = length(r.times)
        for o in obs
            @test size(r.series[o.name]) == (o.ncomp, nm)
        end
        # :m is the MC magnetization convention — matches mean_spins (both are
        # active-site means; summation order may differ, hence ≈ not ==)
        for k = 1:nm
            @test SVector{3,Float64}(r.series[:m][:, k]) ≈ r.mean_spins[k] rtol = 1e-14
        end
        # the observable contract feeds the SCE energy: with b_ext = 0 the
        # recorded dynamical energy IS the SCE energy, bitwise
        @test vec(r.series[:energy]) == r.energies
    end

    @testset "custom observables, Zeeman split" begin
        bz = 4.0
        probB = LLGProblem(H; magmom = 2.0, b_ext = (0.0, 0.0, bz))
        obs = [Observable(:ez1, 1, (cfg, E, H) -> cfg[1][3]),
               Observable(:s12, 3, (cfg, E, H) -> cfg[1] + cfg[2]),
               Observable(:esce, 1, (cfg, E, H) -> E)]
        r = run_llg(probB, config0; dt = 0.01, nsteps = 50, measure_interval = 10,
                    observables = obs)
        @test r.series[:ez1][1, end] == r.config[1][3]
        @test SVector{3,Float64}(r.series[:s12][:, end]) ==
              r.config[1] + r.config[2]
        # dynamical energy = SCE energy + Zeeman (all sites active here)
        mz = sum(c[3] for c in r.config)
        @test r.energies[end] ≈
              r.series[:esce][1, end] - 2.0 * SD.MU_B_EV_T * bz * mz atol = 1e-14
    end

    @testset "validation" begin
        dup = [Observable(:x, 1, (cfg, E, H) -> E),
               Observable(:x, 1, (cfg, E, H) -> E)]
        @test_throws ArgumentError run_llg(prob, config0; dt = 0.01, nsteps = 1,
                                           observables = dup)
        bad = [Observable(:bad, 2, (cfg, E, H) -> [1.0, 2.0, 3.0])]
        @test_throws DimensionMismatch run_llg(prob, config0; dt = 0.01,
                                               nsteps = 1, observables = bad)
    end
end
