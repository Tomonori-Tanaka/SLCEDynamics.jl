# Stochastic-LLG gates: (e) the single-spin Boltzmann distribution — absolute
# agreement with the analytic quadrature AND α-independence (a wrong (1+α²) in the
# fluctuation–dissipation constant shows up as a (1+α²)× temperature error — 2×
# at α = 1, far above the statistical resolution here) — plus noise determinism,
# validation, and (d) the cross-package equilibrium gate vs SCEMonteCarlo
# Metropolis in test order.

@testset "stochastic LLG (gates e, d)" begin
    K = 0.05
    kt = 0.03
    H1 = MC.TiledHamiltonian(_uniaxial_model(K); dims = (1, 1, 1))
    obs1 = [Observable(:energy, 1, (c, E, H) -> E),
            Observable(:ez2, 1, (c, E, H) -> c[1][3]^2)]
    Eu(u) = total_energy(H1, _uniaxial_config(u))
    ez2_exact = _boltzmann_average(Eu, kt, (u, E) -> u^2)
    e_exact = _boltzmann_average(Eu, kt, (u, E) -> E)
    prob1(α) = LLGProblem(H1; magmom = 2.0, alpha = α)

    @testset "single-spin Boltzmann, α-independence (gate e)" begin
        for (α, seed) in ((1.0, 42), (0.5, 43))
            r = run_llg(prob1(α), _uniaxial_config(0.3); dt = 0.05,
                        nsteps = 300_000, kT = kt, seed = seed,
                        measure_interval = 10, observables = obs1)
            st = equilibrium_stats(r; evaluables = Evaluable[])
            # 3σ against the analytic quadrature (measured ~0.1σ at these seeds)
            @test abs(st[:ez2].mean[1] - ez2_exact) < 3 * st[:ez2].err[1]
            @test abs(st[:energy].mean[1] - e_exact) < 3 * st[:energy].err[1]
            # τ_int-aware errors are meaningful (autocorrelation resolved)
            @test st[:ez2].err[1] < 0.01
        end
    end

    @testset "noise determinism and stream properties" begin
        p = prob1(1.0)
        c0 = _uniaxial_config(0.3)
        r1 = run_llg(p, c0; dt = 0.05, nsteps = 500, kT = kt, seed = 7)
        r2 = run_llg(p, c0; dt = 0.05, nsteps = 500, kT = kt, seed = 7)
        @test r1.config == r2.config && r1.energies == r2.energies
        @test r1.kT == kt && r1.seed == UInt64(7)
        r3 = run_llg(p, c0; dt = 0.05, nsteps = 500, kT = kt, seed = 8)
        @test r3.config != r1.config
        rt = run_llg(p, c0; dt = 0.05, nsteps = 500, kT = kt, seed = 7,
                     ntasks = 4)
        @test rt.config == r1.config
        # temperature[K] route resolves through KB_EV
        rk = run_llg(p, c0; dt = 0.05, nsteps = 10, kT = kt, seed = 7)
        rT = run_llg(p, c0; dt = 0.05, nsteps = 10,
                     temperature = kt / MC.KB_EV, seed = 7)
        @test rk.config == rT.config && rT.kT ≈ kt
        # a default seed is drawn and recorded
        rd = run_llg(p, c0; dt = 0.05, nsteps = 10, kT = kt)
        @test rd.seed != 0
        # Depondt stays on the sphere under noise
        rn = run_llg(p, c0; dt = 0.05, nsteps = 2000, kT = kt, seed = 7,
                     renorm_interval = 0)
        @test maximum(abs(norm(e) - 1) for e in rn.config) < 1e-12
    end

    @testset "thermostat validation" begin
        p = prob1(1.0)
        c0 = _uniaxial_config(0.3)
        # α = 0 has no thermostat
        @test_throws ArgumentError run_llg(prob1(0.0), c0; dt = 0.05, nsteps = 1,
                                           kT = kt)
        # seed without a thermostat is an error
        @test_throws ArgumentError run_llg(p, c0; dt = 0.05, nsteps = 1, seed = 1)
        # negative seed and noise-counter capacity fail fast
        @test_throws ArgumentError run_llg(p, c0; dt = 0.05, nsteps = 1,
                                           kT = kt, seed = -1)
        @test_throws ArgumentError run_llg(p, c0; dt = 0.05, nsteps = 2^48,
                                           kT = kt, seed = 1)
        # exactly one of temperature/kT; scalars only
        @test_throws ArgumentError run_llg(p, c0; dt = 0.05, nsteps = 1,
                                           kT = kt, temperature = 300.0)
        @test_throws ArgumentError run_llg(p, c0; dt = 0.05, nsteps = 1,
                                           kT = [0.01, 0.02])
        # equilibrium_stats rejects deterministic runs and bad kwargs
        rdet = run_llg(prob1(0.0), c0; dt = 0.05, nsteps = 10)
        @test_throws ArgumentError equilibrium_stats(rdet)
        rth = run_llg(p, c0; dt = 0.05, nsteps = 100, kT = kt, seed = 1,
                      observables = obs1)
        @test_throws ArgumentError equilibrium_stats(rth; discard = 100)
        @test_throws ArgumentError equilibrium_stats(rth; nbins = 1)
        # an evaluable whose input was not recorded
        bad = Evaluable(:c, [:not_recorded], (m, kT, n) -> 0.0)
        @test_throws ArgumentError equilibrium_stats(rth; evaluables = [bad])
    end

    @testset "standard observables + default evaluables end to end" begin
        # the documented pairing: record standard_observables, then
        # equilibrium_stats with ALL defaults yields the standard evaluables
        H = MC.TiledHamiltonian(_dimer_model(-0.02); dims = (1, 1, 1))
        prob = LLGProblem(H; magmom = 2.0, alpha = 1.0)
        up = SVector(0.0, 0.0, 1.0)
        c0 = MC.SpinConfig([up, up, up, up])
        r = run_llg(prob, c0; dt = 0.05, nsteps = 20_000, kT = 0.05, seed = 5,
                    measure_interval = 10,
                    observables = standard_observables(H))
        st = equilibrium_stats(r)
        for name in (:specific_heat, :susceptibility, :binder)
            @test haskey(st, name) && isfinite(st[name].mean[1])
        end
        @test st[:energy].count == length(r.times) - length(r.times) ÷ 2
    end

    @testset "sLLG ≡ Metropolis equilibrium (gate d, dimer)" begin
        H = MC.TiledHamiltonian(_dimer_model(-0.02); dims = (1, 1, 1))
        kt2 = 0.05
        obs = [Observable(:energy, 1, (c, E, Hh) -> E),
               Observable(:e12, 1, (c, E, Hh) -> dot(c[1], c[2]))]
        mc = run_mc(H; kT = kt2, seed = 11, sweeps_therm = 5_000,
                    sweeps_measure = 100_000, observables = obs,
                    evaluables = MC.Evaluable[])
        mst = mc.points[1].stats
        prob = LLGProblem(H; magmom = 2.0, alpha = 1.0)
        up = SVector(0.0, 0.0, 1.0)
        c0 = MC.SpinConfig([up, up, up, up])
        r = run_llg(prob, c0; dt = 0.05, nsteps = 400_000, kT = kt2, seed = 12,
                    measure_interval = 10, observables = obs)
        st = equilibrium_stats(r; evaluables = Evaluable[])
        for name in (:energy, :e12)
            σ = sqrt(st[name].err[1]^2 + mst[name].err[1]^2)
            @test abs(st[name].mean[1] - mst[name].mean[1]) < 3σ
            @test st[name].err[1] < 0.02        # errors actually resolved
        end
    end
end
