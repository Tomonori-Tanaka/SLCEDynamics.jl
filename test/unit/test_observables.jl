# User-defined observables: `run_llg` accepts the SAME `Observable(name, ncomp, f)`
# definitions as the SLCEMonteCarlo drivers (`f(config, energy, H)` with the SLCE
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
        # the observable contract feeds the SLCE energy: with b_ext = 0 the
        # recorded dynamical energy IS the SLCE energy, bitwise
        @test vec(r.series[:energy]) == r.energies
    end

    @testset "custom observables, Zeeman split" begin
        bz = 4.0
        probB = LLGProblem(H; magmom = 2.0, b_ext = (0.0, 0.0, bz))
        obs = [Observable(:ez1, 1, v -> v.config[1][3]),
               Observable(:s12, 3, v -> v.config[1] + v.config[2]),
               Observable(:esce, 1, v -> v.energy)]
        r = run_llg(probB, config0; dt = 0.01, nsteps = 50, measure_interval = 10,
                    observables = obs)
        @test r.series[:ez1][1, end] == r.config[1][3]
        @test SVector{3,Float64}(r.series[:s12][:, end]) ==
              r.config[1] + r.config[2]
        # dynamical energy = SLCE energy + Zeeman (all sites active here)
        mz = sum(c[3] for c in r.config)
        @test r.energies[end] ≈
              r.series[:esce][1, end] - 2.0 * SD.MU_B_EV_T * bz * mz atol = 1e-14
    end

    @testset "an Evaluable's scope picks the site count it is normalized by" begin
        # `stats.jl` used to pass `n_active` to every evaluable regardless of scope.
        # Invisible on any Hamiltonian this package can currently run — a site is
        # active-but-not-spin-active only if it carries displacement content, and a
        # joint `TiledHamiltonian` dies at the first measurement (see CLAUDE.md's
        # parked gap) — so the counts always coincide today. The gate constructs the
        # divergent result directly rather than pretending otherwise: the moment
        # spin–lattice dynamics lands, `χ` and Binder normalized by the total active
        # count are wrong by the ratio of the two, silently.
        nm = 200
        series = Dict(:x => reshape(collect(1.0:nm), 1, nm))
        result = SD.LLGResult(collect(1.0:nm), zeros(nm),
                           fill(SVector{3,Float64}(0, 0, 1), nm), series,
                           config0, nm, 0.01, 1, 0.05, UInt64(7),
                           10, 4,                       # n_active, n_spin_active
                           "cpu", "classical")
        @test result.n_active != result.n_spin_active         # teeth: the two must differ
        evs = [Evaluable(:n_spin, [:x], (m, kT, n) -> float(n)),
               Evaluable(:n_energy, [:x], (m, kT, n) -> float(n); scope = :energy)]
        st = equilibrium_stats(result; evaluables = evs, discard = 0, nbins = 4)
        @test st[:n_spin].mean[1] == 4.0
        @test st[:n_energy].mean[1] == 10.0
    end

    @testset "validation" begin
        dup = [Observable(:x, 1, v -> v.energy),
               Observable(:x, 1, v -> v.energy)]
        @test_throws ArgumentError run_llg(prob, config0; dt = 0.01, nsteps = 1,
                                           observables = dup)
        bad = [Observable(:bad, 2, v -> [1.0, 2.0, 3.0])]
        @test_throws DimensionMismatch run_llg(prob, config0; dt = 0.01,
                                               nsteps = 1, observables = bad)
        # ncomp = 1 returning a vector gets the same NAMED error, not a bare
        # MethodError from the assignment (audit #9)
        badv = [Observable(:badv, 1, v -> [1.0])]
        err = try
            run_llg(prob, config0; dt = 0.01, nsteps = 1, observables = badv)
            nothing
        catch e
            e
        end
        @test err isa DimensionMismatch
        @test occursin("must return a Real", err.msg)
    end
end
