# Checkpoint / resume gates: bit-identity (==, never ≈), the extension rule, and
# the schema / mismatch guards. Resume bit-identity rests on every per-step
# effect being a pure function of the absolute step index (the stateless Philox
# noise counter, the renormalization cadence, the measurement grid) — these
# tests pin that end to end through genuine crash-shaped mid-run files.

function _assert_same_llg(a::LLGResult, b::LLGResult)
    @test a.times == b.times
    @test a.energies == b.energies
    @test a.mean_spins == b.mean_spins
    @test sort(collect(keys(a.series))) == sort(collect(keys(b.series)))
    for k in keys(a.series)
        @test a.series[k] == b.series[k]
    end
    @test a.config == b.config
    @test a.nsteps == b.nsteps && a.dt == b.dt
    @test a.measure_interval == b.measure_interval
    @test isequal(a.kT, b.kT) && a.seed == b.seed && a.n_active == b.n_active
end

@testset "checkpoint / resume" begin
    dir = mktempdir()
    H = MC.TiledHamiltonian(_biquadratic_model(0); dims = (1, 1, 1))
    prob = LLGProblem(H; magmom = 2.0, alpha = 0.7)
    c0 = _rand_config(MersenneTwister(3), H)
    obs = [Observable(:e12, 1, (c, E, h) -> dot(c[1], c[2])),
           Observable(:m, 3, (c, E, h) -> (c[1] + c[2]) / 2)]
    # mi = 7 with nsteps = 1000 puts the final measurement OFF the grid — the
    # trace-layout edge every resume path must respect
    kw = (; dt = 0.02, nsteps = 1000, kT = 0.02, seed = 5, measure_interval = 7,
          observables = obs, renorm_interval = 100)

    @testset "writes do not perturb; completed-file resume reconstructs" begin
        path = joinpath(dir, "sllg.jld2")
        a = run_llg(prob, c0; kw...)
        b = run_llg(prob, c0; kw..., checkpoint = path, checkpoint_interval = 137)
        _assert_same_llg(a, b)                  # checkpointing is a pure observer
        @test isfile(path)
        c = resume(path, prob; observables = obs)     # completed → no stepping
        _assert_same_llg(a, c)
    end

    @testset "mid-run crash resume is bit-identical" begin
        path = joinpath(dir, "crash.jld2")
        a = run_llg(prob, c0; kw...)
        # an observable that fails on its 81st measurement (step 7·80 = 560)
        # leaves the last periodic tick (step 548) on disk — a genuine mid-run
        # crash file produced through the public API alone
        nmeas = Ref(0)
        boom = Observable(:e12, 1, (c, E, h) -> begin
                              nmeas[] += 1
                              nmeas[] > 80 && error("boom")
                              dot(c[1], c[2])
                          end)
        @test_throws ErrorException run_llg(prob, c0; kw...,
                                            observables = [boom, obs[2]],
                                            checkpoint = path,
                                            checkpoint_interval = 137)
        @test isfile(path)
        c = resume(path, prob; observables = obs)
        _assert_same_llg(a, c)
    end

    @testset "deterministic run: completed resume and extension" begin
        pd = LLGProblem(H; magmom = 2.0, alpha = 0.1)
        path = joinpath(dir, "det.jld2")
        kwd = (; dt = 0.02, nsteps = 500, measure_interval = 10,
               observables = obs)
        a = run_llg(pd, c0; kwd...)
        run_llg(pd, c0; kwd..., checkpoint = path, checkpoint_interval = 123)
        c = resume(path, pd; observables = obs)
        _assert_same_llg(a, c)
        long = run_llg(pd, c0; kwd..., nsteps = 800)
        ext = resume(path, pd; observables = obs, nsteps = 800)
        _assert_same_llg(long, ext)
    end

    @testset "extension ≡ uninterrupted (thermostatted)" begin
        path = joinpath(dir, "ext.jld2")
        kwe = (; dt = 0.02, kT = 0.02, seed = 9, measure_interval = 10,
               observables = obs)
        run_llg(prob, c0; kwe..., nsteps = 300, checkpoint = path)
        long = run_llg(prob, c0; kwe..., nsteps = 900)
        # absolute-step noise counters ⇒ continuing the 300-step file to 900 is
        # bit-identical to the single uninterrupted 900-step run
        ext = resume(path, prob; observables = obs, nsteps = 900)
        _assert_same_llg(long, ext)
        # the resumed run kept checkpointing to the same path — now a completed
        # 900-step file (idempotent retry-loop behavior)
        again = resume(path, prob; observables = obs)
        _assert_same_llg(long, again)
    end

    @testset "off-grid completed run refuses extension" begin
        path = joinpath(dir, "offgrid.jld2")
        run_llg(prob, c0; dt = 0.02, nsteps = 205, kT = 0.02, seed = 2,
                measure_interval = 10, checkpoint = path)
        @test_throws ArgumentError resume(path, prob; nsteps = 400)
        r = resume(path, prob)                  # plain reload still works
        @test r.nsteps == 205 && r.times[end] == 205 * 0.02
    end

    @testset "truncating resume to a mid-run file's own step" begin
        path = joinpath(dir, "trunc.jld2")
        nmeas = Ref(0)
        boom = Observable(:e12, 1, (c, E, h) -> begin
                              nmeas[] += 1
                              nmeas[] > 40 && error("boom")     # step 7·40 = 280
                              dot(c[1], c[2])
                          end)
        @test_throws ErrorException run_llg(prob, c0; kw...,
                                            observables = [boom, obs[2]],
                                            checkpoint = path,
                                            checkpoint_interval = 137)
        # file is at tick step 274; resuming to exactly 274 must supply the
        # off-grid final measurement the loop would never reach
        t = resume(path, prob; observables = obs, nsteps = 274,
                   checkpoint = nothing)
        tref = run_llg(prob, c0; kw..., nsteps = 274)
        _assert_same_llg(tref, t)
    end

    @testset "inactive-site parameters may differ (unvalidated placeholders)" begin
        # the dimer fixture: sites 3–4 inactive; resume must compare
        # magmom/alpha/g on ACTIVE sites only
        Hd = MC.TiledHamiltonian(_dimer_model(-0.02); dims = (1, 1, 1))
        up = SVector(0.0, 0.0, 1.0)
        cd0 = MC.SpinConfig([up, normalize(SVector(1.0, 0.0, 1.0)), up, up])
        pa = LLGProblem(Hd; magmom = 2.0, alpha = 0.7)
        path = joinpath(dir, "dimer.jld2")
        a = run_llg(pa, cd0; dt = 0.05, nsteps = 200, kT = 0.03, seed = 4,
                    checkpoint = path)
        pb = LLGProblem(Hd; magmom = [2.0, 2.0, 9.0, 9.0],
                        alpha = [0.7, 0.7, 0.3, 0.3])
        b = resume(path, pb)
        _assert_same_llg(a, b)
    end

    @testset "schema and mismatch guards" begin
        path = joinpath(dir, "guard.jld2")
        run_llg(prob, c0; kw..., nsteps = 100, checkpoint = path)
        @test_throws ArgumentError resume(joinpath(dir, "nope.jld2"), prob)
        # an MC checkpoint fed to the LLG resume: refused by KIND (not by a
        # confusing schema-version message)
        mcpath = joinpath(dir, "mc.jld2")
        MC.run_mc(H; kT = 0.5, sweeps_therm = 20, sweeps_measure = 40,
                  nbins = 4, seed = 1, checkpoint = mcpath,
                  checkpoint_interval = 20)
        err = try
            resume(mcpath, prob)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("kind", err.msg)
        # fingerprint mismatch: different dims
        H2 = MC.TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
        @test_throws ErrorException resume(path,
                                           LLGProblem(H2; magmom = 2.0,
                                                      alpha = 0.7);
                                           observables = obs)
        # problem-parameter mismatch: same Hamiltonian, different damping
        @test_throws ErrorException resume(path,
                                           LLGProblem(H; magmom = 2.0,
                                                      alpha = 0.5);
                                           observables = obs)
        # observable mismatch (functions are re-supplied; names/ncomps pinned)
        @test_throws ErrorException resume(path, prob)
        # target below the completed step
        @test_throws ArgumentError resume(path, prob; observables = obs,
                                          nsteps = 50)
        # cadence and ntasks validation
        @test_throws ArgumentError run_llg(prob, c0; kw..., checkpoint = path,
                                           checkpoint_interval = -1)
        @test_throws ArgumentError resume(path, prob; observables = obs,
                                          checkpoint_interval = -1)
        @test_throws ArgumentError resume(path, prob; observables = obs,
                                          ntasks = 0)
    end
end
