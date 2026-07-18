# Quantum (colored-noise) thermostat — milestone-1 tier: filter machinery,
# counter map, classical byte-compat pin, wiring gates. The physics gates
# (PSD certificate, Larmor occupation, Einstein specific heat) land with the
# pinned Barker–Bauer fit constants; until then `_build_quantum_filter` is the
# identity placeholder and a quantum run is bitwise the classical one — itself
# the wiring gate here.

# A stable, deliberately non-trivial 2-section cascade for the unit gates
# (Jury: |a2| < 1 and |a1| < 1 + a2 for both sections).
_qt_test_sections() = [SD._Biquad(0.8, 0.3, -0.1, -0.5, 0.06),
                       SD._Biquad(1.1, -0.2, 0.05, -0.9, 0.25)]

function _qt_test_filter()
    sections = _qt_test_sections()
    A, B, _, _ = SD._filter_state_space(sections)
    return SD.ColoredNoiseFilter(sections,
                                 SD._stationary_sqrt(SD._stationary_cov(A, B)))
end

@testset "quantum thermostat" begin
    @testset "construction and validation" begin
        idb = SD._Biquad(1.0, 0.0, 0.0, 0.0, 0.0)
        L2 = Matrix(1.0 * I, 2, 2)
        @test_throws ArgumentError SD.ColoredNoiseFilter(SD._Biquad[],
                                                         zeros(0, 0))
        many = fill(idb, SD._QT_MAX_NSECTIONS + 1)
        @test_throws ArgumentError SD.ColoredNoiseFilter(many,
            zeros(2 * length(many), 2 * length(many)))
        @test_throws ArgumentError SD.ColoredNoiseFilter(
            [SD._Biquad(NaN, 0.0, 0.0, 0.0, 0.0)], L2)
        # unstable: |a2| ≥ 1, then |a1| ≥ 1 + a2
        @test_throws ArgumentError SD.ColoredNoiseFilter(
            [SD._Biquad(1.0, 0.0, 0.0, 0.0, 1.0)], L2)
        @test_throws ArgumentError SD.ColoredNoiseFilter(
            [SD._Biquad(1.0, 0.0, 0.0, -2.0, 0.9)], L2)
        @test_throws DimensionMismatch SD.ColoredNoiseFilter([idb], zeros(3, 3))
        @test_throws ArgumentError SD.ColoredNoiseFilter([idb],
                                                         fill(NaN, 2, 2))
        # the builder is a pure function of (kT, dt)
        f1 = SD._build_quantum_filter(0.01, 1.0)
        f2 = SD._build_quantum_filter(0.01, 1.0)
        @test f1.sections == f2.sections
        @test f1.L == f2.L
        @test sprint(show, f1) == "ColoredNoiseFilter(1 sections)"
        # defensive copies
        secs = _qt_test_sections()
        Lx = zeros(4, 4)
        fc = SD.ColoredNoiseFilter(secs, Lx)
        Lx[1, 1] = 99.0
        @test fc.L[1, 1] == 0.0
    end

    @testset "cascade ≡ state-space (algebraic twin)" begin
        sections = _qt_test_sections()
        m = 2 * length(sections)
        A, B, h, d = SD._filter_state_space(sections)
        rng = MersenneTwister(42)
        x = zeros(m, 1)                       # one site, one component (off = 0)
        svec = zeros(m)
        for _ = 1:200
            xi = randn(rng)
            y_ss = dot(h, svec) + d * xi      # output reads the PRE-update state
            y = SD._qt_cascade!(x, sections, 1, 0, xi)
            svec = A * svec + B * xi
            @test isapprox(y, y_ss; rtol = 1e-12, atol = 1e-14)
            @test isapprox(x[:, 1], svec; rtol = 1e-12, atol = 1e-14)
        end
    end

    @testset "Lyapunov stationary covariance and square root" begin
        sections = _qt_test_sections()
        A, B, h, d = SD._filter_state_space(sections)
        P = SD._stationary_cov(A, B)
        @test norm(P - (A * P * A' + B * B')) <= 1e-12 * max(1.0, norm(P))
        L = SD._stationary_sqrt(P)
        @test norm(L * L' - P) <= 1e-12 * max(1.0, norm(P))
        # the identity (pure-feedthrough) filter has P = 0 exactly — the
        # clamped square root must not throw (plain cholesky would)
        Ai, Bi, _, _ = SD._filter_state_space([SD._Biquad(1.0, 0.0, 0.0, 0.0,
                                                          0.0)])
        Pi = SD._stationary_cov(Ai, Bi)
        @test Pi == zeros(2, 2)
        @test SD._stationary_sqrt(Pi) == zeros(2, 2)
        # seeded stream check: lag-0 variance of the stationary cascade output
        # equals h·P·hᵀ + d² (the closed form) within 4σ
        filt = _qt_test_filter()
        rng = MersenneTwister(7)
        nsamp = 200_000
        x = filt.L * randn(rng, 4)            # stationary start
        xm = reshape(x, 4, 1)
        acc = 0.0
        for _ = 1:nsamp
            y = SD._qt_cascade!(xm, filt.sections, 1, 0, randn(rng))
            acc += y * y
        end
        var_ref = dot(h, P * h) + d^2
        # var of y² estimate ≈ 2·var_ref²/n per sample (Gaussian, ignoring
        # autocorrelation — take 4σ with a generous correlation factor 5)
        tol = 4 * sqrt(5 * 2 / nsamp) * var_ref
        @test abs(acc / nsamp - var_ref) <= tol
    end

    @testset "counter map: stationary init draws (step 0, slots 2…)" begin
        H = MC.TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
        seed = UInt64(0x5eed)
        # L = I makes the init state the raw ζ draws — pins the counters exactly
        filt = SD.ColoredNoiseFilter([SD._Biquad(1.0, 0.0, 0.0, 0.0, 0.0)],
                                     Matrix(1.0 * I, 2, 2))
        fs = SD._init_filter_state(filt, H, seed)
        @test size(fs.x) == (6, MC.n_sites(H))
        for s = 1:MC.n_sites(H)
            if H.site_active[s]
                zeta = Float64[]
                for b = 0:2                   # 3·NS = 3 blocks: slots 2, 3, 4
                    blk = SD.philox_block(seed, (UInt32(s), UInt32(0),
                                                 UInt32(2 + b), SD._DOMAIN_SD))
                    n1, n2 = SD.philox_normal2(blk)
                    push!(zeta, n1, n2)
                end
                @test fs.x[:, s] == zeta
            else
                @test all(fs.x[:, s] .=== 0.0)   # never drawn, exact +0.0
            end
        end
    end

    @testset "step fill: slots 0/1 shared white draws + cascade reference" begin
        H = MC.TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
        n = MC.n_sites(H)
        seed = UInt64(0xabcdef)
        sigma = [H.site_active[s] ? 0.5 + 0.1 * s : 0.0 for s = 1:n]
        filt = _qt_test_filter()
        m = 2 * length(filt.sections)
        fs = SD._init_filter_state(filt, H, seed)
        xref = copy(fs.x)
        gth = fill(zero(SVector{3,Float64}), n)
        for step = 1:5
            SD._fill_noise_quantum!(gth, H, sigma, fs.x, filt, seed, step)
            for s = 1:n
                if !H.site_active[s]
                    @test gth[s] === zero(SVector{3,Float64})
                    @test all(xref[:, s] .== 0.0)
                    continue
                end
                # straight-line reference: the SAME slots-0/1 draws as the
                # classical fill, each pushed through an independent DF2T pass
                c1, c2 = SD._noise_ctrs(s, step)
                n1, n2 = SD.philox_normal2(SD.philox_block(seed, c1))
                n3, _ = SD.philox_normal2(SD.philox_block(seed, c2))
                ys = zeros(3)
                for (c, xi) in enumerate((n1, n2, n3))
                    off = (c - 1) * m
                    u = xi
                    for (j, bq) in enumerate(filt.sections)
                        p1 = off + 2 * j - 1
                        p2 = off + 2 * j
                        out = bq.b0 * u + xref[p1, s]
                        xref[p1, s] = bq.b1 * u - bq.a1 * out + xref[p2, s]
                        xref[p2, s] = bq.b2 * u - bq.a2 * out
                        u = out
                    end
                    ys[c] = u
                end
                @test gth[s] === sigma[s] * SVector(ys[1], ys[2], ys[3])
                @test fs.x[:, s] == xref[:, s]
            end
        end
    end

    @testset "classical byte-compat pin (Philox words, portable)" begin
        # Frozen literals of the classical counter/stream contract: pure integer
        # Philox (Random123-KAT-stable), NOT libm-dependent normals. If this
        # moves, seeded classical trajectories moved — breaking-note territory.
        seed = 0x00000000deadbeef
        c1, c2 = SD._noise_ctrs(2, 5)
        @test c1 === (0x00000002, 0x00000005, 0x00000000, 0x53440000)
        @test c2 === (0x00000002, 0x00000005, 0x00000001, 0x53440000)
        @test SD.philox_block(seed, c1) ===
              (0xc2d528f9, 0x758f0a89, 0xc98bcbda, 0x23c90690)
        @test SD.philox_block(seed, c2) ===
              (0x4977531e, 0xe5112710, 0x86a5a718, 0x4084dd88)
        @test SD._DOMAIN_SD === 0x53440000
        # the fill maps those words through the package's own Box–Muller facade
        H = MC.TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
        n = MC.n_sites(H)
        sigma = [H.site_active[s] ? 1.25 : 0.0 for s = 1:n]
        gth = fill(zero(SVector{3,Float64}), n)
        SD._fill_noise!(gth, H, sigma, seed, 5)
        n1, n2 = SD.philox_normal2(SD.philox_block(seed, c1))
        n3, _ = SD.philox_normal2(SD.philox_block(seed, c2))
        @test gth[2] === 1.25 * SVector(n1, n2, n3)
    end

    @testset "wiring gate: identity filter ⇒ quantum ≡ classical bitwise" begin
        H = MC.TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
        prob = LLGProblem(H; magmom = 2.0, alpha = 0.3)
        rng = MersenneTwister(3)
        cfg = _rand_config(rng, H)
        for integrator in (DepondtMertens(), HeunProjected())
            rc = run_llg(prob, cfg; dt = 1.0, nsteps = 300, kT = 0.005,
                         seed = 17, integrator)
            rq = run_llg(prob, cfg; dt = 1.0, nsteps = 300, kT = 0.005,
                         seed = 17, integrator,
                         thermostat = SD.QuantumThermostat())
            @test rq.config == rc.config
            @test rq.energies == rc.energies
            @test rq.times == rc.times
            @test rc.thermostat == "classical"
            @test rq.thermostat == "quantum"
            @test occursin("quantum", sprint(show, rq))
            @test !occursin("quantum", sprint(show, rc))
        end
        # determinism: same seed reproduces; a different seed does not
        r1 = run_llg(prob, cfg; dt = 1.0, nsteps = 200, kT = 0.005, seed = 4,
                     thermostat = SD.QuantumThermostat())
        r2 = run_llg(prob, cfg; dt = 1.0, nsteps = 200, kT = 0.005, seed = 4,
                     thermostat = SD.QuantumThermostat())
        r3 = run_llg(prob, cfg; dt = 1.0, nsteps = 200, kT = 0.005, seed = 5,
                     thermostat = SD.QuantumThermostat())
        @test r1.config == r2.config
        @test r1.config != r3.config
    end

    @testset "validation" begin
        H = MC.TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
        prob = LLGProblem(H; magmom = 2.0, alpha = 0.3)
        up = SVector(0.0, 0.0, 1.0)
        cfg = MC.SpinConfig([up for _ = 1:MC.n_sites(H)])
        # quantum needs a temperature
        @test_throws ArgumentError run_llg(prob, cfg; dt = 1.0, nsteps = 10,
                                           thermostat = SD.QuantumThermostat())
        # τ = kT·dt/ħ guard (0.5 eV × 1 fs / ħ ≈ 0.76 > 0.1)
        @test_throws ArgumentError run_llg(prob, cfg; dt = 1.0, nsteps = 10,
                                           kT = 0.5, seed = 1,
                                           thermostat = SD.QuantumThermostat())
        # checkpointing a quantum run awaits schema v3
        @test_throws ArgumentError run_llg(prob, cfg; dt = 1.0, nsteps = 10,
                                           kT = 0.005, seed = 1,
                                           thermostat = SD.QuantumThermostat(),
                                           checkpoint = tempname() * ".jld2")
        # classical runs are untouched by the kwarg's existence
        r = run_llg(prob, cfg; dt = 1.0, nsteps = 10, kT = 0.005, seed = 1,
                    thermostat = SD.ClassicalThermostat())
        @test r.thermostat == "classical"
    end

    @testset "equilibrium_stats refuses Boltzmann evaluables on quantum runs" begin
        H = MC.TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
        prob = LLGProblem(H; magmom = 2.0, alpha = 0.5)
        rng = MersenneTwister(11)
        cfg = _rand_config(rng, H)
        obs = MC.standard_observables(H)
        rq = run_llg(prob, cfg; dt = 1.0, nsteps = 2000, kT = 0.005, seed = 8,
                     observables = obs, thermostat = SD.QuantumThermostat())
        @test_throws ArgumentError equilibrium_stats(rq)   # default evaluables
        raw = equilibrium_stats(rq; evaluables = Evaluable[])
        @test haskey(raw, :energy)
        forced = equilibrium_stats(rq; allow_evaluables = true)
        @test haskey(forced, :specific_heat)
        # classical runs are untouched
        rc = run_llg(prob, cfg; dt = 1.0, nsteps = 2000, kT = 0.005, seed = 8,
                     observables = obs)
        @test haskey(equilibrium_stats(rc), :specific_heat)
    end
end
