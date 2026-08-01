# Quantum (colored-noise) thermostat: filter machinery, counter map, classical
# byte-compat pin (Q-M1), and the Q-M4 physics tiers — the deterministic filter
# certificate (F1–F4, against the SHIPPED Barker–Bauer constants) and the
# dynamics gates (G1–G5, against the exact linear-response integral of the
# shipped filter's own closed-form discrete PSD — see qt_predictions.jl for
# the derivation and its numerical verification record).

include("qt_predictions.jl")

# A stable, deliberately non-trivial 2-section cascade for the unit gates
# (Jury: |a2| < 1 and |a1| < 1 + a2 for both sections).
_qt_test_sections() = [SD._Biquad(0.8, 0.3, -0.1, -0.5, 0.06),
                       SD._Biquad(1.1, -0.2, 0.05, -0.9, 0.25)]

function _qt_test_filter()
    sections = _qt_test_sections()
    return SD.ColoredNoiseFilter(sections, SD._stationary_factor(sections))
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
        @test sprint(show, f1) ==
              "ColoredNoiseFilter($(length(SD._QT_S_BIQUADS)) sections)"
        # defensive copies
        secs = _qt_test_sections()
        Lx = zeros(4, 4)
        fc = SD.ColoredNoiseFilter(secs, Lx)
        Lx[1, 1] = 99.0
        @test fc.L[1, 1] == 0.0
    end

    @testset "cascade ≡ state-space (algebraic twin)" begin
        # Consistency gate, not a correctness oracle: BOTH sides come from the
        # implementation (`_qt_cascade!` vs `_filter_state_space`), so a shared
        # misconception would pass here. Correctness is anchored by F2 below —
        # the impulse response against a test-local closed-form |H(z)|² — and by
        # the Lyapunov residual on `_stationary_cov`.
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
        L = SD._stationary_factor(sections)
        P = L * L'
        # `L` is a Cholesky factor of the extended-precision solution, so this
        # asserts that the ROUNDED factor still satisfies the equation the
        # extended-precision solve was given.
        @test norm(P - (A * P * A' + B * B')) <= 1e-12 * max(1.0, norm(P))
        # PSD is now structural (a real factor cannot produce a negative
        # eigenvalue), which is the property the old clamped eigendecomposition
        # could only approximate — see Q-F7 for the scan across τ.
        @test minimum(eigvals(Symmetric(P))) >= 0.0
        # the identity (pure-feedthrough) filter has P = 0 exactly — the factor
        # path must return zeros rather than let `cholesky` throw on it
        Ai, Bi, _, _ = SD._filter_state_space([SD._Biquad(1.0, 0.0, 0.0, 0.0,
                                                          0.0)])
        @test SD._stationary_cov(Ai, Bi) == zeros(2, 2)
        @test SD._stationary_factor([SD._Biquad(1.0, 0.0, 0.0, 0.0, 0.0)]) ==
              zeros(2, 2)
        # seeded stream check: lag-0 variance of the stationary cascade output
        # equals h·P·hᵀ + d² (the closed form) within 4σ
        noise_filter = _qt_test_filter()
        rng = MersenneTwister(7)
        nsamp = 200_000
        x = noise_filter.L * randn(rng, 4)            # stationary start
        xm = reshape(x, 4, 1)
        acc = 0.0
        for _ = 1:nsamp
            y = SD._qt_cascade!(xm, noise_filter.sections, 1, 0, randn(rng))
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
        noise_filter = SD.ColoredNoiseFilter([SD._Biquad(1.0, 0.0, 0.0, 0.0, 0.0)],
                                     Matrix(1.0 * I, 2, 2))
        fs = SD._init_filter_state(noise_filter, H, seed)
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
        noise_filter = _qt_test_filter()
        m = 2 * length(noise_filter.sections)
        fs = SD._init_filter_state(noise_filter, H, seed)
        xref = copy(fs.x)
        gth = fill(zero(SVector{3,Float64}), n)
        for step = 1:5
            SD._fill_noise_quantum!(gth, H, sigma, fs.x, noise_filter, seed, step)
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
                    for (j, biquad) in enumerate(noise_filter.sections)
                        p1 = off + 2 * j - 1
                        p2 = off + 2 * j
                        out = biquad.b0 * u + xref[p1, s]
                        xref[p1, s] = biquad.b1 * u - biquad.a1 * out + xref[p2, s]
                        xref[p2, s] = biquad.b2 * u - biquad.a2 * out
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

    @testset "quantum vs classical runs, determinism" begin
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
            # the colored filter reshapes the shared white draws — a same-seed
            # quantum run is a genuinely different trajectory
            @test rq.config != rc.config
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

    @testset "F1: shipped-filter PSD certificate" begin
        # the discrete PSD is exactly θ_fit(x_w), x_w = c·tan(ωΔt/2) — the fit
        # certificate transfers verbatim: rel ≤ 1.5e-2 on x_w ∈ [0.01, 6.6],
        # abs ≤ 3e-4 on (6.6, 200] (+ small Float64 z-evaluation headroom; the
        # 1e-12-exact identity claims live in dev/fit_qtb_filter.jl's 256-bit
        # check — see the decision record)
        for tau in (0.005, 0.0152, 0.05, 0.1)
            kt = 0.01
            dtf = tau * SD.HBAR_EV_FS / kt
            noise_filter = SD._build_quantum_filter(kt, dtf)
            @test length(noise_filter.sections) == length(SD._QT_S_BIQUADS)
            Pd = _qt_filter_psd(SD._filter_coeffs(noise_filter), dtf)
            c = 2.0 / tau
            omega_of_xw(xw) = (2.0 / dtf) * atan(xw / c)
            relmax = 0.0
            for xw in range(0.01, 6.6; length = 601)
                v = Pd(omega_of_xw(xw))
                relmax = max(relmax, abs(v - _qt_theta(xw)) / _qt_theta(xw))
                @test v > 0.0
            end
            @test relmax <= 1.6e-2
            absmax = 0.0
            for xw in exp.(range(log(6.6), log(200.0); length = 301))
                absmax = max(absmax, abs(Pd(omega_of_xw(xw)) - _qt_theta(xw)))
            end
            @test absmax <= 3.5e-4
            # DC gain 1 (Float64 z-form conditioning bounds the tolerance)
            @test abs(Pd(0.0) - 1.0) <= 1e-6
        end
    end

    @testset "F2: impulse response ≡ closed-form transfer function" begin
        kt = 0.01
        dtf = 0.05 * SD.HBAR_EV_FS / kt          # τ = 0.05 (moderate tails)
        noise_filter = SD._build_quantum_filter(kt, dtf)
        Pd = _qt_filter_psd(SD._filter_coeffs(noise_filter), dtf)
        m = 2 * length(noise_filter.sections)
        x = zeros(m, 1)
        ntap = 150_000
        h = Vector{Float64}(undef, ntap)
        for k = 1:ntap
            h[k] = SD._qt_cascade!(x, noise_filter.sections, 1, 0, k == 1 ? 1.0 : 0.0)
        end
        for omdt in (0.001, 0.01, 0.05, 0.2, 0.5, 1.5)
            om = omdt / dtf                       # ω [rad/fs] at ω·dt = omdt
            acc = 0.0 + 0.0im
            for k = 1:ntap
                acc += h[k] * cis(-om * (k - 1) * dtf)
            end
            @test isapprox(abs2(acc), Pd(om); rtol = 1e-5)
        end
    end

    # Q-F6 — the stationary law against an oracle the implementation cannot
    # reach. `Var(y)` of a stable LTI filter driven by unit white noise is
    # `(1/2π)∮|H_d(z)|² dθ`, which this computes from the STORED biquad
    # coefficients alone: no state space, no Lyapunov solve, no `L`. That
    # independence is the point. The predecessor gate (F4 below) compared a
    # stream started from `x = L·ζ` against `dot(h, P*h) + d²` with both sides
    # reading the same `L`, so it passed self-consistently while the shipped
    # thermal-noise power was wrong by up to +325 % — a reference sharing the
    # core routine is not an oracle (`~/Packages/CLAUDE.md`, Testing).
    #
    # An EMPIRICAL oracle is impossible here, which is why this one is analytic:
    # the filter's memory is `1/(1−ρ²) ~ 1e6` steps, so measuring the stationary
    # variance by burn-in would need ~1e7 steps for ~10 independent samples,
    # i.e. σ ≈ 45 %.
    #
    # The τ grid is FIXED and stratified across the accepted range, chosen before
    # measuring so that no point is selected for passing. The tolerance is set by
    # the ORACLE's own resolution (see the self-check), not by the
    # implementation's error, which is ~1e-16. Mutation resolved (verified by
    # actually making it): assembling and solving at Float64 turns all three
    # Q-gates red at the first τ. It does so by REFUSING — the Float64 matrix is
    # genuinely indefinite, so a factor-based solve cannot proceed — whereas the
    # predecessor, which formed `P` and took a clamped eigendecomposition, sailed
    # past that and returned a variance wrong by 1.5e-2 to 2.2e-2 relative here.
    # Both are caught; only one of them was ever loud.
    @testset "Q-F6: stationary variance vs the contour-integral oracle" begin
        function var_contour(sections, npts)
            acc = 0.0
            for k = 0:(npts - 1)
                z = cis(2π * k / npts)
                H = 1.0 + 0im
                for s in sections
                    H *= (s.b0 + s.b1 / z + s.b2 / z^2) /
                         (1 + s.a1 / z + s.a2 / z^2)
                end
                acc += abs2(H)
            end
            return acc / npts
        end
        for tau in (SD._QT_MIN_TAU, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1)
            noise_filter = SD._build_quantum_filter(tau * SD.HBAR_EV_FS, 1.0)
            _, _, h, d = SD._filter_state_space(noise_filter.sections)
            got = dot(h, (noise_filter.L * noise_filter.L') * h) + d^2
            coarse = var_contour(noise_filter.sections, 2^21)
            fine = var_contour(noise_filter.sections, 2^23)
            # The quadrature proves its OWN resolution, because a grid too coarse
            # for the smallest τ would silently under-resolve the peak rather
            # than fail: the poles sit at `1 − ρ = 7.3e-3·τ`, i.e. a width of
            # 7.4e-7 in θ at the bound, against a 2^23 spacing of 7.5e-7. Both
            # bounds below are the ORACLE's convergence with headroom, measured
            # against a 2^25 reference — `|coarse/fine − 1|` reaches 2.0e-5 and
            # `|got/fine − 1|` reaches 2e-7, both at the smallest τ and both
            # falling by orders as τ grows. The implementation's own error is
            # ~1e-16 and is nowhere near either bound.
            @test abs(coarse / fine - 1) < 1e-4
            @test abs(got / fine - 1) < 1e-6
        end
    end

    # Q-F7 — PSD-ness is a mathematical property of the stationary covariance of
    # a stable filter, so the expected answer is "no negative eigenvalue" at
    # every τ, from theory rather than from a run. It is cheap and scans densely,
    # but it is NOT sufficient on its own and must not be read as such: at the
    # worst point of the old defect (τ = 1.70e-4, variance wrong by +325 %) the
    # eigenvalue ratio was −1.3e-14 and looked healthy. Q-F6 is the primary gate.
    @testset "Q-F7: the stationary covariance is PSD across the accepted range" begin
        for tau in exp10.(range(log10(SD._QT_MIN_TAU), log10(SD._QT_MAX_TAU);
                                length = 60))
            L = SD._build_quantum_filter(tau * SD.HBAR_EV_FS, 1.0).L
            ev = eigvals(Symmetric(L * L'))
            @test minimum(ev) >= 0.0          # structural: L is a real factor
            @test maximum(ev) > 0.0
        end
    end

    # Q-F8 — the shipped working precision is CONVERGED, not merely pinned:
    # raising it further changes the shipped Float64 by no more than its own last
    # bit. That is a statement about the computation, so it needs no captured
    # output and no recapture date — unlike a byte pin, it stays true when the
    # biquads are re-fitted.
    #
    # The bound is a few ulp rather than exact equality, and that is not slack:
    # rounding a p-bit result to Float64 is a double rounding, so whenever the
    # exact value sits near a Float64 boundary the last bit can flip with p at
    # ANY precision. Demanding bit equality here would be demanding that no
    # entry ever lands near a boundary, which is a property of the numbers, not
    # of the convergence being tested.
    #
    # The other direction (that the precision is NEEDED) is not asserted here but
    # in Q-F6, whose mutation note records what the Float64 solve actually
    # produces: a variance wrong by 1.5e-2 to 2.2e-2 relative at these very τ.
    # Asserting "low precision throws" would be fragile — BigFloat at a Float64
    # mantissa is not Float64 arithmetic, and whether the Cholesky happens to
    # fail is not the property worth gating.
    @testset "Q-F8: the Lyapunov precision is converged, not just pinned" begin
        for tau in (SD._QT_MIN_TAU, 1e-3, 1e-1)
            sections = SD._build_quantum_filter(tau * SD.HBAR_EV_FS, 1.0).sections
            shipped = SD._stationary_factor(sections)
            scale = maximum(abs, shipped)
            for higher in (192, 256)
                @test maximum(abs, SD._stationary_factor(sections;
                                                         precision = higher) .-
                                   shipped) <= 8 * eps(Float64) * scale
            end
        end
    end

    @testset "F3/F4: shipped-filter stationary law" begin
        kt = 0.01
        dtf = 1.0                                 # τ ≈ 0.0152 (the G1 setting)
        noise_filter = SD._build_quantum_filter(kt, dtf)
        A, B, h, d = SD._filter_state_space(noise_filter.sections)
        P = noise_filter.L * noise_filter.L'
        @test norm(P - (A * P * A' + B * B')) <= 1e-10 * max(1.0, norm(P))
        # F4 is a CONSISTENCY CHECK, not an oracle, and that distinction is the
        # whole reason the +325 % covariance defect survived: both sides below
        # read the same `L`, so they agree even when `L` is wrong. It stays
        # because it fences the stream against the closed form; correctness of
        # the closed form itself is Q-F6's job.
        # F4: seeded stream variance vs the closed form h·P·hᵀ + d²
        rng = MersenneTwister(11)
        x = noise_filter.L * randn(rng, length(B))
        xm = reshape(copy(x), length(B), 1)
        nsamp = 200_000
        acc = 0.0
        for _ = 1:nsamp
            y = SD._qt_cascade!(xm, noise_filter.sections, 1, 0, randn(rng))
            acc += y * y
        end
        var_ref = dot(h, P * h) + d^2
        tol = 4 * sqrt(5 * 2 / nsamp) * var_ref
        @test abs(acc / nsamp - var_ref) <= tol
    end

    # --- dynamics gates: measured vs the exact linear-response integral of the
    # SHIPPED filter's closed-form discrete PSD (the θ-fit error is F1's
    # business; the derivation + its verification record: qt_predictions.jl).
    # Statistical errors via equilibrium_stats' binning; tolerance = 3σ plus a
    # 1.5% systematic allowance (measured O(dt)/anharmonic bound).

    # measured ⟨E⟩ − E₀ (via b·⟨1 − e_z⟩) and its error from a run's :ez series
    function _qt_gate_run(prob, H, kt, dtf, nsteps, alpha_run; seed = 21,
                          mi = 20)
        up = SVector(0.0, 0.0, 1.0)
        cfg = MC.SpinConfig([up for _ = 1:MC.n_sites(H)])
        obs = [Observable(:ez, 1, v -> v.config[1][3])]
        result = run_llg(prob, cfg; dt = dtf, nsteps, kT = kt, seed,
                      measure_interval = mi, observables = obs,
                      thermostat = SD.QuantumThermostat())
        st = equilibrium_stats(result; evaluables = Evaluable[])[:ez]
        return 1.0 - st.mean[1], st.err[1]
    end

    _qt_shipped_psd(kt, dtf) =
        _qt_filter_psd(SD._filter_coeffs(SD._build_quantum_filter(kt, dtf)),
                       dtf)

    @testset "G1: Larmor occupation vs the α-broadened integral" begin
        # ħω̃ = 0.03 eV (B = 259.14 T at g = 2), μ = 20 ⇒ b = 0.3 eV — the
        # linear-regime lever b/kT = 30 kills sphere anharmonicity while
        # x₀ = ħω₀/kT ≈ 3 stays quantum
        H = MC.TiledHamiltonian(_uniaxial_model(1e-300); dims = (1, 1, 1))
        Bz = 0.03 / (2 * SD.MU_B_EV_T)
        kt = 0.01
        dtf = 1.0
        omega_t = _larmor_omega(2.0, Bz)
        b = 20.0 * SD.MU_B_EV_T * Bz
        Pd = _qt_shipped_psd(kt, dtf)
        for (alpha, nsteps) in ((0.1, 600_000), (0.5, 600_000))
            prob = LLGProblem(H; magmom = 20.0, alpha,
                              b_ext = (0.0, 0.0, Bz))
            one_m_ez, err = _qt_gate_run(prob, H, kt, dtf, nsteps, alpha)
            meas = b * one_m_ez
            pred = _qt_predict_occupation(kt, dtf, omega_t, alpha, Pd)
            @test abs(meas - pred) <= 3 * b * err + 0.015 * pred
            # quantum suppression vs classical equipartition (⟨E⟩−E₀ = kT):
            # θ(3) ≈ 0.16 — a classical result would miss by tens of σ
            @test meas < 0.5 * kt
        end
        # the α-dependence IS the QTB approximation (naive n_BE is α-blind):
        # α = 0.5's prediction is ~1.3× α = 0.1's — assert the integral moves
        p1 = _qt_predict_occupation(kt, dtf, omega_t, 0.1, Pd)
        p5 = _qt_predict_occupation(kt, dtf, omega_t, 0.5, Pd)
        @test p5 / p1 > 1.2
    end

    @testset "G2: Einstein specific heat from two temperatures" begin
        H = MC.TiledHamiltonian(_uniaxial_model(1e-300); dims = (1, 1, 1))
        Bz = 0.03 / (2 * SD.MU_B_EV_T)
        b = 20.0 * SD.MU_B_EV_T * Bz
        omega_t = _larmor_omega(2.0, Bz)
        dtf = 1.0
        alpha = 0.05
        prob = LLGProblem(H; magmom = 20.0, alpha, b_ext = (0.0, 0.0, Bz))
        es = Float64[]
        errs = Float64[]
        preds = Float64[]
        for kt in (0.009, 0.011)
            one_m_ez, err = _qt_gate_run(prob, H, kt, dtf, 2_000_000, alpha)
            push!(es, b * one_m_ez)
            push!(errs, b * err)
            # each temperature builds ITS OWN filter (τ changes with kT)
            push!(preds, _qt_predict_occupation(kt, dtf, omega_t, alpha,
                                                _qt_shipped_psd(kt, dtf)))
        end
        c_meas = (es[2] - es[1]) / 0.002          # d⟨E⟩/dkT [kB units]
        c_pred = (preds[2] - preds[1]) / 0.002
        sigma_c = hypot(errs[1], errs[2]) / 0.002
        @test abs(c_meas - c_pred) <= 3 * sigma_c + 0.05 * c_pred
        # the killer assertion: quantum suppression of the specific heat
        # (Einstein value ≈ 0.50 at x₀ = 3; classical value is exactly 1)
        @test c_meas < 0.6
        @test c_pred < 0.55
    end

    @testset "G3: classical recovery at x₀ ≈ 0.02" begin
        H = MC.TiledHamiltonian(_uniaxial_model(1e-300); dims = (1, 1, 1))
        Bz = 0.002 / (2 * SD.MU_B_EV_T)           # ħω̃ = 0.002 eV
        kt = 0.1
        dtf = 0.5                                 # τ = 0.076 (guard-compliant)
        alpha = 0.5
        mu = 2000.0                               # b = 2 eV ⇒ b/kT = 20
        b = mu * SD.MU_B_EV_T * Bz
        omega_t = _larmor_omega(2.0, Bz)
        prob = LLGProblem(H; magmom = mu, alpha, b_ext = (0.0, 0.0, Bz))
        one_m_ez, err = _qt_gate_run(prob, H, kt, dtf, 4_000_000, alpha;
                                     mi = 40)
        meas = b * one_m_ez
        pred = _qt_predict_occupation(kt, dtf, omega_t, alpha,
                                      _qt_shipped_psd(kt, dtf))
        @test abs(meas - pred) <= 3 * b * err + 0.015 * pred
        # quadrature-level recovery statement (θ(0.02) ≈ 0.99; the residual
        # deficit is the α-broadening tail — see the decision record): the
        # sub-percent classical-limit claim is F1's low-x certificate, not a
        # statistics gate
        @test pred / kt > 0.97
    end

    @testset "G4: dimer two-mode occupations" begin
        # SALC coefficient −0.3/(2√3) ⇒ J_eff = −0.3 (the isotropic l = 1 SALC
        # carries a 2√3 normalization — always measure with _dimer_J)
        model = _dimer_model(-0.3 / (2 * sqrt(3.0)))
        H = MC.TiledHamiltonian(model; dims = (1, 1, 1))
        J = _dimer_J(H)
        @test isapprox(J, -0.3; rtol = 1e-10)
        mu = 60.0
        Bz = 0.01 / (2 * SD.MU_B_EV_T)            # ħω_u·(1+α²) = 0.01 eV
        kt = 0.01
        dtf = 1.0
        alpha = 0.1
        prob = LLGProblem(H; magmom = mu, alpha, b_ext = (0.0, 0.0, Bz))
        up = SVector(0.0, 0.0, 1.0)
        cfg = MC.SpinConfig([up for _ = 1:MC.n_sites(H)])
        obs = [Observable(:e12, 1, v -> dot(v.config[1], v.config[2])),
               Observable(:ezsum, 1, v -> v.config[1][3] + v.config[2][3])]
        result = run_llg(prob, cfg; dt = dtf, nsteps = 3_000_000, kT = kt,
                      seed = 33, measure_interval = 40, observables = obs,
                      thermostat = SD.QuantumThermostat())
        st = equilibrium_stats(result; evaluables = Evaluable[])
        pred = _qt_predict_dimer(kt, dtf, J, Bz, mu, 2.0, alpha,
                                 _qt_shipped_psd(kt, dtf))
        m12 = 1.0 - st[:e12].mean[1]
        e12_err = st[:e12].err[1]
        @test abs(m12 - pred.one_m_e12) <= 3 * e12_err + 0.02 * pred.one_m_e12
        mez = 2.0 - st[:ezsum].mean[1]
        ez_err = st[:ezsum].err[1]
        @test abs(mez - pred.sum_one_m_ez) <=
              3 * ez_err + 0.02 * pred.sum_one_m_ez
        # classical equipartition (2 modes ⇒ 2kT total) is far away: the
        # measured total energy must sit near the quantum sum
        b_u = mu * SD.MU_B_EV_T * Bz
        E_meas = b_u * mez + abs(J) * m12
        @test E_meas < 0.6 * 2 * kt
        @test abs(E_meas - pred.E) <= 3 * (b_u * ez_err + abs(J) * e12_err) +
                                      0.03 * pred.E
    end

    @testset "G5: MC-mismatch tripwire (classical-only cross-checks)" begin
        model = _dimer_model(-0.3 / (2 * sqrt(3.0)))
        H = MC.TiledHamiltonian(model; dims = (1, 1, 1))
        J = _dimer_J(H)
        mu = 60.0
        kt = 0.01
        dtf = 1.0
        alpha = 0.1
        # classical MC: exact Boltzmann ⟨E⟩ − E₀ = kT·(1 − 2y/(e^{2y}−1)) ≈ kT
        eobs = [Observable(:energy, 1, v -> v.energy)]
        mc = MC.run_mc(H; kT = kt, seed = 3, sweeps_therm = 10_000,
                       sweeps_measure = 50_000, observables = eobs,
                       evaluables = MC.Evaluable[])
        mst = mc.points[1].stats
        E0 = MC.total_energy(H, MC.SpinConfig([SVector(0.0, 0.0, 1.0)
                                               for _ = 1:MC.n_sites(H)]))
        E_mc = mst[:energy].mean[1] - E0
        mc_err = mst[:energy].err[1]
        @test isapprox(E_mc, _qt_classical_exact(kt, abs(J)); atol = 5 * mc_err)
        # quantum sLLG on the same model at the same kT (b = 0: the uniform
        # mode is the free-rotation zero mode; only the optical mode counts)
        prob = LLGProblem(H; magmom = mu, alpha)
        up = SVector(0.0, 0.0, 1.0)
        cfg = MC.SpinConfig([up for _ = 1:MC.n_sites(H)])
        obs = [Observable(:e, 1, v -> v.energy)]
        result = run_llg(prob, cfg; dt = dtf, nsteps = 1_000_000, kT = kt,
                      seed = 8, measure_interval = 20, observables = obs,
                      thermostat = SD.QuantumThermostat())
        st = equilibrium_stats(result; evaluables = Evaluable[])[:e]
        E_q = st.mean[1] - E0
        q_err = st.err[1]
        pred = _qt_predict_dimer(kt, dtf, J, 0.0, mu, 2.0, alpha,
                                 _qt_shipped_psd(kt, dtf))
        @test abs(E_q - pred.E) <= 3 * q_err + 0.02 * pred.E
        # the tripwire: quantum and classical MUST disagree, loudly — this is
        # what documents that MC cross-validation is classical-only
        @test (E_mc - E_q) / hypot(mc_err, q_err) > 5
    end

    @testset "G6: ring S(q,ω) intensities (example tier)" begin
        # 4-ring + field: modes m = 0..3 with b_m = b + 2|J|(1 − cos(2πm/4)),
        # spanning x = ħω/kT ≈ 1.0 / 2.2 / 3.5 — the ω-integrated inelastic
        # S⁺⁻(q_m) measures each mode's occupation ⟨|c_m|²⟩ = 2E_m/b_m, so one
        # spectrum exhibits the quantum suppression ACROSS the dispersion
        # (θ(x₂)/θ(x₀) ≈ 0.2; classically all modes carry kT).
        model = _ring_model(-0.375 / (2 * sqrt(3.0)))
        H = MC.TiledHamiltonian(model; dims = (1, 1, 1))
        J = _ring_J(H)
        @test isapprox(J, -0.375; rtol = 1e-10)
        mu = 120.0                                 # large μ ⇒ small ⟨|m|²⟩ at
        kt = 0.01                                  # fixed x (renorm systematic)
        dtf = 1.0
        alpha = 0.1
        Bz = 0.6 / (mu * SD.MU_B_EV_T)            # b = 0.6 eV ⇒ ħω̃₀ = 0.01 eV
        prob = LLGProblem(H; magmom = mu, alpha, b_ext = (0.0, 0.0, Bz))
        up = SVector(0.0, 0.0, 1.0)
        cfg = MC.SpinConfig([up for _ = 1:MC.n_sites(H)])
        # equal-time mode intensities |F₊(q_m)|² (unitary DFT, N = 4) carry the
        # error bars; the trajectory feeds the S(q,ω) twin of the same numbers
        mode_obs = [Observable(Symbol(:i, m), 1,
                        let ph = [cis(-2π * m * (j - 1) / 4) for j = 1:4]
                            v -> abs2(sum(ph[j] * (v.config[j][1] +
                                                   im * v.config[j][2])
                                          for j = 1:4)) / 4
                        end) for m = 0:2]
        mi = 15
        nt = 131_072 + 2_000                       # 2¹⁷ analysis frames + burn-in
        result = run_llg(prob, cfg; dt = dtf, nsteps = mi * (nt - 1), kT = kt,
                      seed = 27, measure_interval = mi,
                      observables = vcat(mode_obs, [trajectory_observable(H)]),
                      thermostat = SD.QuantumThermostat())
        pred = _qt_predict_ring(kt, dtf, J, Bz, mu, 2.0, alpha, 4,
                                _qt_shipped_psd(kt, dtf))
        st = equilibrium_stats(result; discard = 2_000, evaluables = Evaluable[])
        r = structure_factor(result, H, _dimer_crystal(),
                             [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]];
                             window = :none, discard = 2_000)
        @test r.nfft == 131_072
        spm = sqw_plusminus(r)
        p = 2.0 / (SD.HBAR_EV_FS * mu * (1 + alpha^2))
        for m = 0:2
            Ipred = pred.I_m[m + 1]
            Ihat = st[Symbol(:i, m)].mean[1]
            err = st[Symbol(:i, m)].err[1]
            # 4% systematic: linearization (per-site ⟨|m|²⟩ ≈ 1.5%), thermal
            # frequency renormalization, and the ≤ 0.4% integrator bias
            @test abs(Ihat - Ipred) <= 3 * err + 0.04 * Ipred
            # the S(q,ω) route: ω-integrated inelastic bin sum ≡ the same
            # occupation (Parseval; +1% for the elastic/window seam)
            Isqw = sum(spm[m + 1, :]) / (r.nfft * r.dt_meas)
            @test abs(Isqw - Ipred) <= 3 * err + 0.05 * Ipred
            # the peak sits at the linear mode frequency ω_m = p·b_m
            if m >= 1
                wpk = r.omegas[argmax(spm[m + 1, :])]
                @test abs(wpk - p * pred.b_m[m + 1]) <= 0.05 * p * pred.b_m[m + 1]
            end
        end
        # quantum suppression across the dispersion: classically every mode
        # carries kT (I_m^cl = 2kT/b_m), so the zone-boundary/uniform intensity
        # ratio would be b₀/b₂ — the measured ratio must sit far below it
        i0 = st[:i0].mean[1]
        i2 = st[:i2].mean[1]
        @test i2 / i0 < 0.5 * pred.b_m[1] / pred.b_m[3]
        @test i2 < 0.35 * 2 * kt / pred.b_m[3]
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
        # lower conditioning bound (τ ≈ 1.5e-6 ≪ 1e-4): a clear refusal, not
        # an opaque section-stability error
        @test_throws ArgumentError run_llg(prob, cfg; dt = 1.0, nsteps = 10,
                                           kT = 1e-6, seed = 1,
                                           thermostat = SD.QuantumThermostat())
        # checkpointing a quantum run works since schema v3 (gates live in
        # test_checkpoint.jl's "quantum thermostat (schema v3)" testset)
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
