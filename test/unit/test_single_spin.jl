# Validation gate (a): a single spin in a constant external field has an analytic
# solution — uniform Larmor precession at α = 0, the log-tangent damped spiral at
# α > 0. The SLCE coupling of the fixture is made negligible (J = 1e-20) so the
# Zeeman term dominates; Depondt–Mertens is EXACT for a constant rotation vector,
# so the α = 0 trajectory must match pointwise to rounding.

# Analytic damped-precession state at time t: spin started at (θ0, φ0 = 0) in a
# +z field B, precession ω_p = γB/(1+α²), decay tan(θ/2) = tan(θ0/2)·e^(−t/τ),
# τ = (1+α²)/(αγB).
function _damped_analytic(θ0, t, ωl, α)
    ωp = ωl / (1 + α^2)
    τ = (1 + α^2) / (α * ωl)
    θ = 2 * atan(tan(θ0 / 2) * exp(-t / τ))
    φ = ωp * t
    return SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
end

@testset "single spin in a constant field (gate a)" begin
    H = MC.TiledHamiltonian(_dimer_model(1e-20); dims = (1, 1, 1))
    B = 10.0
    ωl = _larmor_omega(2.0, B)                     # rad/fs, magmom-independent
    up = SVector(0.0, 0.0, 1.0)
    e0 = SVector(1.0, 0.0, 0.0)

    @testset "α = 0: Larmor precession, Depondt exact per step" begin
        prob = LLGProblem(H; magmom = 2.0, b_ext = (0.0, 0.0, B))
        config0 = MC.SpinConfig([e0, e0, up, up])
        nsteps = 3600                              # ≈ one period (T = 3572.4 fs)
        res = run_llg(prob, config0; dt = 1.0, nsteps = nsteps,
                      renorm_interval = 0)
        t = nsteps * 1.0
        analytic = SVector(cos(ωl * t), sin(ωl * t), 0.0)
        # constant rotation vector ⇒ the only error is rounding accumulation
        @test res.config[1] ≈ analytic atol = 1e-9
        @test res.config[2] ≈ analytic atol = 1e-9
        # precession sense: starting at +x it moves toward +y (γ > 0, B ∥ +z)
        early = run_llg(prob, config0; dt = 1.0, nsteps = 10)
        @test early.config[1][2] > 0
        # magmom does not change the Larmor frequency (pure Zeeman)
        prob2 = LLGProblem(H; magmom = 5.0, b_ext = (0.0, 0.0, B))
        res2 = run_llg(prob2, config0; dt = 1.0, nsteps = nsteps,
                       renorm_interval = 0)
        @test res2.config[1] ≈ analytic atol = 1e-9
    end

    @testset "α > 0: damped spiral matches the analytic solution" begin
        α = 0.5
        prob = LLGProblem(H; magmom = 2.0, alpha = α, b_ext = (0.0, 0.0, B))
        config0 = MC.SpinConfig([e0, e0, up, up])
        t_end = 500.0
        res = run_llg(prob, config0; dt = 1.0, nsteps = 500)
        analytic = _damped_analytic(π / 2, t_end, ωl, α)
        @test res.config[1] ≈ analytic rtol = 1e-4
        # observed order 2: halving dt shrinks the error ≈ 4×
        err(dt) = norm(run_llg(prob, config0; dt = dt,
                               nsteps = round(Int, t_end / dt)).config[1] -
                       analytic)
        r = err(2.0) / err(1.0)
        @test 3.0 < r < 5.0
        # energy decreases monotonically toward alignment with B
        @test res.energies[end] < res.energies[1]
        @test all(diff(res.energies) .< 1e-12)
    end

    @testset "HeunProjected agrees at order 2" begin
        α = 0.5
        prob = LLGProblem(H; magmom = 2.0, alpha = α, b_ext = (0.0, 0.0, B))
        config0 = MC.SpinConfig([e0, e0, up, up])
        analytic = _damped_analytic(π / 2, 500.0, ωl, α)
        err(dt) = norm(run_llg(prob, config0; dt = dt,
                               nsteps = round(Int, 500.0 / dt),
                               integrator = HeunProjected()).config[1] - analytic)
        @test err(1.0) < 1e-3
        r = err(2.0) / err(1.0)
        @test 3.0 < r < 5.0
    end

    @testset "inactive sites stay bitwise frozen (Zeeman included)" begin
        prob = LLGProblem(H; magmom = 2.0, b_ext = (3.0, 0.0, 0.0))
        e34 = normalize(SVector(0.3, -0.5, 0.8))
        config0 = MC.SpinConfig([e0, e0, e34, e34])
        res = run_llg(prob, config0; dt = 1.0, nsteps = 200)
        @test !H.site_active[3] && !H.site_active[4]
        @test res.config[3] == e34                # no Zeeman on inactive sites
        @test res.config[4] == e34
    end
end
